#include "ReduceScanCommon.h"

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ReduceOpConversion : public ConvertOpToLLVMPattern<triton::ReduceOp> {
public:
  ReduceOpConversion(LLVMTypeConverter &typeConverter,
                     const TargetInfoBase &targetInfo, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::ReduceOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ReduceOpHelper helper(op);
    Location loc = op->getLoc();
    auto accs = unpackInputs(loc, op, adaptor, rewriter);
    unsigned axis = op.getAxis();

    auto *ctx = op.getContext();

    LinearLayout regLl = triton::gpu::toLinearLayout(helper.getSrcTy());
    // Remove broadcasting in registers as SliceLayout removes them
    regLl = regLl.removeZeroBasesAlongDim(str_attr("register"));

    // First reduce all the values along the axis within each thread.
    std::tie(regLl, accs) =
        reduceWithinThreads(op, std::move(regLl), std::move(accs), rewriter);

    // Then reduce across threads within a warp.
    std::tie(regLl, accs) =
        reduceWithinWarps(op, std::move(regLl), std::move(accs), rewriter);

    // reducedRegLaneLayout is used in the AllocationAnalysis to get the size
    // of the scratch space.
    assert(regLl ==
           ReduceOpHelper::reducedRegLaneLayout(helper.getSrcTy(), axis));

    // If we still need to reduce across warps / blocks:
    // Create a temporary layout for reduction within warps.
    // By construction of tmpLl, we will iterate at most 2 times, as the maximum
    // number of warps * blocks is 64 * 16 = 32 * 32.
    // That is, they fit in 2 rounds of warp reductions.
    // Moreover, if we do two rounds, getInterWarpReductionLayout will make sure
    // that the first one does not cross CTAs.
    auto kAxis = *(regLl.getOutDimNames().begin() + axis);
    auto kBlock = StringAttr::get(ctx, "block");
    bool lastCvtCrossesCTAs = false;
    int i = 0;
    while (regLl.getOutDimSize(kAxis) != 1) {
      LinearLayout tmpLl =
          ReduceOpHelper::getInterWarpReductionLayout(regLl, axis);

      // Emit a barrier if we are reusing the shmem
      if (i > 0) {
        sync(rewriter, loc, lastCvtCrossesCTAs, op);
      }
      accs = convertLayoutValues(loc, rewriter, op, regLl, tmpLl, accs,
                                 getTypeConverter(), targetInfo);
      lastCvtCrossesCTAs = !mlir::isCvtDimSync(regLl, tmpLl, kBlock);

      std::tie(regLl, accs) =
          reduceWithinWarps(op, std::move(tmpLl), std::move(accs), rewriter);
      ++i;
    }
    assert(i <= 2 && "expected at most 2 rounds of warp reductions");
    // Remove the axis dimension, which at this point is of size 1.
    regLl = removeStandardDim(regLl, axis);

    // Convert to output layout if we didn't fit the warp bases within zero
    // bases in the tmpLl
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op.getResult()[0].getType())) {
      auto outputLayout = triton::gpu::toLinearLayout(resultTy);
      if (regLl != outputLayout) {
        // Reuse the shmem
        sync(rewriter, loc, lastCvtCrossesCTAs, op);
        accs = convertLayoutValues(loc, rewriter, op, regLl, outputLayout, accs,
                                   getTypeConverter(), targetInfo);
      }
    }

    packResults(op, accs, rewriter);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;

  // Reduce values using a tree of the given arity. Arity=3 generates
  // combine(combine(a, b), c) groups that LLVM folds into ternary
  // instructions (e.g. v_maximum3_f32 on AMD).
  SmallVector<Value> treeReduce(Location loc,
                                ConversionPatternRewriter &rewriter,
                                Region &combineOp,
                                SmallVector<SmallVector<Value>> values,
                                unsigned arity) const {
    assert(!values.empty() && arity >= 2);
    while (values.size() > 1) {
      SmallVector<SmallVector<Value>> next;
      for (size_t i = 0; i < values.size(); i += arity) {
        size_t remaining = values.size() - i;
        size_t groupSize = std::min(static_cast<size_t>(arity), remaining);
        if (groupSize == 1) {
          next.push_back(std::move(values[i]));
        } else {
          SmallVector<Value> acc = std::move(values[i]);
          for (size_t j = 1; j < groupSize; ++j)
            accumulate(loc, rewriter, combineOp, acc, values[i + j]);
          next.push_back(std::move(acc));
        }
      }
      values = std::move(next);
    }
    return values.front();
  }

  void accumulate(Location loc, ConversionPatternRewriter &rewriter,
                  Region &combineOp, SmallVector<Value> &acc, ValueRange cur,
                  Value pred = {}) const {
    auto results = applyCombineOp(loc, rewriter, combineOp, acc, cur, pred);
    if (acc.size() < results.size()) {
      acc.resize(results.size());
    }
    for (unsigned i = 0; i < acc.size(); ++i) {
      acc[i] = results[i];
    }
  }

  SmallVector<SmallVector<Value>>
  unpackInputs(Location loc, triton::ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const {
    auto operands = adaptor.getOperands();
    SmallVector<SmallVector<Value>> srcValues(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      srcValues[i] = unpackUniqueTensorElements(loc, operands[i], rewriter);
    }
    return srcValues;
  }

  void sync(ConversionPatternRewriter &rewriter, Location loc, bool crossCTA,
            Operation *sourceOp) const {
    if (crossCTA) {
      targetInfo.clusterBarrier(loc, rewriter, sourceOp);
    } else {
      targetInfo.barrier(loc, rewriter, triton::gpu::AddrSpace::Local);
    }
  }

  void packVectorized(SmallVector<SmallVector<Value>> &accs,
                      ConversionPatternRewriter &rewriter) const {
    auto loc = accs.front().front().getLoc();
    for (auto &acc : accs) {
      SmallVector<Value> packedAcc;
      for (unsigned reg = 0; reg < acc.size(); reg += 2) {
        auto vector = packLLVector(loc, {acc[reg], acc[reg + 1]}, rewriter);
        packedAcc.emplace_back(std::move(vector));
      }
      acc = std::move(packedAcc);
    }
  }

  using InThreadVectorizeOpKind = ReduceOpHelper::InThreadVectorizeOpKind;

  template <typename Op>
  static Value createBinaryOp(OpBuilder &builder, Location loc, Value lhs,
                              Value rhs) {
    return Op::create(builder, loc, lhs.getType(), lhs, rhs);
  }

  Value createInThreadVectorizedCombineOp(OpBuilder &builder, Location loc,
                                          InThreadVectorizeOpKind kind,
                                          Value lhs, Value rhs) const {
    using CreateOpFn = Value (*)(OpBuilder &, Location, Value, Value);
    static const DenseMap<InThreadVectorizeOpKind, CreateOpFn> builders = {
        {InThreadVectorizeOpKind::AddF, createBinaryOp<LLVM::FAddOp>},
        {InThreadVectorizeOpKind::MulF, createBinaryOp<LLVM::FMulOp>},
        {InThreadVectorizeOpKind::MinNumF, createBinaryOp<LLVM::MinNumOp>},
        {InThreadVectorizeOpKind::MaxNumF, createBinaryOp<LLVM::MaxNumOp>},
        {InThreadVectorizeOpKind::MinimumF, createBinaryOp<LLVM::MinimumOp>},
        {InThreadVectorizeOpKind::MaximumF, createBinaryOp<LLVM::MaximumOp>},
        {InThreadVectorizeOpKind::AddI, createBinaryOp<LLVM::AddOp>},
        {InThreadVectorizeOpKind::MulI, createBinaryOp<LLVM::MulOp>},
        {InThreadVectorizeOpKind::MinSI, createBinaryOp<LLVM::SMinOp>},
        {InThreadVectorizeOpKind::MaxSI, createBinaryOp<LLVM::SMaxOp>},
        {InThreadVectorizeOpKind::MinUI, createBinaryOp<LLVM::UMinOp>},
        {InThreadVectorizeOpKind::MaxUI, createBinaryOp<LLVM::UMaxOp>},
    };
    auto createOp = builders.lookup(kind);
    if (!createOp)
      llvm::report_fatal_error("Unsupported in-thread vectorize op kind");
    return createOp(builder, loc, lhs, rhs);
  }

  std::unique_ptr<Region>
  createVectorCombineRegion(triton::ReduceOp op,
                            bool supportBitwidth32Elementwise) const {
    auto kind = ReduceOpHelper(op).getInThreadVectorizeOpKind(
        targetInfo.supportBitwidth16Elementwise(),
        supportBitwidth32Elementwise);
    if (kind == InThreadVectorizeOpKind::None)
      return nullptr;

    auto loc = op.getLoc();
    auto vecTy = vec_ty(op.getInputTypes().front().getElementType(), 2);
    auto storage = std::make_unique<Region>();
    auto *block = new Block();
    storage->push_back(block);
    Value lhs = block->addArgument(vecTy, loc);
    Value rhs = block->addArgument(vecTy, loc);

    OpBuilder builder(op.getContext());
    builder.setInsertionPointToStart(block);
    Value result =
        createInThreadVectorizedCombineOp(builder, loc, kind, lhs, rhs);
    triton::ReduceReturnOp::create(builder, loc, ValueRange{result});
    return storage;
  }

  void unpackVectorized(Location loc, SmallVector<SmallVector<Value>> &accs,
                        ConversionPatternRewriter &rewriter,
                        Region *reduction) const {
    for (auto &acc : accs) {
      SmallVector<Value> unpacked;
      for (Value val : acc) {
        auto elems = unpackLLVector(loc, val, rewriter);
        assert(elems.size() == 2 && "expected a 2-lane packed vector");
        if (reduction) {
          SmallVector<Value> cur = {elems[0]};
          accumulate(loc, rewriter, *reduction, cur, {elems[1]});
          unpacked.emplace_back(cur[0]);
        } else {
          unpacked.emplace_back(elems[0]);
          unpacked.emplace_back(elems[1]);
        }
      }
      acc = std::move(unpacked);
    }
  }

  template <typename ReductionOp>
  static Value
  createVectorReduction(Location loc, ConversionPatternRewriter &rewriter,
                        Operation *combiner, ArrayRef<Value> values) {
    Type elementType = values.front().getType();
    Value vector = packLLVector(loc, values, rewriter);
    if constexpr (std::is_same_v<ReductionOp, LLVM::vector_reduce_fadd> ||
                  std::is_same_v<ReductionOp, LLVM::vector_reduce_fmul> ||
                  std::is_same_v<ReductionOp, LLVM::vector_reduce_fmin> ||
                  std::is_same_v<ReductionOp, LLVM::vector_reduce_fmax> ||
                  std::is_same_v<ReductionOp, LLVM::vector_reduce_fminimum> ||
                  std::is_same_v<ReductionOp, LLVM::vector_reduce_fmaximum>) {
      auto flags = arith::convertArithFastMathFlagsToLLVM(
          cast<arith::ArithFastMathInterface>(combiner)
              .getFastMathFlagsAttr()
              .getValue());
      if constexpr (std::is_same_v<ReductionOp, LLVM::vector_reduce_fadd> ||
                    std::is_same_v<ReductionOp, LLVM::vector_reduce_fmul>) {
        // Triton reductions already permit reassociation. Use -0.0 for addition
        // so an all-negative-zero group retains its sign without assuming nsz.
        flags = flags | LLVM::FastmathFlags::reassoc;
        double identity =
            std::is_same_v<ReductionOp, LLVM::vector_reduce_fadd> ? -0.0 : 1.0;
        Value start = LLVM::ConstantOp::create(
            rewriter, loc, elementType,
            rewriter.getFloatAttr(elementType, identity));
        return ReductionOp::create(rewriter, loc, elementType, start, vector,
                                   flags);
      } else {
        return ReductionOp::create(rewriter, loc, elementType, vector, flags);
      }
    } else {
      return ReductionOp::create(rewriter, loc, elementType, vector);
    }
  }

  // Only bypass a region that consists of a builtin combiner and its return.
  // getSingleCombiner also accepts regions with extra operations (e.g. prints),
  // which must still execute through the ordinary combine-region path.
  Value tryVectorReduction(triton::ReduceOp op, ArrayRef<Value> values,
                           ConversionPatternRewriter &rewriter) const {
    Operation *combiner = op.getSingleCombiner();
    if (!combiner ||
        !llvm::hasSingleElement(op.getCombineOp().front().without_terminator()))
      return {};

    auto loc = op.getLoc();
    return llvm::TypeSwitch<Operation *, Value>(combiner)
        .Case<arith::AddIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_add>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MulIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_mul>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::AndIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_and>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::OrIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_or>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::XOrIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_xor>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MinSIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_smin>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MaxSIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_smax>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MinUIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_umin>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MaxUIOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_umax>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::AddFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fadd>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MulFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fmul>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MinNumFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fmin>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MaxNumFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fmax>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MinimumFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fminimum>(
              loc, rewriter, combiner, values);
        })
        .Case<arith::MaximumFOp>([&](auto) {
          return createVectorReduction<LLVM::vector_reduce_fmaximum>(
              loc, rewriter, combiner, values);
        })
        .Default([](Operation *) { return Value{}; });
  }

  // Experimental full-width reduction; keep the upstream packed/tree lowering
  // in reduceWithinThreads unchanged when TRITON_LLVM_VECTORIZE is disabled.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreadsVectorized(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    // Keep every local group contiguous, including when independent outputs
    // were originally interleaved in the register layout.
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, false);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    Operation *combinerOp = op.getSingleCombiner();
    unsigned arity =
        combinerOp ? targetInfo.getReductionTreeArity(combinerOp) : 2;

    Region &combineRegion = op.getCombineOp();
    // Extra operations in the combiner region may have observable effects.
    bool isPureBooleanCombiner =
        isa_and_nonnull<arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                        arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                        arith::MaxUIOp>(combinerOp) &&
        combinerOp->getResult(0).getType().isInteger(1) &&
        llvm::hasSingleElement(combineRegion.front().without_terminator());
    bool usePredicateChains =
        isPureBooleanCombiner && targetInfo.preferThreePredicateChains();

    // Prefer a full-width vector reduction for a simple builtin combiner.
    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      if (numOperands == 1 && !usePredicateChains) {
        auto values = ArrayRef<Value>(accs.front()).slice(regBase, axisPack);
        if (Value result = tryVectorReduction(op, values, rewriter)) {
          reduced.front().push_back(result);
          continue;
        }
      }
      // Custom and tuple combiners retain the original tree reduction.
      // Transpose from [opIdx][reg] into [reg][opIdx]
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        if (usePredicateChains && i >= 3)
          accumulate(loc, rewriter, combineRegion, vals[i % 3], cur);
        else
          vals.push_back(std::move(cur));
      }
      auto acc =
          treeReduce(loc, rewriter, op.getCombineOp(), std::move(vals), arity);
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Update the layout, killing the axis bases along registers.
    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
    layout = layout.removeZeroBasesAlongDim(kReg);
    return {std::move(layout), std::move(accs)};
  }

  // Reduce along the op axis for elements that are in the same thread. The
  // accumulated value is stored in accs.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinThreads(triton::ReduceOp op, LinearLayout layout,
                      SmallVector<SmallVector<Value>> accs,
                      ConversionPatternRewriter &rewriter) const {
    if (triton::tools::getBoolEnv("TRITON_LLVM_VECTORIZE"))
      return reduceWithinThreadsVectorized(op, std::move(layout),
                                           std::move(accs), rewriter);

    auto *ctx = op.getContext();
    auto loc = op.getLoc();
    unsigned axis = op.getAxis();
    auto kReg = str_attr("register");
    auto linearAttr = triton::gpu::LinearEncodingAttr::get(ctx, layout);
    auto basesPerDim = linearAttr.basesPerDim(kReg, /*skipBroadcast=*/true);
    unsigned axisPack = basesPerDim[axis];
    if (axisPack == 1) {
      return {std::move(layout), std::move(accs)};
    }

    // Create the vectorized region if needed.
    auto vectorCombineRegion =
        axisPack >= 4 ? createVectorCombineRegion(
                            op, targetInfo.supportBitwidth32Elementwise())
                      : nullptr;
    bool vectorize = vectorCombineRegion != nullptr;

    // Bring the registers that move the axis to the front.
    auto perm = ReduceOpHelper::moveAxisBasesToFront(layout, axis, vectorize);
    if (!perm.isIdentity()) {
      layout = perm.apply(layout);
      for (auto &vals : accs) {
        vals = perm.apply(vals);
      }
    }

    // Pack the inputs into vector values.
    if (vectorize)
      packVectorized(accs, rewriter);

    // If we pack along the reduction axis, we need to process half the
    // registers.
    const auto &regBases = layout.getBases().lookup(kReg);
    bool packAlongAxis = vectorize && regBases.front()[axis] != 0;
    if (packAlongAxis)
      axisPack /= 2;

    Region &combineRegion =
        vectorCombineRegion ? *vectorCombineRegion : op.getCombineOp();

    Operation *combinerOp = vectorCombineRegion
                                ? &vectorCombineRegion->front().front()
                                : op.getSingleCombiner();
    unsigned arity =
        combinerOp ? targetInfo.getReductionTreeArity(combinerOp) : 2;

    // Extra operations in the combiner region may have observable effects.
    bool isPureBooleanCombiner =
        isa_and_nonnull<arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                        arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                        arith::MaxUIOp>(combinerOp) &&
        combinerOp->getResult(0).getType().isInteger(1) &&
        llvm::hasSingleElement(combineRegion.front().without_terminator());
    bool usePredicateChains =
        isPureBooleanCombiner && targetInfo.preferThreePredicateChains();

    // Perform a tree reduction
    unsigned numOperands = accs.size();
    SmallVector<SmallVector<Value>> reduced(numOperands);
    unsigned regs = accs.front().size();
    for (unsigned regBase = 0; regBase < regs; regBase += axisPack) {
      // Transpose from [opIdx][reg] into [reg][opIdx]
      SmallVector<SmallVector<Value>> vals;
      for (unsigned i = 0; i < axisPack; ++i) {
        SmallVector<Value> cur(numOperands);
        for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
          cur[opIdx] = accs[opIdx][regBase + i];
        }
        if (usePredicateChains && i >= 3)
          accumulate(loc, rewriter, combineRegion, vals[i % 3], cur);
        else
          vals.push_back(std::move(cur));
      }
      auto acc =
          treeReduce(loc, rewriter, combineRegion, std::move(vals), arity);
      for (unsigned opIdx = 0; opIdx < numOperands; ++opIdx) {
        reduced[opIdx].push_back(acc[opIdx]);
      }
    }
    accs = std::move(reduced);

    // Unpack the vector values into the accumulator values.
    // Reduce one last time via the scalar combine op if we packed along the
    // axis.
    if (vectorize) {
      Region *reduceAfterUnpacking =
          packAlongAxis ? &op.getCombineOp() : nullptr;
      unpackVectorized(loc, accs, rewriter, reduceAfterUnpacking);
    }

    // Update the layout, killing the axis bases along registers.
    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, axis, kReg);
    layout = layout.removeZeroBasesAlongDim(kReg);
    return {std::move(layout), std::move(accs)};
  }

  // Reduce across threads within each warp.
  std::pair<LinearLayout, SmallVector<SmallVector<Value>>>
  reduceWithinWarps(triton::ReduceOp op, LinearLayout layout,
                    SmallVector<SmallVector<Value>> accs,
                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = op.getContext();
    auto kLane = str_attr("lane");
    const auto &laneBases = layout.getBases().lookup(kLane);
    unsigned reduceLaneIdMask = 0;
    unsigned broadcastLaneIdMask = 0;
    for (unsigned bit = 0; bit < laneBases.size(); ++bit) {
      if (laneBases[bit][op.getAxis()] != 0) {
        reduceLaneIdMask |= 1u << bit;
      } else if (llvm::all_of(laneBases[bit],
                              [](int32_t x) { return x == 0; })) {
        broadcastLaneIdMask |= 1u << bit;
      }
    }
    if (reduceLaneIdMask == 0) {
      return {std::move(layout), std::move(accs)};
    }

    warpReduce(op, reduceLaneIdMask, broadcastLaneIdMask, accs, rewriter);

    layout = ReduceOpHelper::zeroBasesAlongDimAndReorder(layout, op.getAxis(),
                                                         kLane);
    return {std::move(layout), std::move(accs)};
  }

  void warpReduce(triton::ReduceOp op, unsigned reduceLaneIdMask,
                  unsigned broadcastLaneIdMask,
                  SmallVector<SmallVector<Value>> &accs,
                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto vectorCombineRegion =
        createVectorCombineRegion(op, /*supportBitwidth32Elementwise=*/false);
    unsigned regs = accs.front().size();
    for (unsigned reg = 0; reg < regs; ++reg) {
      SmallVector<Value> acc(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        acc[i] = accs[i][reg];
      }
      if (!targetInfo.warpReduce(rewriter, loc, acc, op, reduceLaneIdMask,
                                 broadcastLaneIdMask)) {
        // Shuffle pairs of independent outputs in one 32-bit register.
        if (vectorCombineRegion && reg + 1 < regs) {
          for (unsigned i = 0; i < op.getNumOperands(); ++i)
            acc[i] = packLLVector(loc, {acc[i], accs[i][reg + 1]}, rewriter);
          shuffleAndCombine(op, reduceLaneIdMask, acc, *vectorCombineRegion,
                            rewriter);
          for (unsigned i = 0; i < op.getNumOperands(); ++i) {
            auto values = unpackLLVector(loc, acc[i], rewriter);
            accs[i][reg] = values[0];
            accs[i][reg + 1] = values[1];
          }
          ++reg;
          continue;
        }
        shuffleAndCombine(op, reduceLaneIdMask, acc, op.getCombineOp(),
                          rewriter);
      }
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        accs[i][reg] = acc[i];
      }
    }
  }

  void shuffleAndCombine(triton::ReduceOp op, unsigned reduceLaneIdMask,
                         SmallVector<Value> &acc, Region &combineRegion,
                         ConversionPatternRewriter &rewriter) const {
    // No reduction to do
    if (reduceLaneIdMask == 0)
      return;
    auto moduleOp = op->getParentOfType<ModuleOp>();
    unsigned warpSize =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    assert(reduceLaneIdMask < warpSize &&
           "expected reduce lane ID mask to be strictly less than warp size");
    // Not that it matters a lot, but a more reasonable iteration order would be
    // from bit 0 to bit llvm::Log2_32(warpSize) - 1. Changing this breaks a ton
    // of bitwise comparisons so we stick with the legacy inverse order
    for (int bit = llvm::Log2_32(warpSize) - 1; bit >= 0; --bit) {
      unsigned mask = 1u << bit;
      if ((reduceLaneIdMask & mask) == 0)
        continue;
      SmallVector<Value> shfl(op.getNumOperands());
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
        Value value = acc[i];
        if (auto ty = dyn_cast<VectorType>(value.getType())) {
          assert(ty.getNumElements() * ty.getElementTypeBitWidth() == 32);
          value = b.bitcast(value, i32_ty);
        }
        shfl[i] = targetInfo.shuffleXor(rewriter, op.getLoc(), value, mask);
        shfl[i] = b.bitcast(shfl[i], acc[i].getType());
      }
      accumulate(op.getLoc(), rewriter, combineRegion, acc, shfl);
    }
    if (!op.isCommutative()) {
      // The lane with all reduction bits clear combines each subtree in the
      // same order. Broadcast its result instead of ordering both operands at
      // every step of the tree.
      auto b = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      Value leader = b.and_(getLaneId(rewriter, op.getLoc()),
                            b.i32_val(~reduceLaneIdMask));
      for (Value &value : acc)
        value = targetInfo.shuffleIdx(rewriter, op.getLoc(), value, leader);
    }
  }

  // Pack the accumulator values and replace the reduce op with the result.
  void packResults(triton::ReduceOp op, SmallVector<SmallVector<Value>> &accs,
                   ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    SmallVector<Value> results(op.getNumOperands());
    for (unsigned i = 0; i < op.getNumOperands(); ++i) {
      results[i] =
          packUniqueTensorElements(loc, getTypeConverter(), accs[i], rewriter,
                                   op.getResult()[i].getType());
    }
    rewriter.replaceOp(op, results);
  }
};
} // namespace

void mlir::triton::populateReduceOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ReduceOpConversion>(typeConverter, targetInfo, benefit);
}
