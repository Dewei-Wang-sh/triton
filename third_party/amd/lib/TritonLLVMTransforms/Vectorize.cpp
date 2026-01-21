#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "TritonLLVMTransforms/Passes.h"
#include "llvm/ADT/StringSwitch.h"

#define DEBUG_TYPE "tritonllvm-vectorize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_TRITONLLVMVECTORIZE
#include "TritonLLVMTransforms/Passes.h.inc"

namespace {
class ConvertUndef : public OpConversionPattern<LLVM::UndefOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::UndefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (newTy != op.getType()) {
      rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, newTy);
      return success();
    }
    return failure();
  }
};

class ConvertInsertValue : public OpConversionPattern<LLVM::InsertValueOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::InsertValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type newTy = getTypeConverter()->convertType(op.getType());
    if (newTy != op.getType()) {
      auto pos = op.getPosition();
      assert(pos.size() == 1 &&
             "Only single index insertvalue is supported in this pass");
      Value newPos = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                              rewriter.getI32Type(), pos[0]);
      rewriter.replaceOpWithNewOp<LLVM::InsertElementOp>(
          op, newTy, adaptor.getContainer(), adaptor.getValue(), newPos);
      return success();
    }
    return failure();
  }
};

class ConvertExtractValue : public OpConversionPattern<LLVM::ExtractValueOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::ExtractValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type oldTy = op.getContainer().getType();
    Type newTy = getTypeConverter()->convertType(oldTy);
    if (newTy != oldTy) {
      auto pos = op.getPosition();
      assert(pos.size() == 1 &&
             "Only single index insertvalue is supported in this pass");
      Value newPos = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                              rewriter.getI32Type(), pos[0]);
      rewriter.replaceOpWithNewOp<LLVM::ExtractElementOp>(
          op, op.getType(), adaptor.getContainer(), newPos);
      return success();
    }
    return failure();
  }
};

class ConvertBranch : public OpConversionPattern<LLVM::BrOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LLVM::BrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool needsConversion = false;
    for (auto operand : op.getDestOperands()) {
      Type newTy = getTypeConverter()->convertType(operand.getType());
      if (newTy != operand.getType()) {
        needsConversion = true;
        break;
      }
    }

    if (needsConversion) {
      Block *successor = op.getSuccessor();

      // Convert successor block argument types
      TypeConverter::SignatureConversion sigConversion(
          successor->getNumArguments());
      for (unsigned i = 0, e = successor->getNumArguments(); i < e; ++i) {
        Type newType = getTypeConverter()->convertType(
            successor->getArgument(i).getType());
        sigConversion.addInputs(i, newType);
      }
      auto newSuccessor =
          rewriter.applySignatureConversion(successor, sigConversion);

      rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, adaptor.getDestOperands(),
                                              newSuccessor);
      return success();
    }
    return failure();
  }
};

// ------------------------------------------------------------
// Helper functions to check Value group pattern
// ------------------------------------------------------------
// Check if all values in a range are equal
bool allEqual(ArrayRef<Value> vals) {
  if (vals.empty())
    return true;
  Value first = vals[0];
  for (Value v : vals.drop_front()) {
    if (v != first)
      return false;
  }
  return true;
}

// Check if the entire sequence is K repetitions of the first P elements
bool isRepeatingPattern(ArrayRef<Value> values, size_t period) {
  if (values.size() % period != 0)
    return false;
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] != values[i % period])
      return false;
  }
  return true;
}

// Check if sequence is made of constant blocks of size `blockSize`
bool isBlockConstantPattern(ArrayRef<Value> values, size_t blockSize) {
  if (values.size() % blockSize != 0)
    return false;
  size_t numBlocks = values.size() / blockSize;
  for (size_t b = 0; b < numBlocks; ++b) {
    ArrayRef<Value> block = values.slice(b * blockSize, blockSize);
    if (!allEqual(block))
      return false;
  }
  return true;
}

// Infer the best sub-vector size
size_t inferSubVectorSize(ArrayRef<Value> values) {
  size_t N = values.size();
  if (N <= 1)
    return N;

  assert(llvm::isPowerOf2_64(N) && "Total size must be power of two");
  int logN = llvm::Log2_64(N);
  for (int i = logN; i >= 1; --i) {
    size_t subSize = size_t{1} << i;
    // Block-constant works for any size (including full vector)
    if (isBlockConstantPattern(values, subSize)) {
      return subSize;
    }
  }
  for (int i = 1; i < logN; ++i) {
    size_t subSize = size_t{1} << i;
    // Repeating pattern only counts if it actually repeats (≥2 times)
    if (isRepeatingPattern(values, subSize)) {
      return subSize;
    }
  }
  return 1;
}

// Split values into chunks of size `subSize`
bool groupValuesIntoChunks(ArrayRef<Value> values, size_t subSize,
                           SmallVectorImpl<SmallVector<Value>> &groups) {
  if (subSize == 0 || values.size() % subSize != 0) {
    return false;
  }
  groups.clear();
  size_t numGroups = values.size() / subSize;
  groups.reserve(numGroups);
  for (size_t i = 0; i < numGroups; ++i) {
    auto start = values.begin() + i * subSize;
    groups.emplace_back(start, start + subSize);
  }
  return true;
}

Value vectorizeFromScalar(Value scalar, int64_t numElements) {
  OpBuilder b(scalar.getDefiningOp());
  Location loc = scalar.getLoc();
  auto scalarTy = scalar.getType();
  auto vecTy = VectorType::get(numElements, scalarTy);

  /// vectorize scalar constant
  if (auto cst = dyn_cast<LLVM::ConstantOp>(scalar.getDefiningOp())) {
    Attribute attr = cst.getValue();
    auto vecAttr =
        SplatElementsAttr::get(vecTy, attr); // attr.cast<TypedAttr>()
    auto cstVec = LLVM::ConstantOp::create(b, loc, vecTy, vecAttr);
    return cstVec;
  }
  /// vectorize scalar value
  Value undef = LLVM::UndefOp::create(b, loc, vecTy);
  Value zero = LLVM::ConstantOp::create(b, loc, b.getI32Type(), 0);
  // But note: insertelement expects i32/i64 index — use i32 for consistency
  // with LLVM
  Value inserted = LLVM::InsertElementOp::create(b, loc, undef, scalar, zero);
  SmallVector<int32_t> mask(numElements, 0);
  // auto maskVecTy = VectorType::get({numElements}, b.getI32Type());
  // auto maskAttr = DenseIntElementsAttr::get(maskVecTy, maskVals);
  // Value mask = LLVM::ConstantOp::create(b, loc, maskVecTy, maskAttr);
  return LLVM::ShuffleVectorOp::create(b, loc, vecTy, inserted, undef, mask);
}

Value vectorizeFromSubVector(SmallVector<Value> subVectors) {
  assert(!subVectors.empty() && "No sub-vectors provided");
  assert(llvm::isPowerOf2_64(subVectors.size()) &&
         "Number of sub-vectors must be power of two");
  OpBuilder b(subVectors[0].getDefiningOp());
  Location loc = subVectors[0].getDefiningOp()->getLoc();

  // Work on a mutable list
  SmallVector<Value> current = subVectors;

  // Get element type and sub-vector length
  auto vecType = cast<VectorType>(current[0].getType());
  unsigned subLen = vecType.getNumElements();
  Type elementType = vecType.getElementType();

  // Binary tree reduction
  while (current.size() > 1) {
    SmallVector<Value> next;
    for (size_t i = 0; i < current.size(); i += 2) {
      Value left = current[i];
      Value right = current[i + 1];

      // New vector length = 2 * subLen
      unsigned newLen = 2 * subLen;
      auto newVecType = VectorType::get(newLen, elementType);

      SmallVector<int32_t> mask(newLen);
      std::iota(mask.begin(), mask.end(), 0);
      // auto maskAttr = b.getI32ArrayAttr(mask);
      Value shuffled =
          LLVM::ShuffleVectorOp::create(b, loc, newVecType, left, right, mask);
      next.push_back(shuffled);
    }
    current = std::move(next);
    subLen *= 2; // double the segment size
  }

  return current[0]; // final assembled vector
}

// generic vectorize
Value vectorizeFromScalars(SmallVectorImpl<Value> &scalars) {
  OpBuilder b(scalars[0].getDefiningOp());
  auto loc = scalars[0].getLoc();
  auto i32Type = b.getI32Type();
  auto vecType = VectorType::get(scalars.size(), scalars[0].getType());

  Value vector = LLVM::UndefOp::create(b, loc, vecType);
  for (size_t i = 0; i < scalars.size(); ++i) {
    auto idx = LLVM::ConstantOp::create(
        b, loc, b.getI32Type(), b.getI32IntegerAttr(static_cast<int32_t>(i)));
    vector =
        LLVM::InsertElementOp::create(b, loc, vecType, vector, scalars[i], idx);
  }
  return vector;
}

// create llvm op with vectorized src (binary ops)
Value createLLVMVecOpBinary(OpBuilder &b, Location loc, StringRef opName,
                            Value lhs, Value rhs, Type type) {
  return llvm::StringSwitch<Value>(opName)
      // Integer arithmetic
      .Case("add", LLVM::AddOp::create(b, loc, type, lhs, rhs))
      .Case("sub", LLVM::SubOp::create(b, loc, type, lhs, rhs))
      .Case("mul", LLVM::MulOp::create(b, loc, type, lhs, rhs))
      .Case("udiv", LLVM::UDivOp::create(b, loc, type, lhs, rhs))
      .Case("sdiv", LLVM::SDivOp::create(b, loc, type, lhs, rhs))
      .Case("urem", LLVM::URemOp::create(b, loc, type, lhs, rhs))
      .Case("srem", LLVM::SRemOp::create(b, loc, type, lhs, rhs))

      // Bitwise
      .Case("and", LLVM::AndOp::create(b, loc, type, lhs, rhs))
      .Case("or", LLVM::OrOp::create(b, loc, type, lhs, rhs))
      .Case("xor", LLVM::XOrOp::create(b, loc, type, lhs, rhs))
      .Case("shl", LLVM::ShlOp::create(b, loc, type, lhs, rhs))
      .Case("lshr", LLVM::LShrOp::create(b, loc, type, lhs, rhs))
      .Case("ashr", LLVM::AShrOp::create(b, loc, type, lhs, rhs))

      // Floating-point arithmetic
      .Case("fadd", LLVM::FAddOp::create(b, loc, type, lhs, rhs))
      .Case("fsub", LLVM::FSubOp::create(b, loc, type, lhs, rhs))
      .Case("fmul", LLVM::FMulOp::create(b, loc, type, lhs, rhs))
      .Case("fdiv", LLVM::FDivOp::create(b, loc, type, lhs, rhs))
      .Case("frem", LLVM::FRemOp::create(b, loc, type, lhs, rhs))

      .Default(Value{});
}

// create llvm op with vectorized src (unary ops)
Value createLLVMVecOpUnary(OpBuilder &b, Location loc, StringRef opName,
                           Value operand, Type type) {
  return llvm::StringSwitch<Value>(opName)
      .Case("fneg", LLVM::FNegOp::create(b, loc, type, operand))
      .Case("trunc", LLVM::TruncOp::create(b, loc, type, operand))
      .Case("zext", LLVM::ZExtOp::create(b, loc, type, operand))
      .Case("sext", LLVM::SExtOp::create(b, loc, type, operand))
      .Case("fpext", LLVM::FPExtOp::create(b, loc, type, operand))
      .Case("fptrunc", LLVM::FPTruncOp::create(b, loc, type, operand))
      .Case("sitofp", LLVM::SIToFPOp::create(b, loc, type, operand))
      .Case("uitofp", LLVM::UIToFPOp::create(b, loc, type, operand))
      .Case("fptosi", LLVM::FPToSIOp::create(b, loc, type, operand))
      .Case("fptoui", LLVM::FPToUIOp::create(b, loc, type, operand))
      .Case("bitcast", LLVM::BitcastOp::create(b, loc, type, operand))
      .Case("ptrtoint", LLVM::PtrToIntOp::create(b, loc, type, operand))
      .Case("inttoptr", LLVM::IntToPtrOp::create(b, loc, type, operand))

      .Default(Value{});
}

bool checkPositionSequentialFromZero(SmallVectorImpl<Value> &positions) {
  SmallVector<int32_t> pos;
  for (auto p : positions) {
    APInt cstVal;
    if (!matchPattern(p, m_ConstantInt(&cstVal))) {
      return false;
    }
    pos.push_back(cstVal.getSExtValue());
  }
  return llvm::equal(pos, llvm::seq<int32_t>(pos.size()));
}

FailureOr<Value> tryVectorizeValues(SmallVector<Value> &values,
                                    bool enforce = true) {
  // srcs can be all same, group same, group repeating, all different
  auto subSize = inferSubVectorSize(values);
  // all same value
  if (subSize == values.size()) {
    Value vecSrc = vectorizeFromScalar(values[0], values.size());
    return vecSrc;
    // grouped values
  } else if (subSize > 1) {
    SmallVector<SmallVector<Value>> groups;
    groupValuesIntoChunks(values, subSize, groups);
    SmallVector<Value> vecSubVals;
    for (auto group : groups) {
      auto subVecVal = tryVectorizeValues(group, enforce);
      vecSubVals.push_back(*subVecVal);
    }
    Value vecVals = vectorizeFromSubVector(vecSubVals);
    return vecVals;
  }
  assert(subSize == 1 && "unexpected subSize in vectorization");
  // all different values
  // value can be constant, extractElement, binaryOp, unaryOp
  auto op = values[0].getDefiningOp();
  auto opName = op->getName();
  // collect src
  SmallVector<Value> src0;
  SmallVector<Value> src1;
  bool vectorizable = true;
  for (auto src : values) {
    auto op = src.getDefiningOp();
    if (opName != op->getName()) {
      vectorizable = false;
      break;
    } else if (isa<LLVM::ConstantOp>(op)) {
      src0.push_back(src);
    } else if (auto extract = dyn_cast<LLVM::ExtractElementOp>(op)) {
      src0.push_back(extract.getVector());
      src1.push_back(extract.getPosition());
    } else if (isa<LLVM::TruncOp, LLVM::ZExtOp, LLVM::SExtOp>(op)) {
      src0.push_back(op->getOperand(0));
    } else if (isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::UDivOp,
                   LLVM::SDivOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp>(op)) {
      src0.push_back(op->getOperand(0));
      src1.push_back(op->getOperand(1));
    } else {
      assert(0 && "unsupported op in vectorization");
      return failure();
    }
  }

  // early return
  if (!enforce && !vectorizable)
    return failure();

  OpBuilder b(op);
  // vectorize values
  if (isa<LLVM::ConstantOp>(op)) {
    // use the generic form for now
    return vectorizeFromScalars(src0);
  } else if (isa<LLVM::ExtractElementOp>(op)) {
    bool res = checkPositionSequentialFromZero(src1);
    if (!res)
      return failure();
    llvm::SetVector<Value> uniqueSrc0(src0.begin(), src0.end());
    assert(uniqueSrc0.size() == 1 &&
           "only support all same vector in vectorization");
    return src0[0];
  } else {
    FailureOr<Value> vecSrc0;
    FailureOr<Value> vecSrc1;
    if (!src0.empty())
      vecSrc0 = tryVectorizeValues(src0);
    if (!src1.empty())
      vecSrc1 = tryVectorizeValues(src1);

    auto loc = op->getLoc();
    auto name = opName.stripDialect().str();
    auto dstType = VectorType::get(values.size(), op->getResult(0).getType());
    // unary op
    if (!failed(vecSrc0) && src1.empty()) {
      return createLLVMVecOpUnary(b, loc, name, *vecSrc0, dstType);
    }
    // binary op
    if (!failed(vecSrc0) && !failed(vecSrc1)) {
      return createLLVMVecOpBinary(b, loc, name, *vecSrc0, *vecSrc1, dstType);
    }
    // port attrs
  }

  if (enforce)
    return vectorizeFromScalars(values);
  return failure();
}

std::tuple<Value, SmallVector<Value>, SmallVector<Value>,
           SmallVector<Operation *>>
getChainOps(LLVM::InsertElementOp insert) {
  SmallVector<Value> srcs;
  SmallVector<Value> positions;
  SmallVector<Operation *> ops;
  LLVM::InsertElementOp next = insert;
  Value res;
  do {
    srcs.push_back(next.getValue());
    positions.push_back(next.getPosition());
    ops.push_back(next.getOperation());
    res = next.getRes();
  } while (res.hasOneUse() &&
           (next = dyn_cast<LLVM::InsertElementOp>(*res.user_begin())) && next);
  return {res, srcs, positions, ops};
}

LogicalResult tryVectorizeInsertChain(LLVM::InsertElementOp insert) {
  auto [dst, srcs, positions, ops] = getChainOps(insert);
  if (srcs.size() != insert.getVector().getType().getNumElements() ||
      srcs.size() == 1)
    return failure();
  bool res = checkPositionSequentialFromZero(positions);
  if (!res)
    return failure();

  FailureOr<Value> vecSrc = tryVectorizeValues(srcs, false);
  if (failed(vecSrc))
    return failure();
  dst.replaceAllUsesWith(*vecSrc);
  for (auto op : llvm::reverse(ops))
    op->erase();
  return success();
}

} // anonymous namespace

class TritonLLVMVectorize
    : public impl::TritonLLVMVectorizeBase<TritonLLVMVectorize> {
public:
  using TritonLLVMVectorizeBase::TritonLLVMVectorizeBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // step 1: convert struct type to vector type
    mlir::TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](LLVM::LLVMStructType type) -> Type {
      auto body = type.getBody();
      if (body.empty())
        return type;
      llvm::SmallSetVector<Type, 4> uniqueTypes(body.begin(), body.end());
      if (uniqueTypes.size() == 1) {
        return VectorType::get({static_cast<int64_t>(body.size())}, body[0]);
      }
      return type;
    });

    mlir::ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<LLVM::UndefOp, LLVM::InsertValueOp,
                                 LLVM::ExtractValueOp, LLVM::BrOp>(
        [&](Operation *op) { return converter.isLegal(op); });

    RewritePatternSet patterns(context);
    patterns.add<ConvertUndef, ConvertInsertValue, ConvertExtractValue,
                 ConvertBranch>(converter, context);

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      return signalPassFailure();

    // step 2: iteratively vectorize ops
    auto funcOps = m.getOps<LLVM::LLVMFuncOp>();
    for (auto funcOp : funcOps) {
      bool changed = false;
      funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto insert = dyn_cast<LLVM::InsertElementOp>(op)) {
          (void)tryVectorizeInsertChain(insert);
        }
      });
    }
  }
};
} // namespace mlir
