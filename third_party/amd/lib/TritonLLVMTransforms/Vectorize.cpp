#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

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
bool isGroupConstantPattern(ArrayRef<Value> values, size_t groupSize) {
  if (values.size() % groupSize != 0)
    return false;
  size_t numGroups = values.size() / groupSize;
  for (size_t b = 0; b < numGroups; ++b) {
    ArrayRef<Value> group = values.slice(b * groupSize, groupSize);
    if (!allEqual(group))
      return false;
  }
  return true;
}

// Infer the best sub-vector size
std::pair<size_t, bool> inferSubVectorSize(ArrayRef<Value> values) {
  size_t N = values.size();
  if (N <= 1)
    return {N, false};

  assert(llvm::isPowerOf2_64(N) && "Total size must be power of two");
  int logN = llvm::Log2_64(N);
  for (int i = logN; i >= 1; --i) {
    size_t subSize = size_t{1} << i;
    // Group-constant works for any size (including full vector)
    if (isGroupConstantPattern(values, subSize)) {
      return {subSize, false};
    }
  }
  for (int i = 1; i < logN; ++i) {
    size_t subSize = size_t{1} << i;
    // Repeating pattern only counts if it actually repeats (≥2 times)
    if (isRepeatingPattern(values, subSize)) {
      return {subSize, true};
    }
  }
  return {1, false};
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

Value vectorizeFromScalar(Value scalar, int64_t numElements, Operation *srcOp) {
  OpBuilder b(srcOp);
  Location loc = srcOp->getLoc();
  auto scalarTy = scalar.getType();
  auto vecTy = VectorType::get(numElements, scalarTy);

  /// vectorize scalar constant
  if (auto cst = dyn_cast<LLVM::ConstantOp>(scalar.getDefiningOp())) {
    if (scalarTy.isFloat()) {
      APFloat cstVal(0.0);
      (void)matchPattern(scalar, m_ConstantFloat(&cstVal));
      auto fpAttr = b.getFloatAttr(scalarTy, cstVal);
      auto dense =
          DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), fpAttr);
      auto cstVec = LLVM::ConstantOp::create(b, loc, vecTy, dense);
      return cstVec;
    } else {
      assert(scalarTy.isInteger() &&
             "only support float or integer scalar vectorization");
      APInt cstVal;
      (void)matchPattern(scalar, m_ConstantInt(&cstVal));
      auto intAttr = b.getIntegerAttr(scalarTy, cstVal.getSExtValue());
      auto dense =
          DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), intAttr);
      auto cstVec = LLVM::ConstantOp::create(b, loc, vecTy, dense);
      return cstVec;
    }
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

Value vectorizeFromSubVector(SmallVector<Value> subVectors, Operation *srcOp) {
  assert(!subVectors.empty() && "No sub-vectors provided");
  if (subVectors.size() == 1)
    return subVectors[0];
  assert(llvm::isPowerOf2_64(subVectors.size()) &&
         "Number of sub-vectors must be power of two");
  OpBuilder b(srcOp);
  Location loc = srcOp->getLoc();

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
Value vectorizeFromScalars(SmallVectorImpl<Value> &scalars, Operation *srcOp) {
  OpBuilder b(srcOp);
  auto loc = srcOp->getLoc();
  auto vecType = VectorType::get(scalars.size(), scalars[0].getType());
  Value vector = LLVM::UndefOp::create(b, loc, vecType);
  for (size_t i = 0; i < scalars.size(); ++i) {
    auto idx = LLVM::ConstantOp::create(
        b, loc, b.getI32Type(), b.getI32IntegerAttr(static_cast<int32_t>(i)));
    vector =
        LLVM::InsertElementOp::create(b, loc, vecType, vector, scalars[i], idx);
    // visitedInsert.insert(vector.getDefiningOp());
  }
  return vector;
}

// create llvm op with vectorized src (binary ops)
Value createLLVMVecOpBinary(OpBuilder &b, Location loc, StringRef opName,
                            Value lhs, Value rhs, Type type) {
  // Note: avoid llvm::StringSwitch with create(...), since that eagerly
  // builds all candidate ops. Use explicit string comparisons instead.

  // Integer arithmetic
  if (opName == "add")
    return LLVM::AddOp::create(b, loc, type, lhs, rhs);
  if (opName == "sub")
    return LLVM::SubOp::create(b, loc, type, lhs, rhs);
  if (opName == "mul")
    return LLVM::MulOp::create(b, loc, type, lhs, rhs);
  if (opName == "udiv")
    return LLVM::UDivOp::create(b, loc, type, lhs, rhs);
  if (opName == "sdiv")
    return LLVM::SDivOp::create(b, loc, type, lhs, rhs);
  if (opName == "urem")
    return LLVM::URemOp::create(b, loc, type, lhs, rhs);
  if (opName == "srem")
    return LLVM::SRemOp::create(b, loc, type, lhs, rhs);

  // Bitwise
  if (opName == "and")
    return LLVM::AndOp::create(b, loc, type, lhs, rhs);
  if (opName == "or")
    return LLVM::OrOp::create(b, loc, type, lhs, rhs);
  if (opName == "xor")
    return LLVM::XOrOp::create(b, loc, type, lhs, rhs);
  if (opName == "shl")
    return LLVM::ShlOp::create(b, loc, type, lhs, rhs);
  if (opName == "lshr")
    return LLVM::LShrOp::create(b, loc, type, lhs, rhs);
  if (opName == "ashr")
    return LLVM::AShrOp::create(b, loc, type, lhs, rhs);

  // Floating-point arithmetic
  if (opName == "fadd")
    return LLVM::FAddOp::create(b, loc, type, lhs, rhs);
  if (opName == "fsub")
    return LLVM::FSubOp::create(b, loc, type, lhs, rhs);
  if (opName == "fmul")
    return LLVM::FMulOp::create(b, loc, type, lhs, rhs);
  if (opName == "fdiv")
    return LLVM::FDivOp::create(b, loc, type, lhs, rhs);
  if (opName == "frem")
    return LLVM::FRemOp::create(b, loc, type, lhs, rhs);

  return Value{};
}

// create llvm op with vectorized src (unary ops)
Value createLLVMVecOpUnary(OpBuilder &b, Location loc, StringRef opName,
                           Value operand, Type type) {
  // Same reasoning as above: use explicit comparisons to only build
  // the op we actually need.

  if (opName == "fneg")
    return LLVM::FNegOp::create(b, loc, type, operand);
  if (opName == "trunc")
    return LLVM::TruncOp::create(b, loc, type, operand);
  if (opName == "zext")
    return LLVM::ZExtOp::create(b, loc, type, operand);
  if (opName == "sext")
    return LLVM::SExtOp::create(b, loc, type, operand);
  if (opName == "fpext")
    return LLVM::FPExtOp::create(b, loc, type, operand);
  if (opName == "fptrunc")
    return LLVM::FPTruncOp::create(b, loc, type, operand);
  if (opName == "sitofp")
    return LLVM::SIToFPOp::create(b, loc, type, operand);
  if (opName == "uitofp")
    return LLVM::UIToFPOp::create(b, loc, type, operand);
  if (opName == "fptosi")
    return LLVM::FPToSIOp::create(b, loc, type, operand);
  if (opName == "fptoui")
    return LLVM::FPToUIOp::create(b, loc, type, operand);
  if (opName == "bitcast")
    return LLVM::BitcastOp::create(b, loc, type, operand);
  if (opName == "ptrtoint")
    return LLVM::PtrToIntOp::create(b, loc, type, operand);
  if (opName == "inttoptr")
    return LLVM::IntToPtrOp::create(b, loc, type, operand);

  return Value{};
}

std::tuple<bool, SmallVector<APInt>, SmallVector<int32_t>>
checkPositionSequentialFromZero(SmallVectorImpl<Value> &positions) {
  SmallVector<int32_t> cst;
  SmallVector<APInt> pos;
  for (auto p : positions) {
    APInt cstVal;
    if (!matchPattern(p, m_ConstantInt(&cstVal))) {
      return {false, {}, {}};
    }
    pos.push_back(cstVal);
    cst.push_back(cstVal.getSExtValue());
  }
  return {llvm::equal(cst, llvm::seq<int32_t>(cst.size())), pos, cst};
}

FailureOr<Value> tryVectorizeValues(SmallVector<Value> &values,
                                    Operation *srcOp, bool enforce = true) {
  // srcs can be all same, group same, group repeating, all different
  auto [subSize, isRepeating] = inferSubVectorSize(values);
  llvm::dbgs() << "try to vectorize from: " << values[0] << "\n";
  llvm::dbgs() << "sub size: " << subSize << " in " << values.size() << "\n";
  // all same value
  if (subSize == values.size()) {
    Value vecSrc = vectorizeFromScalar(values[0], values.size(), srcOp);
    llvm::dbgs() << "vector created from scalar: " << vecSrc << "\n";
    return vecSrc;
    // grouped values
  } else if (subSize > 1) {
    SmallVector<SmallVector<Value>> groups;
    groupValuesIntoChunks(values, subSize, groups);
    SmallVector<Value> vecSubVals;
    auto firstSubVec = tryVectorizeValues(groups[0], srcOp, enforce);
    vecSubVals.push_back(*firstSubVec);
    for (unsigned i = 1; i < groups.size(); ++i) {
      if (isRepeating) {
        vecSubVals.push_back(*firstSubVec);
        continue;
      } else {
        auto subVecVal = tryVectorizeValues(groups[i], srcOp, enforce);
        vecSubVals.push_back(*subVecVal);
      }
    }
    Value vecVals = vectorizeFromSubVector(vecSubVals, srcOp);
    llvm::dbgs() << "vector created from subVector: " << vecVals << "\n";
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
  // SmallVector<Attribute> gepCstIndices;
  for (auto val : values) {
    auto op = val.getDefiningOp();
    if (opName != op->getName()) {
      vectorizable = false;
      break;
    } else if (isa<LLVM::ConstantOp>(op)) {
      src0.push_back(val);
    } else if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
      // skip it for now
      vectorizable = false;
      break;
      // src0.push_back(gep.getBase());
      // assert(gep.getIndices().size() == 1 &&
      //        "only support single index gep in vectorization");
      // auto idx = gep.getIndices().front();
      // if (isa<IntegerAttr>(idx))
      //   gepCstIndices.push_back(idx);
      // else
      //   src1.push_back(idx);
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

  OpBuilder b(srcOp);
  auto loc = srcOp->getLoc();
  // vectorize values
  if (isa<LLVM::ConstantOp>(op)) {
    auto [res, pos, cst] = checkPositionSequentialFromZero(src0);
    auto vecTy = VectorType::get(src0.size(), src0[0].getType());
    auto dense = DenseElementsAttr::get(vecTy, llvm::ArrayRef(pos));
    Value cstVec = LLVM::ConstantOp::create(b, loc, vecTy, dense);
    llvm::dbgs() << "vectorized constant: " << cstVec << "\n";
    return cstVec;
  } else if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
    // auto [subSize, isRepeating] = inferSubVectorSize(src0);
    // Value vecBase = vectorizeFromScalar(src0[0], src0.size(), op);
    // Value vecIdx;
    // // indices can be dynamic value or static constant
    // if (!gepCstIndices.empty()) {
    //   assert(gepCstIndices.size() == src0.size() &&
    //          "all gep indices should be constant in vectorization");
    //   vecIdx = LLVM::ConstantOp::create(b, loc,
    //                                  VectorType::get(src0.size(),
    //                                  b.getI32Type()), DenseElementsAttr::get(
    //                                      VectorType::get(src0.size(),
    //                                      b.getI32Type()), gepCstIndices));
    // } else {
    //   assert(src1.size() == src0.size() &&
    //          "all gep indices should be collected in vectorization");
    //   vecIdx = tryVectorizeValues(src1, );
    // }
    // Value vecGep = LLVM::GEPOp::create(b, loc, vecBase.getType(),
    //                                    gep.getElemType() vecBase, vecIdx);
    // llvm::dbgs() << "vectorized gep: " << vecGep << "\n";
    // return vecGep;
  } else if (isa<LLVM::ExtractElementOp>(op)) {
    auto [subSize, isRepeating] = inferSubVectorSize(src0);
    // make sure it's group constant
    if (subSize == 1 || isRepeating)
      return failure();
    SmallVector<SmallVector<Value>> vectorGroups;
    SmallVector<SmallVector<Value>> positionGroups;
    groupValuesIntoChunks(src0, subSize, vectorGroups);
    groupValuesIntoChunks(src1, subSize, positionGroups);
    SmallVector<Value> vecSubVals;
    for (unsigned i = 0; i < vectorGroups.size(); ++i) {
      auto vecGroup = vectorGroups[i];
      auto posGroup = positionGroups[i];
      auto [res, pos, cst] = checkPositionSequentialFromZero(posGroup);
      auto srcVecTy = cast<VectorType>(vecGroup[0].getType());
      auto dstVecTy =
          VectorType::get(vecGroup.size(), srcVecTy.getElementType());
      // full match
      if (res && srcVecTy == dstVecTy) {
        vecSubVals.push_back(vecGroup[0]);
        // partial match
      } else {
        assert(cst.size() == vecGroup.size() && "not all constant");
        Value subVal = LLVM::ShuffleVectorOp::create(
            b, loc, dstVecTy, vecGroup[0], vecGroup[0], cst);
        vecSubVals.push_back(subVal);
      }
    }
    Value vecVals = vectorizeFromSubVector(vecSubVals, srcOp);
    llvm::dbgs() << "vector created from subVector: " << vecVals << "\n";
    return vecVals;
    // llvm::SetVector<Value> uniqueSrc0(src0.begin(), src0.end());
    // assert(uniqueSrc0.size() == 1 &&
    //        "only support all same vector in vectorization");
    // return src0[0];
  } else {
    FailureOr<Value> vecSrc0;
    FailureOr<Value> vecSrc1;
    if (!src0.empty())
      vecSrc0 = tryVectorizeValues(src0, op);
    if (!src1.empty())
      vecSrc1 = tryVectorizeValues(src1, op);

    auto name = opName.stripDialect().str();
    auto dstType = VectorType::get(values.size(), op->getResult(0).getType());
    // unary op
    if (!failed(vecSrc0) && src1.empty()) {
      Value vec = createLLVMVecOpUnary(b, loc, name, *vecSrc0, dstType);
      llvm::dbgs() << "Created LLVM op: " << vec << "\n";
      return vec;
    }
    // binary op
    if (!failed(vecSrc0) && !failed(vecSrc1)) {
      Value vec =
          createLLVMVecOpBinary(b, loc, name, *vecSrc0, *vecSrc1, dstType);
      llvm::dbgs() << "Created LLVM op: " << vec << "\n";
      return vec;
    }
    // port attrs
  }

  if (enforce) {
    llvm::dbgs() << "enforce vectorization fallback to generic\n";
    return vectorizeFromScalars(values, srcOp);
  }
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

LogicalResult
tryVectorizeInsertChain(LLVM::InsertElementOp insert,
                        llvm::SetVector<Operation *> &visitedInsert) {
  llvm::dbgs() << "check insert chain from : " << insert << "\n";
  auto [dst, srcs, positions, ops] = getChainOps(insert);
  visitedInsert.insert(ops.begin(), ops.end());
  if (srcs.size() != insert.getVector().getType().getNumElements() ||
      srcs.size() == 1)
    return failure();
  bool res = std::get<0>(checkPositionSequentialFromZero(positions));
  if (!res)
    return failure();

  llvm::dbgs() << "try to vectorize the insert chain" << "\n";
  FailureOr<Value> vecSrc = tryVectorizeValues(srcs, insert, false);
  if (failed(vecSrc)) {
    llvm::dbgs() << "====================" << "\n";
    llvm::dbgs() << "potential but failed" << "\n";
    llvm::dbgs() << "====================" << "\n";
    return failure();
  }
  dst.replaceAllUsesWith(*vecSrc);
  return success();
}

} // anonymous namespace

class TritonLLVMVectorize
    : public impl::TritonLLVMVectorizeBase<TritonLLVMVectorize> {
private:
  llvm::SetVector<Operation *> visitedInsert;

public:
  using TritonLLVMVectorizeBase::TritonLLVMVectorizeBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Ensure no stale state if this pass instance is reused.
    visitedInsert.clear();

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

    // llvm::dbgs() << "Module after struct to vector convert: " << m << "\n";

    // step 2: iteratively vectorize ops from insertelement chains
    SmallVector<LLVM::InsertElementOp> insertOps;
    auto funcOps = m.getOps<LLVM::LLVMFuncOp>();
    for (auto funcOp : funcOps) {
      funcOp.walk<WalkOrder::PreOrder>(
          [&](LLVM::InsertElementOp op) { insertOps.push_back(op); });
    }
    for (auto insert : insertOps) {
      if (!visitedInsert.contains(insert))
        (void)tryVectorizeInsertChain(insert, visitedInsert);
    }
    // step 3: canonicalize after vectorization
    OpPassManager pm;
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(runPipeline(pm, m)))
      return signalPassFailure();
    // step 4: print constant human readable
  }
};
} // namespace mlir
