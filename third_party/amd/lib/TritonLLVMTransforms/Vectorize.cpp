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

#define DEBUG_TYPE "triton-vectorize-llvm"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace mlir{

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
      TypeConverter::SignatureConversion sigConversion(successor->getNumArguments());
      for (unsigned i = 0, e = successor->getNumArguments(); i < e; ++i) {
        Type newType = getTypeConverter()->convertType(successor->getArgument(i).getType());
        sigConversion.addInputs(i, newType);
      }
      auto newSuccessor = rewriter.applySignatureConversion(successor, sigConversion);
      
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(op, adaptor.getDestOperands(),
                                               newSuccessor);
      return success();
    }
    return failure();
  }
};

// #include "TritonCombine.inc"

} // anonymous namespace

class TritonLLVMVectorize
    : public impl::TritonLLVMVectorizeBase<TritonLLVMVectorize> {
public:
  using TritonLLVMVectorizeBase::TritonLLVMVectorizeBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // step 0: convert struct type to vector type
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
    
    // step 1: iteratively vectorize ops
  }
};
} // namespace mlir