#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

using namespace mlir;

namespace {

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                    mlir::gpu::Dimension::y,
                                                    mlir::gpu::Dimension::z};
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    Value blockId =
        ::mlir::gpu::GridDimOp::create(rewriter, loc, dims[op.getAxisAsInt()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);
    return success();
  }
};

struct CondBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::CondBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::CondBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterCondBarBlock =
        currentBlock->splitBlock(rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterCondBarBlock);
    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, adaptor.getPred(), trueBlock,
                           afterCondBarBlock);

    // conditional barrier
    rewriter.setInsertionPointToStart(trueBlock);
    ROCDL::SBarrierOp::create(rewriter, loc);
    LLVM::BrOp::create(rewriter, loc, afterCondBarBlock);
    rewriter.eraseOp(op);
    return success();
  }
};

// OR the thread's own elements, then ballot across the wave. The ballot mask is
// wave-uniform (SGPR), so a branch on the result is a scalar branch.
struct WaveAnyOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::WaveAnyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::amdgpu::WaveAnyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> elems =
        unpackLLElements(loc, adaptor.getPred(), rewriter);
    Value any = b.false_val();
    for (Value v : elems)
      any = b.or_(any, v);
    unsigned waveSize = triton::gpu::lookupThreadsPerWarp(rewriter);
    Type maskTy = rewriter.getIntegerType(waveSize);
    Value mask = ROCDL::BallotOp::create(rewriter, loc, maskTy, any);
    Value zero = LLVM::ConstantOp::create(rewriter, loc, maskTy,
                                          rewriter.getIntegerAttr(maskTy, 0));
    rewriter.replaceOp(op, b.icmp_ne(mask, zero));
    return success();
  }
};

} // namespace

void mlir::triton::AMD::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
  patterns.add<CondBarrierOpConversion>(typeConverter, benefit);
  patterns.add<WaveAnyOpConversion>(typeConverter, benefit);
}
