#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUMatchTargetSizePass
    : public TritonGPUMatchTargetSizeBase<TritonGPUMatchTargetSizePass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    /// preprocess: remove all the encoding attr
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!llvm::any_of(op->getResultTypes(), [&](Type type) {
            if (isa<RankedTensorType>(type)) {
              return true;
            } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
              auto pointeeType = ptrType.getPointeeType();
              return isa<RankedTensorType>(pointeeType) ? true : false;
            } else {
              return false;
            }
          }))
        ;
      else if (auto forOp = dyn_cast<scf::ForOp>(op))
        transformScfForOp(forOp);
      else if (auto cstOp = dyn_cast<arith::ConstantOp>(op))
        transformArithConstantOp(cstOp);
      else
        transformGenericOp(op);
      return WalkResult::advance();
    });

    m.dump();

    /// transform op to match target llvm/spirv size
    // for dot split k, n to make the inner register contiguous
    for (auto func : m.getOps<tt::FuncOp>()) {
      // fixme: handle load/store later
      func.walk([&](tt::DotOp dot) { splitDotOp(dot); });
    }
  }

private:
  Type convertType(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      if (tensorType.getEncoding())
        return RankedTensorType::get(tensorType.getShape(),
                                     tensorType.getElementType());

    } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      auto newType = convertType(ptrType.getPointeeType());
      auto newPtrType =
          tt::PointerType::get(newType, ptrType.getAddressSpace());
      return newPtrType;
    }
    return type;
  }
  void transformScfForOp(scf::ForOp op) {
    auto body = op.getBody();
    for (auto [lhs, rhs] :
         llvm::zip(body->getArguments().drop_front(1), op.getInitArgs()))
      lhs.setType(rhs.getType());
    for (auto result : op->getResults()) {
      result.setType(convertType(result.getType()));
    }
    return;
  }
  void transformArithConstantOp(arith::ConstantOp op) {
    auto newType = convertType(op.getType());
    auto value = cast<DenseElementsAttr>(op.getValue());
    value = value.resizeSplat(newType.cast<ShapedType>());
    OpBuilder b(op);
    auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();
    return;
  }
  void transformGenericOp(Operation *op) {
    for (auto result : op->getResults()) {
      result.setType(convertType(result.getType()));
    }
    // updateRootInplace
    return;
  }
  Operation *getDefiningOp(Value val) {
    if (auto op = val.getDefiningOp()) {
      return op;
    } else if (auto arg = dyn_cast<BlockArgument>(val)) {
      auto ownerOp = arg.getOwner()->getParentOp();
      // support scf ForOp for now
      auto forOp = cast<scf::ForOp>(ownerOp);
      auto init = forOp.getInits()[arg.getArgNumber() - 1];
      return getDefiningOp(init);
    } else {
      assert(0 && "add more support");
      return nullptr;
    }
  }
  void splitDotOp(tt::DotOp dot) {
    assert(dotSize.size() == 3 && "target-size should have m, n ,k");
    auto aType = dot.getA().getType().cast<RankedTensorType>();
    auto bType = dot.getB().getType().cast<RankedTensorType>();
    auto cType = dot.getC().getType().cast<RankedTensorType>();
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto cShape = cType.getShape();
    auto m = aShape[0];
    auto n = bShape[1];
    auto k = aShape[1];
    auto mStep = dotSize[0];
    auto nStep = dotSize[1];
    auto kStep = dotSize[2];
    OpBuilder b(dot);
    auto loc = dot.getLoc();
    auto packValues = [&](ArrayRef<int64_t> values) {
      SmallVector<OpFoldResult> newValues = llvm::to_vector<4>(
          llvm::map_range(values, [&](int64_t v) -> OpFoldResult {
            return b.getI64IntegerAttr(v);
          }));
      return newValues;
    };
    auto getSubC = [&](int mm, int nn) -> Value {
      auto defOp = getDefiningOp(dot.getC());
      if (auto cst = dyn_cast<arith::ConstantOp>(defOp)) {
        auto subType =
            RankedTensorType::get({mStep, nStep}, cType.getElementType());
        auto val = cast<DenseElementsAttr>(cst.getValue());
        val = val.resizeSplat(subType);
        auto subCst = b.create<arith::ConstantOp>(loc, subType, val);
        return subCst;
      } else {
        auto subC = b.create<tensor::ExtractSliceOp>(
            loc, dot.getC(), packValues({mm, nn}), packValues({mStep, nStep}),
            packValues({1, 1}));
        return subC;
      }
    };
    Value newC = b.create<tensor::EmptyOp>(loc, ArrayRef({m, n}),
                                           cType.getElementType());

    // n first, so that we can use larger store
    for (auto nn = 0; nn < n; nn += nStep) {
      for (auto mm = 0; mm < m; mm += mStep) {
        Value subC = getSubC(mm, nn);
        for (auto kk = 0; kk < k; kk += kStep) {
          // decide get {mm,kk} from which load slice
          // which is mm/mLoad, kk/kStep
          auto subA = b.create<tensor::ExtractSliceOp>(
              loc, dot.getA(), packValues({mm, kk}), packValues({mStep, kStep}),
              packValues({1, 1}));
          auto subB = b.create<tensor::ExtractSliceOp>(
              loc, dot.getA(), packValues({kk, nn}), packValues({kStep, nStep}),
              packValues({1, 1}));
          subC =
              b.create<tt::DotOp>(loc, subA, subB, subC, dot.getAllowTF32Attr(),
                                  dot.getMaxNumImpreciseAccAttr());
          subA.dump();
          subB.dump();
          subC.dump();
        }
        newC = b.create<tensor::InsertSliceOp>(
            loc, subC, newC, packValues({mm, nn}), packValues({mStep, nStep}),
            packValues({1, 1}));
        newC.dump();
      }
    }
    dot->replaceAllUsesWith(newC.getDefiningOp());
    dot->erase();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUMatchTargetSizePass() {
  return std::make_unique<TritonGPUMatchTargetSizePass>();
}

// for kk -> k kStep
// for mm -> m  mLoadStep
//    loadA.push_back()

// for nn -> n nStep
// for kk -> k kLoadStep
//    loadB.push_back()

// for nn -> n nStep
// for mm -> m mStep
//   subC
//   for kk -> k kStep
//     subA = extract_from loadA[(kk/kStep) *(m/mLoadStep) + (mm/mLoadStep)],
//     slice_offset = [mm, 0]
//     subB = extract_from loadB[(nn/nStep) *(k/kLoadStep) + (kk/kLoadStep)],
//     slice_offset = [kk, 0]
//     subC = subA * subB + subC
//   if kk == kLoadStep - kStep