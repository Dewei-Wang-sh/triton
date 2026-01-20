#ifndef TRITONLLVM_PASSES
#define TRITONLLVM_PASSES

include "mlir/Pass/PassBase.td"

def TritonLLVMVectorize : Pass<"tritonllvm-vectorize", "mlir::ModuleOp"> {
  let summary = "vectorize llvm ir";

  let description = [{
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}


#endif
