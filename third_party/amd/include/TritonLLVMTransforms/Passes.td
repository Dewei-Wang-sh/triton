#ifndef TRITONLLVM_PASSES
#define TRITONLLVM_PASSES

include "mlir/Pass/PassBase.td"

def TritonLLVMVectorize : Pass<"tritonllvm-vectorize", "mlir::ModuleOp"> {
  let summary = "vectorize llvm ir";

  let description = [{
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::ROCDL::ROCDLDialect"];

  let options = [
    Option<"setFP32DenormalMode", "set-fp32-denormal-mode", "bool", "false",
           "Set the FP32 environment from allow-flush-denorm before vectorizing">,
    Option<"allowFlushDenorm", "allow-flush-denorm", "bool", "false",
           "Allow sign-preserving FP32 denormal flushing when setting the environment">
  ];
}


#endif
