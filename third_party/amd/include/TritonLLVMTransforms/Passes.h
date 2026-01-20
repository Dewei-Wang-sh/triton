#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONLLVMTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONLLVMTRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonLLVMTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir::triton {

// Generate the pass class declarations.
#define GEN_PASS_DECL_TRITONLLVMVECTORIZE
#include "TritonLLVMTransforms/Passes.h.inc"

} // namespace mlir::triton

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonLLVMTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONLLVMTRANSFORMS_PASSES_H_
