# Triton SIMD Path for dense operations (GEMM)

## tritongpu distribute to warp

this pass distribute the thread-block(work-group) workload to the warps

[RFC for this pass](https://github.com/openai/triton/issues/2729)

[test before this pass](https://github.com/yo-yo-hu/triton/blob/distribute/test/TritonGPU/distribute-to-warps.mlir)

[test after this pass](https://github.com/yo-yo-hu/triton/blob/distribute/test/TritonGPU/distribute-to-warps.output.mlir)

## convert tritongpu to spirv simd Intrinsics

this pass directly maps ttg ir that match target size to corresponding simd Intrinsics.

For example, 8x16xf32 = tt.dot 8x16xf16, 16x16xf16 can be directly mapped to a dot Intrinsic.

[pass link](https://github.com/intel/intel-xpu-backend-for-triton/blob/gemm_simd/lib/Conversion/TritonGPUToSPIRV/TritonGPUToVC.cpp)

[test before this pass](https://github.com/intel/intel-xpu-backend-for-triton/blob/gemm_simd/gemm_test/matmul.8x16x1024.mlir)

[test after this pass](https://github.com/intel/intel-xpu-backend-for-triton/blob/gemm_simd/gemm_test/matmul.8x16x1024.spirv.mlir)

### comment about the Intrinsics

VC means Vector Compute which means SIMD

raw_send2_v64i32 is for tt.load 8x16xf16

raw_send2_v128i32 is for tt.load 16x16xf16

dpas2_v128f32 is for tt.load 8x16xf32

raw_sends2_noresult_v64i32is for tt.store 8x16xf16
