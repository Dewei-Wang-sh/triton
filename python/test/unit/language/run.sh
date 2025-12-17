


TRITON_HIP_USE_ASYNC_COPY=1 pytest -s -v test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[bfloat16] &> log.gemm.64x64.mlir

triton-opt -tritonamdgpu-pipeline="use_async_copy=true" tmp.vec8.beforepipeline.mlir  -debug-only="tritonamdgpu-pipeline-expand-loops,tritonamdgpu-pipeline-lower-loops" &> out.vec8.pipeline.mlir
