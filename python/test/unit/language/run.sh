TRITON_HIP_USE_ASYNC_COPY=1 pytest -v test_tensor_descriptor.py::test_tensor_descriptor_reshape_matmul[float16] &> log.mlir

triton-opt -tritonamdgpu-pipeline="use_async_copy=true" tmp.beforepipeline.mlir
