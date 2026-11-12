// RUN: env TRITON_LLVM_VECTORIZE=0 triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=SCALAR
// RUN: env TRITON_LLVM_VECTORIZE=1 triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=VECTOR

// With TRITON_LLVM_VECTORIZE, buffer_load_to_local computes the masked byte
// offsets of all its instructions as one vector (scale to bytes, then select
// out of bounds for masked lanes) and extracts one element per instruction.
// Without it, every instruction gets its own scalar mul + select.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // SCALAR-LABEL: buffer_load_to_local_masked
  // SCALAR-NOT: vector<8xi32>
  // SCALAR-COUNT-8: rocdl.raw.ptr.buffer.load.async.lds
  // VECTOR-LABEL: buffer_load_to_local_masked
  // VECTOR-COUNT-8: llvm.insertelement {{.*}} : vector<8xi32>
  // VECTOR: %[[BYTES:.*]] = llvm.mul {{.*}} : vector<8xi32>
  // VECTOR: %[[MASKED:.*]] = llvm.select {{.*}}, %[[BYTES]], {{.*}} : vector<8xi1>, vector<8xi32>
  // VECTOR-COUNT-8: llvm.extractelement %[[MASKED]]
  // VECTOR-COUNT-8: rocdl.raw.ptr.buffer.load.async.lds
  // VECTOR-NOT: rocdl.raw.ptr.buffer.load.async.lds
  tt.func public @buffer_load_to_local_masked(%arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                %arg2: tensor<32x64xi32, #blocked>,
                                %arg3: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>,
                                %mask: tensor<32x64xi1, #blocked>) {
    %0 = amdg.buffer_load_to_local %arg1[%arg2] mask = %mask into %arg3 : !tt.ptr<f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    tt.return
  }
}
