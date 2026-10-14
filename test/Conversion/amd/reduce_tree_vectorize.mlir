// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a -cse | FileCheck %s --check-prefix=LOCAL
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 -cse | FileCheck %s --check-prefix=LOCAL
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 -cse | FileCheck %s --check-prefix=LOCAL
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 -cse | FileCheck %s --check-prefix=LOCAL
// RUN: triton-opt %s -split-input-file --tritonllvm-vectorize | FileCheck %s --check-prefix=LLVM-VECTORIZE --implicit-check-not=builtin.unrealized_conversion_cast

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // LOCAL-LABEL: reduce_f16
  // LOCAL: "llvm.intr.vector.reduce.fadd"(%{{.*}}, %{{.*}}) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f16, vector<4xf16>) -> f16
  tt.func public @reduce_f16(%arg0: tensor<1x256xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x256xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // LOCAL-LABEL: reduce_f32
  // LOCAL: "llvm.intr.vector.reduce.fadd"(%{{.*}}, %{{.*}}) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<4xf32>) -> f32
  tt.func public @reduce_f32(%arg0: tensor<1x256xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x256xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}

// -----

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // LOCAL-LABEL: reduce_f16_vector_reduce
  // LOCAL: "llvm.intr.vector.reduce.fadd"(%{{.*}}, %{{.*}}) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f16, vector<4xf16>) -> f16
  tt.func public @reduce_f16_vector_reduce(%arg0: tensor<1x128xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x128xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // LOCAL-LABEL: reduce_f32_vector_reduce
  // LOCAL: "llvm.intr.vector.reduce.fadd"(%{{.*}}, %{{.*}}) <{fastmathFlags = #llvm.fastmath<reassoc>}> : (f32, vector<4xf32>) -> f32
  tt.func public @reduce_f32_vector_reduce(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // Full local reductions are left for LLVM to select the target tree.

  // LOCAL-LABEL: reduce_maximum_f32_vector_reduce
  // LOCAL: llvm.intr.vector.reduce.fmaximum(%{{.*}}) : (vector<4xf32>) -> f32
  tt.func public @reduce_maximum_f32_vector_reduce(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maximumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // LOCAL-LABEL: reduce_minimum_f32_vector_reduce
  // LOCAL: llvm.intr.vector.reduce.fminimum(%{{.*}}) : (vector<4xf32>) -> f32
  tt.func public @reduce_minimum_f32_vector_reduce(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minimumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // LOCAL-LABEL: reduce_maxnum_f32_vector_reduce
  // LOCAL: llvm.intr.vector.reduce.fmax(%{{.*}}) : (vector<4xf32>) -> f32
  tt.func public @reduce_maxnum_f32_vector_reduce(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maxnumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // LOCAL-LABEL: reduce_minnum_f32_vector_reduce
  // LOCAL: llvm.intr.vector.reduce.fmin(%{{.*}}) : (vector<4xf32>) -> f32
  tt.func public @reduce_minnum_f32_vector_reduce(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minnumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // Check the standalone port before extending its operation coverage.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_llvm_integer_add
  // LLVM-VECTORIZE-NEXT: %[[SUM:.*]] = llvm.add %arg0, %arg1 : vector<2xi32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[SUM]] : vector<2xi32>
  llvm.func @vectorize_llvm_integer_add(%a: vector<2xi32>, %b: vector<2xi32>) -> vector<2xi32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %a0 = llvm.extractelement %a[%c0 : i32] : vector<2xi32>
    %a1 = llvm.extractelement %a[%c1 : i32] : vector<2xi32>
    %b0 = llvm.extractelement %b[%c0 : i32] : vector<2xi32>
    %b1 = llvm.extractelement %b[%c1 : i32] : vector<2xi32>
    %x0 = llvm.add %a0, %b0 : i32
    %x1 = llvm.add %a1, %b1 : i32
    %s = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %s0 = llvm.insertvalue %x0, %s[0] : !llvm.struct<(i32, i32)>
    %s1 = llvm.insertvalue %x1, %s0[1] : !llvm.struct<(i32, i32)>
    %r0 = llvm.extractvalue %s1[0] : !llvm.struct<(i32, i32)>
    %r1 = llvm.extractvalue %s1[1] : !llvm.struct<(i32, i32)>
    %u = llvm.mlir.undef : vector<2xi32>
    %v0 = llvm.insertelement %r0, %u[%c0 : i32] : vector<2xi32>
    %v1 = llvm.insertelement %r1, %v0[%c1 : i32] : vector<2xi32>
    llvm.return %v1 : vector<2xi32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // Preserve the intrinsic result and scalar arithmetic even when the scalar
  // results are subsequently packed into an existing vector.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_permlane32_result
  // LLVM-VECTORIZE-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
  // LLVM-VECTORIZE-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // LLVM-VECTORIZE-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : vector<2xi32>
  // LLVM-VECTORIZE: %[[SWAP:.*]] = llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"
  // LLVM-VECTORIZE-SAME: -> !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[LO:.*]] = llvm.extractvalue %[[SWAP]][0] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[HI:.*]] = llvm.extractvalue %[[SWAP]][1] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[ADDLO:.*]] = llvm.add %[[LO]], %[[ONE]] : i32
  // LLVM-VECTORIZE-NEXT: %[[ADDHI:.*]] = llvm.add %[[HI]], %[[ONE]] : i32
  // LLVM-VECTORIZE-NEXT: %[[VLO:.*]] = llvm.insertelement %[[ADDLO]], %[[UNDEF]][%[[ZERO]] : i32] : vector<2xi32>
  // LLVM-VECTORIZE-NEXT: %[[VHI:.*]] = llvm.insertelement %[[ADDHI]], %[[VLO]][%[[ONE]] : i32] : vector<2xi32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[VHI]] : vector<2xi32>
  llvm.func @preserve_permlane32_result(%a: i32, %b: i32) -> vector<2xi32> {
    %false = llvm.mlir.constant(false) : i1
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %pair = llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"(%a, %b, %false, %false) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
    %lo = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i32)>
    %hi = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i32)>
    %x0 = llvm.add %lo, %c1 : i32
    %x1 = llvm.add %hi, %c1 : i32
    %u = llvm.mlir.undef : vector<2xi32>
    %v0 = llvm.insertelement %x0, %u[%c0 : i32] : vector<2xi32>
    %v1 = llvm.insertelement %x1, %v0[%c1 : i32] : vector<2xi32>
    llvm.return %v1 : vector<2xi32>
  }

  // A scalar consumer keeps its direct extraction without a vector adapter.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_permlane16_scalar
  // LLVM-VECTORIZE-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
  // LLVM-VECTORIZE-NEXT: %[[SWAP16:.*]] = llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"(%arg0, %arg1, %[[FALSE]], %[[FALSE]])
  // LLVM-VECTORIZE-SAME: -> !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[HIGH:.*]] = llvm.extractvalue %[[SWAP16]][1] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[HIGH]] : i32
  llvm.func @preserve_permlane16_scalar(%a: i32, %b: i32) -> i32 {
    %false = llvm.mlir.constant(false) : i1
    %pair = llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"(%a, %b, %false, %false) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
    %hi = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i32)>
    llvm.return %hi : i32
  }

  // The Gluon MLA reduction exchanges a pair, reduces its scalar fields, and
  // repeats with permlane16. Keep that sequence unchanged at the MLIR level.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_mla_permlane_sequence
  // LLVM-VECTORIZE-NEXT: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
  // LLVM-VECTORIZE-NEXT: %[[BITS:.*]] = llvm.bitcast %arg0 : f32 to i32
  // LLVM-VECTORIZE-NEXT: %[[PAIR32:.*]] = llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"(%[[BITS]], %[[BITS]], %[[FALSE]], %[[FALSE]]) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[LO32:.*]] = llvm.extractvalue %[[PAIR32]][0] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[HI32:.*]] = llvm.extractvalue %[[PAIR32]][1] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[FLO32:.*]] = llvm.bitcast %[[LO32]] : i32 to f32
  // LLVM-VECTORIZE-NEXT: %[[FHI32:.*]] = llvm.bitcast %[[HI32]] : i32 to f32
  // LLVM-VECTORIZE-NEXT: %[[MAX32:.*]] = llvm.intr.maxnum(%[[FLO32]], %[[FHI32]]) : (f32, f32) -> f32
  // LLVM-VECTORIZE-NEXT: %[[BITS32:.*]] = llvm.bitcast %[[MAX32]] : f32 to i32
  // LLVM-VECTORIZE-NEXT: %[[PAIR16:.*]] = llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"(%[[BITS32]], %[[BITS32]], %[[FALSE]], %[[FALSE]]) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[LO16:.*]] = llvm.extractvalue %[[PAIR16]][0] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[HI16:.*]] = llvm.extractvalue %[[PAIR16]][1] : !llvm.struct<(i32, i32)>
  // LLVM-VECTORIZE-NEXT: %[[FLO16:.*]] = llvm.bitcast %[[LO16]] : i32 to f32
  // LLVM-VECTORIZE-NEXT: %[[FHI16:.*]] = llvm.bitcast %[[HI16]] : i32 to f32
  // LLVM-VECTORIZE-NEXT: %[[MAX16:.*]] = llvm.intr.maxnum(%[[FLO16]], %[[FHI16]]) : (f32, f32) -> f32
  // LLVM-VECTORIZE-NEXT: llvm.return %[[MAX16]] : f32
  llvm.func @preserve_mla_permlane_sequence(%x: f32) -> f32 {
    %false = llvm.mlir.constant(false) : i1
    %bits = llvm.bitcast %x : f32 to i32
    %pair32 = llvm.call_intrinsic "llvm.amdgcn.permlane32.swap"(%bits, %bits, %false, %false) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
    %lo32 = llvm.extractvalue %pair32[0] : !llvm.struct<(i32, i32)>
    %hi32 = llvm.extractvalue %pair32[1] : !llvm.struct<(i32, i32)>
    %flo32 = llvm.bitcast %lo32 : i32 to f32
    %fhi32 = llvm.bitcast %hi32 : i32 to f32
    %max32 = llvm.intr.maxnum(%flo32, %fhi32) : (f32, f32) -> f32
    %bits32 = llvm.bitcast %max32 : f32 to i32
    %pair16 = llvm.call_intrinsic "llvm.amdgcn.permlane16.swap"(%bits32, %bits32, %false, %false) : (i32, i32, i1, i1) -> !llvm.struct<(i32, i32)>
    %lo16 = llvm.extractvalue %pair16[0] : !llvm.struct<(i32, i32)>
    %hi16 = llvm.extractvalue %pair16[1] : !llvm.struct<(i32, i32)>
    %flo16 = llvm.bitcast %lo16 : i32 to f32
    %fhi16 = llvm.bitcast %hi16 : i32 to f32
    %max16 = llvm.intr.maxnum(%flo16, %fhi16) : (f32, f32) -> f32
    llvm.return %max16 : f32
  }

  // The rule covers any call_intrinsic, including vector-capable intrinsics
  // with homogeneous struct results; it is not a permlane name blacklist.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_intrinsic_tuple
  // LLVM-VECTORIZE-NEXT: %[[PAIR:.*]] = llvm.call_intrinsic "llvm.sincos"(%arg0) : (f32) -> !llvm.struct<(f32, f32)>
  // LLVM-VECTORIZE-NEXT: %[[SIN:.*]] = llvm.extractvalue %[[PAIR]][0] : !llvm.struct<(f32, f32)>
  // LLVM-VECTORIZE-NEXT: %[[COS:.*]] = llvm.extractvalue %[[PAIR]][1] : !llvm.struct<(f32, f32)>
  // LLVM-VECTORIZE-NEXT: %[[SUM:.*]] = llvm.fadd %[[SIN]], %[[COS]] : f32
  // LLVM-VECTORIZE-NEXT: llvm.return %[[SUM]] : f32
  llvm.func @preserve_intrinsic_tuple(%x: f32) -> f32 {
    %pair = llvm.call_intrinsic "llvm.sincos"(%x) : (f32) -> !llvm.struct<(f32, f32)>
    %sin = llvm.extractvalue %pair[0] : !llvm.struct<(f32, f32)>
    %cos = llvm.extractvalue %pair[1] : !llvm.struct<(f32, f32)>
    %sum = llvm.fadd %sin, %cos : f32
    llvm.return %sum : f32
  }

  // The arithmetic reconstruction stage must also leave scalar intrinsic
  // calls alone when their results are packed into a vector.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_scalar_intrinsics
  // LLVM-VECTORIZE: %[[X0:.*]] = llvm.extractelement %arg0[{{.*}}] : vector<2xf32>
  // LLVM-VECTORIZE-NEXT: %[[X1:.*]] = llvm.extractelement %arg0[{{.*}}] : vector<2xf32>
  // LLVM-VECTORIZE-NEXT: %[[EXP0:.*]] = llvm.call_intrinsic "llvm.exp"(%[[X0]]) : (f32) -> f32
  // LLVM-VECTORIZE-NEXT: %[[EXP1:.*]] = llvm.call_intrinsic "llvm.exp"(%[[X1]]) : (f32) -> f32
  // LLVM-VECTORIZE-NEXT: %[[V0:.*]] = llvm.insertelement %[[EXP0]], {{.*}} : vector<2xf32>
  // LLVM-VECTORIZE-NEXT: %[[V1:.*]] = llvm.insertelement %[[EXP1]], %[[V0]][{{.*}}] : vector<2xf32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[V1]] : vector<2xf32>
  llvm.func @preserve_scalar_intrinsics(%x: vector<2xf32>) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %exp0 = llvm.call_intrinsic "llvm.exp"(%x0) : (f32) -> f32
    %exp1 = llvm.call_intrinsic "llvm.exp"(%x1) : (f32) -> f32
    %u = llvm.mlir.undef : vector<2xf32>
    %v0 = llvm.insertelement %exp0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %exp1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // Already-vector intrinsics also remain legal in their original form.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_vector_intrinsic
  // LLVM-VECTORIZE-NEXT: %[[EXP:.*]] = llvm.call_intrinsic "llvm.exp"(%arg0) : (vector<2xf32>) -> vector<2xf32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[EXP]] : vector<2xf32>
  llvm.func @preserve_vector_intrinsic(%x: vector<2xf32>) -> vector<2xf32> {
    %exp = llvm.call_intrinsic "llvm.exp"(%x) : (vector<2xf32>) -> vector<2xf32>
    llvm.return %exp : vector<2xf32>
  }

  // Disabled with tuple/vector materializations; these cases need adapters.
  // Return types are also fixed boundaries: the converted producer must be
  // materialized back to the declared struct type.
  // ADAPTER-ONLY-LABEL: llvm.func @vectorize_tuple_return
  // ADAPTER-ONLY-SAME: -> !llvm.struct<(i32, i32)>
  // ADAPTER-ONLY: %[[X0:.*]] = llvm.extractelement %arg0[{{.*}}] : vector<2xi32>
  // ADAPTER-ONLY: %[[S0:.*]] = llvm.insertvalue %[[X0]], {{.*}}[0] : !llvm.struct<(i32, i32)>
  // ADAPTER-ONLY: %[[X1:.*]] = llvm.extractelement %arg0[{{.*}}] : vector<2xi32>
  // ADAPTER-ONLY: %[[S1:.*]] = llvm.insertvalue %[[X1]], %[[S0]][1] : !llvm.struct<(i32, i32)>
  // ADAPTER-ONLY: llvm.return %[[S1]] : !llvm.struct<(i32, i32)>
  // llvm.func @vectorize_tuple_return(%x: vector<2xi32>) -> !llvm.struct<(i32, i32)> {
  //   %c0 = llvm.mlir.constant(0 : i32) : i32
  //   %c1 = llvm.mlir.constant(1 : i32) : i32
  //   %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xi32>
  //   %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xi32>
  //   %u = llvm.mlir.undef : !llvm.struct<(i32, i32)>
  //   %s0 = llvm.insertvalue %x0, %u[0] : !llvm.struct<(i32, i32)>
  //   %s1 = llvm.insertvalue %x1, %s0[1] : !llvm.struct<(i32, i32)>
  //   llvm.return %s1 : !llvm.struct<(i32, i32)>
  // }

  // Pointer aggregates are outside numeric tuple vectorization.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_pointer_tuple
  // LLVM-VECTORIZE: llvm.extractvalue %arg0[0] : !llvm.struct<(ptr, ptr)>
  // LLVM-VECTORIZE: llvm.return {{.*}} : !llvm.ptr
  llvm.func @preserve_pointer_tuple(%p: !llvm.struct<(ptr, ptr)>) -> !llvm.ptr {
    %x = llvm.extractvalue %p[0] : !llvm.struct<(ptr, ptr)>
    llvm.return %x : !llvm.ptr
  }

  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_packed_tuple
  // LLVM-VECTORIZE: llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32)>
  llvm.func @preserve_packed_tuple(%p: !llvm.struct<packed (i32, i32)>) -> i32 {
    %x = llvm.extractvalue %p[1] : !llvm.struct<packed (i32, i32)>
    llvm.return %x : i32
  }

  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_named_tuple
  // LLVM-VECTORIZE: llvm.extractvalue %arg0[0] : !llvm.struct<"fixed_pair", (i32, i32)>
  llvm.func @preserve_named_tuple(%p: !llvm.struct<"fixed_pair", (i32, i32)>) -> i32 {
    %x = llvm.extractvalue %p[0] : !llvm.struct<"fixed_pair", (i32, i32)>
    llvm.return %x : i32
  }

  // Nested homogeneous aggregates cannot be vector elements.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_nested_tuple
  // LLVM-VECTORIZE: llvm.extractvalue %arg0[1] : !llvm.struct<(array<2 x i32>, array<2 x i32>)>
  llvm.func @preserve_nested_tuple(%p: !llvm.struct<(array<2 x i32>, array<2 x i32>)>) -> !llvm.array<2 x i32> {
    %x = llvm.extractvalue %p[1] : !llvm.struct<(array<2 x i32>, array<2 x i32>)>
    llvm.return %x : !llvm.array<2 x i32>
  }

  // Disabled with tuple/vector materializations; these cases need adapters.
  // Nested numeric tuples also stay aggregates, including when a converted
  // inner tuple is inserted back into its outer container.
  // ADAPTER-ONLY-LABEL: llvm.func @preserve_nested_numeric_tuple
  // ADAPTER-ONLY-SAME: %arg0: !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  // ADAPTER-ONLY: llvm.extractvalue %arg0[0, 1] : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  // ADAPTER-ONLY: llvm.insertvalue {{.*}}, %arg0[1] : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  // ADAPTER-ONLY: llvm.return {{.*}} : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  // llvm.func @preserve_nested_numeric_tuple(%p: !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>) -> !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)> {
  //   %a = llvm.extractvalue %p[0] : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  //   %x = llvm.mlir.constant(7 : i32) : i32
  //   %a1 = llvm.insertvalue %x, %a[0] : !llvm.struct<(i32, i32)>
  //   %r = llvm.insertvalue %a1, %p[1] : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  //   llvm.return %r : !llvm.struct<(struct<(i32, i32)>, struct<(i32, i32)>)>
  // }

  // Disabled with tuple/vector materializations; these cases need adapters.
  // A non-power-of-two tuple may be packed without invoking the binary-tree
  // arithmetic reconstruction. Check floating-point numeric fields as well.
  // ADAPTER-ONLY-LABEL: llvm.func @vectorize_float_triple
  // ADAPTER-ONLY-SAME: %arg0: !llvm.struct<(f32, f32, f32)>
  // ADAPTER-ONLY: llvm.extractvalue %arg0[0] : !llvm.struct<(f32, f32, f32)>
  // ADAPTER-ONLY: llvm.extractvalue %arg0[1] : !llvm.struct<(f32, f32, f32)>
  // ADAPTER-ONLY: llvm.extractvalue %arg0[2] : !llvm.struct<(f32, f32, f32)>
  // ADAPTER-ONLY: llvm.return {{.*}} : vector<3xf32>
  // llvm.func @vectorize_float_triple(%p: !llvm.struct<(f32, f32, f32)>) -> vector<3xf32> {
  //   %c0 = llvm.mlir.constant(0 : i32) : i32
  //   %c1 = llvm.mlir.constant(1 : i32) : i32
  //   %c2 = llvm.mlir.constant(2 : i32) : i32
  //   %x0 = llvm.extractvalue %p[0] : !llvm.struct<(f32, f32, f32)>
  //   %x1 = llvm.extractvalue %p[1] : !llvm.struct<(f32, f32, f32)>
  //   %x2 = llvm.extractvalue %p[2] : !llvm.struct<(f32, f32, f32)>
  //   %u = llvm.mlir.undef : vector<3xf32>
  //   %v0 = llvm.insertelement %x0, %u[%c0 : i32] : vector<3xf32>
  //   %v1 = llvm.insertelement %x1, %v0[%c1 : i32] : vector<3xf32>
  //   %v2 = llvm.insertelement %x2, %v1[%c2 : i32] : vector<3xf32>
  //   llvm.return %v2 : vector<3xf32>
  // }

}

// -----

// LLVM-VECTORIZE: #[[$LOOP:.*]] = #llvm.loop_annotation<disableNonforced = false, mustProgress = true>
#loop = #llvm.loop_annotation<disableNonforced = false, mustProgress = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // A merge reached by both br and cond_br must receive vectors on both edges.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_mixed_predecessors
  // LLVM-VECTORIZE: llvm.cond_br %arg0, ^{{.*}}, ^[[MERGE:bb[0-9]+]]({{.*}} : vector<1xf32>)
  // LLVM-VECTORIZE: llvm.br ^[[MERGE]]({{.*}} : vector<1xf32>)
  // LLVM-VECTORIZE: ^[[MERGE]](%[[MERGED:.*]]: vector<1xf32>):
  // LLVM-VECTORIZE: llvm.extractelement %[[MERGED]][{{.*}}] : vector<1xf32>
  llvm.func @vectorize_mixed_predecessors(%condition: i1, %x: f32, %y: f32) -> f32 {
    %u = llvm.mlir.undef : !llvm.struct<(f32)>
    %initial = llvm.insertvalue %x, %u[0] : !llvm.struct<(f32)>
    llvm.br ^header(%initial : !llvm.struct<(f32)>)
  ^header(%carried: !llvm.struct<(f32)>):
    llvm.cond_br %condition, ^other, ^merge(%carried : !llvm.struct<(f32)>)
  ^other:
    %alternate = llvm.insertvalue %y, %u[0] : !llvm.struct<(f32)>
    llvm.br ^merge(%alternate : !llvm.struct<(f32)>)
  ^merge(%merged: !llvm.struct<(f32)>):
    %result = llvm.extractvalue %merged[0] : !llvm.struct<(f32)>
    llvm.return %result : f32
  }

  // Convert both conditional destinations, keep scalar arguments, and preserve
  // branch weights and the loop annotation on the unconditional backedge.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_tuple_loop
  // LLVM-VECTORIZE: llvm.br ^[[HEADER:bb[0-9]+]]({{.*}} : vector<1xf32>, i32)
  // LLVM-VECTORIZE: ^[[HEADER]](%[[CARRIED:.*]]: vector<1xf32>, %[[ITER:.*]]: i32):
  // LLVM-VECTORIZE: llvm.cond_br %{{.*}} weights([9, 1]), ^[[BODY:bb[0-9]+]](%[[CARRIED]], %[[ITER]] : vector<1xf32>, i32), ^[[EXIT:bb[0-9]+]](%[[CARRIED]] : vector<1xf32>)
  // LLVM-VECTORIZE: ^[[BODY]](%[[VALUE:.*]]: vector<1xf32>, %{{.*}}: i32):
  // LLVM-VECTORIZE: llvm.extractelement %[[VALUE]][{{.*}}] : vector<1xf32>
  // LLVM-VECTORIZE: llvm.br ^[[HEADER]]({{.*}} : vector<1xf32>, i32) {loop_annotation = #[[$LOOP]]}
  // LLVM-VECTORIZE: ^[[EXIT]](%[[RESULT:.*]]: vector<1xf32>):
  // LLVM-VECTORIZE: llvm.extractelement %[[RESULT]][{{.*}}] : vector<1xf32>
  llvm.func @vectorize_tuple_loop(%x: f32, %count: i32) -> f32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %u = llvm.mlir.undef : !llvm.struct<(f32)>
    %initial = llvm.insertvalue %x, %u[0] : !llvm.struct<(f32)>
    llvm.br ^header(%initial, %zero : !llvm.struct<(f32)>, i32)
  ^header(%carried: !llvm.struct<(f32)>, %iteration: i32):
    %continue = llvm.icmp "slt" %iteration, %count : i32
    llvm.cond_br %continue weights([9, 1]), ^body(%carried, %iteration : !llvm.struct<(f32)>, i32), ^exit(%carried : !llvm.struct<(f32)>)
  ^body(%value: !llvm.struct<(f32)>, %i: i32):
    %scalar = llvm.extractvalue %value[0] : !llvm.struct<(f32)>
    %sum = llvm.fadd %scalar, %x : f32
    %next = llvm.insertvalue %sum, %u[0] : !llvm.struct<(f32)>
    %next_i = llvm.add %i, %one : i32
    llvm.br ^header(%next, %next_i : !llvm.struct<(f32)>, i32) {loop_annotation = #loop}
  ^exit(%result: !llvm.struct<(f32)>):
    %scalar_result = llvm.extractvalue %result[0] : !llvm.struct<(f32)>
    llvm.return %scalar_result : f32
  }

  // A conditional backedge uses a signature already converted by the entry br.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_conditional_backedge
  // LLVM-VECTORIZE: llvm.br ^[[HEADER:bb[0-9]+]]({{.*}} : vector<1xf32>, i32)
  // LLVM-VECTORIZE: ^[[HEADER]](%{{.*}}: vector<1xf32>, %{{.*}}: i32):
  // LLVM-VECTORIZE: llvm.cond_br %{{.*}} weights([7, 1]), ^[[HEADER]](%[[NEXT:.*]], %{{.*}} : vector<1xf32>, i32), ^[[EXIT:bb[0-9]+]](%[[NEXT]] : vector<1xf32>) {loop_annotation = #[[$LOOP]]}
  // LLVM-VECTORIZE: ^[[EXIT]](%[[RESULT:.*]]: vector<1xf32>):
  // LLVM-VECTORIZE: llvm.extractelement %[[RESULT]][{{.*}}] : vector<1xf32>
  llvm.func @vectorize_conditional_backedge(%x: f32, %count: i32) -> f32 {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %u = llvm.mlir.undef : !llvm.struct<(f32)>
    %initial = llvm.insertvalue %x, %u[0] : !llvm.struct<(f32)>
    llvm.br ^header(%initial, %zero : !llvm.struct<(f32)>, i32)
  ^header(%carried: !llvm.struct<(f32)>, %iteration: i32):
    %scalar = llvm.extractvalue %carried[0] : !llvm.struct<(f32)>
    %sum = llvm.fadd %scalar, %x : f32
    %next = llvm.insertvalue %sum, %u[0] : !llvm.struct<(f32)>
    %next_i = llvm.add %iteration, %one : i32
    %continue = llvm.icmp "slt" %next_i, %count : i32
    llvm.cond_br %continue weights([7, 1]), ^header(%next, %next_i : !llvm.struct<(f32)>, i32), ^exit(%next : !llvm.struct<(f32)>) {loop_annotation = #loop}
  ^exit(%result: !llvm.struct<(f32)>):
    %scalar_result = llvm.extractvalue %result[0] : !llvm.struct<(f32)>
    llvm.return %scalar_result : f32
  }

  // Both edges target the same block but forward different values. Convert its
  // signature once and retain the correct operand list for each edge.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_shared_successor
  // LLVM-VECTORIZE: %[[TRUE:.*]] = llvm.insertelement %arg1, {{.*}} : vector<1xf32>
  // LLVM-VECTORIZE: %[[FALSE:.*]] = llvm.insertelement %arg2, {{.*}} : vector<1xf32>
  // LLVM-VECTORIZE: llvm.cond_br %arg0 weights([3, 2]), ^[[MERGE:bb[0-9]+]](%[[TRUE]] : vector<1xf32>), ^[[MERGE]](%[[FALSE]] : vector<1xf32>)
  // LLVM-VECTORIZE: ^[[MERGE]](%[[MERGED:.*]]: vector<1xf32>):
  // LLVM-VECTORIZE: llvm.extractelement %[[MERGED]][{{.*}}] : vector<1xf32>
  llvm.func @vectorize_shared_successor(%condition: i1, %x: f32, %y: f32) -> f32 {
    %u = llvm.mlir.undef : !llvm.struct<(f32)>
    %true_value = llvm.insertvalue %x, %u[0] : !llvm.struct<(f32)>
    %false_value = llvm.insertvalue %y, %u[0] : !llvm.struct<(f32)>
    llvm.cond_br %condition weights([3, 2]), ^merge(%true_value : !llvm.struct<(f32)>), ^merge(%false_value : !llvm.struct<(f32)>)
  ^merge(%merged: !llvm.struct<(f32)>):
    %result = llvm.extractvalue %merged[0] : !llvm.struct<(f32)>
    llvm.return %result : f32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_exp2_ftz
  // LLVM-VECTORIZE: %[[RESULT:.*]] = llvm.intr.exp2(%arg0) : (vector<2xf32>) -> vector<2xf32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[RESULT]] : vector<2xf32>
  llvm.func @vectorize_exp2_ftz(%x: vector<2xf32>) -> vector<2xf32> attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = preservesign, float_input_mode = preservesign>} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %e0 = rocdl.exp2 %x0 f32 -> f32
    %e1 = rocdl.exp2 %x1 f32 -> f32
    %u = llvm.mlir.undef : vector<2xf32>
    %v0 = llvm.insertelement %e0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %e1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_exp2_ieee
  // LLVM-VECTORIZE: rocdl.exp2
  // LLVM-VECTORIZE: rocdl.exp2
  // LLVM-VECTORIZE-NOT: llvm.intr.exp2
  // LLVM-VECTORIZE: llvm.return
  llvm.func @preserve_exp2_ieee(%x: vector<2xf32>) -> vector<2xf32> attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = ieee, float_input_mode = ieee>} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %e0 = rocdl.exp2 %x0 f32 -> f32
    %e1 = rocdl.exp2 %x1 f32 -> f32
    %u = llvm.mlir.undef : vector<2xf32>
    %v0 = llvm.insertelement %e0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %e1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_exp2_unspecified
  // LLVM-VECTORIZE: rocdl.exp2
  // LLVM-VECTORIZE: rocdl.exp2
  // LLVM-VECTORIZE-NOT: llvm.intr.exp2
  // LLVM-VECTORIZE: llvm.return
  llvm.func @preserve_exp2_unspecified(%x: vector<2xf32>) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %e0 = rocdl.exp2 %x0 f32 -> f32
    %e1 = rocdl.exp2 %x1 f32 -> f32
    %u = llvm.mlir.undef : vector<2xf32>
    %v0 = llvm.insertelement %e0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %e1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_generic_exp2
  // LLVM-VECTORIZE: %[[RESULT:.*]] = llvm.intr.exp2(%arg0) {fastmathFlags = #llvm.fastmath<afn>} : (vector<2xf32>) -> vector<2xf32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[RESULT]] : vector<2xf32>
  llvm.func @vectorize_generic_exp2(%x: vector<2xf32>) -> vector<2xf32> attributes {denormal_fpenv = #llvm.denormal_fpenv<default_output_mode = ieee, default_input_mode = ieee, float_output_mode = ieee, float_input_mode = ieee>} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %e0 = llvm.intr.exp2(%x0) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %e1 = llvm.intr.exp2(%x1) {fastmathFlags = #llvm.fastmath<afn>} : (f32) -> f32
    %u = llvm.mlir.undef : vector<2xf32>
    %v0 = llvm.insertelement %e0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %e1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // Interleaved lanes need an insertion point after both operands. This also
  // covers a scalar block argument, distinct FP constants, and FP flags.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_fp_exp_chain
  // LLVM-VECTORIZE-DAG: llvm.mlir.constant(dense<[1.250000e+00, -0.000000e+00]> : vector<2xf32>)
  // LLVM-VECTORIZE: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf32>
  // LLVM-VECTORIZE: llvm.fsub {{.*}} : vector<2xf32>
  // LLVM-VECTORIZE: %[[RESULT:.*]] = llvm.intr.exp2({{.*}}) : (vector<2xf32>) -> vector<2xf32>
  // LLVM-VECTORIZE-NEXT: llvm.return %[[RESULT]] : vector<2xf32>
  llvm.func @vectorize_fp_exp_chain(%x: vector<2xf32>, %scale: f32) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %a = llvm.mlir.constant(1.25 : f32) : f32
    %b = llvm.mlir.constant(-0.0 : f32) : f32
    %u = llvm.mlir.undef : vector<2xf32>
    %x0 = llvm.extractelement %x[%c0 : i32] : vector<2xf32>
    %m0 = llvm.fmul %x0, %scale {fastmathFlags = #llvm.fastmath<contract>} : f32
    %s0 = llvm.fsub %m0, %a : f32
    %e0 = llvm.intr.exp2(%s0) : (f32) -> f32
    %v0 = llvm.insertelement %e0, %u[%c0 : i32] : vector<2xf32>
    %x1 = llvm.extractelement %x[%c1 : i32] : vector<2xf32>
    %m1 = llvm.fmul %x1, %scale {fastmathFlags = #llvm.fastmath<contract>} : f32
    %s1 = llvm.fsub %m1, %b : f32
    %e1 = llvm.intr.exp2(%s1) : (f32) -> f32
    %v1 = llvm.insertelement %e1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // Scalar arguments and unsupported producers may be packed as operands of
  // vector arithmetic without trying to reconstruct the producer itself.
  // LLVM-VECTORIZE-LABEL: llvm.func @vectorize_fp_scalar_leaves
  // LLVM-VECTORIZE: llvm.select
  // LLVM-VECTORIZE: llvm.insertelement
  // LLVM-VECTORIZE: llvm.fadd {{.*}} : vector<2xf32>
  // LLVM-VECTORIZE: llvm.return {{.*}} : vector<2xf32>
  llvm.func @vectorize_fp_scalar_leaves(%condition: i1, %a: f32, %b: f32, %c: f32) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %u = llvm.mlir.undef : vector<2xf32>
    %s = llvm.select %condition, %a, %b : i1, f32
    %f0 = llvm.fadd %s, %c : f32
    %f1 = llvm.fadd %b, %c : f32
    %v0 = llvm.insertelement %f0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %f1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // Different fast-math contracts must not become one flagged vector op.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_mixed_fp_flags
  // LLVM-VECTORIZE: llvm.fadd {{.*}} : f32
  // LLVM-VECTORIZE: llvm.fadd {{.*}} {fastmathFlags = #llvm.fastmath<nnan>} : f32
  // LLVM-VECTORIZE-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // LLVM-VECTORIZE: llvm.return {{.*}} : vector<2xf32>
  llvm.func @preserve_mixed_fp_flags(%a: f32, %b: f32, %c: f32) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %u = llvm.mlir.undef : vector<2xf32>
    %f0 = llvm.fadd %a, %b : f32
    %f1 = llvm.fadd %a, %c {fastmathFlags = #llvm.fastmath<nnan>} : f32
    %v0 = llvm.insertelement %f0, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %f1, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }

  // A mismatched producer must not build a vector from a partial constant list.
  // LLVM-VECTORIZE-LABEL: llvm.func @preserve_mixed_constant_argument
  // LLVM-VECTORIZE: llvm.insertelement %arg0, {{.*}} : vector<2xf32>
  // LLVM-VECTORIZE: llvm.return {{.*}} : vector<2xf32>
  llvm.func @preserve_mixed_constant_argument(%a: f32) -> vector<2xf32> {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %u = llvm.mlir.undef : vector<2xf32>
    %f = llvm.mlir.constant(1.25 : f32) : f32
    %v0 = llvm.insertelement %f, %u[%c0 : i32] : vector<2xf32>
    %v1 = llvm.insertelement %a, %v0[%c1 : i32] : vector<2xf32>
    llvm.return %v1 : vector<2xf32>
  }
}
