// -----// IR Dump Before TritonGPUPipeline (tritongpu-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%sm_scale: f32 loc("sm_scale"), %M: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("M"), %Z: i32 loc("Z"), %H: i32 {tt.divisibility = 16 : i32} loc("H"), %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_q"), %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_k"), %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_v"), %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_o"), %N_CTX: i32 {tt.divisibility = 16 : i32} loc("N_CTX")) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi64, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<1x128xi64, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<1x128xi64, #blocked>
    %cst_3 = arith.constant dense<128> : tensor<128x1xi64, #blocked>
    %cst_4 = arith.constant dense<0> : tensor<64x1xi64, #blocked>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi64, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_10 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %H : i32
    %3 = arith.remsi %1, %H : i32
    %4 = arith.muli %Z, %H : i32
    %5 = arith.muli %4, %N_CTX : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.muli %N_CTX, %H : i32
    %8 = arith.muli %2, %7 : i32
    %9 = arith.muli %3, %N_CTX : i32
    %10 = arith.addi %8, %9 : i32
    %11 = arith.muli %0, %c128_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %16 = tt.splat %11 : i32 -> tensor<128xi32, #blocked1>
    %17 = arith.addi %16, %15 : tensor<128xi32, #blocked1>
    %18 = arith.mulf %sm_scale, %cst_6 : f32
    %19 = arith.extsi %12 : i32 to i64
    %20 = tt.splat %19 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = arith.extsi %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.extsi %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = arith.addi %20, %21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi64, #blocked>
    %25 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked>
    %26 = arith.cmpi sge, %24, %cst_0 : tensor<128x1xi64, #blocked>
    %27 = tt.splat %6 : i64 -> tensor<128x1xi64, #blocked>
    %28 = arith.cmpi slt, %24, %27 : tensor<128x1xi64, #blocked>
    %29 = arith.andi %26, %28 : tensor<128x1xi1, #blocked>
    %30 = tt.broadcast %29 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %31 = arith.cmpi sge, %25, %cst_1 : tensor<1x128xi64, #blocked>
    %32 = arith.cmpi slt, %25, %cst_2 : tensor<1x128xi64, #blocked>
    %33 = arith.andi %31, %32 : tensor<1x128xi1, #blocked>
    %34 = tt.broadcast %33 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %35 = arith.andi %30, %34 : tensor<128x128xi1, #blocked>
    %36 = tt.splat %desc_q : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %37 = arith.muli %24, %cst_3 : tensor<128x1xi64, #blocked>
    %38 = tt.broadcast %37 : tensor<128x1xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %39 = tt.broadcast %25 : tensor<1x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %40 = arith.addi %38, %39 : tensor<128x128xi64, #blocked>
    %41 = tt.addptr %36, %40 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    %42 = tt.load %41, %35, %cst_7 : tensor<128x128x!tt.ptr<f16>, #blocked>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %44 = arith.extsi %43 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %6 : i64 -> tensor<64x1xi64, #blocked>
    %46 = tt.broadcast %33 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %47 = tt.splat %desc_k : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %48 = tt.broadcast %25 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
    %49 = ttg.convert_layout %42 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %50 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %51 = tt.splat %18 : f32 -> tensor<128x64xf32, #mma>
    %52 = tt.splat %desc_v : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %offsetv_y:4 = scf.for %offsetv_y_12 = %c0_i32 to %N_CTX step %c64_i32 iter_args(%arg10 = %cst_9, %arg11 = %cst_11, %arg12 = %cst_10, %arg13 = %10) -> (tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32)  : i32 {
      %offsetk_y = arith.extsi %arg13 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
      %67 = tt.splat %offsetk_y {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %68 = arith.addi %67, %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = tt.expand_dims %68 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
      %70 = arith.cmpi sge, %69, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #blocked>
      %71 = arith.cmpi slt, %69, %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #blocked>
      %72 = arith.andi %70, %71 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #blocked>
      %73 = tt.broadcast %72 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %74 = arith.andi %73, %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi1, #blocked>
      %75 = arith.muli %69, %cst_5 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #blocked>
      %76 = tt.broadcast %75 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %77 = arith.addi %76, %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi64, #blocked>
      %78 = tt.addptr %47, %77 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %79 = tt.load %78, %74, %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #blocked>
      %80 = ttg.convert_layout %79 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #linear>
      %81 = tt.trans %80 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<64x128xf16, #linear> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %82 = tt.dot %49, %81, %cst {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %83 = "tt.reduce"(%82) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %106 = arith.maxnumf %arg14, %arg15 : f32
        tt.reduce.return %106 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %84 = arith.mulf %83, %50 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %85 = arith.maxnumf %arg12, %84 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %86 = arith.mulf %82, %51 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #mma>
      %87 = tt.expand_dims %85 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %88 = tt.broadcast %87 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
      %89 = arith.subf %86, %88 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #mma>
      %90 = math.exp2 %89 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #mma>
      %91 = arith.subf %arg12, %85 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %92 = math.exp2 %91 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %93 = "tt.reduce"(%90) <{axis = 1 : i32}> ({
      ^bb0(%arg14: f32, %arg15: f32):
        %106 = arith.addf %arg14, %arg15 : f32
        tt.reduce.return %106 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %94 = tt.expand_dims %92 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %95 = tt.broadcast %94 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
      %96 = arith.mulf %arg10, %95 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #mma>
      %97 = tt.addptr %52, %77 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %98 = tt.load %97, %74, %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #blocked>
      %99 = arith.truncf %90 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %100 = ttg.convert_layout %99 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %101 = ttg.convert_layout %98 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %102 = tt.dot %100, %101, %96 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %103 = arith.mulf %arg11, %92 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %104 = arith.addf %103, %93 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %105 = arith.addi %arg13, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
      scf.yield %102, %104, %85, %105 : tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32
    } {tt.scheduled_max_stage = 1 : i32}
    %53 = math.log2 %offsetv_y#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %54 = arith.addf %offsetv_y#2, %53 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %55 = tt.expand_dims %offsetv_y#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %56 = tt.broadcast %55 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %57 = arith.divf %offsetv_y#0, %56 : tensor<128x128xf32, #mma>
    %58 = arith.muli %1, %N_CTX : i32
    %59 = tt.addptr %M, %58 : !tt.ptr<f32>, i32
    %60 = tt.splat %59 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %61 = tt.addptr %60, %17 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %62 = ttg.convert_layout %54 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked1>
    tt.store %61, %62 : tensor<128x!tt.ptr<f32>, #blocked1>
    %63 = arith.truncf %57 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %64 = tt.splat %desc_o : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %65 = tt.addptr %64, %40 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    %66 = ttg.convert_layout %63 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked>
    tt.store %65, %66, %35 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: LowerLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_1 = arith.constant dense<0> : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_2 = arith.constant dense<128> : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_3 = arith.constant dense<128> : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_4 = arith.constant dense<0> : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %cst_10 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %arg2, %arg3 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.muli %arg8, %arg3 : i32
    %8 = arith.muli %2, %7 : i32
    %9 = arith.muli %3, %arg8 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = arith.muli %0, %c128_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %16 = tt.splat %11 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %17 = arith.addi %16, %15 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %18 = arith.mulf %arg0, %cst_6 : f32
    %19 = arith.extsi %12 : i32 to i64
    %20 = tt.splat %19 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %21 = arith.extsi %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %22 = arith.extsi %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %23 = arith.addi %20, %21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = arith.cmpi sge, %24, %cst_0 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %27 = tt.splat %6 : i64 -> tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %28 = arith.cmpi slt, %24, %27 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %29 = arith.andi %26, %28 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %30 = tt.broadcast %29 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %31 = arith.cmpi sge, %25, %cst_1 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %32 = arith.cmpi slt, %25, %cst_2 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %33 = arith.andi %31, %32 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %34 = tt.broadcast %33 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %35 = arith.andi %30, %34 : tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %36 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %37 = arith.muli %24, %cst_3 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %38 = tt.broadcast %37 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %39 = tt.broadcast %25 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = arith.addi %38, %39 : tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %41 = tt.addptr %36, %40 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %42 = tt.load %41, %35, %cst_7 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %44 = arith.extsi %43 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %45 = tt.splat %6 : i64 -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %46 = tt.broadcast %33 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %47 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %48 = tt.broadcast %25 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %49 = ttg.convert_layout %42 : tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
    %50 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %51 = tt.splat %18 : f32 -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %52 = tt.splat %arg6 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %54 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32_12 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32_13 = arith.constant 0 : i32
    %c0_i32_14 = arith.constant 0 : i32
    %55:6 = scf.for %arg9 = %c0_i32 to %arg8 step %c64_i32 iter_args(%arg10 = %cst_9, %arg11 = %cst_11, %arg12 = %cst_10, %arg13 = %10, %arg14 = %c-1_i32, %arg15 = %c-1_i32) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, i32, i32, i32)  : i32 {
      %c1_i32_15 = arith.constant {loop.cluster = 2 : i32, loop.stage = 0 : i32} 1 : i32
      %71 = arith.addi %arg14, %c1_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %72 = arith.cmpi sge, %71, %c1_i32_15 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %73 = arith.select %72, %c0_i32_12, %71 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %74 = arith.addi %arg15, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %75 = arith.cmpi sge, %74, %c1_i32_15 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %76 = arith.select %75, %c0_i32_12, %74 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %77 = arith.extsi %arg13 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
      %78 = tt.splat %77 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %79 = arith.addi %78, %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %80 = tt.expand_dims %79 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %81 = arith.cmpi sge, %80, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %82 = arith.cmpi slt, %80, %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %83 = arith.andi %81, %82 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %84 = tt.broadcast %83 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %85 = arith.andi %84, %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %86 = arith.muli %80, %cst_5 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %87 = tt.broadcast %86 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %88 = arith.addi %87, %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %89 = tt.addptr %47, %88 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %90 = ttg.memdesc_index %53[%73] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %91 = ttg.async_copy_global_to_local %89, %90 mask %85 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %92 = ttg.async_commit_group tokens %91 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %93 = ttg.async_wait %92 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
      %94 = ttg.memdesc_index %53[%76] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %95 = ttg.local_load %94 token %93 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %96 = ttg.convert_layout %95 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf16, #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>>
      %97 = tt.trans %96 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<64x128xf16, #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %98 = tt.dot %49, %97, %cst {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %99 = "tt.reduce"(%98) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %127 = arith.maxnumf %arg16, %arg17 : f32
        tt.reduce.return %127 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %100 = arith.mulf %99, %50 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %101 = arith.maxnumf %arg12, %100 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %102 = arith.mulf %98, %51 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %103 = tt.expand_dims %101 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %104 = tt.broadcast %103 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %105 = arith.subf %102, %104 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %106 = math.exp2 %105 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %107 = arith.subf %arg12, %101 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %108 = math.exp2 %107 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %109 = "tt.reduce"(%106) <{axis = 1 : i32}> ({
      ^bb0(%arg16: f32, %arg17: f32):
        %127 = arith.addf %arg16, %arg17 : f32
        tt.reduce.return %127 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %110 = tt.expand_dims %108 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %111 = tt.broadcast %110 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %112 = arith.mulf %arg10, %111 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %113 = tt.addptr %52, %88 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %114 = ttg.memdesc_index %54[%73] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %115 = ttg.async_copy_global_to_local %113, %114 mask %85 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %116 = ttg.async_commit_group tokens %115 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %117 = ttg.async_wait %116 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
      %118 = ttg.memdesc_index %54[%76] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %119 = ttg.local_load %118 token %117 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %120 = arith.truncf %106 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> to tensor<128x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %121 = ttg.convert_layout %120 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %122 = ttg.convert_layout %119 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %123 = tt.dot %121, %122, %112 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %124 = arith.mulf %arg11, %108 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %125 = arith.addf %124, %109 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %126 = arith.addi %arg13, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
      scf.yield %123, %125, %101, %126, %73, %76 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, i32, i32, i32
    } {tt.scheduled_max_stage = 1 : i32}
    %56 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %54 : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %53 : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %57 = math.log2 %55#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %58 = arith.addf %55#2, %57 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %59 = tt.expand_dims %55#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %60 = tt.broadcast %59 : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %61 = arith.divf %55#0, %60 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %62 = arith.muli %1, %arg8 : i32
    %63 = tt.addptr %arg1, %62 : !tt.ptr<f32>, i32
    %64 = tt.splat %63 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %65 = tt.addptr %64, %17 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %66 = ttg.convert_layout %58 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    tt.store %65, %66 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %67 = arith.truncf %61 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> to tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %68 = tt.splat %arg7 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %69 = tt.addptr %68, %40 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %70 = ttg.convert_layout %67 : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %69, %70, %35 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}


// -----// SoftwarePipeliner internal IR Dump After: ExpandLoops
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg7: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_1 = arith.constant dense<0> : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_2 = arith.constant dense<128> : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_3 = arith.constant dense<128> : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_4 = arith.constant dense<0> : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %cst_10 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %arg2, %arg3 : i32
    %5 = arith.muli %4, %arg8 : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.muli %arg8, %arg3 : i32
    %8 = arith.muli %2, %7 : i32
    %9 = arith.muli %3, %arg8 : i32
    %10 = arith.addi %8, %9 : i32
    %11 = arith.muli %0, %c128_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %16 = tt.splat %11 : i32 -> tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %17 = arith.addi %16, %15 : tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %18 = arith.mulf %arg0, %cst_6 : f32
    %19 = arith.extsi %12 : i32 to i64
    %20 = tt.splat %19 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %21 = arith.extsi %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %22 = arith.extsi %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %23 = arith.addi %20, %21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %25 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %26 = arith.cmpi sge, %24, %cst_0 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %27 = tt.splat %6 : i64 -> tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %28 = arith.cmpi slt, %24, %27 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %29 = arith.andi %26, %28 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %30 = tt.broadcast %29 : tensor<128x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %31 = arith.cmpi sge, %25, %cst_1 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %32 = arith.cmpi slt, %25, %cst_2 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %33 = arith.andi %31, %32 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %34 = tt.broadcast %33 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %35 = arith.andi %30, %34 : tensor<128x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %36 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %37 = arith.muli %24, %cst_3 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %38 = tt.broadcast %37 : tensor<128x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %39 = tt.broadcast %25 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %40 = arith.addi %38, %39 : tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %41 = tt.addptr %36, %40 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %42 = tt.load %41, %35, %cst_7 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %44 = arith.extsi %43 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %45 = tt.splat %6 : i64 -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %46 = tt.broadcast %33 : tensor<1x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %47 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %48 = tt.broadcast %25 : tensor<1x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %49 = ttg.convert_layout %42 : tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
    %50 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %51 = tt.splat %18 : f32 -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %52 = tt.splat %arg6 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %54 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %55 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    %56 = arith.cmpi sge, %c0_i32, %c1_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
    %57 = arith.select %56, %c0_i32, %c0_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
    %58 = arith.extsi %10 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
    %59 = tt.splat %58 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %60 = arith.addi %59, %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %61 = tt.expand_dims %60 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %62 = arith.cmpi sge, %61, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %63 = arith.cmpi slt, %61, %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %64 = arith.andi %62, %63 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %65 = tt.broadcast %64 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %66 = arith.andi %65, %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %67 = arith.muli %61, %cst_5 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %68 = tt.broadcast %67 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %69 = arith.addi %68, %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %70 = tt.addptr %47, %69 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %71 = ttg.memdesc_index %53[%57] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %72 = tt.splat %55 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %73 = arith.andi %72, %66 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %74 = ttg.async_copy_global_to_local %70, %71 mask %73 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %75 = ttg.async_commit_group tokens %74 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %76 = tt.addptr %52, %69 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %77 = ttg.memdesc_index %54[%57] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %78 = tt.splat %55 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %79 = arith.andi %78, %66 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %80 = ttg.async_copy_global_to_local %76, %77 mask %79 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %81 = ttg.async_commit_group tokens %80 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %82:9 = scf.for %arg9 = %c0_i32 to %arg8 step %c64_i32 iter_args(%arg10 = %cst_9, %arg11 = %cst_11, %arg12 = %cst_10, %arg13 = %10, %arg14 = %57, %arg15 = %c-1_i32, %arg16 = %c1_i32, %arg17 = %75, %arg18 = %81) -> (tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %98 = arith.subi %arg8, %c64_i32 : i32
      %99 = arith.cmpi slt, %arg9, %98 : i32
      %100 = arith.addi %arg15, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %101 = arith.cmpi sge, %100, %arg16 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %102 = arith.select %101, %c0_i32, %100 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
      %103 = ttg.async_wait %arg17 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
      %104 = ttg.memdesc_index %53[%102] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %105 = ttg.local_load %104 token %103 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %106 = ttg.convert_layout %105 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf16, #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>>
      %107 = tt.trans %106 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<64x128xf16, #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %108 = tt.dot %49, %107, %cst {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %109 = "tt.reduce"(%108) <{axis = 1 : i32}> ({
      ^bb0(%arg19: f32, %arg20: f32):
        %160 = arith.maxnumf %arg19, %arg20 : f32
        tt.reduce.return %160 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %110 = arith.mulf %109, %50 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %111 = arith.maxnumf %arg12, %110 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %112 = arith.mulf %108, %51 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %113 = tt.expand_dims %111 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %114 = tt.broadcast %113 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %115 = arith.subf %112, %114 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %116 = math.exp2 %115 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %117 = arith.subf %arg12, %111 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %118 = math.exp2 %117 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %119 = "tt.reduce"(%116) <{axis = 1 : i32}> ({
      ^bb0(%arg19: f32, %arg20: f32):
        %160 = arith.addf %arg19, %arg20 : f32
        tt.reduce.return %160 : f32
      }) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %120 = tt.expand_dims %118 {axis = 1 : i32, loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %121 = tt.broadcast %120 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %122 = arith.mulf %arg10, %121 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %123 = ttg.async_wait %arg18 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
      %124 = ttg.memdesc_index %54[%102] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %125 = ttg.local_load %124 token %123 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %126 = arith.truncf %116 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> to tensor<128x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %127 = ttg.convert_layout %126 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %128 = ttg.convert_layout %125 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<64x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>>
      %129 = tt.dot %127, %128, %122 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>, kWidth = 2}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
      %130 = arith.mulf %arg11, %118 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %131 = arith.addf %130, %119 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
      %132 = arith.addi %arg13, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
      %133 = arith.addi %arg14, %c1_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %134 = arith.cmpi sge, %133, %c1_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %135 = arith.select %134, %c0_i32, %133 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %136 = arith.extsi %132 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
      %137 = tt.splat %136 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %138 = arith.addi %137, %44 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
      %139 = tt.expand_dims %138 {axis = 1 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %140 = arith.cmpi sge, %139, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %141 = arith.cmpi slt, %139, %45 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %142 = arith.andi %140, %141 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %143 = tt.broadcast %142 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %144 = arith.andi %143, %46 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %145 = arith.muli %139, %cst_5 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %146 = tt.broadcast %145 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %147 = arith.addi %146, %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %148 = tt.addptr %47, %147 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %149 = ttg.memdesc_index %53[%135] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %150 = tt.splat %99 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %151 = arith.andi %150, %144 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %152 = ttg.async_copy_global_to_local %148, %149 mask %151 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %153 = ttg.async_commit_group tokens %152 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      %154 = tt.addptr %52, %147 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %155 = ttg.memdesc_index %54[%135] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %156 = tt.splat %99 : i1 -> tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %157 = arith.andi %156, %144 : tensor<64x128xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
      %158 = ttg.async_copy_global_to_local %154, %155 mask %157 other %cst_8 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
      %159 = ttg.async_commit_group tokens %158 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
      scf.yield %129, %131, %111, %132, %135, %102, %c1_i32, %153, %159 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token
    } {tt.scheduled_max_stage = 1 : i32}
    %83 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %54 : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    ttg.local_dealloc %53 : !ttg.memdesc<1x64x128xf16, #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
    %84 = math.log2 %82#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %85 = arith.addf %82#2, %84 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>>
    %86 = tt.expand_dims %82#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %87 = tt.broadcast %86 : tensor<128x1xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %88 = arith.divf %82#0, %87 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %89 = arith.muli %1, %arg8 : i32
    %90 = tt.addptr %arg1, %89 : !tt.ptr<f32>, i32
    %91 = tt.splat %90 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %92 = tt.addptr %91, %17 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>, tensor<128xi32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %93 = ttg.convert_layout %85 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>}>> -> tensor<128xf32, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    tt.store %92, %93 : tensor<128x!tt.ptr<f32>, #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>>
    %94 = arith.truncf %88 : tensor<128x128xf32, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> to tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>>
    %95 = tt.splat %arg7 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %96 = tt.addptr %95, %40 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<128x128xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %97 = ttg.convert_layout %94 : tensor<128x128xf16, #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>> -> tensor<128x128xf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.store %96, %97, %35 : tensor<128x128x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    tt.return
  }
}


// -----// IR Dump Before Canonicalizer (canonicalize) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 8], [0, 16], [0, 32], [0, 64], [8, 0], [16, 0], [32, 0]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [[0, 0], [0, 0]], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%sm_scale: f32 loc("sm_scale"), %M: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("M"), %Z: i32 loc("Z"), %H: i32 {tt.divisibility = 16 : i32} loc("H"), %desc_q: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_q"), %desc_k: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_k"), %desc_v: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_v"), %desc_o: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("desc_o"), %N_CTX: i32 {tt.divisibility = 16 : i32} loc("N_CTX")) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %offsetv_y = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi64, #blocked>
    %cst_1 = arith.constant dense<0> : tensor<1x128xi64, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<1x128xi64, #blocked>
    %cst_3 = arith.constant dense<128> : tensor<128x1xi64, #blocked>
    %cst_4 = arith.constant dense<0> : tensor<64x1xi64, #blocked>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi64, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_6 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %cst_7 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_10 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_11 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %H : i32
    %3 = arith.remsi %1, %H : i32
    %4 = arith.muli %Z, %H : i32
    %5 = arith.muli %4, %N_CTX : i32
    %6 = arith.extsi %5 : i32 to i64
    %7 = arith.muli %N_CTX, %H : i32
    %8 = arith.muli %2, %7 : i32
    %9 = arith.muli %3, %N_CTX : i32
    %10 = arith.addi %8, %9 : i32
    %11 = arith.muli %0, %c128_i32 : i32
    %12 = arith.addi %10, %11 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %16 = tt.splat %11 : i32 -> tensor<128xi32, #blocked1>
    %17 = arith.addi %16, %15 : tensor<128xi32, #blocked1>
    %18 = arith.mulf %sm_scale, %cst_6 : f32
    %19 = arith.extsi %12 : i32 to i64
    %20 = tt.splat %19 : i64 -> tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = arith.extsi %13 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.extsi %14 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %23 = arith.addi %20, %21 : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<128xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi64, #blocked>
    %25 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi64, #blocked>
    %26 = arith.cmpi sge, %24, %cst_0 : tensor<128x1xi64, #blocked>
    %27 = tt.splat %6 : i64 -> tensor<128x1xi64, #blocked>
    %28 = arith.cmpi slt, %24, %27 : tensor<128x1xi64, #blocked>
    %29 = arith.andi %26, %28 : tensor<128x1xi1, #blocked>
    %30 = tt.broadcast %29 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %31 = arith.cmpi sge, %25, %cst_1 : tensor<1x128xi64, #blocked>
    %32 = arith.cmpi slt, %25, %cst_2 : tensor<1x128xi64, #blocked>
    %33 = arith.andi %31, %32 : tensor<1x128xi1, #blocked>
    %34 = tt.broadcast %33 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %35 = arith.andi %30, %34 : tensor<128x128xi1, #blocked>
    %36 = tt.splat %desc_q : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %37 = arith.muli %24, %cst_3 : tensor<128x1xi64, #blocked>
    %38 = tt.broadcast %37 : tensor<128x1xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %39 = tt.broadcast %25 : tensor<1x128xi64, #blocked> -> tensor<128x128xi64, #blocked>
    %40 = arith.addi %38, %39 : tensor<128x128xi64, #blocked>
    %41 = tt.addptr %36, %40 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    %42 = tt.load %41, %35, %cst_7 : tensor<128x128x!tt.ptr<f16>, #blocked>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %44 = arith.extsi %43 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %6 : i64 -> tensor<64x1xi64, #blocked>
    %46 = tt.broadcast %33 : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %47 = tt.splat %desc_k : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %48 = tt.broadcast %25 : tensor<1x128xi64, #blocked> -> tensor<64x128xi64, #blocked>
    %49 = ttg.convert_layout %42 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %50 = tt.splat %18 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %51 = tt.splat %18 : f32 -> tensor<128x64xf32, #mma>
    %52 = tt.splat %desc_v : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %54 = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %offsetv_y_12 = arith.cmpi sgt, %N_CTX, %c0_i32 : i32
    %offsetk_y = arith.extsi %10 : i32 to i64
    %55 = tt.splat %offsetk_y : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %56 = arith.addi %55, %44 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %57 = tt.expand_dims %56 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
    %58 = arith.cmpi sge, %57, %cst_4 : tensor<64x1xi64, #blocked>
    %59 = arith.cmpi slt, %57, %45 : tensor<64x1xi64, #blocked>
    %60 = arith.andi %58, %59 : tensor<64x1xi1, #blocked>
    %61 = tt.broadcast %60 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %62 = arith.andi %61, %46 : tensor<64x128xi1, #blocked>
    %63 = arith.muli %57, %cst_5 : tensor<64x1xi64, #blocked>
    %64 = tt.broadcast %63 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked>
    %65 = arith.addi %64, %48 : tensor<64x128xi64, #blocked>
    %66 = tt.addptr %47, %65 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
    %67 = ttg.memdesc_index %53[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %offsetv_y_13 = tt.splat %offsetv_y_12 : i1 -> tensor<64x128xi1, #blocked>
    %offsetv_y_14 = arith.andi %offsetv_y_13, %62 : tensor<64x128xi1, #blocked>
    %68 = ttg.async_copy_global_to_local %66, %67 mask %offsetv_y_14 other %cst_8 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    %69 = ttg.async_commit_group tokens %68
    %70 = tt.addptr %52, %65 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
    %71 = ttg.memdesc_index %54[%c0_i32] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %offsetv_y_15 = tt.splat %offsetv_y_12 : i1 -> tensor<64x128xi1, #blocked>
    %offsetv_y_16 = arith.andi %offsetv_y_15, %62 : tensor<64x128xi1, #blocked>
    %72 = ttg.async_copy_global_to_local %70, %71 mask %offsetv_y_16 other %cst_8 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    %73 = ttg.async_commit_group tokens %72
    %offsetv_y_17:9 = scf.for %offsetv_y_19 = %c0_i32 to %N_CTX step %c64_i32 iter_args(%arg10 = %cst_9, %arg11 = %cst_11, %arg12 = %cst_10, %arg13 = %10, %offsetv_y_20 = %c0_i32, %offsetv_y_21 = %offsetv_y, %offsetv_y_22 = %c1_i32, %arg17 = %69, %arg18 = %73) -> (tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %offsetv_y_23 = arith.subi %N_CTX, %c64_i32 : i32
      %offsetv_y_24 = arith.cmpi slt, %offsetv_y_19, %offsetv_y_23 : i32
      %offsetv_y_25 = arith.addi %offsetv_y_21, %c1_i32 : i32
      %offsetv_y_26 = arith.cmpi sge, %offsetv_y_25, %offsetv_y_22 : i32
      %offsetv_y_27 = arith.select %offsetv_y_26, %c0_i32, %offsetv_y_25 : i32
      %88 = ttg.async_wait %arg17, %arg18 {num = 0 : i32}
      %89 = ttg.memdesc_index %53[%offsetv_y_27] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %90 = ttg.local_load %89 token %88 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #blocked>
      %91 = ttg.convert_layout %90 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #linear>
      %92 = tt.trans %91 {order = array<i32: 1, 0>} : tensor<64x128xf16, #linear> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %93 = tt.dot %49, %92, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      %94 = "tt.reduce"(%93) <{axis = 1 : i32}> ({
      ^bb0(%arg19: f32, %arg20: f32):
        %136 = arith.maxnumf %arg19, %arg20 : f32
        tt.reduce.return %136 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %95 = arith.mulf %94, %50 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %96 = arith.maxnumf %arg12, %95 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %97 = arith.mulf %93, %51 : tensor<128x64xf32, #mma>
      %98 = tt.expand_dims %96 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %99 = tt.broadcast %98 : tensor<128x1xf32, #mma> -> tensor<128x64xf32, #mma>
      %100 = arith.subf %97, %99 : tensor<128x64xf32, #mma>
      %101 = math.exp2 %100 : tensor<128x64xf32, #mma>
      %102 = arith.subf %arg12, %96 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %103 = math.exp2 %102 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %104 = "tt.reduce"(%101) <{axis = 1 : i32}> ({
      ^bb0(%arg19: f32, %arg20: f32):
        %136 = arith.addf %arg19, %arg20 : f32
        tt.reduce.return %136 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %105 = tt.expand_dims %103 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %106 = tt.broadcast %105 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
      %107 = arith.mulf %arg10, %106 : tensor<128x128xf32, #mma>
      %108 = ttg.memdesc_index %54[%offsetv_y_27] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %109 = ttg.local_load %108 token %88 : !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<64x128xf16, #blocked>
      %110 = arith.truncf %101 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %111 = ttg.convert_layout %110 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %112 = ttg.convert_layout %109 : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %113 = tt.dot %111, %112, %107 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %114 = arith.mulf %arg11, %103 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %115 = arith.addf %114, %104 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %116 = arith.addi %arg13, %c64_i32 : i32
      %offsetv_y_28 = arith.addi %offsetv_y_20, %c1_i32 : i32
      %offsetv_y_29 = arith.cmpi sge, %offsetv_y_28, %c1_i32 : i32
      %offsetv_y_30 = arith.select %offsetv_y_29, %c0_i32, %offsetv_y_28 : i32
      %offsetk_y_31 = arith.extsi %116 : i32 to i64
      %117 = tt.splat %offsetk_y_31 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %118 = arith.addi %117, %44 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
      %119 = tt.expand_dims %118 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
      %120 = arith.cmpi sge, %119, %cst_4 : tensor<64x1xi64, #blocked>
      %121 = arith.cmpi slt, %119, %45 : tensor<64x1xi64, #blocked>
      %122 = arith.andi %120, %121 : tensor<64x1xi1, #blocked>
      %123 = tt.broadcast %122 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %124 = arith.andi %123, %46 : tensor<64x128xi1, #blocked>
      %125 = arith.muli %119, %cst_5 : tensor<64x1xi64, #blocked>
      %126 = tt.broadcast %125 : tensor<64x1xi64, #blocked> -> tensor<64x128xi64, #blocked>
      %127 = arith.addi %126, %48 : tensor<64x128xi64, #blocked>
      %128 = tt.addptr %47, %127 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %129 = ttg.memdesc_index %53[%offsetv_y_30] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %offsetv_y_32 = tt.splat %offsetv_y_24 : i1 -> tensor<64x128xi1, #blocked>
      %offsetv_y_33 = arith.andi %offsetv_y_32, %124 : tensor<64x128xi1, #blocked>
      %130 = ttg.async_copy_global_to_local %128, %129 mask %offsetv_y_33 other %cst_8 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
      %131 = ttg.async_commit_group tokens %130
      %132 = tt.addptr %52, %127 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi64, #blocked>
      %133 = ttg.memdesc_index %54[%offsetv_y_30] : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %offsetv_y_34 = tt.splat %offsetv_y_24 : i1 -> tensor<64x128xi1, #blocked>
      %offsetv_y_35 = arith.andi %offsetv_y_34, %124 : tensor<64x128xi1, #blocked>
      %134 = ttg.async_copy_global_to_local %132, %133 mask %offsetv_y_35 other %cst_8 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
      %135 = ttg.async_commit_group tokens %134
      scf.yield %113, %115, %96, %116, %offsetv_y_30, %offsetv_y_27, %c1_i32, %131, %135 : tensor<128x128xf32, #mma>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token
    }
    %offsetv_y_18 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %54 : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    ttg.local_dealloc %53 : !ttg.memdesc<1x64x128xf16, #shared, #smem, mutable>
    %74 = math.log2 %offsetv_y_17#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %75 = arith.addf %offsetv_y_17#2, %74 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %76 = tt.expand_dims %offsetv_y_17#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
    %77 = tt.broadcast %76 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
    %78 = arith.divf %offsetv_y_17#0, %77 : tensor<128x128xf32, #mma>
    %79 = arith.muli %1, %N_CTX : i32
    %80 = tt.addptr %M, %79 : !tt.ptr<f32>, i32
    %81 = tt.splat %80 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %82 = tt.addptr %81, %17 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %83 = ttg.convert_layout %75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128xf32, #blocked1>
    tt.store %82, %83 : tensor<128x!tt.ptr<f32>, #blocked1>
    %84 = arith.truncf %78 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %85 = tt.splat %desc_o : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %86 = tt.addptr %85, %40 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi64, #blocked>
    %87 = ttg.convert_layout %84 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked>
    tt.store %86, %87, %35 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
