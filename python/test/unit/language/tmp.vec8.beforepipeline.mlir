
// -----// IR Dump Before TritonAMDGPUPipeline (tritonamdgpu-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 8]], warp = [[0, 0, 0], [1, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 8]], warp = [[1, 0, 0], [0, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_reshape(%a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"), %b_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"), %c_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"), %M: i32 {tt.divisibility = 16 : i32} loc("M"), %N: i32 {tt.divisibility = 16 : i32} loc("N"), %K: i32 {tt.divisibility = 16 : i32} loc("K")) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #linear>
    %cst_0 = arith.constant dense<0> : tensor<2x32x64xi64, #blocked>
    %cst_1 = arith.constant dense<false> : tensor<2x32x64xi1, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<2x1x1xi64, #blocked>
    %cst_3 = arith.constant dense<2> : tensor<2x1x1xi64, #blocked>
    %cst_4 = arith.constant dense<0> : tensor<1x32x1xi64, #blocked>
    %cst_5 = arith.constant dense<0> : tensor<1x1x64xi64, #blocked>
    %cst_6 = arith.constant dense<0> : tensor<64x1xi64, #blocked1>
    %cst_7 = arith.constant dense<0> : tensor<1x64xi64, #blocked1>
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_8 = arith.constant dense<0.000000e+00> : tensor<2x32x64xbf16, #blocked>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %M, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %N, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.addi %K, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %M, %c2_i32 : i32
    %9 = arith.muli %8, %K : i32
    %10 = arith.extsi %9 : i32 to i64
    %K_10 = arith.extsi %K : i32 to i64
    %11 = arith.extsi %8 : i32 to i64
    %12 = arith.divsi %N, %c2_i32 : i32
    %13 = arith.muli %12, %K : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.extsi %12 : i32 to i64
    %N_11 = arith.extsi %N : i32 to i64
    %M_12 = arith.extsi %M : i32 to i64
    %16 = arith.subi %0, %c4_i32 : i32
    %17 = arith.muli %4, %c8_i32 : i32
    %18 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %19 = arith.extsi %18 : tensor<2xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> to tensor<2xi64, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %20 = tt.expand_dims %19 {axis = 1 : i32} : tensor<2xi64, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<2x1xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %21 = tt.expand_dims %20 {axis = 2 : i32} : tensor<2x1xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<2x1x1xi64, #blocked>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %23 = arith.extsi %22 : tensor<32xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> to tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %25 = arith.extsi %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %26 = arith.cmpi sge, %21, %cst_2 : tensor<2x1x1xi64, #blocked>
    %27 = arith.cmpi slt, %21, %cst_3 : tensor<2x1x1xi64, #blocked>
    %28 = arith.andi %26, %27 : tensor<2x1x1xi1, #blocked>
    %29 = tt.broadcast %28 : tensor<2x1x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %30 = tt.splat %11 : i64 -> tensor<1x32x1xi64, #blocked>
    %31 = tt.splat %K_10 : i64 -> tensor<1x1x64xi64, #blocked>
    %32 = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<2x32x64x!tt.ptr<bf16>, #blocked>
    %33 = tt.splat %10 : i64 -> tensor<2x1x1xi64, #blocked>
    %34 = arith.muli %21, %33 : tensor<2x1x1xi64, #blocked>
    %35 = tt.broadcast %34 : tensor<2x1x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %36 = tt.splat %K_10 : i64 -> tensor<1x32x1xi64, #blocked>
    %37 = tt.splat %15 : i64 -> tensor<1x32x1xi64, #blocked>
    %38 = tt.splat %b_ptr : !tt.ptr<bf16> -> tensor<2x32x64x!tt.ptr<bf16>, #blocked>
    %39 = tt.splat %14 : i64 -> tensor<2x1x1xi64, #blocked>
    %40 = arith.muli %21, %39 : tensor<2x1x1xi64, #blocked>
    %41 = tt.broadcast %40 : tensor<2x1x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %42 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %43 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %44 = arith.extsi %42 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %45 = arith.extsi %43 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %46 = tt.splat %M_12 : i64 -> tensor<64x1xi64, #blocked1>
    %47 = tt.splat %N_11 : i64 -> tensor<1x64xi64, #blocked1>
    %48 = tt.splat %c_ptr : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked1>
    %49 = tt.splat %N_11 : i64 -> tensor<64x1xi64, #blocked1>
    %accumulator = arith.cmpi eq, %6, %c0_i32 : i32
    scf.if %accumulator {
      %tile_id_c = scf.for %tile_id = %0 to %7 step %c4_i32 iter_args(%tile_id_c_13 = %16) -> (i32)  : i32 {
        %50 = arith.addi %tile_id_c_13, %c4_i32 : i32
        %51 = arith.divsi %50, %17 : i32
        %52 = arith.muli %51, %c8_i32 : i32
        %53 = arith.subi %2, %52 : i32
        %54 = arith.minsi %53, %c8_i32 : i32
        %55 = arith.remsi %50, %54 : i32
        %56 = arith.addi %52, %55 : i32
        %57 = arith.remsi %50, %17 : i32
        %58 = arith.divsi %57, %54 : i32
        %59 = arith.muli %56, %c64_i32 : i32
        %60 = arith.muli %58, %c64_i32 : i32
        %61 = arith.extsi %59 : i32 to i64
        %62 = arith.extsi %60 : i32 to i64
        %63 = tt.splat %61 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %64 = arith.addi %63, %44 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi64, #blocked1>
        %66 = tt.splat %62 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %67 = arith.addi %66, %45 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %68 = tt.expand_dims %67 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1>
        %69 = arith.cmpi sge, %65, %cst_6 : tensor<64x1xi64, #blocked1>
        %70 = arith.cmpi slt, %65, %46 : tensor<64x1xi64, #blocked1>
        %71 = arith.andi %69, %70 : tensor<64x1xi1, #blocked1>
        %72 = tt.broadcast %71 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
        %73 = arith.cmpi sge, %68, %cst_7 : tensor<1x64xi64, #blocked1>
        %74 = arith.cmpi slt, %68, %47 : tensor<1x64xi64, #blocked1>
        %75 = arith.andi %73, %74 : tensor<1x64xi1, #blocked1>
        %76 = tt.broadcast %75 : tensor<1x64xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
        %77 = arith.andi %72, %76 : tensor<64x64xi1, #blocked1>
        %78 = arith.muli %65, %49 : tensor<64x1xi64, #blocked1>
        %79 = tt.broadcast %78 : tensor<64x1xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
        %80 = tt.broadcast %68 : tensor<1x64xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
        %81 = arith.addi %79, %80 : tensor<64x64xi64, #blocked1>
        %82 = tt.addptr %48, %81 : tensor<64x64x!tt.ptr<bf16>, #blocked1>, tensor<64x64xi64, #blocked1>
        %83 = ttg.convert_layout %82 : tensor<64x64x!tt.ptr<bf16>, #blocked1> -> tensor<64x64x!tt.ptr<bf16>, #linear>
        %84 = ttg.convert_layout %77 : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #linear>
        tt.store %83, %cst, %84 : tensor<64x64x!tt.ptr<bf16>, #linear>
        scf.yield %50 : i32
      } {tt.flatten}
    } else {
      %tile_id_c = arith.subi %7, %0 : i32
      %tile_id_c_13 = arith.ceildivsi %tile_id_c, %c4_i32 : i32
      %tile_id_c_14 = arith.maxsi %6, %c1_i32 : i32
      %tile_id_c_15 = arith.muli %tile_id_c_13, %tile_id_c_14 : i32
      %tile_id_c_16 = arith.subi %0, %c4_i32 : i32
      %tile_id_c_17 = arith.subi %tile_id_c_14, %c1_i32 : i32
      %tile_id_c_18 = arith.subi %tile_id_c_14, %c1_i32 : i32
      %tile_id_c_19:9 = scf.for %tile_id_c_20 = %c0_i32 to %tile_id_c_15 step %c1_i32 iter_args(%arg7 = %c0_i32, %tile_id_c_21 = %tile_id_c_16, %arg9 = %16, %arg10 = %c0_i32, %arg11 = %cst_9, %arg12 = %cst_1, %arg13 = %cst_0, %arg14 = %cst_1, %arg15 = %cst_0) -> (i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>)  : i32 {
        %tile_id_c_22 = arith.cmpi eq, %arg7, %c0_i32 : i32
        %tile_id_c_23 = arith.select %tile_id_c_22, %c0_i32, %arg10 : i32
        %tile_id_c_24:5 = scf.if %tile_id_c_22 -> (tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, i32) {
          %tile_id_c_32 = arith.addi %tile_id_c_21, %c4_i32 : i32
          %77 = arith.divsi %tile_id_c_32, %17 : i32
          %78 = arith.muli %77, %c8_i32 : i32
          %79 = arith.subi %2, %78 : i32
          %80 = arith.minsi %79, %c8_i32 : i32
          %81 = arith.remsi %tile_id_c_32, %80 : i32
          %82 = arith.addi %78, %81 : i32
          %83 = arith.remsi %tile_id_c_32, %17 : i32
          %84 = arith.divsi %83, %80 : i32
          %85 = arith.muli %82, %c32_i32 : i32
          %86 = arith.muli %84, %c32_i32 : i32
          %87 = arith.extsi %85 : i32 to i64
          %88 = tt.splat %87 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
          %89 = arith.addi %88, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
          %90 = tt.expand_dims %89 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
          %91 = tt.expand_dims %90 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
          %92 = arith.cmpi sge, %91, %cst_4 : tensor<1x32x1xi64, #blocked>
          %93 = arith.cmpi slt, %91, %30 : tensor<1x32x1xi64, #blocked>
          %94 = arith.andi %92, %93 : tensor<1x32x1xi1, #blocked>
          %95 = tt.broadcast %94 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
          %96 = arith.andi %29, %95 : tensor<2x32x64xi1, #blocked>
          %97 = arith.muli %91, %36 : tensor<1x32x1xi64, #blocked>
          %98 = tt.broadcast %97 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
          %99 = arith.extsi %86 : i32 to i64
          %100 = tt.splat %99 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
          %101 = arith.addi %100, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
          %102 = tt.expand_dims %101 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
          %103 = tt.expand_dims %102 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
          %104 = arith.cmpi sge, %103, %cst_4 : tensor<1x32x1xi64, #blocked>
          %105 = arith.cmpi slt, %103, %37 : tensor<1x32x1xi64, #blocked>
          %106 = arith.andi %104, %105 : tensor<1x32x1xi1, #blocked>
          %107 = tt.broadcast %106 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
          %108 = arith.andi %29, %107 : tensor<2x32x64xi1, #blocked>
          %109 = arith.muli %103, %36 : tensor<1x32x1xi64, #blocked>
          %110 = tt.broadcast %109 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
          scf.yield %96, %98, %108, %110, %tile_id_c_32 : tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, i32
        } else {
          scf.yield %arg12, %arg13, %arg14, %arg15, %tile_id_c_21 : tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, i32
        }
        %50 = arith.muli %tile_id_c_23, %c64_i32 : i32
        %51 = arith.extsi %50 : i32 to i64
        %52 = tt.splat %51 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
        %53 = arith.addi %52, %25 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
        %54 = tt.expand_dims %53 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %55 = tt.expand_dims %54 {axis = 1 : i32} : tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x64xi64, #blocked>
        %56 = arith.cmpi sge, %55, %cst_5 : tensor<1x1x64xi64, #blocked>
        %57 = arith.cmpi slt, %55, %31 : tensor<1x1x64xi64, #blocked>
        %58 = arith.andi %56, %57 : tensor<1x1x64xi1, #blocked>
        %59 = tt.broadcast %58 : tensor<1x1x64xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
        %60 = arith.andi %tile_id_c_24#0, %59 : tensor<2x32x64xi1, #blocked>
        %61 = tt.broadcast %55 : tensor<1x1x64xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
        %62 = arith.addi %tile_id_c_24#1, %61 : tensor<2x32x64xi64, #blocked>
        %63 = arith.addi %35, %62 : tensor<2x32x64xi64, #blocked>
        %64 = tt.addptr %32, %63 : tensor<2x32x64x!tt.ptr<bf16>, #blocked>, tensor<2x32x64xi64, #blocked>
        %65 = tt.load %64, %60, %cst_8 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<2x32x64x!tt.ptr<bf16>, #blocked>
        %66 = ttg.convert_layout %65 : tensor<2x32x64xbf16, #blocked> -> tensor<2x32x64xbf16, #linear1>
        %67 = tt.reshape %66 : tensor<2x32x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        %68 = arith.andi %tile_id_c_24#2, %59 : tensor<2x32x64xi1, #blocked>
        %69 = arith.addi %tile_id_c_24#3, %61 : tensor<2x32x64xi64, #blocked>
        %70 = arith.addi %41, %69 : tensor<2x32x64xi64, #blocked>
        %71 = tt.addptr %38, %70 : tensor<2x32x64x!tt.ptr<bf16>, #blocked>, tensor<2x32x64xi64, #blocked>
        %72 = tt.load %71, %68, %cst_8 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<2x32x64x!tt.ptr<bf16>, #blocked>
        %73 = ttg.convert_layout %72 : tensor<2x32x64xbf16, #blocked> -> tensor<2x32x64xbf16, #linear2>
        %74 = tt.reshape %73 : tensor<2x32x64xbf16, #linear2> -> tensor<64x64xbf16, #linear3>
        %75 = tt.trans %74 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear3> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %76 = tt.dot %67, %75, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
        %tile_id_c_25 = arith.addi %tile_id_c_23, %c1_i32 : i32
        %tile_id_c_26 = arith.cmpi eq, %arg7, %tile_id_c_17 : i32
        %tile_id_c_27 = arith.select %tile_id_c_26, %cst_9, %76 : tensor<64x64xf32, #mma>
        %tile_id_c_28 = scf.if %tile_id_c_26 -> (i32) {
          %77 = arith.addi %arg9, %c4_i32 : i32
          %78 = arith.divsi %77, %17 : i32
          %79 = arith.muli %78, %c8_i32 : i32
          %80 = arith.subi %2, %79 : i32
          %81 = arith.minsi %80, %c8_i32 : i32
          %82 = arith.remsi %77, %81 : i32
          %83 = arith.addi %79, %82 : i32
          %84 = arith.remsi %77, %17 : i32
          %85 = arith.divsi %84, %81 : i32
          %86 = arith.muli %83, %c64_i32 : i32
          %87 = arith.muli %85, %c64_i32 : i32
          %88 = arith.truncf %76 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
          %89 = arith.extsi %86 : i32 to i64
          %90 = arith.extsi %87 : i32 to i64
          %91 = tt.splat %89 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %92 = arith.addi %91, %44 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %93 = tt.expand_dims %92 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi64, #blocked1>
          %94 = tt.splat %90 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %95 = arith.addi %94, %45 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %96 = tt.expand_dims %95 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi64, #blocked1>
          %97 = arith.cmpi sge, %93, %cst_6 : tensor<64x1xi64, #blocked1>
          %98 = arith.cmpi slt, %93, %46 : tensor<64x1xi64, #blocked1>
          %99 = arith.andi %97, %98 : tensor<64x1xi1, #blocked1>
          %100 = tt.broadcast %99 : tensor<64x1xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
          %101 = arith.cmpi sge, %96, %cst_7 : tensor<1x64xi64, #blocked1>
          %102 = arith.cmpi slt, %96, %47 : tensor<1x64xi64, #blocked1>
          %103 = arith.andi %101, %102 : tensor<1x64xi1, #blocked1>
          %104 = tt.broadcast %103 : tensor<1x64xi1, #blocked1> -> tensor<64x64xi1, #blocked1>
          %105 = arith.andi %100, %104 : tensor<64x64xi1, #blocked1>
          %106 = arith.muli %93, %49 : tensor<64x1xi64, #blocked1>
          %107 = tt.broadcast %106 : tensor<64x1xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
          %108 = tt.broadcast %96 : tensor<1x64xi64, #blocked1> -> tensor<64x64xi64, #blocked1>
          %109 = arith.addi %107, %108 : tensor<64x64xi64, #blocked1>
          %110 = tt.addptr %48, %109 : tensor<64x64x!tt.ptr<bf16>, #blocked1>, tensor<64x64xi64, #blocked1>
          %111 = ttg.convert_layout %110 : tensor<64x64x!tt.ptr<bf16>, #blocked1> -> tensor<64x64x!tt.ptr<bf16>, #linear>
          %112 = ttg.convert_layout %88 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
          %113 = ttg.convert_layout %105 : tensor<64x64xi1, #blocked1> -> tensor<64x64xi1, #linear>
          tt.store %111, %112, %113 : tensor<64x64x!tt.ptr<bf16>, #linear>
          scf.yield %77 : i32
        } else {
          scf.yield %arg9 : i32
        }
        %tile_id_c_29 = arith.addi %arg7, %c1_i32 : i32
        %tile_id_c_30 = arith.cmpi eq, %arg7, %tile_id_c_18 : i32
        %tile_id_c_31 = arith.select %tile_id_c_30, %c0_i32, %tile_id_c_29 : i32
        scf.yield %tile_id_c_31, %tile_id_c_24#4, %tile_id_c_28, %tile_id_c_25, %tile_id_c_27, %tile_id_c_24#0, %tile_id_c_24#1, %tile_id_c_24#2, %tile_id_c_24#3 : i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>
      } {tt.scheduled_max_stage = 1 : i32}
    }
    tt.return
  }
}
