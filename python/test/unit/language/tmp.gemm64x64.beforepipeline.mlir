// -----// IR Dump Before TritonAMDGPUPipeline (tritonamdgpu-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_reshape(%a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"), %b_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"), %c_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"), %M: i32 {tt.divisibility = 16 : i32} loc("M"), %N: i32 {tt.divisibility = 16 : i32} loc("N"), %K: i32 {tt.divisibility = 16 : i32} loc("K")) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #linear>
    %cst_0 = arith.constant dense<0> : tensor<64x64xi64, #blocked>
    %cst_1 = arith.constant dense<false> : tensor<64x64xi1, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<64x1xi64, #blocked>
    %cst_3 = arith.constant dense<0> : tensor<1x64xi64, #blocked>
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %M, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %N, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.addi %K, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %K_6 = arith.extsi %K : i32 to i64
    %M_7 = arith.extsi %M : i32 to i64
    %N_8 = arith.extsi %N : i32 to i64
    %8 = arith.subi %0, %c4_i32 : i32
    %9 = arith.muli %4, %c8_i32 : i32
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = arith.extsi %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.extsi %11 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.splat %M_7 : i64 -> tensor<64x1xi64, #blocked>
    %15 = tt.splat %K_6 : i64 -> tensor<1x64xi64, #blocked>
    %16 = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %17 = tt.splat %K_6 : i64 -> tensor<64x1xi64, #blocked>
    %18 = tt.splat %N_8 : i64 -> tensor<64x1xi64, #blocked>
    %19 = tt.splat %b_ptr : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %21 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %22 = arith.extsi %20 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.extsi %21 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.splat %M_7 : i64 -> tensor<64x1xi64, #blocked>
    %25 = tt.splat %N_8 : i64 -> tensor<1x64xi64, #blocked>
    %26 = tt.splat %c_ptr : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %27 = tt.splat %N_8 : i64 -> tensor<64x1xi64, #blocked>
    %accumulator = arith.cmpi eq, %6, %c0_i32 : i32
    scf.if %accumulator {
      %tile_id_c = scf.for %tile_id = %0 to %7 step %c4_i32 iter_args(%tile_id_c_9 = %8) -> (i32)  : i32 {
        %28 = arith.addi %tile_id_c_9, %c4_i32 : i32
        %29 = arith.divsi %28, %9 : i32
        %30 = arith.muli %29, %c8_i32 : i32
        %31 = arith.subi %2, %30 : i32
        %32 = arith.minsi %31, %c8_i32 : i32
        %33 = arith.remsi %28, %32 : i32
        %34 = arith.addi %30, %33 : i32
        %35 = arith.remsi %28, %9 : i32
        %36 = arith.divsi %35, %32 : i32
        %37 = arith.muli %34, %c64_i32 : i32
        %38 = arith.muli %36, %c64_i32 : i32
        %39 = arith.extsi %37 : i32 to i64
        %40 = arith.extsi %38 : i32 to i64
        %41 = tt.splat %39 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %42 = arith.addi %41, %22 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %43 = tt.expand_dims %42 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
        %44 = tt.splat %40 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %45 = arith.addi %44, %23 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %46 = tt.expand_dims %45 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
        %47 = arith.cmpi sge, %43, %cst_2 : tensor<64x1xi64, #blocked>
        %48 = arith.cmpi slt, %43, %24 : tensor<64x1xi64, #blocked>
        %49 = arith.andi %47, %48 : tensor<64x1xi1, #blocked>
        %50 = tt.broadcast %49 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %51 = arith.cmpi sge, %46, %cst_3 : tensor<1x64xi64, #blocked>
        %52 = arith.cmpi slt, %46, %25 : tensor<1x64xi64, #blocked>
        %53 = arith.andi %51, %52 : tensor<1x64xi1, #blocked>
        %54 = tt.broadcast %53 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %55 = arith.andi %50, %54 : tensor<64x64xi1, #blocked>
        %56 = arith.muli %43, %27 : tensor<64x1xi64, #blocked>
        %57 = tt.broadcast %56 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %58 = tt.broadcast %46 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %59 = arith.addi %57, %58 : tensor<64x64xi64, #blocked>
        %60 = tt.addptr %26, %59 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %61 = ttg.convert_layout %60 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #linear>
        %62 = ttg.convert_layout %55 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #linear>
        tt.store %61, %cst, %62 : tensor<64x64x!tt.ptr<bf16>, #linear>
        scf.yield %28 : i32
      } {tt.flatten}
    } else {
      %tile_id_c = arith.subi %7, %0 : i32
      %tile_id_c_9 = arith.ceildivsi %tile_id_c, %c4_i32 : i32
      %tile_id_c_10 = arith.maxsi %6, %c1_i32 : i32
      %tile_id_c_11 = arith.muli %tile_id_c_9, %tile_id_c_10 : i32
      %tile_id_c_12 = arith.subi %0, %c4_i32 : i32
      %tile_id_c_13 = arith.subi %tile_id_c_10, %c1_i32 : i32
      %tile_id_c_14 = arith.subi %tile_id_c_10, %c1_i32 : i32
      %tile_id_c_15:9 = scf.for %tile_id_c_16 = %c0_i32 to %tile_id_c_11 step %c1_i32 iter_args(%arg7 = %c0_i32, %tile_id_c_17 = %tile_id_c_12, %arg9 = %8, %arg10 = %c0_i32, %arg11 = %cst_5, %arg12 = %cst_1, %arg13 = %cst_0, %arg14 = %cst_1, %arg15 = %cst_0) -> (i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>)  : i32 {
        %tile_id_c_18 = arith.cmpi eq, %arg7, %c0_i32 : i32
        %tile_id_c_19 = arith.select %tile_id_c_18, %c0_i32, %arg10 : i32
        %tile_id_c_20:5 = scf.if %tile_id_c_18 -> (tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32) {
          %tile_id_c_28 = arith.addi %tile_id_c_17, %c4_i32 : i32
          %50 = arith.divsi %tile_id_c_28, %9 : i32
          %51 = arith.muli %50, %c8_i32 : i32
          %52 = arith.subi %2, %51 : i32
          %53 = arith.minsi %52, %c8_i32 : i32
          %54 = arith.remsi %tile_id_c_28, %53 : i32
          %55 = arith.addi %51, %54 : i32
          %56 = arith.remsi %tile_id_c_28, %9 : i32
          %57 = arith.divsi %56, %53 : i32
          %58 = arith.muli %55, %c64_i32 : i32
          %59 = arith.muli %57, %c64_i32 : i32
          %60 = arith.extsi %58 : i32 to i64
          %61 = tt.splat %60 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %62 = arith.addi %61, %12 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %63 = tt.expand_dims %62 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %64 = arith.cmpi sge, %63, %cst_2 : tensor<64x1xi64, #blocked>
          %65 = arith.cmpi slt, %63, %14 : tensor<64x1xi64, #blocked>
          %66 = arith.andi %64, %65 : tensor<64x1xi1, #blocked>
          %67 = tt.broadcast %66 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %68 = arith.muli %63, %17 : tensor<64x1xi64, #blocked>
          %69 = tt.broadcast %68 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %70 = arith.extsi %59 : i32 to i64
          %71 = tt.splat %70 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %72 = arith.addi %71, %12 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %73 = tt.expand_dims %72 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %74 = arith.cmpi sge, %73, %cst_2 : tensor<64x1xi64, #blocked>
          %75 = arith.cmpi slt, %73, %18 : tensor<64x1xi64, #blocked>
          %76 = arith.andi %74, %75 : tensor<64x1xi1, #blocked>
          %77 = tt.broadcast %76 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %78 = arith.muli %73, %17 : tensor<64x1xi64, #blocked>
          %79 = tt.broadcast %78 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          scf.yield %67, %69, %77, %79, %tile_id_c_28 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
        } else {
          scf.yield %arg12, %arg13, %arg14, %arg15, %tile_id_c_17 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
        }
        %28 = arith.muli %tile_id_c_19, %c64_i32 : i32
        %29 = arith.extsi %28 : i32 to i64
        %30 = tt.splat %29 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %31 = arith.addi %30, %13 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %32 = tt.expand_dims %31 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
        %33 = arith.cmpi sge, %32, %cst_3 : tensor<1x64xi64, #blocked>
        %34 = arith.cmpi slt, %32, %15 : tensor<1x64xi64, #blocked>
        %35 = arith.andi %33, %34 : tensor<1x64xi1, #blocked>
        %36 = tt.broadcast %35 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %37 = arith.andi %tile_id_c_20#0, %36 : tensor<64x64xi1, #blocked>
        %38 = tt.broadcast %32 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %39 = arith.addi %tile_id_c_20#1, %38 : tensor<64x64xi64, #blocked>
        %40 = tt.addptr %16, %39 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %41 = tt.load %40, %37, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
        %42 = arith.andi %tile_id_c_20#2, %36 : tensor<64x64xi1, #blocked>
        %43 = arith.addi %tile_id_c_20#3, %38 : tensor<64x64xi64, #blocked>
        %44 = tt.addptr %19, %43 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %45 = tt.load %44, %42, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
        %46 = ttg.convert_layout %45 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #linear1>
        %47 = tt.trans %46 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %48 = ttg.convert_layout %41 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        %49 = tt.dot %48, %47, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
        %tile_id_c_21 = arith.addi %tile_id_c_19, %c1_i32 : i32
        %tile_id_c_22 = arith.cmpi eq, %arg7, %tile_id_c_13 : i32
        %tile_id_c_23 = arith.select %tile_id_c_22, %cst_5, %49 : tensor<64x64xf32, #mma>
        %tile_id_c_24 = scf.if %tile_id_c_22 -> (i32) {
          %50 = arith.addi %arg9, %c4_i32 : i32
          %51 = arith.divsi %50, %9 : i32
          %52 = arith.muli %51, %c8_i32 : i32
          %53 = arith.subi %2, %52 : i32
          %54 = arith.minsi %53, %c8_i32 : i32
          %55 = arith.remsi %50, %54 : i32
          %56 = arith.addi %52, %55 : i32
          %57 = arith.remsi %50, %9 : i32
          %58 = arith.divsi %57, %54 : i32
          %59 = arith.muli %56, %c64_i32 : i32
          %60 = arith.muli %58, %c64_i32 : i32
          %61 = arith.truncf %49 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
          %62 = arith.extsi %59 : i32 to i64
          %63 = arith.extsi %60 : i32 to i64
          %64 = tt.splat %62 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %65 = arith.addi %64, %22 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %66 = tt.expand_dims %65 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %67 = tt.splat %63 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
          %68 = arith.addi %67, %23 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
          %69 = tt.expand_dims %68 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
          %70 = arith.cmpi sge, %66, %cst_2 : tensor<64x1xi64, #blocked>
          %71 = arith.cmpi slt, %66, %24 : tensor<64x1xi64, #blocked>
          %72 = arith.andi %70, %71 : tensor<64x1xi1, #blocked>
          %73 = tt.broadcast %72 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %74 = arith.cmpi sge, %69, %cst_3 : tensor<1x64xi64, #blocked>
          %75 = arith.cmpi slt, %69, %25 : tensor<1x64xi64, #blocked>
          %76 = arith.andi %74, %75 : tensor<1x64xi1, #blocked>
          %77 = tt.broadcast %76 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %78 = arith.andi %73, %77 : tensor<64x64xi1, #blocked>
          %79 = arith.muli %66, %27 : tensor<64x1xi64, #blocked>
          %80 = tt.broadcast %79 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %81 = tt.broadcast %69 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %82 = arith.addi %80, %81 : tensor<64x64xi64, #blocked>
          %83 = tt.addptr %26, %82 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
          %84 = ttg.convert_layout %83 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #linear>
          %85 = ttg.convert_layout %61 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
          %86 = ttg.convert_layout %78 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #linear>
          tt.store %84, %85, %86 : tensor<64x64x!tt.ptr<bf16>, #linear>
          scf.yield %50 : i32
        } else {
          scf.yield %arg9 : i32
        }
        %tile_id_c_25 = arith.addi %arg7, %c1_i32 : i32
        %tile_id_c_26 = arith.cmpi eq, %arg7, %tile_id_c_14 : i32
        %tile_id_c_27 = arith.select %tile_id_c_26, %c0_i32, %tile_id_c_25 : i32
        scf.yield %tile_id_c_27, %tile_id_c_20#4, %tile_id_c_24, %tile_id_c_21, %tile_id_c_23, %tile_id_c_20#0, %tile_id_c_20#1, %tile_id_c_20#2, %tile_id_c_20#3 : i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>
      } {tt.scheduled_max_stage = 1 : i32}
    }
    tt.return
  }
}
