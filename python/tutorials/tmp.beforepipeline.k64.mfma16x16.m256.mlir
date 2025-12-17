// -----// IR Dump Before TritonAMDGPUPipeline (tritonamdgpu-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} , %K: i32 {tt.divisibility = 16 : i32} , %stride_am: i32 {tt.divisibility = 16 : i32} , %stride_bk: i32 {tt.divisibility = 16 : i32} , %stride_cm: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %N, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.cmpi sge, %3, %c0_i32 : i32
    llvm.intr.assume %5 : i1
    %6 = arith.cmpi sge, %4, %c0_i32 : i32
    llvm.intr.assume %6 : i1
    %7 = arith.cmpi sgt, %stride_am, %c0_i32 : i32
    llvm.intr.assume %7 : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %8 = arith.cmpi sgt, %stride_bk, %c0_i32 : i32
    llvm.intr.assume %8 : i1
    %9 = arith.cmpi sgt, %stride_cm, %c0_i32 : i32
    llvm.intr.assume %9 : i1
    llvm.intr.assume %true : i1
    %10 = arith.muli %3, %c256_i32 : i32
    %11 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %14 = tt.splat %10 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = tt.splat %10 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = arith.addi %14, %11 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = arith.addi %15, %12 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.splat %M : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = arith.remsi %16, %18 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.muli %4, %c256_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = arith.addi %21, %13 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %23 = tt.splat %N : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = arith.remsi %22, %23 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.expand_dims %19 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %26 = tt.splat %stride_am : i32 -> tensor<256x1xi32, #blocked>
    %27 = arith.muli %25, %26 : tensor<256x1xi32, #blocked>
    %28 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %30 = tt.broadcast %27 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %31 = tt.broadcast %29 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %32 = arith.addi %30, %31 : tensor<256x64xi32, #blocked>
    %33 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %34 = tt.addptr %33, %32 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
    %35 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %36 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %37 = tt.expand_dims %35 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %38 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %39 = tt.splat %stride_bk : i32 -> tensor<64x1xi32, #blocked1>
    %40 = arith.muli %37, %39 : tensor<64x1xi32, #blocked1>
    %41 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %42 = tt.broadcast %40 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %43 = tt.broadcast %41 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %44 = arith.addi %42, %43 : tensor<64x256xi32, #blocked1>
    %45 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %46 = tt.addptr %45, %44 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
    %47 = arith.addi %K, %c63_i32 : i32
    %48 = arith.divsi %47, %c64_i32 : i32
    %49 = arith.muli %stride_bk, %c64_i32 : i32
    %50 = tt.splat %49 : i32 -> tensor<64x256xi32, #blocked1>
    %accumulator:3 = scf.for %accumulator_3 = %c0_i32 to %48 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %34, %arg12 = %46) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>)  : i32 {
      %70 = arith.muli %accumulator_3, %c64_i32 : i32
      %71 = arith.subi %K, %70 : i32
      %72 = tt.splat %71 : i32 -> tensor<1x64xi32, #blocked>
      %73 = arith.cmpi slt, %29, %72 : tensor<1x64xi32, #blocked>
      %74 = tt.broadcast %73 : tensor<1x64xi1, #blocked> -> tensor<256x64xi1, #blocked>
      %75 = tt.load %arg11, %74, %cst_0 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<256x64x!tt.ptr<f16>, #blocked>
      %76 = tt.splat %71 : i32 -> tensor<64x1xi32, #blocked1>
      %77 = arith.cmpi slt, %38, %76 : tensor<64x1xi32, #blocked1>
      %78 = tt.broadcast %77 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
      %79 = tt.load %arg12, %78, %cst_1 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %80 = ttg.convert_layout %75 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %81 = ttg.convert_layout %79 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %82 = tt.dot %80, %81, %arg10 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %83 = tt.addptr %arg11, %cst : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
      %84 = tt.addptr %arg12, %50 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      scf.yield %82, %83, %84 : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>
    } {tt.scheduled_max_stage = 1 : i32}
    %51 = arith.truncf %accumulator#0 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %52 = tt.expand_dims %17 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %53 = tt.splat %stride_cm : i32 -> tensor<256x1xi32, #blocked1>
    %54 = arith.muli %53, %52 : tensor<256x1xi32, #blocked1>
    %55 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #blocked1>
    %56 = tt.addptr %55, %54 : tensor<256x1x!tt.ptr<f16>, #blocked1>, tensor<256x1xi32, #blocked1>
    %57 = tt.expand_dims %22 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %58 = tt.broadcast %56 : tensor<256x1x!tt.ptr<f16>, #blocked1> -> tensor<256x256x!tt.ptr<f16>, #blocked1>
    %59 = tt.broadcast %57 : tensor<1x256xi32, #blocked1> -> tensor<256x256xi32, #blocked1>
    %60 = tt.addptr %58, %59 : tensor<256x256x!tt.ptr<f16>, #blocked1>, tensor<256x256xi32, #blocked1>
    %61 = tt.splat %M : i32 -> tensor<256x1xi32, #blocked1>
    %62 = arith.cmpi slt, %52, %61 : tensor<256x1xi32, #blocked1>
    %63 = tt.splat %N : i32 -> tensor<1x256xi32, #blocked1>
    %64 = arith.cmpi slt, %57, %63 : tensor<1x256xi32, #blocked1>
    %65 = tt.broadcast %62 : tensor<256x1xi1, #blocked1> -> tensor<256x256xi1, #blocked1>
    %66 = tt.broadcast %64 : tensor<1x256xi1, #blocked1> -> tensor<256x256xi1, #blocked1>
    %67 = arith.andi %65, %66 : tensor<256x256xi1, #blocked1>
    %68 = ttg.convert_layout %60 : tensor<256x256x!tt.ptr<f16>, #blocked1> -> tensor<256x256x!tt.ptr<f16>, #mma>
    %69 = ttg.convert_layout %67 : tensor<256x256xi1, #blocked1> -> tensor<256x256xi1, #mma>
    tt.store %68, %51, %69 : tensor<256x256x!tt.ptr<f16>, #mma>
    tt.return
  }
}
