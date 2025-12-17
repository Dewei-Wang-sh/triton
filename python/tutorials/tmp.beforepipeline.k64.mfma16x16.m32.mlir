// -----// IR Dump Before TritonAMDGPUPipeline (tritonamdgpu-pipeline) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} , %K: i32 {tt.divisibility = 16 : i32} , %stride_am: i32 {tt.divisibility = 16 : i32} , %stride_bk: i32 {tt.divisibility = 16 : i32} , %stride_cm: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<32x64xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %N, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
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
    %10 = arith.muli %3, %c32_i32 : i32
    %11 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = tt.splat %10 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.addi %12, %11 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.splat %M : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = arith.remsi %13, %14 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.muli %4, %c64_i32 : i32
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %18 = tt.splat %16 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = arith.addi %18, %17 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.splat %N : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %21 = arith.remsi %19, %20 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %22 = tt.expand_dims %15 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %23 = tt.splat %stride_am : i32 -> tensor<32x1xi32, #blocked>
    %24 = arith.muli %22, %23 : tensor<32x1xi32, #blocked>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %26 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %27 = tt.broadcast %24 : tensor<32x1xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %28 = tt.broadcast %26 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %29 = arith.addi %27, %28 : tensor<32x64xi32, #blocked>
    %30 = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %31 = tt.addptr %30, %29 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    %32 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %33 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %34 = tt.expand_dims %32 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %35 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %36 = tt.splat %stride_bk : i32 -> tensor<64x1xi32, #blocked>
    %37 = arith.muli %34, %36 : tensor<64x1xi32, #blocked>
    %38 = tt.expand_dims %21 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %39 = tt.broadcast %37 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %40 = tt.broadcast %38 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %41 = arith.addi %39, %40 : tensor<64x64xi32, #blocked>
    %42 = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %43 = tt.addptr %42, %41 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %44 = arith.addi %K, %c63_i32 : i32
    %45 = arith.divsi %44, %c64_i32 : i32
    %46 = arith.muli %stride_bk, %c64_i32 : i32
    %47 = tt.splat %46 : i32 -> tensor<64x64xi32, #blocked>
    %accumulator:3 = scf.for %accumulator_3 = %c0_i32 to %45 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %31, %arg12 = %43) -> (tensor<32x64xf32, #mma>, tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %67 = arith.muli %accumulator_3, %c64_i32 : i32
      %68 = arith.subi %K, %67 : i32
      %69 = tt.splat %68 : i32 -> tensor<1x64xi32, #blocked>
      %70 = arith.cmpi slt, %26, %69 : tensor<1x64xi32, #blocked>
      %71 = tt.broadcast %70 : tensor<1x64xi1, #blocked> -> tensor<32x64xi1, #blocked>
      %72 = tt.load %arg11, %71, %cst_0 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked>
      %73 = tt.splat %68 : i32 -> tensor<64x1xi32, #blocked>
      %74 = arith.cmpi slt, %35, %73 : tensor<64x1xi32, #blocked>
      %75 = tt.broadcast %74 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %76 = tt.load %arg12, %75, %cst_1 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %77 = ttg.convert_layout %72 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %78 = ttg.convert_layout %76 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %79 = tt.dot %77, %78, %arg10 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x64xf32, #mma>
      %80 = tt.addptr %arg11, %cst : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
      %81 = tt.addptr %arg12, %47 : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      scf.yield %79, %80, %81 : tensor<32x64xf32, #mma>, tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<64x64x!tt.ptr<f16>, #blocked>
    } {tt.scheduled_max_stage = 1 : i32}
    %48 = arith.truncf %accumulator#0 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
    %49 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %50 = tt.splat %stride_cm : i32 -> tensor<32x1xi32, #blocked>
    %51 = arith.muli %50, %49 : tensor<32x1xi32, #blocked>
    %52 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked>
    %53 = tt.addptr %52, %51 : tensor<32x1x!tt.ptr<f16>, #blocked>, tensor<32x1xi32, #blocked>
    %54 = tt.expand_dims %19 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %55 = tt.broadcast %53 : tensor<32x1x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %56 = tt.broadcast %54 : tensor<1x64xi32, #blocked> -> tensor<32x64xi32, #blocked>
    %57 = tt.addptr %55, %56 : tensor<32x64x!tt.ptr<f16>, #blocked>, tensor<32x64xi32, #blocked>
    %58 = tt.splat %M : i32 -> tensor<32x1xi32, #blocked>
    %59 = arith.cmpi slt, %49, %58 : tensor<32x1xi32, #blocked>
    %60 = tt.splat %N : i32 -> tensor<1x64xi32, #blocked>
    %61 = arith.cmpi slt, %54, %60 : tensor<1x64xi32, #blocked>
    %62 = tt.broadcast %59 : tensor<32x1xi1, #blocked> -> tensor<32x64xi1, #blocked>
    %63 = tt.broadcast %61 : tensor<1x64xi1, #blocked> -> tensor<32x64xi1, #blocked>
    %64 = arith.andi %62, %63 : tensor<32x64xi1, #blocked>
    %65 = ttg.convert_layout %57 : tensor<32x64x!tt.ptr<f16>, #blocked> -> tensor<32x64x!tt.ptr<f16>, #mma>
    %66 = ttg.convert_layout %64 : tensor<32x64xi1, #blocked> -> tensor<32x64xi1, #mma>
    tt.store %65, %48, %66 : tensor<32x64x!tt.ptr<f16>, #mma>
    tt.return
  }
}
