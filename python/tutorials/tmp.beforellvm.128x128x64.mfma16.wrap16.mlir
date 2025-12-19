// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [32, 0], [64, 0]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [8, 0], [16, 0]], warp = [[2, 0], [4, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [8, 0], [4, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0]], warp = [[1, 0], [2, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [8, 0], [16, 0], [2, 0], [4, 0], [32, 0], [64, 0]], block = []}>
#shared1 = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [16, 0], [32, 0], [1, 0], [2, 0], [8, 0], [4, 0]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 67520 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} , %M: i32 {tt.divisibility = 16 : i32} , %N: i32 {tt.divisibility = 16 : i32} , %K: i32 {tt.divisibility = 16 : i32} , %stride_am: i32 {tt.divisibility = 16 : i32} , %stride_bk: i32 {tt.divisibility = 16 : i32} , %stride_cm: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c2_i32 = arith.constant 2 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_0 = arith.constant dense<64> : tensor<128x64xi32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %N, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    llvm.intr.assume %true : i1
    %5 = arith.muli %3, %c128_i32 : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %9 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %10 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %11 = tt.splat %5 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %12 = arith.addi %10, %6 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %13 = arith.addi %11, %7 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %14 = tt.splat %M : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %15 = arith.remsi %12, %14 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %16 = arith.muli %4, %c128_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %18 = tt.splat %16 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %19 = arith.addi %17, %8 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %20 = arith.addi %18, %9 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %21 = tt.splat %N : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %22 = arith.remsi %19, %21 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi32, #linear>
    %25 = tt.expand_dims %15 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %26 = tt.splat %stride_am : i32 -> tensor<128x1xi32, #linear>
    %27 = arith.muli %25, %26 : tensor<128x1xi32, #linear>
    %28 = tt.broadcast %27 : tensor<128x1xi32, #linear> -> tensor<128x64xi32, #linear>
    %29 = tt.broadcast %24 : tensor<1x64xi32, #linear> -> tensor<128x64xi32, #linear>
    %30 = arith.addi %28, %29 : tensor<128x64xi32, #linear>
    %31 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear1}>> -> tensor<64x1xi32, #linear1>
    %33 = tt.splat %stride_bk : i32 -> tensor<64x1xi32, #linear1>
    %34 = arith.muli %32, %33 : tensor<64x1xi32, #linear1>
    %35 = tt.broadcast %34 : tensor<64x1xi32, #linear1> -> tensor<64x128xi32, #linear1>
    %36 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #linear1}>> -> tensor<1x128xi32, #linear1>
    %37 = tt.broadcast %36 : tensor<1x128xi32, #linear1> -> tensor<64x128xi32, #linear1>
    %38 = arith.addi %35, %37 : tensor<64x128xi32, #linear1>
    %39 = arith.addi %K, %c63_i32 : i32
    %40 = arith.divsi %39, %c64_i32 : i32
    llvm.intr.assume %true : i1
    %41 = arith.muli %stride_bk, %c64_i32 : i32
    %42 = tt.splat %41 : i32 -> tensor<64x128xi32, #linear1>
    %43 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %44 = ttg.local_alloc {allocation.offset = 33760 : i32} : () -> !ttg.memdesc<2x64x128xf16, #shared1, #smem, mutable>
    %45 = tt.splat %K : i32 -> tensor<1x64xi32, #linear>
    %46 = arith.cmpi slt, %24, %45 : tensor<1x64xi32, #linear>
    %47 = tt.broadcast %46 : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
    %48 = ttg.memdesc_index %43[%c0_i32] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %49 = amdg.buffer_load_to_local %a_ptr[%30] mask = %47 stride = %stride_am into %48 {contiguity = 8 : i32} : <f16>[tensor<128x64xi32, #linear>]  -> <128x64xf16, #shared, #smem, mutable>
    %50 = ttg.async_commit_group tokens %49
    %51 = tt.splat %K : i32 -> tensor<64x1xi32, #linear1>
    %52 = arith.cmpi slt, %32, %51 : tensor<64x1xi32, #linear1>
    %53 = tt.broadcast %52 : tensor<64x1xi1, #linear1> -> tensor<64x128xi1, #linear1>
    %54 = ttg.memdesc_index %44[%c0_i32] : !ttg.memdesc<2x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    %55 = amdg.buffer_load_to_local %b_ptr[%38] mask = %53 stride = %stride_bk into %54 {contiguity = 8 : i32} : <f16>[tensor<64x128xi32, #linear1>]  -> <64x128xf16, #shared1, #smem, mutable>
    %56 = ttg.async_commit_group tokens %55
    %accumulator = arith.subi %40, %c1_i32 : i32
    cf.br ^bb1(%c0_i32, %cst, %c0_i32, %50, %56, %48, %54, %30, %38 : i32, tensor<128x128xf32, #mma>, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, tensor<128x64xi32, #linear>, tensor<64x128xi32, #linear1>)
  ^bb1(%accumulator_1: i32 , %57: tensor<128x128xf32, #mma>, %58: i32, %59: !ttg.async.token, %60: !ttg.async.token, %61: !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, %62: !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, %63: tensor<128x64xi32, #linear>, %64: tensor<64x128xi32, #linear1>):  // 2 preds: ^bb0, ^bb2
    %accumulator_2 = arith.cmpi slt, %accumulator_1, %accumulator : i32
    cf.cond_br %accumulator_2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %65 = amdg.async_wait %59, %60 {num_inst = 0 : i32}
    %66 = arith.addi %63, %cst_0 : tensor<128x64xi32, #linear>
    %67 = arith.addi %64, %42 : tensor<64x128xi32, #linear1>
    %accumulator_3 = arith.addi %58, %c1_i32 : i32
    %accumulator_4 = arith.cmpi slt, %accumulator_3, %c2_i32 : i32
    %accumulator_5 = arith.select %accumulator_4, %accumulator_3, %c0_i32 : i32
    %accumulator_6 = arith.addi %accumulator_1, %c1_i32 : i32
    %68 = arith.muli %accumulator_6, %c64_i32 : i32
    %69 = arith.subi %K, %68 : i32
    %70 = tt.splat %69 : i32 -> tensor<1x64xi32, #linear>
    %71 = arith.cmpi slt, %24, %70 : tensor<1x64xi32, #linear>
    %72 = tt.broadcast %71 : tensor<1x64xi1, #linear> -> tensor<128x64xi1, #linear>
    %73 = ttg.memdesc_index %43[%accumulator_5] : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %74 = amdg.buffer_load_to_local %a_ptr[%66] mask = %72 into %73 {contiguity = 8 : i32} : <f16>[tensor<128x64xi32, #linear>]  -> <128x64xf16, #shared, #smem, mutable>
    %75 = ttg.async_commit_group tokens %74
    %76 = ttg.local_load %61 token %65 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %77 = tt.splat %69 : i32 -> tensor<64x1xi32, #linear1>
    %78 = arith.cmpi slt, %32, %77 : tensor<64x1xi32, #linear1>
    %79 = tt.broadcast %78 : tensor<64x1xi1, #linear1> -> tensor<64x128xi1, #linear1>
    %80 = ttg.memdesc_index %44[%accumulator_5] : !ttg.memdesc<2x64x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>
    %81 = amdg.buffer_load_to_local %b_ptr[%67] mask = %79 into %80 {contiguity = 8 : i32} : <f16>[tensor<64x128xi32, #linear1>]  -> <64x128xf16, #shared1, #smem, mutable>
    %82 = ttg.async_commit_group tokens %81
    %83 = ttg.local_load %62 token %65 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %84 = tt.dot %76, %83, %57 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
    %accumulator_7 = arith.addi %accumulator_1, %c1_i32 : i32
    cf.br ^bb1(%accumulator_7, %84, %accumulator_5, %75, %82, %73, %80, %66, %67 : i32, tensor<128x128xf32, #mma>, i32, !ttg.async.token, !ttg.async.token, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x128xf16, #shared1, #smem, mutable>, tensor<128x64xi32, #linear>, tensor<64x128xi32, #linear1>)
  ^bb3:  // pred: ^bb1
    %85 = amdg.async_wait %59, %60 {num_inst = 0 : i32}
    %86 = ttg.local_load %61 token %85 : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %87 = ttg.local_load %62 token %85 : !ttg.memdesc<64x128xf16, #shared1, #smem, mutable> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %88 = tt.dot %86, %87, %57 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
    ttg.local_dealloc %44 : !ttg.memdesc<2x64x128xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %43 : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %89 = arith.truncf %88 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %90 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %91 = arith.muli %stride_cm, %5 : i32
    %92 = tt.expand_dims %20 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %93 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xi32, #mma>
    %94 = tt.splat %stride_cm : i32 -> tensor<128x1xi32, #mma>
    %95 = arith.muli %94, %93 : tensor<128x1xi32, #mma>
    %96 = tt.broadcast %95 : tensor<128x1xi32, #mma> -> tensor<128x128xi32, #mma>
    %97 = tt.expand_dims %9 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
    %98 = tt.broadcast %97 : tensor<1x128xi32, #mma> -> tensor<128x128xi32, #mma>
    %99 = arith.addi %91, %16 : i32
    %100 = arith.addi %96, %98 : tensor<128x128xi32, #mma>
    %101 = tt.splat %99 : i32 -> tensor<128x128xi32, #mma>
    %102 = arith.addi %101, %100 : tensor<128x128xi32, #mma>
    %103 = tt.splat %M : i32 -> tensor<128x1xi32, #mma>
    %104 = arith.cmpi slt, %90, %103 : tensor<128x1xi32, #mma>
    %105 = tt.splat %N : i32 -> tensor<1x128xi32, #mma>
    %106 = arith.cmpi slt, %92, %105 : tensor<1x128xi32, #mma>
    %107 = tt.broadcast %104 : tensor<128x1xi1, #mma> -> tensor<128x128xi1, #mma>
    %108 = tt.broadcast %106 : tensor<1x128xi1, #mma> -> tensor<128x128xi1, #mma>
    %109 = arith.andi %107, %108 : tensor<128x128xi1, #mma>
    amdg.buffer_store %89, %c_ptr[%102], %109 : tensor<128x128xf16, #mma>
    tt.return
  }
}
