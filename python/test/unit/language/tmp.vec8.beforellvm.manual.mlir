
// -----// IR Dump Before ConvertTritonAMDGPUToLLVM (convert-triton-amdgpu-to-llvm) ('builtin.module' operation) //----- //
#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 8]], warp = [[0, 0, 0], [1, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 16], [0, 0, 32]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 0, 8]], warp = [[1, 0, 0], [0, 0, 0]], block = []}>
#linear3 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [2, 0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32768 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_reshape(%a_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("a_ptr"), %b_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("b_ptr"), %c_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("c_ptr"), %M: i32 {tt.divisibility = 16 : i32} loc("M"), %N: i32 {tt.divisibility = 16 : i32} loc("N"), %K: i32 {tt.divisibility = 16 : i32} loc("K")) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #linear>
    %cst_1 = arith.constant dense<0> : tensor<2x1x1xi64, #blocked>
    %cst_2 = arith.constant dense<2> : tensor<2x1x1xi64, #blocked>
    %cst_3 = arith.constant dense<0> : tensor<1x32x1xi64, #blocked>
    %cst_4 = arith.constant dense<0> : tensor<1x1x64xi64, #blocked>
    %cst_5 = arith.constant dense<0> : tensor<64x1xi64, #linear>
    %cst_6 = arith.constant dense<0> : tensor<1x64xi64, #linear>
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
    %K_7 = arith.extsi %K : i32 to i64
    %11 = arith.extsi %8 : i32 to i64
    %12 = arith.divsi %N, %c2_i32 : i32
    %13 = arith.muli %12, %K : i32
    %14 = arith.extsi %13 : i32 to i64
    %15 = arith.extsi %12 : i32 to i64
    %N_8 = arith.extsi %N : i32 to i64
    %M_9 = arith.extsi %M : i32 to i64
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
    %26 = arith.cmpi sge, %21, %cst_1 : tensor<2x1x1xi64, #blocked>
    %27 = arith.cmpi slt, %21, %cst_2 : tensor<2x1x1xi64, #blocked>
    %28 = arith.andi %26, %27 : tensor<2x1x1xi1, #blocked>
    %29 = tt.broadcast %28 : tensor<2x1x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %30 = tt.splat %11 : i64 -> tensor<1x32x1xi64, #blocked>
    %31 = tt.splat %K_7 : i64 -> tensor<1x1x64xi64, #blocked>
    %32 = tt.splat %K_7 : i64 -> tensor<1x32x1xi64, #blocked>
    %33 = tt.splat %15 : i64 -> tensor<1x32x1xi64, #blocked>
    %34 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %35 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %36 = arith.extsi %34 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %37 = arith.extsi %35 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #linear}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %38 = tt.splat %M_9 : i64 -> tensor<64x1xi64, #linear>
    %39 = tt.splat %N_8 : i64 -> tensor<1x64xi64, #linear>
    %accumulator = arith.cmpi eq, %6, %c0_i32 : i32
    cf.cond_br %accumulator, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    cf.br ^bb2(%0, %16 : i32, i32)
  ^bb2(%tile_id: i32 loc("tile_id"), %tile_id_c: i32 loc("tile_id_c")):  // 2 preds: ^bb1, ^bb3
    %tile_id_c_10 = arith.cmpi slt, %tile_id, %7 : i32
    cf.cond_br %tile_id_c_10, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %40 = arith.addi %tile_id_c, %c4_i32 : i32
    %41 = arith.divsi %40, %17 : i32
    %42 = arith.muli %41, %c8_i32 : i32
    %43 = arith.subi %2, %42 : i32
    %44 = arith.minsi %43, %c8_i32 : i32
    %45 = arith.remsi %40, %44 : i32
    %46 = arith.addi %42, %45 : i32
    %47 = arith.remsi %40, %17 : i32
    %48 = arith.divsi %47, %44 : i32
    %49 = arith.muli %46, %c64_i32 : i32
    %50 = arith.muli %48, %c64_i32 : i32
    %51 = arith.extsi %49 : i32 to i64
    %52 = arith.extsi %50 : i32 to i64
    %53 = tt.splat %51 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %54 = arith.addi %53, %36 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %55 = tt.expand_dims %54 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %56 = tt.splat %52 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %57 = arith.addi %56, %37 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %58 = tt.expand_dims %57 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %59 = arith.cmpi sge, %55, %cst_5 : tensor<64x1xi64, #linear>
    %60 = arith.cmpi slt, %55, %38 : tensor<64x1xi64, #linear>
    %61 = arith.andi %59, %60 : tensor<64x1xi1, #linear>
    %62 = tt.broadcast %61 : tensor<64x1xi1, #linear> -> tensor<64x64xi1, #linear>
    %63 = arith.cmpi sge, %58, %cst_6 : tensor<1x64xi64, #linear>
    %64 = arith.cmpi slt, %58, %39 : tensor<1x64xi64, #linear>
    %65 = arith.andi %63, %64 : tensor<1x64xi1, #linear>
    %66 = tt.broadcast %65 : tensor<1x64xi1, #linear> -> tensor<64x64xi1, #linear>
    %67 = arith.andi %62, %66 : tensor<64x64xi1, #linear>
    %68 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %69 = arith.muli %51, %N_8 : i64
    %70 = tt.splat %N_8 : i64 -> tensor<64x1xi64, #linear>
    %71 = arith.muli %68, %70 : tensor<64x1xi64, #linear>
    %72 = tt.broadcast %71 : tensor<64x1xi64, #linear> -> tensor<64x64xi64, #linear>
    %73 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %74 = tt.broadcast %73 : tensor<1x64xi64, #linear> -> tensor<64x64xi64, #linear>
    %75 = arith.addi %69, %52 : i64
    %76 = arith.addi %72, %74 : tensor<64x64xi64, #linear>
    %77 = tt.splat %75 : i64 -> tensor<64x64xi64, #linear>
    %78 = arith.addi %77, %76 : tensor<64x64xi64, #linear>
    %79 = arith.trunci %78 : tensor<64x64xi64, #linear> to tensor<64x64xi32, #linear>
    amdg.buffer_store %cst_0, %c_ptr[%79], %67 : tensor<64x64xbf16, #linear>
    %tile_id_c_11 = arith.addi %tile_id, %c4_i32 : i32
    cf.br ^bb2(%tile_id_c_11, %40 : i32, i32)
  ^bb4:  // pred: ^bb2
    cf.br ^bb23
  ^bb5:  // pred: ^bb0
    %tile_id_c_12 = arith.subi %7, %0 : i32
    %tile_id_c_13 = arith.ceildivsi %tile_id_c_12, %c4_i32 : i32
    %tile_id_c_14 = arith.maxsi %6, %c1_i32 : i32
    %tile_id_c_15 = arith.muli %tile_id_c_13, %tile_id_c_14 : i32
    %tile_id_c_16 = arith.subi %tile_id_c_14, %c1_i32 : i32
    %80 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable>
    %81 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable>
    %tile_id_c_17 = arith.cmpi sgt, %tile_id_c_15, %c0_i32 : i32
    %82 = arith.divsi %0, %17 : i32
    %83 = arith.muli %82, %c8_i32 : i32
    %84 = arith.subi %2, %83 : i32
    %85 = arith.minsi %84, %c8_i32 : i32
    %86 = arith.remsi %0, %85 : i32
    %87 = arith.addi %83, %86 : i32
    %88 = arith.remsi %0, %17 : i32
    %89 = arith.divsi %88, %85 : i32
    %90 = arith.muli %87, %c32_i32 : i32
    %91 = arith.muli %89, %c32_i32 : i32
    %92 = arith.extsi %90 : i32 to i64
    %93 = tt.splat %92 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %94 = arith.addi %93, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %95 = tt.expand_dims %94 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %96 = tt.expand_dims %95 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
    %97 = arith.cmpi sge, %96, %cst_3 : tensor<1x32x1xi64, #blocked>
    %98 = arith.cmpi slt, %96, %30 : tensor<1x32x1xi64, #blocked>
    %99 = arith.andi %97, %98 : tensor<1x32x1xi1, #blocked>
    %100 = tt.broadcast %99 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %101 = arith.andi %29, %100 : tensor<2x32x64xi1, #blocked>
    %102 = arith.muli %96, %32 : tensor<1x32x1xi64, #blocked>
    %103 = tt.broadcast %102 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %104 = arith.extsi %91 : i32 to i64
    %105 = tt.splat %104 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %106 = arith.addi %105, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %107 = tt.expand_dims %106 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %108 = tt.expand_dims %107 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
    %109 = arith.cmpi sge, %108, %cst_3 : tensor<1x32x1xi64, #blocked>
    %110 = arith.cmpi slt, %108, %33 : tensor<1x32x1xi64, #blocked>
    %111 = arith.andi %109, %110 : tensor<1x32x1xi1, #blocked>
    %112 = tt.broadcast %111 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %113 = arith.andi %29, %112 : tensor<2x32x64xi1, #blocked>
    %114 = arith.muli %108, %32 : tensor<1x32x1xi64, #blocked>
    %115 = tt.broadcast %114 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %116 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %117 = tt.expand_dims %116 {axis = 1 : i32} : tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x64xi64, #blocked>
    %118 = arith.cmpi sge, %117, %cst_4 : tensor<1x1x64xi64, #blocked>
    %119 = arith.cmpi slt, %117, %31 : tensor<1x1x64xi64, #blocked>
    %120 = arith.andi %118, %119 : tensor<1x1x64xi1, #blocked>
    %121 = tt.broadcast %120 : tensor<1x1x64xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %122 = arith.andi %101, %121 : tensor<2x32x64xi1, #blocked>
    %123 = tt.splat %10 : i64 -> tensor<2x1x1xi64, #blocked>
    %124 = arith.muli %21, %123 : tensor<2x1x1xi64, #blocked>
    %125 = tt.broadcast %124 : tensor<2x1x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %126 = tt.expand_dims %23 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %127 = tt.expand_dims %126 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
    %128 = arith.muli %92, %K_7 : i64
    %129 = arith.muli %127, %32 : tensor<1x32x1xi64, #blocked>
    %130 = tt.broadcast %129 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %131 = tt.broadcast %117 : tensor<1x1x64xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %132 = arith.addi %130, %131 : tensor<2x32x64xi64, #blocked>
    %133 = arith.addi %125, %132 : tensor<2x32x64xi64, #blocked>
    %134 = tt.splat %128 : i64 -> tensor<2x32x64xi64, #blocked>
    %135 = arith.addi %134, %133 : tensor<2x32x64xi64, #blocked>
    %136 = arith.trunci %135 : tensor<2x32x64xi64, #blocked> to tensor<2x32x64xi32, #blocked>
    %137 = ttg.memdesc_index %80[%c0_i32] : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
    %tile_id_c_18 = tt.splat %tile_id_c_17 : i1 -> tensor<2x32x64xi1, #blocked>
    %tile_id_c_19 = arith.andi %tile_id_c_18, %122 : tensor<2x32x64xi1, #blocked>
    %138 = amdg.buffer_load_to_local %a_ptr[%136] mask = %tile_id_c_19 into %137 {contiguity = 8 : i32} : <bf16>[tensor<2x32x64xi32, #blocked>]  -> <2x32x64xbf16, #shared, #smem, mutable>
    %139 = ttg.async_commit_group tokens %138
    %140 = arith.andi %113, %121 : tensor<2x32x64xi1, #blocked>
    %141 = tt.splat %14 : i64 -> tensor<2x1x1xi64, #blocked>
    %142 = arith.muli %21, %141 : tensor<2x1x1xi64, #blocked>
    %143 = tt.broadcast %142 : tensor<2x1x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %144 = arith.muli %104, %K_7 : i64
    %145 = arith.addi %143, %132 : tensor<2x32x64xi64, #blocked>
    %146 = tt.splat %144 : i64 -> tensor<2x32x64xi64, #blocked>
    %147 = arith.addi %146, %145 : tensor<2x32x64xi64, #blocked>
    %148 = arith.trunci %147 : tensor<2x32x64xi64, #blocked> to tensor<2x32x64xi32, #blocked>
    %149 = ttg.memdesc_index %81[%c0_i32] : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
    %tile_id_c_20 = arith.andi %tile_id_c_18, %140 : tensor<2x32x64xi1, #blocked>
    %150 = amdg.buffer_load_to_local %b_ptr[%148] mask = %tile_id_c_20 into %149 {contiguity = 8 : i32} : <bf16>[tensor<2x32x64xi32, #blocked>]  -> <2x32x64xbf16, #shared, #smem, mutable>
    %151 = ttg.async_commit_group tokens %150
    %tile_id_c_21 = arith.subi %tile_id_c_15, %c1_i32 : i32
    cf.br ^bb6(%c0_i32, %c0_i32, %0, %16, %cst, %c0_i32, %139, %151, %c0_i32, %137, %149, %103, %101, %115, %113 : i32, i32, i32, i32, tensor<64x64xf32, #mma>, i32, !ttg.async.token, !ttg.async.token, i32, !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>)
  ^bb6(%tile_id_c_22: i32 loc("tile_id_c"), %152: i32, %153: i32, %154: i32, %155: tensor<64x64xf32, #mma>, %156: i32, %157: !ttg.async.token, %158: !ttg.async.token, %159: i32, %160: !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, %161: !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, %162: tensor<2x32x64xi64, #blocked>, %163: tensor<2x32x64xi1, #blocked>, %164: tensor<2x32x64xi64, #blocked>, %165: tensor<2x32x64xi1, #blocked>):  // 2 preds: ^bb5, ^bb15
    %tile_id_c_23 = arith.cmpi slt, %tile_id_c_22, %tile_id_c_21 : i32
    cf.cond_br %tile_id_c_23, ^bb7, ^bb16
  ^bb7:  // pred: ^bb6
    %166 = amdg.async_wait %157, %158 {num_inst = 0 : i32}
    %tile_id_c_24 = arith.addi %159, %c1_i32 : i32
    %tile_id_c_25 = arith.addi %152, %c1_i32 : i32
    %tile_id_c_26 = arith.cmpi eq, %152, %tile_id_c_16 : i32
    %tile_id_c_27 = arith.select %tile_id_c_26, %c0_i32, %tile_id_c_25 : i32
    %tile_id_c_28 = arith.addi %156, %c1_i32 : i32
    %tile_id_c_29 = arith.cmpi slt, %tile_id_c_28, %c2_i32 : i32
    %tile_id_c_30 = arith.select %tile_id_c_29, %tile_id_c_28, %c0_i32 : i32
    %tile_id_c_31 = arith.cmpi eq, %tile_id_c_27, %c0_i32 : i32
    %tile_id_c_32 = arith.select %tile_id_c_31, %c0_i32, %tile_id_c_24 : i32
    cf.cond_br %tile_id_c_31, ^bb8, ^bb9
  ^bb8:  // pred: ^bb7
    %tile_id_c_33 = arith.addi %153, %c4_i32 : i32
    %167 = arith.divsi %tile_id_c_33, %17 : i32
    %168 = arith.muli %167, %c8_i32 : i32
    %169 = arith.subi %2, %168 : i32
    %170 = arith.minsi %169, %c8_i32 : i32
    %171 = arith.remsi %tile_id_c_33, %170 : i32
    %172 = arith.addi %168, %171 : i32
    %173 = arith.remsi %tile_id_c_33, %17 : i32
    %174 = arith.divsi %173, %170 : i32
    %175 = arith.muli %172, %c32_i32 : i32
    %176 = arith.muli %174, %c32_i32 : i32
    %177 = arith.extsi %175 : i32 to i64
    %178 = tt.splat %177 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %179 = arith.addi %178, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %180 = tt.expand_dims %179 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %181 = tt.expand_dims %180 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
    %182 = arith.cmpi sge, %181, %cst_3 : tensor<1x32x1xi64, #blocked>
    %183 = arith.cmpi slt, %181, %30 : tensor<1x32x1xi64, #blocked>
    %184 = arith.andi %182, %183 : tensor<1x32x1xi1, #blocked>
    %185 = tt.broadcast %184 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %186 = arith.andi %29, %185 : tensor<2x32x64xi1, #blocked>
    %187 = arith.muli %181, %32 : tensor<1x32x1xi64, #blocked>
    %188 = tt.broadcast %187 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    %189 = arith.extsi %176 : i32 to i64
    %190 = tt.splat %189 : i64 -> tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %191 = arith.addi %190, %23 : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %192 = tt.expand_dims %191 {axis = 0 : i32} : tensor<32xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>>
    %193 = tt.expand_dims %192 {axis = 2 : i32} : tensor<1x32xi64, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x32x1xi64, #blocked>
    %194 = arith.cmpi sge, %193, %cst_3 : tensor<1x32x1xi64, #blocked>
    %195 = arith.cmpi slt, %193, %33 : tensor<1x32x1xi64, #blocked>
    %196 = arith.andi %194, %195 : tensor<1x32x1xi1, #blocked>
    %197 = tt.broadcast %196 : tensor<1x32x1xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %198 = arith.andi %29, %197 : tensor<2x32x64xi1, #blocked>
    %199 = arith.muli %193, %32 : tensor<1x32x1xi64, #blocked>
    %200 = tt.broadcast %199 : tensor<1x32x1xi64, #blocked> -> tensor<2x32x64xi64, #blocked>
    cf.br ^bb10(%tile_id_c_33, %188, %186, %200, %198 : i32, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>)
  ^bb9:  // pred: ^bb7
    cf.br ^bb10(%153, %162, %163, %164, %165 : i32, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>)
  ^bb10(%tile_id_c_34: i32 loc("tile_id_c"), %tile_id_c_35: tensor<2x32x64xi64, #blocked> loc("tile_id_c"), %tile_id_c_36: tensor<2x32x64xi1, #blocked> loc("tile_id_c"), %tile_id_c_37: tensor<2x32x64xi64, #blocked> loc("tile_id_c"), %tile_id_c_38: tensor<2x32x64xi1, #blocked> loc("tile_id_c")):  // 2 preds: ^bb8, ^bb9
    cf.br ^bb11
  ^bb11:  // pred: ^bb10
    %201 = arith.muli %tile_id_c_32, %c64_i32 : i32
    %202 = arith.extsi %201 : i32 to i64
    %203 = tt.splat %202 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %204 = arith.addi %203, %25 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %205 = tt.expand_dims %204 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %206 = tt.expand_dims %205 {axis = 1 : i32} : tensor<1x64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x64xi64, #blocked>
    %207 = arith.cmpi sge, %206, %cst_4 : tensor<1x1x64xi64, #blocked>
    %208 = arith.cmpi slt, %206, %31 : tensor<1x1x64xi64, #blocked>
    %209 = arith.andi %207, %208 : tensor<1x1x64xi1, #blocked>
    %210 = tt.broadcast %209 : tensor<1x1x64xi1, #blocked> -> tensor<2x32x64xi1, #blocked>
    %211 = arith.andi %tile_id_c_36, %210 : tensor<2x32x64xi1, #blocked>
    %212 = arith.addi %tile_id_c_35, %131 : tensor<2x32x64xi64, #blocked>
    %213 = arith.addi %125, %212 : tensor<2x32x64xi64, #blocked>
    %214 = tt.splat %202 : i64 -> tensor<2x32x64xi64, #blocked>
    %215 = arith.addi %214, %213 : tensor<2x32x64xi64, #blocked>
    %216 = arith.trunci %215 : tensor<2x32x64xi64, #blocked> to tensor<2x32x64xi32, #blocked>
    %217 = ttg.memdesc_index %80[%tile_id_c_30] : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
    %218 = amdg.buffer_load_to_local %a_ptr[%216] mask = %211 into %217 {contiguity = 8 : i32} : <bf16>[tensor<2x32x64xi32, #blocked>]  -> <2x32x64xbf16, #shared, #smem, mutable>
    %219 = ttg.async_commit_group tokens %218
    %220 = ttg.local_load %160 token %166 : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> tensor<2x32x64xbf16, #linear1>
    %221 = tt.reshape %220 : tensor<2x32x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %222 = arith.andi %tile_id_c_38, %210 : tensor<2x32x64xi1, #blocked>
    %223 = arith.addi %tile_id_c_37, %131 : tensor<2x32x64xi64, #blocked>
    %224 = arith.addi %143, %223 : tensor<2x32x64xi64, #blocked>
    %225 = arith.addi %214, %224 : tensor<2x32x64xi64, #blocked>
    %226 = arith.trunci %225 : tensor<2x32x64xi64, #blocked> to tensor<2x32x64xi32, #blocked>
    %227 = ttg.memdesc_index %81[%tile_id_c_30] : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>
    %228 = amdg.buffer_load_to_local %b_ptr[%226] mask = %222 into %227 {contiguity = 8 : i32} : <bf16>[tensor<2x32x64xi32, #blocked>]  -> <2x32x64xbf16, #shared, #smem, mutable>
    %229 = ttg.async_commit_group tokens %228
    %230 = ttg.local_load %161 token %166 : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> tensor<2x32x64xbf16, #linear2>
    %231 = tt.reshape %230 : tensor<2x32x64xbf16, #linear2> -> tensor<64x64xbf16, #linear3>
    %232 = tt.trans %231 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear3> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %233 = tt.dot %221, %232, %155 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
    %tile_id_c_39 = arith.select %tile_id_c_26, %cst, %233 : tensor<64x64xf32, #mma>
    cf.cond_br %tile_id_c_26, ^bb12, ^bb13
  ^bb12:  // pred: ^bb11
    %234 = arith.addi %154, %c4_i32 : i32
    %235 = arith.divsi %234, %17 : i32
    %236 = arith.muli %235, %c8_i32 : i32
    %237 = arith.subi %2, %236 : i32
    %238 = arith.minsi %237, %c8_i32 : i32
    %239 = arith.remsi %234, %238 : i32
    %240 = arith.addi %236, %239 : i32
    %241 = arith.remsi %234, %17 : i32
    %242 = arith.divsi %241, %238 : i32
    %243 = arith.muli %240, %c64_i32 : i32
    %244 = arith.muli %242, %c64_i32 : i32
    %245 = arith.truncf %233 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
    %246 = arith.extsi %243 : i32 to i64
    %247 = arith.extsi %244 : i32 to i64
    %248 = tt.splat %246 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %249 = arith.addi %248, %36 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %250 = tt.expand_dims %249 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %251 = tt.splat %247 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %252 = arith.addi %251, %37 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %253 = tt.expand_dims %252 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %254 = arith.cmpi sge, %250, %cst_5 : tensor<64x1xi64, #linear>
    %255 = arith.cmpi slt, %250, %38 : tensor<64x1xi64, #linear>
    %256 = arith.andi %254, %255 : tensor<64x1xi1, #linear>
    %257 = tt.broadcast %256 : tensor<64x1xi1, #linear> -> tensor<64x64xi1, #linear>
    %258 = arith.cmpi sge, %253, %cst_6 : tensor<1x64xi64, #linear>
    %259 = arith.cmpi slt, %253, %39 : tensor<1x64xi64, #linear>
    %260 = arith.andi %258, %259 : tensor<1x64xi1, #linear>
    %261 = tt.broadcast %260 : tensor<1x64xi1, #linear> -> tensor<64x64xi1, #linear>
    %262 = arith.andi %257, %261 : tensor<64x64xi1, #linear>
    %263 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %264 = arith.muli %246, %N_8 : i64
    %265 = tt.splat %N_8 : i64 -> tensor<64x1xi64, #linear>
    %266 = arith.muli %263, %265 : tensor<64x1xi64, #linear>
    %267 = tt.broadcast %266 : tensor<64x1xi64, #linear> -> tensor<64x64xi64, #linear>
    %268 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %269 = tt.broadcast %268 : tensor<1x64xi64, #linear> -> tensor<64x64xi64, #linear>
    %270 = arith.addi %264, %247 : i64
    %271 = arith.addi %267, %269 : tensor<64x64xi64, #linear>
    %272 = tt.splat %270 : i64 -> tensor<64x64xi64, #linear>
    %273 = arith.addi %272, %271 : tensor<64x64xi64, #linear>
    %274 = arith.trunci %273 : tensor<64x64xi64, #linear> to tensor<64x64xi32, #linear>
    %275 = ttg.convert_layout %245 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
    amdg.buffer_store %275, %c_ptr[%274], %262 : tensor<64x64xbf16, #linear>
    cf.br ^bb14(%234 : i32)
  ^bb13:  // pred: ^bb11
    cf.br ^bb14(%154 : i32)
  ^bb14(%tile_id_c_40: i32 loc("tile_id_c")):  // 2 preds: ^bb12, ^bb13
    cf.br ^bb15
  ^bb15:  // pred: ^bb14
    %tile_id_c_41 = arith.addi %tile_id_c_22, %c1_i32 : i32
    cf.br ^bb6(%tile_id_c_41, %tile_id_c_27, %tile_id_c_34, %tile_id_c_40, %tile_id_c_39, %tile_id_c_30, %219, %229, %tile_id_c_32, %217, %227, %tile_id_c_35, %tile_id_c_36, %tile_id_c_37, %tile_id_c_38 : i32, i32, i32, i32, tensor<64x64xf32, #mma>, i32, !ttg.async.token, !ttg.async.token, i32, !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>, tensor<2x32x64xi64, #blocked>, tensor<2x32x64xi1, #blocked>)
  ^bb16:  // pred: ^bb6
    %tile_id_c_42 = arith.cmpi sge, %tile_id_c_15, %c1_i32 : i32
    %276 = amdg.async_wait %157, %158 {num_inst = 0 : i32}
    %277 = ttg.local_load %160 token %276 : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> tensor<2x32x64xbf16, #linear1>
    %278 = tt.reshape %277 : tensor<2x32x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %279 = ttg.local_load %161 token %276 : !ttg.memdesc<2x32x64xbf16, #shared, #smem, mutable> -> tensor<2x32x64xbf16, #linear2>
    %280 = tt.reshape %279 : tensor<2x32x64xbf16, #linear2> -> tensor<64x64xbf16, #linear3>
    %281 = tt.trans %280 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear3> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    cf.cond_br %tile_id_c_42, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    %282 = tt.dot %278, %281, %155 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
    cf.br ^bb19(%282 : tensor<64x64xf32, #mma>)
  ^bb18:  // pred: ^bb16
    cf.br ^bb19(%155 : tensor<64x64xf32, #mma>)
  ^bb19(%283: tensor<64x64xf32, #mma>):  // 2 preds: ^bb17, ^bb18
    cf.br ^bb20
  ^bb20:  // pred: ^bb19
    %tile_id_c_43 = arith.cmpi eq, %152, %tile_id_c_16 : i32
    %tile_id_c_44 = arith.andi %tile_id_c_42, %tile_id_c_43 : i1
    cf.cond_br %tile_id_c_44, ^bb21, ^bb22
  ^bb21:  // pred: ^bb20
    %284 = arith.addi %154, %c4_i32 : i32
    %285 = arith.divsi %284, %17 : i32
    %286 = arith.muli %285, %c8_i32 : i32
    %287 = arith.subi %2, %286 : i32
    %288 = arith.minsi %287, %c8_i32 : i32
    %289 = arith.remsi %284, %288 : i32
    %290 = arith.addi %286, %289 : i32
    %291 = arith.remsi %284, %17 : i32
    %292 = arith.divsi %291, %288 : i32
    %293 = arith.muli %290, %c64_i32 : i32
    %294 = arith.muli %292, %c64_i32 : i32
    %295 = arith.truncf %283 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
    %296 = arith.extsi %293 : i32 to i64
    %297 = arith.extsi %294 : i32 to i64
    %298 = tt.splat %296 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %299 = arith.addi %298, %36 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>>
    %300 = tt.expand_dims %299 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %301 = tt.splat %297 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %302 = arith.addi %301, %37 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>>
    %303 = tt.expand_dims %302 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %304 = arith.cmpi sge, %300, %cst_5 : tensor<64x1xi64, #linear>
    %305 = arith.cmpi slt, %300, %38 : tensor<64x1xi64, #linear>
    %306 = arith.andi %304, %305 : tensor<64x1xi1, #linear>
    %307 = tt.broadcast %306 : tensor<64x1xi1, #linear> -> tensor<64x64xi1, #linear>
    %308 = arith.cmpi sge, %303, %cst_6 : tensor<1x64xi64, #linear>
    %309 = arith.cmpi slt, %303, %39 : tensor<1x64xi64, #linear>
    %310 = arith.andi %308, %309 : tensor<1x64xi1, #linear>
    %311 = tt.broadcast %310 : tensor<1x64xi1, #linear> -> tensor<64x64xi1, #linear>
    %312 = arith.andi %307, %311 : tensor<64x64xi1, #linear>
    %313 = tt.expand_dims %36 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xi64, #linear>
    %314 = arith.muli %296, %N_8 : i64
    %315 = tt.splat %N_8 : i64 -> tensor<64x1xi64, #linear>
    %316 = arith.muli %313, %315 : tensor<64x1xi64, #linear>
    %317 = tt.broadcast %316 : tensor<64x1xi64, #linear> -> tensor<64x64xi64, #linear>
    %318 = tt.expand_dims %37 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x64xi64, #linear>
    %319 = tt.broadcast %318 : tensor<1x64xi64, #linear> -> tensor<64x64xi64, #linear>
    %320 = arith.addi %314, %297 : i64
    %321 = arith.addi %317, %319 : tensor<64x64xi64, #linear>
    %322 = tt.splat %320 : i64 -> tensor<64x64xi64, #linear>
    %323 = arith.addi %322, %321 : tensor<64x64xi64, #linear>
    %324 = arith.trunci %323 : tensor<64x64xi64, #linear> to tensor<64x64xi32, #linear>
    %325 = ttg.convert_layout %295 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
    amdg.buffer_store %325, %c_ptr[%324], %312 : tensor<64x64xbf16, #linear>
    cf.br ^bb22
  ^bb22:  // 2 preds: ^bb20, ^bb21
    ttg.local_dealloc %81 : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable>
    ttg.local_dealloc %80 : !ttg.memdesc<2x2x32x64xbf16, #shared, #smem, mutable>
    cf.br ^bb23
  ^bb23:  // 2 preds: ^bb4, ^bb22
    tt.return
  }
}
