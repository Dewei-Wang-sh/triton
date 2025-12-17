
[tritonamdgpu-pipeline-lower-loops]: [lowerLoops]deserialized schedule:

---- Ops in stage 0
        cluster: 0:
	%56 = tt.load %55, %52, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 0:
	%60 = tt.load %59, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

---- Ops in stage 1
        cluster: 1:
	%64 = tt.dot %63, %62, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
[tritonamdgpu-pipeline-lower-loops]: Deduce shared encoding for: %56 = tt.load %55, %52, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-pipeline-lower-loops]:  getSharedEncIfAllUsersAreDotEnc current user: %63 = ttg.convert_layout %56 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-pipeline-lower-loops]: Deduced shared encoding candidate from dot layout: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: Deduced shared encoding: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: Populate loadInfo with shared encoding: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: Deduce shared encoding for: %60 = tt.load %59, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonamdgpu-pipeline-lower-loops]:  getSharedEncIfAllUsersAreDotEnc current user: %61 = ttg.convert_layout %60 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
[tritonamdgpu-pipeline-lower-loops]: getDotEncoding user: %62 = tt.trans %61 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
[tritonamdgpu-pipeline-lower-loops]: getDotEncoding user: %64 = tt.dot %63, %62, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
[tritonamdgpu-pipeline-lower-loops]: deduced opIdx: 1; deduced vecSize: 8
[tritonamdgpu-pipeline-lower-loops]: Deduced shared encoding candidate from mfma layout: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: Deduced shared encoding: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: Populate loadInfo with shared encoding: #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
[tritonamdgpu-pipeline-lower-loops]: SingleDotSchedule::updateSchedule
[tritonamdgpu-pipeline-lower-loops]: Init SingleDotSchedule
[tritonamdgpu-pipeline-lower-loops]: Stage schedule:  GLOBAL_LOAD stage = 0, LOCAL_STORE stage = 0, LOCAL_LOAD stage = 1, COMPUTE stage = 1, ASYNC_WAIT stage = 1; total = 2
[tritonamdgpu-pipeline-lower-loops]: deduced max shared memory buffer number = 2
[tritonamdgpu-pipeline-lower-loops]: Cluster schedule:  GLOBAL_LOAD cluster = 1, LOCAL_STORE cluster = 3, LOCAL_LOAD cluster = 1, COMPUTE cluster = 1, ASYNC_WAIT cluster = 0; total = 5

[tritonamdgpu-pipeline-lower-loops]: Coarse schedule stream ops:

---- Ops in stage 0
        cluster: 1:
	%66 = tt.load %60, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%75 = tt.load %69, %67, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%63 = ttg.async_commit_group tokens %62
        cluster: 1:
	%71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%72 = ttg.async_commit_group tokens %71

---- Ops in stage 1
        cluster: 1:
	%79 = tt.dot %78, %77, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
        cluster: 0:
	%64 = ttg.async_wait %63 {num = 0 : i32}
        cluster: 0:
	%73 = ttg.async_wait %72 {num = 0 : i32}

[tritonamdgpu-pipeline-lower-loops]: Coarse schedule with dependencies:

---- Ops in stage 0
        cluster: 1:
	%66 = tt.load %60, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%75 = tt.load %69, %67, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%63 = ttg.async_commit_group tokens %62
        cluster: 1:
	%71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%72 = ttg.async_commit_group tokens %71
        cluster: 1:
	%60 = tt.addptr %19, %59 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%59 = arith.addi %47#1, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%47:5 = scf.if %45 -> (tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32) {
  %87 = arith.addi %arg8, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.extsi %96 : i32 to i64
  %99 = tt.splat %98 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %100 = arith.addi %99, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %102 = arith.cmpi sge, %101, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %103 = arith.cmpi slt, %101, %17 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = arith.andi %102, %103 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %105 = tt.broadcast %104 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %106 = arith.muli %101, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = tt.broadcast %106 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.extsi %97 : i32 to i64
  %109 = tt.splat %108 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %110 = arith.addi %109, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi sge, %111, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.cmpi slt, %111, %21 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = arith.andi %112, %113 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = tt.broadcast %114 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %111, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  scf.yield %105, %107, %115, %117, %87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
} else {
  scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
}
        cluster: 1:
	%45 = arith.cmpi eq, %arg7, %c0_i32 : i32
        cluster: 1:
	%58 = tt.broadcast %52 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%51 = arith.addi %50, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 1:
	%50 = tt.splat %49 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 1:
	%49 = arith.extsi %48 : i32 to i64
        cluster: 1:
	%48 = arith.muli %46, %c64_i32 : i32
        cluster: 1:
	%46 = arith.select %45, %c0_i32, %arg10 : i32
        cluster: 1:
	%61 = ttg.memdesc_index %39[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%44 = arith.select %43, %42, %c0_i32_6 : i32
        cluster: 1:
	%43 = arith.cmpi slt, %42, %c2_i32 : i32
        cluster: 1:
	%42 = arith.addi %arg16, %c1_i32_7 : i32
        cluster: 1:
	%57 = arith.andi %47#0, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%56 = tt.broadcast %55 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%55 = arith.andi %53, %54 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%53 = arith.cmpi sge, %52, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%54 = arith.cmpi slt, %52, %18 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%69 = tt.addptr %22, %68 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%68 = arith.addi %47#3, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%70 = ttg.memdesc_index %40[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 1:
	%67 = arith.andi %47#2, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

---- Ops in stage 1
        cluster: 1:
	%79 = tt.dot %78, %77, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
        cluster: 0:
	%64 = ttg.async_wait %63 {num = 0 : i32}
        cluster: 0:
	%73 = ttg.async_wait %72 {num = 0 : i32}
        cluster: 1:
	%78 = ttg.convert_layout %65 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 1:
	%65 = ttg.local_load %61 token %64 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%77 = tt.trans %76 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 1:
	%76 = ttg.convert_layout %74 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
        cluster: 1:
	%74 = ttg.local_load %70 token %73 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

[tritonamdgpu-pipeline-lower-loops]: Coarse schedule with dist 1:

---- Ops in stage 0
        cluster: 2:
	%66 = tt.load %60, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%75 = tt.load %69, %67, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%63 = ttg.async_commit_group tokens %62
        cluster: 2:
	%71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%72 = ttg.async_commit_group tokens %71
        cluster: 2:
	%60 = tt.addptr %19, %59 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%59 = arith.addi %47#1, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%47:5 = scf.if %45 -> (tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32) {
  %87 = arith.addi %arg8, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.extsi %96 : i32 to i64
  %99 = tt.splat %98 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %100 = arith.addi %99, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %102 = arith.cmpi sge, %101, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %103 = arith.cmpi slt, %101, %17 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = arith.andi %102, %103 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %105 = tt.broadcast %104 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %106 = arith.muli %101, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = tt.broadcast %106 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.extsi %97 : i32 to i64
  %109 = tt.splat %108 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %110 = arith.addi %109, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi sge, %111, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.cmpi slt, %111, %21 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = arith.andi %112, %113 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = tt.broadcast %114 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %111, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  scf.yield %105, %107, %115, %117, %87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
} else {
  scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
}
        cluster: 2:
	%45 = arith.cmpi eq, %arg7, %c0_i32 : i32
        cluster: 2:
	%58 = tt.broadcast %52 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%51 = arith.addi %50, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%50 = tt.splat %49 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%49 = arith.extsi %48 : i32 to i64
        cluster: 2:
	%48 = arith.muli %46, %c64_i32 : i32
        cluster: 2:
	%46 = arith.select %45, %c0_i32, %arg10 : i32
        cluster: 2:
	%61 = ttg.memdesc_index %39[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%44 = arith.select %43, %42, %c0_i32_6 : i32
        cluster: 2:
	%43 = arith.cmpi slt, %42, %c2_i32 : i32
        cluster: 2:
	%42 = arith.addi %arg16, %c1_i32_7 : i32
        cluster: 2:
	%57 = arith.andi %47#0, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%56 = tt.broadcast %55 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%55 = arith.andi %53, %54 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%53 = arith.cmpi sge, %52, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%54 = arith.cmpi slt, %52, %18 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%69 = tt.addptr %22, %68 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%68 = arith.addi %47#3, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%70 = ttg.memdesc_index %40[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%67 = arith.andi %47#2, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

---- Ops in stage 1
        cluster: 2:
	%79 = tt.dot %78, %77, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
        cluster: 0:
	%64 = ttg.async_wait %63 {num = 0 : i32}
        cluster: 0:
	%73 = ttg.async_wait %72 {num = 0 : i32}
        cluster: 2:
	%78 = ttg.convert_layout %65 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%65 = ttg.local_load %61 token %64 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%77 = tt.trans %76 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%76 = ttg.convert_layout %74 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
        cluster: 2:
	%74 = ttg.local_load %70 token %73 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%86 = arith.select %85, %c0_i32, %84 : i32
        cluster: 1:
	%85 = arith.cmpi eq, %arg7, %38 : i32
        cluster: 1:
	%84 = arith.addi %arg7, %c1_i32 : i32
        cluster: 1:
	%80 = arith.addi %46, %c1_i32 : i32

[tritonamdgpu-pipeline-lower-loops]: Final coarse schedule:

---- Ops in stage 0
        cluster: 2:
	%66 = tt.load %60, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%75 = tt.load %69, %67, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%63 = ttg.async_commit_group tokens %62
        cluster: 2:
	%71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%72 = ttg.async_commit_group tokens %71
        cluster: 2:
	%60 = tt.addptr %19, %59 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%59 = arith.addi %47#1, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%47:5 = scf.if %45 -> (tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32) {
  %87 = arith.addi %arg8, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.extsi %96 : i32 to i64
  %99 = tt.splat %98 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %100 = arith.addi %99, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %102 = arith.cmpi sge, %101, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %103 = arith.cmpi slt, %101, %17 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = arith.andi %102, %103 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %105 = tt.broadcast %104 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %106 = arith.muli %101, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = tt.broadcast %106 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.extsi %97 : i32 to i64
  %109 = tt.splat %108 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %110 = arith.addi %109, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi sge, %111, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.cmpi slt, %111, %21 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = arith.andi %112, %113 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = tt.broadcast %114 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %111, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  scf.yield %105, %107, %115, %117, %87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
} else {
  scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
}
        cluster: 2:
	%45 = arith.cmpi eq, %arg7, %c0_i32 : i32
        cluster: 2:
	%58 = tt.broadcast %52 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%51 = arith.addi %50, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%50 = tt.splat %49 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%49 = arith.extsi %48 : i32 to i64
        cluster: 2:
	%48 = arith.muli %46, %c64_i32 : i32
        cluster: 2:
	%46 = arith.select %45, %c0_i32, %arg10 : i32
        cluster: 2:
	%61 = ttg.memdesc_index %39[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%44 = arith.select %43, %42, %c0_i32_6 : i32
        cluster: 2:
	%43 = arith.cmpi slt, %42, %c2_i32 : i32
        cluster: 2:
	%42 = arith.addi %arg16, %c1_i32_7 : i32
        cluster: 2:
	%57 = arith.andi %47#0, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%56 = tt.broadcast %55 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%55 = arith.andi %53, %54 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%53 = arith.cmpi sge, %52, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%54 = arith.cmpi slt, %52, %18 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%69 = tt.addptr %22, %68 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%68 = arith.addi %47#3, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%70 = ttg.memdesc_index %40[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%67 = arith.andi %47#2, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

---- Ops in stage 1
        cluster: 2:
	%79 = tt.dot %78, %77, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
        cluster: 0:
	%64 = ttg.async_wait %63 {num = 0 : i32}
        cluster: 0:
	%73 = ttg.async_wait %72 {num = 0 : i32}
        cluster: 2:
	%78 = ttg.convert_layout %65 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%65 = ttg.local_load %61 token %64 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%77 = tt.trans %76 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%76 = ttg.convert_layout %74 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
        cluster: 2:
	%74 = ttg.local_load %70 token %73 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%86 = arith.select %85, %c0_i32, %84 : i32
        cluster: 1:
	%85 = arith.cmpi eq, %arg7, %38 : i32
        cluster: 1:
	%84 = arith.addi %arg7, %c1_i32 : i32
        cluster: 1:
	%80 = arith.addi %46, %c1_i32 : i32
        cluster: 2:
	%83 = scf.if %81 -> (i32) {
  %87 = arith.addi %arg9, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.truncf %79 : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> to tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
  %99 = arith.extsi %96 : i32 to i64
  %100 = arith.extsi %97 : i32 to i64
  %101 = tt.splat %99 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %102 = arith.addi %101, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %103 = tt.expand_dims %102 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = tt.splat %100 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %105 = arith.addi %104, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %106 = tt.expand_dims %105 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = arith.cmpi sge, %103, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.cmpi slt, %103, %27 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %109 = arith.andi %107, %108 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %110 = tt.broadcast %109 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %111 = arith.cmpi sge, %106, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi slt, %106, %28 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.andi %111, %112 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = tt.broadcast %113 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = arith.andi %110, %114 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %103, %30 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %118 = tt.broadcast %106 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %119 = arith.addi %117, %118 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %120 = tt.addptr %29, %119 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %121 = ttg.convert_layout %120 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  %122 = ttg.convert_layout %98 : tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  %123 = ttg.convert_layout %115 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  tt.store %121, %122, %123 : tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  scf.yield %87 : i32
} else {
  scf.yield %arg9 : i32
}
        cluster: 2:
	%81 = arith.cmpi eq, %arg7, %37 : i32
        cluster: 2:
	%82 = arith.select %81, %cst_5, %79 : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>

[tritonamdgpu-pipeline-lower-loops]: [lowerLoops]updated schedule:

---- Ops in stage 0
        cluster: 2:
	%66 = tt.load %60, %57, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%75 = tt.load %69, %67, %cst_4 {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%63 = ttg.async_commit_group tokens %62
        cluster: 2:
	%71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%72 = ttg.async_commit_group tokens %71
        cluster: 2:
	%60 = tt.addptr %19, %59 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%59 = arith.addi %47#1, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%47:5 = scf.if %45 -> (tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32) {
  %87 = arith.addi %arg8, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.extsi %96 : i32 to i64
  %99 = tt.splat %98 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %100 = arith.addi %99, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %102 = arith.cmpi sge, %101, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %103 = arith.cmpi slt, %101, %17 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = arith.andi %102, %103 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %105 = tt.broadcast %104 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %106 = arith.muli %101, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = tt.broadcast %106 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.extsi %97 : i32 to i64
  %109 = tt.splat %108 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %110 = arith.addi %109, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi sge, %111, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.cmpi slt, %111, %21 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = arith.andi %112, %113 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = tt.broadcast %114 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %111, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  scf.yield %105, %107, %115, %117, %87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
} else {
  scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
}
        cluster: 2:
	%45 = arith.cmpi eq, %arg7, %c0_i32 : i32
        cluster: 2:
	%58 = tt.broadcast %52 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%52 = tt.expand_dims %51 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%51 = arith.addi %50, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%50 = tt.splat %49 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
        cluster: 2:
	%49 = arith.extsi %48 : i32 to i64
        cluster: 2:
	%48 = arith.muli %46, %c64_i32 : i32
        cluster: 2:
	%46 = arith.select %45, %c0_i32, %arg10 : i32
        cluster: 2:
	%61 = ttg.memdesc_index %39[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%44 = arith.select %43, %42, %c0_i32_6 : i32
        cluster: 2:
	%43 = arith.cmpi slt, %42, %c2_i32 : i32
        cluster: 2:
	%42 = arith.addi %arg16, %c1_i32_7 : i32
        cluster: 2:
	%57 = arith.andi %47#0, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%56 = tt.broadcast %55 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%55 = arith.andi %53, %54 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%53 = arith.cmpi sge, %52, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%54 = arith.cmpi slt, %52, %18 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%69 = tt.addptr %22, %68 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%68 = arith.addi %47#3, %58 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%70 = ttg.memdesc_index %40[%44] : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
        cluster: 2:
	%67 = arith.andi %47#2, %56 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

---- Ops in stage 1
        cluster: 2:
	%79 = tt.dot %78, %77, %arg11 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
        cluster: 0:
	%64 = ttg.async_wait %63 {num = 0 : i32}
        cluster: 0:
	%73 = ttg.async_wait %72 {num = 0 : i32}
        cluster: 2:
	%78 = ttg.convert_layout %65 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%65 = ttg.local_load %61 token %64 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 2:
	%77 = tt.trans %76 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
        cluster: 2:
	%76 = ttg.convert_layout %74 : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
        cluster: 2:
	%74 = ttg.local_load %70 token %73 : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
        cluster: 1:
	%86 = arith.select %85, %c0_i32, %84 : i32
        cluster: 1:
	%85 = arith.cmpi eq, %arg7, %38 : i32
        cluster: 1:
	%84 = arith.addi %arg7, %c1_i32 : i32
        cluster: 1:
	%80 = arith.addi %46, %c1_i32 : i32
        cluster: 2:
	%83 = scf.if %81 -> (i32) {
  %87 = arith.addi %arg9, %c4_i32 : i32
  %88 = arith.divsi %87, %12 : i32
  %89 = arith.muli %88, %c8_i32 : i32
  %90 = arith.subi %2, %89 : i32
  %91 = arith.minsi %90, %c8_i32 : i32
  %92 = arith.remsi %87, %91 : i32
  %93 = arith.addi %89, %92 : i32
  %94 = arith.remsi %87, %12 : i32
  %95 = arith.divsi %94, %91 : i32
  %96 = arith.muli %93, %c64_i32 : i32
  %97 = arith.muli %95, %c64_i32 : i32
  %98 = arith.truncf %79 : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> to tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
  %99 = arith.extsi %96 : i32 to i64
  %100 = arith.extsi %97 : i32 to i64
  %101 = tt.splat %99 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %102 = arith.addi %101, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %103 = tt.expand_dims %102 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %104 = tt.splat %100 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %105 = arith.addi %104, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %106 = tt.expand_dims %105 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %107 = arith.cmpi sge, %103, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %108 = arith.cmpi slt, %103, %27 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %109 = arith.andi %107, %108 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %110 = tt.broadcast %109 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %111 = arith.cmpi sge, %106, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %112 = arith.cmpi slt, %106, %28 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %113 = arith.andi %111, %112 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %114 = tt.broadcast %113 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %115 = arith.andi %110, %114 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %116 = arith.muli %103, %30 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %118 = tt.broadcast %106 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %119 = arith.addi %117, %118 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %120 = tt.addptr %29, %119 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %121 = ttg.convert_layout %120 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  %122 = ttg.convert_layout %98 : tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  %123 = ttg.convert_layout %115 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  tt.store %121, %122, %123 : tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
  scf.yield %87 : i32
} else {
  scf.yield %arg9 : i32
}
        cluster: 2:
	%81 = arith.cmpi eq, %arg7, %37 : i32
        cluster: 2:
	%82 = arith.select %81, %cst_5, %79 : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
[tritonamdgpu-pipeline-expand-loops]: Loop before sending to expander:
%41:10 = scf.for %arg6 = %c0_i32 to %35 step %c1_i32 iter_args(%arg7 = %c0_i32, %arg8 = %36, %arg9 = %11, %arg10 = %c0_i32, %arg11 = %cst_5, %arg12 = %cst_1, %arg13 = %cst_0, %arg14 = %cst_1, %arg15 = %cst_0, %arg16 = %c-1_i32) -> (i32, i32, i32, i32, tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32)  : i32 {
  %42 = arith.addi %arg16, %c1_i32_7 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %43 = arith.cmpi slt, %42, %c2_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %44 = arith.select %43, %42, %c0_i32_6 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %45 = arith.cmpi eq, %arg7, %c0_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %46 = arith.select %45, %c0_i32, %arg10 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %47:5 = scf.if %45 -> (tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32) {
    %87 = arith.addi %arg8, %c4_i32 : i32
    %88 = arith.divsi %87, %12 : i32
    %89 = arith.muli %88, %c8_i32 : i32
    %90 = arith.subi %2, %89 : i32
    %91 = arith.minsi %90, %c8_i32 : i32
    %92 = arith.remsi %87, %91 : i32
    %93 = arith.addi %89, %92 : i32
    %94 = arith.remsi %87, %12 : i32
    %95 = arith.divsi %94, %91 : i32
    %96 = arith.muli %93, %c64_i32 : i32
    %97 = arith.muli %95, %c64_i32 : i32
    %98 = arith.extsi %96 : i32 to i64
    %99 = tt.splat %98 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %100 = arith.addi %99, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %101 = tt.expand_dims %100 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %102 = arith.cmpi sge, %101, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %103 = arith.cmpi slt, %101, %17 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %104 = arith.andi %102, %103 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %105 = tt.broadcast %104 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %106 = arith.muli %101, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %107 = tt.broadcast %106 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %108 = arith.extsi %97 : i32 to i64
    %109 = tt.splat %108 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %110 = arith.addi %109, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %111 = tt.expand_dims %110 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %112 = arith.cmpi sge, %111, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %113 = arith.cmpi slt, %111, %21 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %114 = arith.andi %112, %113 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %115 = tt.broadcast %114 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %116 = arith.muli %111, %20 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    scf.yield %105, %107, %115, %117, %87 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
  } else {
    scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
  } {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  %48 = arith.muli %46, %c64_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
  %49 = arith.extsi %48 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 to i64
  %50 = tt.splat %49 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %51 = arith.addi %50, %16 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
  %52 = tt.expand_dims %51 {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %53 = arith.cmpi sge, %52, %cst_3 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %54 = arith.cmpi slt, %52, %18 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %55 = arith.andi %53, %54 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %56 = tt.broadcast %55 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %57 = arith.andi %47#0, %56 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %58 = tt.broadcast %52 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %59 = arith.addi %47#1, %58 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %60 = tt.addptr %19, %59 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %61 = ttg.memdesc_index %39[%44] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
  %62 = ttg.async_copy_global_to_local %60, %61 mask %57 other %cst_4 {contiguity = 8 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
  %63 = ttg.async_commit_group tokens %62 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  %64 = ttg.async_wait %63 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
  %65 = ttg.local_load %61 token %64 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %66 = tt.load %60, %57, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %67 = arith.andi %47#2, %56 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %68 = arith.addi %47#3, %58 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %69 = tt.addptr %22, %68 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %70 = ttg.memdesc_index %40[%44] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !ttg.memdesc<2x64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
  %71 = ttg.async_copy_global_to_local %69, %70 mask %67 other %cst_4 {contiguity = 8 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> <64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable>
  %72 = ttg.async_commit_group tokens %71 {loop.cluster = 2 : i32, loop.stage = 0 : i32}
  %73 = ttg.async_wait %72 {loop.cluster = 0 : i32, loop.stage = 1 : i32, num = 0 : i32}
  %74 = ttg.local_load %70 token %73 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x64xbf16, #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>, #ttg.shared_memory, mutable> -> tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %75 = tt.load %69, %67, %cst_4 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
  %76 = ttg.convert_layout %74 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>>
  %77 = tt.trans %76 {loop.cluster = 2 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
  %78 = ttg.convert_layout %65 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>>
  %79 = tt.dot %78, %77, %arg11 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>, kWidth = 8}>> -> tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
  %80 = arith.addi %46, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
  %81 = arith.cmpi eq, %arg7, %37 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : i32
  %82 = arith.select %81, %cst_5, %79 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
  %83 = scf.if %81 -> (i32) {
    %87 = arith.addi %arg9, %c4_i32 : i32
    %88 = arith.divsi %87, %12 : i32
    %89 = arith.muli %88, %c8_i32 : i32
    %90 = arith.subi %2, %89 : i32
    %91 = arith.minsi %90, %c8_i32 : i32
    %92 = arith.remsi %87, %91 : i32
    %93 = arith.addi %89, %92 : i32
    %94 = arith.remsi %87, %12 : i32
    %95 = arith.divsi %94, %91 : i32
    %96 = arith.muli %93, %c64_i32 : i32
    %97 = arith.muli %95, %c64_i32 : i32
    %98 = arith.truncf %79 : tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> to tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>
    %99 = arith.extsi %96 : i32 to i64
    %100 = arith.extsi %97 : i32 to i64
    %101 = tt.splat %99 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %102 = arith.addi %101, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %103 = tt.expand_dims %102 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %104 = tt.splat %100 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %105 = arith.addi %104, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>>
    %106 = tt.expand_dims %105 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>}>> -> tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %107 = arith.cmpi sge, %103, %cst_2 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %108 = arith.cmpi slt, %103, %27 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %109 = arith.andi %107, %108 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %110 = tt.broadcast %109 : tensor<64x1xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %111 = arith.cmpi sge, %106, %cst_3 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %112 = arith.cmpi slt, %106, %28 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %113 = arith.andi %111, %112 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %114 = tt.broadcast %113 : tensor<1x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %115 = arith.andi %110, %114 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %116 = arith.muli %103, %30 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %117 = tt.broadcast %116 : tensor<64x1xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %118 = tt.broadcast %106 : tensor<1x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %119 = arith.addi %117, %118 : tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %120 = tt.addptr %29, %119 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %121 = ttg.convert_layout %120 : tensor<64x64x!tt.ptr<bf16>, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
    %122 = ttg.convert_layout %98 : tensor<64x64xbf16, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>> -> tensor<64x64xbf16, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
    %123 = ttg.convert_layout %115 : tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>> -> tensor<64x64xi1, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
    tt.store %121, %122, %123 : tensor<64x64x!tt.ptr<bf16>, #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>>
    scf.yield %87 : i32
  } else {
    scf.yield %arg9 : i32
  } {loop.cluster = 2 : i32, loop.stage = 1 : i32}
  %84 = arith.addi %arg7, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
  %85 = arith.cmpi eq, %arg7, %38 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
  %86 = arith.select %85, %c0_i32, %84 {loop.cluster = 1 : i32, loop.stage = 1 : i32} : i32
  scf.yield %86, %47#4, %83, %80, %82, %47#0, %47#1, %47#2, %47#3, %44 : i32, i32, i32, i32, tensor<64x64xf32, #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi1, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, tensor<64x64xi64, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>, i32
} {tt.scheduled_max_stage = 1 : i32}
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_reshape(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
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
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg3 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.subi %0, %c4_i32 : i32
    %12 = arith.muli %4, %c8_i32 : i32
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = arith.extsi %13 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = arith.extsi %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.splat %9 : i64 -> tensor<64x1xi64, #blocked>
    %18 = tt.splat %8 : i64 -> tensor<1x64xi64, #blocked>
    %19 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %20 = tt.splat %8 : i64 -> tensor<64x1xi64, #blocked>
    %21 = tt.splat %10 : i64 -> tensor<64x1xi64, #blocked>
    %22 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25 = arith.extsi %23 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.extsi %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
    %27 = tt.splat %9 : i64 -> tensor<64x1xi64, #blocked>
    %28 = tt.splat %10 : i64 -> tensor<1x64xi64, #blocked>
    %29 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %30 = tt.splat %10 : i64 -> tensor<64x1xi64, #blocked>
    %31 = arith.cmpi eq, %6, %c0_i32 : i32
    scf.if %31 {
      %32 = scf.for %arg6 = %0 to %7 step %c4_i32 iter_args(%arg7 = %11) -> (i32)  : i32 {
        %33 = arith.addi %arg7, %c4_i32 : i32
        %34 = arith.divsi %33, %12 : i32
        %35 = arith.muli %34, %c8_i32 : i32
        %36 = arith.subi %2, %35 : i32
        %37 = arith.minsi %36, %c8_i32 : i32
        %38 = arith.remsi %33, %37 : i32
        %39 = arith.addi %35, %38 : i32
        %40 = arith.remsi %33, %12 : i32
        %41 = arith.divsi %40, %37 : i32
        %42 = arith.muli %39, %c64_i32 : i32
        %43 = arith.muli %41, %c64_i32 : i32
        %44 = arith.extsi %42 : i32 to i64
        %45 = arith.extsi %43 : i32 to i64
        %46 = tt.splat %44 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %47 = arith.addi %46, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
        %49 = tt.splat %45 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %50 = arith.addi %49, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %51 = tt.expand_dims %50 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
        %52 = arith.cmpi sge, %48, %cst_2 : tensor<64x1xi64, #blocked>
        %53 = arith.cmpi slt, %48, %27 : tensor<64x1xi64, #blocked>
        %54 = arith.andi %52, %53 : tensor<64x1xi1, #blocked>
        %55 = tt.broadcast %54 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %56 = arith.cmpi sge, %51, %cst_3 : tensor<1x64xi64, #blocked>
        %57 = arith.cmpi slt, %51, %28 : tensor<1x64xi64, #blocked>
        %58 = arith.andi %56, %57 : tensor<1x64xi1, #blocked>
        %59 = tt.broadcast %58 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %60 = arith.andi %55, %59 : tensor<64x64xi1, #blocked>
        %61 = arith.muli %48, %30 : tensor<64x1xi64, #blocked>
        %62 = tt.broadcast %61 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %63 = tt.broadcast %51 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %64 = arith.addi %62, %63 : tensor<64x64xi64, #blocked>
        %65 = tt.addptr %29, %64 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %66 = ttg.convert_layout %65 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #linear>
        %67 = ttg.convert_layout %60 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #linear>
        tt.store %66, %cst, %67 : tensor<64x64x!tt.ptr<bf16>, #linear>
        scf.yield %33 : i32
      } {tt.flatten}
    } else {
      %32 = arith.subi %7, %0 : i32
      %33 = arith.ceildivsi %32, %c4_i32 : i32
      %34 = arith.maxsi %6, %c1_i32 : i32
      %35 = arith.muli %33, %34 : i32
      %36 = arith.subi %0, %c4_i32 : i32
      %37 = arith.subi %34, %c1_i32 : i32
      %38 = arith.subi %34, %c1_i32 : i32
      %39 = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
      %40 = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
      %41 = arith.cmpi sgt, %35, %c0_i32 : i32
      %42 = arith.cmpi slt, %c0_i32, %c2_i32 : i32
      %43 = arith.select %42, %c0_i32, %c0_i32 : i32
      %44:5 = scf.if %true -> (tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32) {
        %97 = arith.divsi %0, %12 : i32
        %98 = arith.muli %97, %c8_i32 : i32
        %99 = arith.subi %2, %98 : i32
        %100 = arith.minsi %99, %c8_i32 : i32
        %101 = arith.remsi %0, %100 : i32
        %102 = arith.addi %98, %101 : i32
        %103 = arith.remsi %0, %12 : i32
        %104 = arith.divsi %103, %100 : i32
        %105 = arith.muli %102, %c64_i32 : i32
        %106 = arith.muli %104, %c64_i32 : i32
        %107 = arith.extsi %105 : i32 to i64
        %108 = tt.splat %107 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %109 = arith.addi %108, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %110 = tt.expand_dims %109 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
        %111 = arith.cmpi sge, %110, %cst_2 : tensor<64x1xi64, #blocked>
        %112 = arith.cmpi slt, %110, %17 : tensor<64x1xi64, #blocked>
        %113 = arith.andi %111, %112 : tensor<64x1xi1, #blocked>
        %114 = tt.broadcast %113 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %115 = arith.muli %110, %20 : tensor<64x1xi64, #blocked>
        %116 = tt.broadcast %115 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %117 = arith.extsi %106 : i32 to i64
        %118 = tt.splat %117 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %119 = arith.addi %118, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %120 = tt.expand_dims %119 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
        %121 = arith.cmpi sge, %120, %cst_2 : tensor<64x1xi64, #blocked>
        %122 = arith.cmpi slt, %120, %21 : tensor<64x1xi64, #blocked>
        %123 = arith.andi %121, %122 : tensor<64x1xi1, #blocked>
        %124 = tt.broadcast %123 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %125 = arith.muli %120, %20 : tensor<64x1xi64, #blocked>
        %126 = tt.broadcast %125 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
        scf.yield %114, %116, %124, %126, %0 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
      } else {
        scf.yield %cst_1, %cst_0, %cst_1, %cst_0, %36 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
      }
      %45 = arith.muli %c0_i32, %c64_i32 : i32
      %46 = arith.extsi %45 : i32 to i64
      %47 = tt.splat %46 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %48 = arith.addi %47, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
      %49 = tt.expand_dims %48 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
      %50 = arith.cmpi sge, %49, %cst_3 : tensor<1x64xi64, #blocked>
      %51 = arith.cmpi slt, %49, %18 : tensor<1x64xi64, #blocked>
      %52 = arith.andi %50, %51 : tensor<1x64xi1, #blocked>
      %53 = tt.broadcast %52 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %54 = arith.andi %44#0, %53 : tensor<64x64xi1, #blocked>
      %55 = tt.broadcast %49 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
      %56 = arith.addi %44#1, %55 : tensor<64x64xi64, #blocked>
      %57 = tt.addptr %19, %56 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
      %58 = ttg.memdesc_index %39[%43] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
      %59 = tt.splat %41 : i1 -> tensor<64x64xi1, #blocked>
      %60 = arith.andi %59, %54 : tensor<64x64xi1, #blocked>
      %61 = ttg.async_copy_global_to_local %57, %58 mask %60 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable>
      %62 = ttg.async_commit_group tokens %61
      %63 = tt.splat %41 : i1 -> tensor<64x64xi1, #blocked>
      %64 = arith.andi %63, %54 : tensor<64x64xi1, #blocked>
      %65 = tt.load %57, %64, %cst_4 {amd.pipeliner_part = "prologue"} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      %66 = arith.andi %44#2, %53 : tensor<64x64xi1, #blocked>
      %67 = arith.addi %44#3, %55 : tensor<64x64xi64, #blocked>
      %68 = tt.addptr %22, %67 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
      %69 = ttg.memdesc_index %40[%43] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
      %70 = tt.splat %41 : i1 -> tensor<64x64xi1, #blocked>
      %71 = arith.andi %70, %66 : tensor<64x64xi1, #blocked>
      %72 = ttg.async_copy_global_to_local %68, %69 mask %71 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable>
      %73 = ttg.async_commit_group tokens %72
      %74 = tt.splat %41 : i1 -> tensor<64x64xi1, #blocked>
      %75 = arith.andi %74, %66 : tensor<64x64xi1, #blocked>
      %76 = tt.load %68, %75, %cst_4 {amd.pipeliner_part = "prologue"} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      %77 = arith.subi %35, %c1_i32 : i32
      %78:15 = scf.for %arg6 = %c0_i32 to %77 step %c1_i32 iter_args(%arg7 = %c0_i32, %arg8 = %44#4, %arg9 = %11, %arg10 = %c0_i32, %arg11 = %cst_5, %arg12 = %44#0, %arg13 = %44#1, %arg14 = %44#2, %arg15 = %44#3, %arg16 = %43, %arg17 = %62, %arg18 = %73, %arg19 = %c0_i32, %arg20 = %58, %arg21 = %69) -> (i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32, !ttg.async.token, !ttg.async.token, i32, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>)  : i32 {
        %97 = ttg.async_wait %arg17, %arg18 {num = 0 : i32}
        %98 = arith.addi %arg19, %c1_i32 : i32
        %99 = arith.addi %arg7, %c1_i32 : i32
        %100 = arith.cmpi eq, %arg7, %38 : i32
        %101 = arith.select %100, %c0_i32, %99 : i32
        %102 = arith.addi %arg16, %c1_i32 : i32
        %103 = arith.cmpi slt, %102, %c2_i32 : i32
        %104 = arith.select %103, %102, %c0_i32 : i32
        %105 = arith.cmpi eq, %101, %c0_i32 : i32
        %106 = arith.select %105, %c0_i32, %98 : i32
        %107:5 = scf.if %105 -> (tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32) {
          %139 = arith.addi %arg8, %c4_i32 : i32
          %140 = arith.divsi %139, %12 : i32
          %141 = arith.muli %140, %c8_i32 : i32
          %142 = arith.subi %2, %141 : i32
          %143 = arith.minsi %142, %c8_i32 : i32
          %144 = arith.remsi %139, %143 : i32
          %145 = arith.addi %141, %144 : i32
          %146 = arith.remsi %139, %12 : i32
          %147 = arith.divsi %146, %143 : i32
          %148 = arith.muli %145, %c64_i32 : i32
          %149 = arith.muli %147, %c64_i32 : i32
          %150 = arith.extsi %148 : i32 to i64
          %151 = tt.splat %150 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %152 = arith.addi %151, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %153 = tt.expand_dims %152 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %154 = arith.cmpi sge, %153, %cst_2 : tensor<64x1xi64, #blocked>
          %155 = arith.cmpi slt, %153, %17 : tensor<64x1xi64, #blocked>
          %156 = arith.andi %154, %155 : tensor<64x1xi1, #blocked>
          %157 = tt.broadcast %156 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %158 = arith.muli %153, %20 : tensor<64x1xi64, #blocked>
          %159 = tt.broadcast %158 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %160 = arith.extsi %149 : i32 to i64
          %161 = tt.splat %160 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %162 = arith.addi %161, %15 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %163 = tt.expand_dims %162 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %164 = arith.cmpi sge, %163, %cst_2 : tensor<64x1xi64, #blocked>
          %165 = arith.cmpi slt, %163, %21 : tensor<64x1xi64, #blocked>
          %166 = arith.andi %164, %165 : tensor<64x1xi1, #blocked>
          %167 = tt.broadcast %166 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %168 = arith.muli %163, %20 : tensor<64x1xi64, #blocked>
          %169 = tt.broadcast %168 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          scf.yield %157, %159, %167, %169, %139 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
        } else {
          scf.yield %arg12, %arg13, %arg14, %arg15, %arg8 : tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32
        }
        %108 = arith.muli %106, %c64_i32 : i32
        %109 = arith.extsi %108 : i32 to i64
        %110 = tt.splat %109 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %111 = arith.addi %110, %16 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %112 = tt.expand_dims %111 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
        %113 = arith.cmpi sge, %112, %cst_3 : tensor<1x64xi64, #blocked>
        %114 = arith.cmpi slt, %112, %18 : tensor<1x64xi64, #blocked>
        %115 = arith.andi %113, %114 : tensor<1x64xi1, #blocked>
        %116 = tt.broadcast %115 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %117 = arith.andi %107#0, %116 : tensor<64x64xi1, #blocked>
        %118 = tt.broadcast %112 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %119 = arith.addi %107#1, %118 : tensor<64x64xi64, #blocked>
        %120 = tt.addptr %19, %119 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %121 = ttg.memdesc_index %39[%104] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %122 = ttg.async_copy_global_to_local %120, %121 mask %117 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable>
        %123 = ttg.async_commit_group tokens %122
        %124 = ttg.local_load %arg20 token %97 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
        %125 = arith.andi %107#2, %116 : tensor<64x64xi1, #blocked>
        %126 = arith.addi %107#3, %118 : tensor<64x64xi64, #blocked>
        %127 = tt.addptr %22, %126 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %128 = ttg.memdesc_index %40[%104] : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
        %129 = ttg.async_copy_global_to_local %127, %128 mask %125 other %cst_4 {contiguity = 8 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared, #smem, mutable>
        %130 = ttg.async_commit_group tokens %129
        %131 = ttg.local_load %arg21 token %97 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
        %132 = ttg.convert_layout %131 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #linear1>
        %133 = tt.trans %132 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
        %134 = ttg.convert_layout %124 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
        %135 = tt.dot %134, %133, %arg11 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
        %136 = arith.cmpi eq, %arg7, %37 : i32
        %137 = arith.select %136, %cst_5, %135 : tensor<64x64xf32, #mma>
        %138 = scf.if %136 -> (i32) {
          %139 = arith.addi %arg9, %c4_i32 : i32
          %140 = arith.divsi %139, %12 : i32
          %141 = arith.muli %140, %c8_i32 : i32
          %142 = arith.subi %2, %141 : i32
          %143 = arith.minsi %142, %c8_i32 : i32
          %144 = arith.remsi %139, %143 : i32
          %145 = arith.addi %141, %144 : i32
          %146 = arith.remsi %139, %12 : i32
          %147 = arith.divsi %146, %143 : i32
          %148 = arith.muli %145, %c64_i32 : i32
          %149 = arith.muli %147, %c64_i32 : i32
          %150 = arith.truncf %135 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
          %151 = arith.extsi %148 : i32 to i64
          %152 = arith.extsi %149 : i32 to i64
          %153 = tt.splat %151 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %154 = arith.addi %153, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
          %155 = tt.expand_dims %154 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
          %156 = tt.splat %152 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
          %157 = arith.addi %156, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
          %158 = tt.expand_dims %157 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
          %159 = arith.cmpi sge, %155, %cst_2 : tensor<64x1xi64, #blocked>
          %160 = arith.cmpi slt, %155, %27 : tensor<64x1xi64, #blocked>
          %161 = arith.andi %159, %160 : tensor<64x1xi1, #blocked>
          %162 = tt.broadcast %161 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %163 = arith.cmpi sge, %158, %cst_3 : tensor<1x64xi64, #blocked>
          %164 = arith.cmpi slt, %158, %28 : tensor<1x64xi64, #blocked>
          %165 = arith.andi %163, %164 : tensor<1x64xi1, #blocked>
          %166 = tt.broadcast %165 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
          %167 = arith.andi %162, %166 : tensor<64x64xi1, #blocked>
          %168 = arith.muli %155, %30 : tensor<64x1xi64, #blocked>
          %169 = tt.broadcast %168 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %170 = tt.broadcast %158 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
          %171 = arith.addi %169, %170 : tensor<64x64xi64, #blocked>
          %172 = tt.addptr %29, %171 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
          %173 = ttg.convert_layout %172 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #linear>
          %174 = ttg.convert_layout %150 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
          %175 = ttg.convert_layout %167 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #linear>
          tt.store %173, %174, %175 : tensor<64x64x!tt.ptr<bf16>, #linear>
          scf.yield %139 : i32
        } else {
          scf.yield %arg9 : i32
        }
        scf.yield %101, %107#4, %138, %98, %137, %107#0, %107#1, %107#2, %107#3, %104, %123, %130, %106, %121, %128 : i32, i32, i32, i32, tensor<64x64xf32, #mma>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, tensor<64x64xi1, #blocked>, tensor<64x64xi64, #blocked>, i32, !ttg.async.token, !ttg.async.token, i32, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
      }
      %79 = arith.addi %35, %c1_i32 : i32
      %80 = arith.addi %79, %c-1_i32 : i32
      %81 = arith.cmpi sge, %80, %c1_i32 : i32
      %82 = ttg.async_wait %78#10, %78#11 {num = 0 : i32}
      %83 = arith.addi %78#12, %c1_i32 : i32
      %84 = arith.addi %78#0, %c1_i32 : i32
      %85 = arith.cmpi eq, %78#0, %38 : i32
      %86 = arith.select %85, %c0_i32, %84 : i32
      %87 = ttg.local_load %78#13 token %82 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
      %88 = ttg.local_load %78#14 token %82 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #blocked>
      %89 = ttg.convert_layout %88 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #linear1>
      %90 = tt.trans %89 {order = array<i32: 1, 0>} : tensor<64x64xbf16, #linear1> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %91 = ttg.convert_layout %87 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %92 = scf.if %81 -> (tensor<64x64xf32, #mma>) {
        %97 = tt.dot %91, %90, %78#4 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
        scf.yield %97 : tensor<64x64xf32, #mma>
      } else {
        scf.yield %78#4 : tensor<64x64xf32, #mma>
      }
      %93 = arith.cmpi eq, %78#0, %37 : i32
      %94 = arith.select %93, %cst_5, %92 : tensor<64x64xf32, #mma>
      %95 = arith.andi %81, %93 : i1
      %96 = scf.if %95 -> (i32) {
        %97 = arith.addi %78#2, %c4_i32 : i32
        %98 = arith.divsi %97, %12 : i32
        %99 = arith.muli %98, %c8_i32 : i32
        %100 = arith.subi %2, %99 : i32
        %101 = arith.minsi %100, %c8_i32 : i32
        %102 = arith.remsi %97, %101 : i32
        %103 = arith.addi %99, %102 : i32
        %104 = arith.remsi %97, %12 : i32
        %105 = arith.divsi %104, %101 : i32
        %106 = arith.muli %103, %c64_i32 : i32
        %107 = arith.muli %105, %c64_i32 : i32
        %108 = arith.truncf %92 : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
        %109 = arith.extsi %106 : i32 to i64
        %110 = arith.extsi %107 : i32 to i64
        %111 = tt.splat %109 : i64 -> tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %112 = arith.addi %111, %25 : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>>
        %113 = tt.expand_dims %112 {axis = 1 : i32} : tensor<64xi64, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi64, #blocked>
        %114 = tt.splat %110 : i64 -> tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %115 = arith.addi %114, %26 : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>>
        %116 = tt.expand_dims %115 {axis = 0 : i32} : tensor<64xi64, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi64, #blocked>
        %117 = arith.cmpi sge, %113, %cst_2 : tensor<64x1xi64, #blocked>
        %118 = arith.cmpi slt, %113, %27 : tensor<64x1xi64, #blocked>
        %119 = arith.andi %117, %118 : tensor<64x1xi1, #blocked>
        %120 = tt.broadcast %119 : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %121 = arith.cmpi sge, %116, %cst_3 : tensor<1x64xi64, #blocked>
        %122 = arith.cmpi slt, %116, %28 : tensor<1x64xi64, #blocked>
        %123 = arith.andi %121, %122 : tensor<1x64xi1, #blocked>
        %124 = tt.broadcast %123 : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
        %125 = arith.andi %120, %124 : tensor<64x64xi1, #blocked>
        %126 = arith.muli %113, %30 : tensor<64x1xi64, #blocked>
        %127 = tt.broadcast %126 : tensor<64x1xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %128 = tt.broadcast %116 : tensor<1x64xi64, #blocked> -> tensor<64x64xi64, #blocked>
        %129 = arith.addi %127, %128 : tensor<64x64xi64, #blocked>
        %130 = tt.addptr %29, %129 : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi64, #blocked>
        %131 = ttg.convert_layout %130 : tensor<64x64x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #linear>
        %132 = ttg.convert_layout %108 : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #linear>
        %133 = ttg.convert_layout %125 : tensor<64x64xi1, #blocked> -> tensor<64x64xi1, #linear>
        tt.store %131, %132, %133 : tensor<64x64x!tt.ptr<bf16>, #linear>
        scf.yield %97 : i32
      } else {
        scf.yield %78#2 : i32
      }
      ttg.local_dealloc %40 : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
      ttg.local_dealloc %39 : !ttg.memdesc<2x64x64xbf16, #shared, #smem, mutable>
    }
    tt.return
  }
}

