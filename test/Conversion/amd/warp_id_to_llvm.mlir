// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a  | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942  | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950  | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

// ttg.warp_id lowers to rocdl.wave.id on every target, hoisted to the function
// entry. LLVM makes it uniform during instruction selection.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: @wave_id_wave64
tt.func public @wave_id_wave64() {
  //      CHECK: rocdl.wave.id : i32
  //  CHECK-NOT: rocdl.readfirstlane
  //      CHECK: llvm.br ^[[LOOP:bb[0-9]+]](
  // CHECK-NEXT: ^[[LOOP]](

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @wave_id_wave32
tt.func public @wave_id_wave32() {
  //      CHECK: rocdl.wave.id : i32
  //  CHECK-NOT: rocdl.readfirstlane
  //      CHECK: llvm.br ^[[LOOP:bb[0-9]+]](
  // CHECK-NEXT: ^[[LOOP]](

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %1 = "ttg.warp_id"() : () -> i32
    scf.yield
  }
  tt.return
}

}
