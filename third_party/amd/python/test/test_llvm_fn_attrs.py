from pathlib import Path

import pytest
import triton
import triton.language as tl

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)

ATTN_FWD_TTIR = str(Path(__file__).parent / "attn_fwd.ttir")


def test_llvm_fn_attrs_set_single_llir_attribute():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' not in baseline.asm["llir"]
    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]


def test_llvm_fn_attrs_set_multiple_llir_attributes():
    target = triton.runtime.driver.active.get_current_target()
    llvm_fn_attrs = ",".join([
        "amdgpu-sched-strategy=iterative-ilp",
        "triton-test-attr=enabled",
        "triton-bare-attr",
    ])

    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": llvm_fn_attrs},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]
    assert '"triton-test-attr"="enabled"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"=' not in with_attrs.asm["llir"]


def test_llvm_fn_attrs_change_amdgcn():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_sched_attr = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert baseline.asm["amdgcn"] != with_sched_attr.asm["amdgcn"]


@pytest.mark.parametrize("allow_flush_denorm", [False, True])
def test_fp32_denormal_mode_controls_exp2_lowering(allow_flush_denorm):

    @triton.jit
    def exp_kernel(x, y):
        offsets = tl.arange(0, 128)
        # tl.exp uses generic llvm.exp2; tl.exp2 currently uses the native
        # AMD intrinsic, which flushes denormals regardless of this option.
        tl.store(y + offsets, tl.exp(tl.load(x + offsets)))

    source = triton.compiler.ASTSource(exp_kernel, signature={"x": "*fp32", "y": "*fp32"})
    compiled = triton.compile(
        source,
        target=triton.runtime.driver.active.get_current_target(),
        options={"allow_flush_denorm": allow_flush_denorm},
    )

    llir = compiled.asm["llir"]
    expected_mode = "float: preservesign" if allow_flush_denorm else "ieee"
    assert f"denormal_fpenv({expected_mode})" in llir
    assert "denormal-fp-math-f32" not in llir
    assert "@llvm.exp2." in llir

    amdgcn = compiled.asm["amdgcn"]
    assert "v_exp_f32" in amdgcn
    # IEEE mode scales around v_exp_f32 to recover subnormal results.
    assert ("v_ldexp_f32" in amdgcn) == (not allow_flush_denorm)
