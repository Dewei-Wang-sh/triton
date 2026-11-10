"""Explicit synchronization primitives for the Plumb WYSIWYG Gluon front end.

With these, block ping-pong is written in the kernel instead of being synthesized
by ``warp_pipeline_stage`` + ``ConvertWarpPipeline``:

    group = gl.warp_id() // 4
    if group == 1:
        gl.amd.s_barrier()          # phase shift
    for ...:
        ... compute stage ...
        gl.amd.hint.sched_barrier()
        gl.amd.local_barrier()      # stage boundary
        gl.amd.hint.sched_barrier()
        ... memory stage ...
    if group == 0:
        gl.amd.s_barrier()          # reconverge
"""

from .. import _core as ttgl
from .._core import builtin
from .._semantic import _check

__all__ = ["s_barrier", "local_barrier", "wave_any"]


@builtin
def s_barrier(_semantic=None):
    """Execution rendezvous of all warps in the CTA, with no memory fence.

    Lowers to ``rocdl.s.barrier`` / ``s_barrier``. It does not make any memory
    operation visible; pair it with an explicit wait when ordering is needed.
    Calling it from a branch taken by only some warps is how a phase shift
    between warp groups is expressed.
    """
    _semantic.builder.create_amd_s_barrier()


@builtin
def local_barrier(_semantic=None):
    """CTA barrier that fences only LDS (``ttg.barrier local``).

    Lowers to an LDS-scoped release/acquire around ``s_barrier``, which on
    gfx9 becomes ``s_waitcnt lgkmcnt(0); s_barrier``. ``gl.barrier()`` fences
    all address spaces instead.
    """
    _semantic.builder.create_local_barrier()


@builtin
def wave_any(pred, _semantic=None):
    """True if any element held by any lane of the calling wave is true.

    ``pred`` is a distributed ``int1`` tensor. The result is a scalar ``int1``
    that is uniform within a wave but may differ between waves, so
    ``if gl.amd.wave_any(p): ...`` is a per-wave scalar branch.

    Unlike a branch inside ``gl.map_elementwise``, it does not mask lanes: when
    the branch is taken, every lane of the wave executes the body.
    """
    _check(isinstance(pred, ttgl.tensor) and isinstance(pred.type, ttgl.distributed_type),
           lambda: f"wave_any expects a distributed tensor, got {pred}")
    _check(pred.dtype == ttgl.int1, lambda: f"wave_any expects an int1 tensor, got {pred.dtype}")
    return ttgl.tensor(_semantic.builder.create_amd_wave_any(pred.handle), ttgl.int1)
