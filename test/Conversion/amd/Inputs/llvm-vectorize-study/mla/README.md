# MLA LLVM vectorization review test

This directory preserves the original MLA stage1 compilation and the validated
result after the LLVM vectorization changes. It includes the capture/replay
driver, raw post-SymbolDCE IR, standalone vectorized IR, and both stage1 caches.
These are saved results from October 13–14, 2026, not newly measured performance.
The large snapshots live under `Inputs` and are not automatically run by lit.

## Original and after

| | Original | After |
|---|---|---|
| Triton commit | `21e1b37d82fe010550b2fb3f08c8e600f73ae355` | `0492a1e548128dacb86529bca646a9fccf306483` |
| Raw LLVM-dialect IR immediately after SymbolDCE | [original/mla.after.SymbolDCE.mlir](original/mla.after.SymbolDCE.mlir) | [after/mla.after.SymbolDCE.mlir](after/mla.after.SymbolDCE.mlir) |
| Standalone vectorization output | — | [after/mla.vectorized.mlir](after/mla.vectorized.mlir) |
| Stage1 compiler cache | [original/cache/](original/cache/) | [after/cache/](after/cache/) |
| Optimized LLVM IR | [original LLIR](original/cache/_mla_gluon.llir) | [after LLIR](after/cache/_mla_gluon.llir) |
| Assembly | [original assembly](original/cache/_mla_gluon.amdgcn) | [after assembly](after/cache/_mla_gluon.amdgcn) |
| Capture / correctness record | [original/capture.json](original/capture.json) | [after/capture.json](after/capture.json) |
| `allow_flush_denorm` | `False` | `True` |
| Recorded GPU | GPU1 / gfx950 | GPU4 / gfx950 |
| Stage1 registers / spills | 481 / 0 | 488 / 0 |
| Max absolute error against FP32 PyTorch | 0.002318739891052246 | 0.002318739891052246 |
| RMS error | 0.00016516199177647013 | 0.00016516253390689022 |

Each cache includes `_mla_gluon` source, Gluon IR, TTGIR, LLIR, assembly, HSACO,
metadata and its grouping JSON. The cached `.source` is the Gluon module supplied
to compilation; the executable host test is [capture_mla.py](capture_mla.py).
Only stage1 cache files are archived here. Capture records also describe the
normally compiled stage2 reducer (14 registers, zero spills).

The original/after comparison includes all compiler changes and the denormal
setting change. It does not isolate the vectorization pass alone. Normal
`make_llir` still does not invoke the pass; the after cache was produced by
substituting the standalone vectorized module before LLVM translation. The
driver asserts exactly one `_mla_gluon` substitution and leaves the reducer on
the normal compilation pipeline.

Raw IR, cache files and capture JSON are preserved byte-for-byte. Absolute paths
inside IR debug locations and capture metadata are historical provenance, not
paths needed to run this copy. The original record's `standalone_replay` describes
an unsuccessful early experiment; its `correctness` describes the successful
original kernel. The after record has the successful final replay. Full original
dump logs remain in the source workspace; new commands below save their own logs
and caches under ignored `run/` directories.

## Environment

- A working gfx950 GPU and ROCm-enabled PyTorch.
- Aiter at `9a5b187c66ab64c6feec30f218c134d2c617050e`, with
  `aiter.ops.triton.gluon.mla_gluon` importable.
- Triton built at the corresponding revision above. For the after run, this
  branch's current compiler has the same implementation as `0492a1e548`.
- The pinned LLVM revision is `b010a18d2b648cab83c83967ff26b8fde11acdc6`.
  Put its `mlir-translate` and `opt`, and the built `triton-opt`, on `PATH`.
  A system ROCm LLVM may not understand these IR files.

The recorded environment used Python 3.10. The driver records revisions from the
actual imported Triton/Aiter checkouts, so review each generated `capture.json`
to confirm which installation ran. It reports `null` for a revision if an
installation has no Git checkout.

All commands below run from this directory:

```bash
cd "$(git rev-parse --show-toplevel)/test/Conversion/amd/Inputs/llvm-vectorize-study/mla"
```

The fixed workload is BF16, batch 128, 64 heads, context 1024, Q/K width 576,
V/output width 512, random page permutation, and seed 123. At the pinned Aiter
revision this selects the bh64 kernel with two KV splits. Validation compares
the complete wrapper output against FP32 PyTorch attention with
`atol=rtol=0.01`. `MLA_GPU` can select a local device; the examples default to GPU4.

## Run the original kernel

Use an environment importing Triton **21e1b37d82** and the pinned Aiter revision.
Running this command with the current branch instead gives the current normal
pipeline, not the archived original compiler.

```bash
mkdir -p run/original
CUDA_VISIBLE_DEVICES="${MLA_GPU:-4}" \
TRITON_CACHE_DIR="$PWD/run/original/cache" TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=_mla_gluon AMDGCN_ENABLE_DUMP=1 \
python3 capture_mla.py --output-dir run/original \
  > run/original/ir_dump.log 2>&1
```

This intentionally omits `--allow-flush-denorm`, matching the saved original.
It writes a new `capture.json`, full dump log, and stage1/stage2 caches.

## Recreate standalone vectorized IR

Use the built tools from this branch. The input below is the saved **after**
post-SymbolDCE dump: it already includes the earlier reduction and BF16 lowering
changes. The archived original input is for the whole-project comparison.

The raw dump has no FP32 denormal environment yet; normal `make_llir` sets it
after LLVM translation. For standalone vectorization, attach the explicit
sign-preserving FP32 environment to the input stream to match
`allow_flush_denorm=True`. The archived input is not modified.

```bash
mkdir -p run/standalone
python3 - <<'PY' > run/standalone/vectorize.log 2>&1
from pathlib import Path
import subprocess

source = Path('after/mla.after.SymbolDCE.mlir').read_text()
anchor = ') attributes {noinline = false,'
assert source.count(anchor) == 1
env = ('denormal_fpenv = #llvm.denormal_fpenv<'
       'default_output_mode = ieee, default_input_mode = ieee, '
       'float_output_mode = preservesign, float_input_mode = preservesign>, ')
source = source.replace(anchor, ') attributes {' + env + 'noinline = false,', 1)
subprocess.run(['triton-opt', '--tritonllvm-vectorize', '-o',
                'run/standalone/mla.vectorized.mlir'],
               input=source, text=True, check=True)
PY
mlir-translate --mlir-to-llvmir run/standalone/mla.vectorized.mlir \
  -o run/standalone/mla.vectorized.ll
opt -passes=verify -disable-output run/standalone/mla.vectorized.ll
```

These checks require no GPU. Inspect the transformation with:

```bash
diff -u after/mla.after.SymbolDCE.mlir after/mla.vectorized.mlir
diff -u original/cache/_mla_gluon.llir after/cache/_mla_gluon.llir
```

`diff` returns 1 when differences are present. The after output has three shared
vector16 exp2 operations, six local vector reductions, three vector16 plus one
vector128 BF16 conversions, and no unresolved conversion casts.

## Run the vectorized replay

Use this branch's Triton build and the pinned Aiter revision. This runs the
archived vectorized IR through the normal remaining lowering, LLVM optimization
and code generation before executing it on the GPU:

```bash
mkdir -p run/after
CUDA_VISIBLE_DEVICES="${MLA_GPU:-4}" \
TRITON_CACHE_DIR="$PWD/run/after/cache" TRITON_ALWAYS_COMPILE=1 \
MLIR_ENABLE_DUMP=_mla_gluon AMDGCN_ENABLE_DUMP=1 \
python3 capture_mla.py --allow-flush-denorm --output-dir run/after \
  --replay-mlir after/mla.vectorized.mlir > run/after/ir_dump.log 2>&1
```

To run the newly generated IR instead, replace the replay path with
`run/standalone/mla.vectorized.mlir`. Verify `correctness.passed: true` and
`replay_count: 1` in the new `run/after/capture.json`; the log must include
`MLA_REPLAY`. The new cache holds the actual optimized LLIR and binary used.

For a same-compiler, same-denormal-setting comparison without the standalone
pass, run the driver with `--allow-flush-denorm` but without `--replay-mlir`, using
separate output/cache paths such as `run/normal/`. This comparison is different
from the archived `original/` snapshot.

## Existing compiler regressions

The focused regression tests are already part of the branch. From the Triton
repository root, after building native changes:

```bash
make
TRITON_TEST_BUILD=$(PYTHONPATH=./python python3 -c \
  'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
lit -v "$TRITON_TEST_BUILD/test/Conversion/amd/reduce_tree_vectorize.mlir" \
  "$TRITON_TEST_BUILD/test/Conversion/amd/fp_to_fp.mlir" \
  "$TRITON_TEST_BUILD/test/Conversion/reduce_to_llvm.mlir"
```

The saved final checkpoint passed its compiler checks, BF16 rounding checks and
MLA numerical replay. Performance was not benchmarked. This review bundle adds
the evidence and runnable commands without enabling the pass in the normal
pipeline or adding the large MLA workload to automatic CI.

Packaging checks on October 16: all 21 archived files matched the source
snapshots byte-for-byte. The standalone command above regenerated the saved
vectorized IR exactly and passed LLVM translation and verification. Driver help,
portable revision recording, cache grouping and relative documentation links
were checked. The historical GPU runs were not repeated for this packaging change.
