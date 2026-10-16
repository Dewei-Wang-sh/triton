"""Run the existing MLA wrapper and validate a small bh64 workload."""
import importlib
import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
import triton


def git_revision(module_path):
    """Record the checkout that actually supplies the imported module."""
    result = subprocess.run(
        ['git', '-C', str(Path(module_path).resolve().parent), 'rev-parse', 'HEAD'],
        text=True, capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path,
                    default=Path(__file__).resolve().parent / 'run',
                    help='Capture output directory (default: run/ beside this script)')
parser.add_argument('--allow-flush-denorm', action='store_true')
parser.add_argument('--replay-mlir', type=Path,
                    help='Use standalone post-SymbolDCE MLA IR for this fixed workload')
options = parser.parse_args()
if not os.environ.get('TRITON_CACHE_DIR'):
    parser.error('Set TRITON_CACHE_DIR to a dedicated output cache directory')
out = options.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
mla = importlib.import_module('aiter.ops.triton.gluon.mla_gluon')
record = {
    'triton_commit': git_revision(triton.__file__),
    'aiter_commit': git_revision(mla.__file__),
    'kernel_source': mla.__file__,
    'triton_source': triton.__file__,
    'gpu': torch.cuda.get_device_name(),
    'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'cache': os.environ['TRITON_CACHE_DIR'],
    'compiled': {},
}
if options.replay_mlir:
    from triton._C.libtriton import ir, llvm, amd, passes

    if not options.allow_flush_denorm:
        parser.error('--replay-mlir requires --allow-flush-denorm for this experiment')
    replay_path = options.replay_mlir.resolve()
    original_to_module = llvm.to_module
    record['replay_mlir'] = str(replay_path)
    record['replay_count'] = 0

    def replay_to_module(mod, context):
        if 'llvm.func @_mla_gluon(' in str(mod):
            replay = ir.parse_mlir_module(str(replay_path), mod.context)
            assert 'float_input_mode = preservesign' in str(replay)
            assert 'float_output_mode = preservesign' in str(replay)
            # Finish the passes following SymbolDCE before LLVM translation.
            pm = ir.pass_manager(mod.context)
            passes.llvmir.add_di_scope(pm)
            amd.passes.ttgpuir.add_builtin_func_to_llvmir(
                pm, triton.runtime.driver.active.get_current_target().arch, True)
            passes.convert.add_reconcile_unrealized_casts(pm)
            pm.run(replay, 'standalone_mla_replay')
            record['replay_count'] += 1
            print(f'MLA_REPLAY {replay_path}', flush=True)
            return original_to_module(replay, context)
        return original_to_module(mod, context)

    llvm.to_module = replay_to_module

for name in ('_mla_gluon', '_mla_softmax_reducev_kernel'):
    jit = getattr(mla, name)
    original = jit.run
    def capture(*args, _original=original, _name=name, **kwargs):
        kwargs['allow_flush_denorm'] = options.allow_flush_denorm
        kernel = _original(*args, **kwargs)
        record['compiled'][_name] = {
            'hash': kernel.hash, 'metadata': kernel.metadata._asdict(),
            'grid': kwargs.get('grid'), 'n_regs': kernel.n_regs,
            'n_spills': kernel.n_spills,
        }
        return kernel
    jit.run = capture

torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = False
batch, heads, context = 128, 64, 1024
q = torch.randn((batch, heads, 576), device='cuda', dtype=torch.bfloat16)
kv = torch.randn((batch * context, 576), device='cuda', dtype=torch.bfloat16)
pages = torch.randperm(batch * context, device='cuda', dtype=torch.int32).view(batch, context)
lengths = torch.full((batch,), context, device='cuda', dtype=torch.int32)
output = torch.empty((batch, heads, 512), device='cuda', dtype=torch.bfloat16)
record['workload'] = {'batch': batch, 'heads': heads, 'context': context, 'seed': 123,
                      'dtype': 'bf16', 'q_stride': q.stride(), 'kv_stride': kv.stride(),
                      'page_mapping': 'random permutation'}
mla.mla_gluon(q[..., :512], q[..., 512:], kv, output, pages, lengths,
              576 ** -0.5, min_kv_seq_len=context)
torch.cuda.synchronize()
if options.replay_mlir:
    assert record['replay_count'] == 1, 'Expected exactly one standalone MLA substitution'
max_abs, sum_squares, elements = 0.0, 0.0, 0
for start in range(0, batch, 4):
    k = kv[pages[start:start + 4].long()].float()
    scores = torch.matmul(q[start:start + 4].float(), k.transpose(1, 2)) * (576 ** -0.5)
    reference = torch.matmul(torch.softmax(scores, dim=-1), k[..., :512])
    actual = output[start:start + 4].float()
    torch.testing.assert_close(actual, reference, atol=0.01, rtol=0.01)
    diff = actual - reference
    max_abs = max(max_abs, diff.abs().max().item())
    sum_squares += diff.square().sum().item()
    elements += diff.numel()
record['correctness'] = {'passed': True, 'reference': 'FP32 PyTorch attention',
                         'atol': 0.01, 'rtol': 0.01, 'max_abs': max_abs,
                         'rms': (sum_squares / elements) ** 0.5}
(out / 'capture.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
print('MLA_CAPTURE ' + json.dumps(record, default=str), flush=True)
