"""Launch vLLM 0.19.1 with the scorer's exact flags, adapted to ONE 24 GB RTX 3090.

The argv is produced by the official `adk_submission.VllmServer.build_cmd()` with the getting-started
notebook's VllmConfig (NB1 cell 8), so parsers, reasoning-config, dtype, LoRA flags and
--no-scheduler-reserve-full-isl are byte-identical to the scorer. A profile then appends single-GPU
flags; every addition is printed as a deviation. Run with the runner venv (it has adk_submission) and
point --vllm-python at the vLLM venv's interpreter:

    uv run --project harness/runner python -m harness.serve.serve \
        --model-dir <models>/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2 \
        --vllm-python ~/venvs/vllm/bin/python --profile 3090 [--submission <dir|zip>] [--print]

Profiles (see docs/runner.md §3):
  scorer      exactly NB1 on 1 GPU (util 0.90, no TP flag). 31B will not fit KV at 32k on 24 GB.
  3090        31B: util 0.945, 1 seq, 512 batched tokens, compile off + FULL_DECODE_ONLY CUDA graph (~34.6 tok/s, bf16 KV at 32k).
  small       12B/E4B/E2B: util 0.90, --max-num-seqs 4.
  a100        31B on ONE A100-80GB: scorer flags
              unchanged (util 0.90, vLLM default --max-num-seqs/--max-num-batched-tokens, bf16 KV, vision on). No
              single-GPU memory hacks.
  4xl4        31B on FOUR L4: the Kaggle scorer's own hardware, so argv = a100's plus the
              scorer's `--tensor-parallel-size 4` (NB1 picks tp=4 when 4 GPUs are visible). No other flags.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

NB1_SERVER_ENV = {
    'TRANSFORMERS_NO_TF': '1',
    'TRANSFORMERS_NO_FLAX': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1',
    'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1200',
    'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True',
    'LITELLM_TELEMETRY': 'False',
    'OPENAI_API_KEY': 'EMPTY',
    'LOCAL_API_KEY': 'EMPTY',
    'OTEL_SDK_DISABLED': 'true',
}

ALIASES = ('gemma-4-31b-it-qat-w4a16-ct', 'gemma-4-12b-it-qat-w4a16-ct',
           'gemma-4-e4b-it-qat-w4a16-ct', 'gemma-4-e2b-it-qat-w4a16-ct')
# NB1 MODEL_VLLM_EXTRA_ARGS
MODEL_EXTRA_ARGS = {
    'gemma-4-12b-it-qat-w4a16-ct': ['--hf-overrides', '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'],
}
NO_MM = json.dumps({'image': 0, 'video': 0})
PROFILES: dict[str, dict] = {
    'scorer': {'gpu_memory_utilization': 0.90, 'extra': []},
    # verified 2026-11-02 (RTX 3090 under WSL2): bf16 KV at 32k, 4.08 GiB KV (1.11x), ~34.6 tok/s single stream (~Kaggle 4xL4).
    # torch.compile (mode 3) costs ~2.6 GiB; mode 0 + one FULL_DECODE_ONLY CUDA graph keeps memory and speed.
    # fp8 KV is impossible on Ampere here (Triton e4m3 unsupported on sm86), so there is no fp8 profile.
    '3090': {'gpu_memory_utilization': 0.945,
             'extra': ['--max-num-seqs', '1', '--max-num-batched-tokens', '512', '--limit-mm-per-prompt', NO_MM,
                       '--compilation-config', json.dumps({'mode': 0, 'cudagraph_mode': 'FULL_DECODE_ONLY',
                                                          'cudagraph_capture_sizes': [1]})]},
    'small': {'gpu_memory_utilization': 0.90, 'extra': ['--max-num-seqs', '4', '--limit-mm-per-prompt', NO_MM]},
    # one A100-80GB: argv equals the scorer's minus --tensor-parallel-size; vLLM 0.19.1 gives A100 the same scheduler
    # defaults as the L4 (max_num_batched_tokens 2048, max_num_seqs 256: arg_utils.py "a100" special case)
    'a100': {'gpu_memory_utilization': 0.90, 'extra': []},
    # four L4 = the scorer's hardware: NB1 passes tensor_parallel_size=4 when >=4 GPUs are visible (docs/harness-contracts.md)
    '4xl4': {'gpu_memory_utilization': 0.90, 'extra': [], 'tp': 4},
}
PROFILE_NOTES = {
    'a100': ['1x A100-80GB (72 GB at util 0.90) instead of 4x L4 (4 x 21.6 GB at 0.90, TP=4): the KV pool is much larger '
             'and there is no tensor-parallel all-reduce, so outputs can differ only at floating-point noise level',
             'attention backend / kernels are chosen for sm_80 and may differ from the L4 (sm_89) ones; timing is not '
             'comparable to Kaggle'],
    '4xl4': ['4x L4 (sm_89, PCIe, 24 GB each) with TP=4 = the scorer hardware class; host CPU/RAM/PCIe topology and the '
             'NCCL transport (P2P vs SHM) depend on the rented machine, so timing is close to Kaggle but not identical'],
}
DEVIATION_NOTES = {
    '--max-num-seqs': 'caps concurrent sequences (vLLM default 256 here); lowers sampler/cudagraph memory; '
                      'no effect on outputs, limits throughput to ~N parallel tasks',
    '--max-num-batched-tokens': 'smaller prefill chunk; shrinks the sliding-window KV reserve per request; '
                                'slower prefill, same outputs',
    '--limit-mm-per-prompt': 'text-only: no image/video inputs (the agent never sends any); frees the vision '
                             'encoder budget (inferred, verify in the vLLM log)',
    '--kv-cache-dtype': 'fp8 KV cache: halves KV memory, small numerics change vs scorer bf16 KV',
    '--gpu-memory-utilization': 'scorer 0.90 of each 24 GB L4 with TP=4; here one 24 GB card',
}


def _alias_of(model_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    for part in reversed(model_dir.parts):
        if part.lower() in ALIASES:
            return part.lower()
    raise SystemExit(f'cannot infer the model alias from {model_dir}; pass --alias one of {ALIASES}')


def _show_env(k: str, v: str) -> str:
    """Env values may be secrets (OPENAI_API_KEY is in NB1_SERVER_ENV): never print those."""
    return '<redacted>' if any(w in k.upper() for w in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) else repr(v)


def nb1_env(ambient: dict[str, str], keep_ambient: bool = False) -> tuple[dict[str, str], list[str]]:
    """(server env, deviation notes). The scorer (NB1) env wins over pre-exported variables by default (audit-2 S2:
    `setdefault` let e.g. an ambient PYTORCH_CUDA_ALLOC_CONF or TRANSFORMERS_OFFLINE=0 silently replace the scorer's).
    Every ambient value that differs is listed: overridden by default, kept (a real deviation) with keep_ambient."""
    env = dict(ambient)
    devs: list[str] = []
    for k, v in NB1_SERVER_ENV.items():
        cur = ambient.get(k)
        if cur is not None and cur != v:
            if keep_ambient:
                devs.append(f'env {k}={_show_env(k, cur)} kept from the ambient env by --keep-ambient-env '
                            f'(scorer: {_show_env(k, v)})')
                continue
            devs.append(f'env {k}: ambient {_show_env(k, cur)} overridden to the scorer value {_show_env(k, v)} '
                        '(not a deviation; pass --keep-ambient-env to keep the ambient value)')
        env[k] = v
    return env, devs


def build(args: argparse.Namespace) -> tuple[list[str], dict[str, str], list[str]]:
    from adk_submission import VllmConfig, VllmServer, discover_adapters
    try:
        from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS
    except ImportError:  # slim serving env (Modal image) without swegemma's heavy deps; same constant as swegemma.config
        ALLOWED_ADAPTER_EXTENSIONS = frozenset({'.safetensors'})

    model_dir = Path(args.model_dir).expanduser().resolve()
    alias = _alias_of(model_dir, args.alias)
    adapters = None
    adapter_map: dict = {}
    if args.submission:
        sub = Path(args.submission).expanduser().resolve()
        if sub.suffix == '.zip':
            import zipfile
            tmp = Path(tempfile.mkdtemp(prefix='gemma4-sub-'))
            with zipfile.ZipFile(sub) as zf:
                zf.extractall(tmp)
            sub = tmp
        adapters = discover_adapters(str(sub), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
        adapter_map = getattr(adapters, 'adapters', None) or {}
    ranks = [r for info in adapter_map.values() if (r := getattr(info, 'rank', None)) is not None and r > 0]
    prof = PROFILES[args.profile]
    util = args.gpu_memory_utilization or prof['gpu_memory_utilization']
    prof_extra = list(prof['extra'])
    if alias in MODEL_EXTRA_ARGS and '--limit-mm-per-prompt' in prof_extra:
        # 12B is served text-only (Gemma4ForCausalLM override): no multimodal limits to set
        i = prof_extra.index('--limit-mm-per-prompt')
        del prof_extra[i:i + 2]
    extra = [*MODEL_EXTRA_ARGS.get(alias, []), *prof_extra, *shlex.split(args.extra or '')]
    cfg = VllmConfig(
        model=str(model_dir), port=args.port, host=args.host,
        tool_call_parser='gemma4', reasoning_parser='gemma4', max_model_len=args.max_model_len,
        dtype='bfloat16', gpu_memory_utilization=util, enable_auto_tool_choice=True,
        enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)),
        max_lora_rank=max(ranks) if ranks else 128, scheduler_reserve_full_isl=False,
        tensor_parallel_size=int(prof.get('tp', 1)), startup_timeout=1200, extra_args=extra,
    )
    srv = VllmServer(cfg, adapter_manifest=adapters)
    cmd = srv.build_cmd()
    try:
        os.unlink(srv.log_path)
    except OSError:
        pass
    cmd[0] = args.vllm_python
    env, env_devs = nb1_env(dict(os.environ), keep_ambient=bool(getattr(args, 'keep_ambient_env', False)))

    tp = int(prof.get('tp', 1))
    deviations = ['--tensor-parallel-size omitted (scorer: 4 on 4xL4; 1 GPU here)'] if tp == 1 else []
    if util != 0.90:
        deviations.append(f'--gpu-memory-utilization {util}: {DEVIATION_NOTES["--gpu-memory-utilization"]}')
    for flag in [a for a in extra if a.startswith('--') and a not in ('--hf-overrides',)]:
        deviations.append(f'{flag}: {DEVIATION_NOTES.get(flag, "user extra flag")}')
    if args.max_model_len != 32768:
        deviations.append(f'--max-model-len {args.max_model_len} (scorer 32768): requests with prompt + '
                          'max_tokens above it fail; agent behaviour changes')
    deviations.extend(PROFILE_NOTES.get(args.profile, []))
    deviations.extend(env_devs)
    if args.host != '127.0.0.1':
        deviations.append(f'--host {args.host} (network-visible)')
    if env.get('VLLM_API_KEY'):
        deviations.append('API key enforced via the VLLM_API_KEY env var (vLLM default for --api-key; not in argv)')
    return cmd, env, deviations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model-dir', required=True, help='.../gemma-4/other/<alias>/2 (Kaggle layout) or an HF snapshot dir')
    ap.add_argument('--alias', choices=ALIASES, help='model alias if not inferable from --model-dir')
    ap.add_argument('--vllm-python', default=os.environ.get('VLLM_PYTHON', sys.executable),
                    help='python of the vLLM 0.19.1 venv (default $VLLM_PYTHON)')
    ap.add_argument('--profile', choices=sorted(PROFILES), default='3090')
    ap.add_argument('--submission', help='submission dir/zip whose adapters/ are mounted as LoRA modules')
    ap.add_argument('--gpu-memory-utilization', type=float)
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--extra', help='extra vLLM flags, one quoted string')
    ap.add_argument('--print', action='store_true', help='print the command and deviations, do not start')
    ap.add_argument('--log', help='also tee server output to this file (default: stdout only)')
    ap.add_argument('--keep-ambient-env', action='store_true',
                    help='keep pre-exported values of the scorer env vars (NB1_SERVER_ENV) instead of overriding them; '
                         'each kept value is printed as a deviation')
    args = ap.parse_args(argv)

    cmd, env, deviations = build(args)
    print('# vLLM command (scorer argv from adk_submission.VllmServer.build_cmd + profile):')
    print(' '.join(shlex.quote(c) for c in cmd))
    print('# deviations from the scorer:')
    for d in deviations:
        print(f'#   - {d}')
    if args.print:
        return 0
    sys.stdout.flush()
    if args.log:
        # keep vLLM in the foreground; tee via a shell so Ctrl-C stops the server
        line = ' '.join(shlex.quote(c) for c in cmd) + f' 2>&1 | tee -a {shlex.quote(args.log)}'
        os.execvpe('/bin/bash', ['/bin/bash', '-o', 'pipefail', '-c', line], env)
    os.execvpe(cmd[0], cmd, env)
    return 0


if __name__ == '__main__':
    sys.exit(main())
