#!/usr/bin/env bash
set -Eeuo pipefail
run=$1
runtime_python=${TSR_PYTHON:?Missing configured Python environment}
export PATH="$(dirname "$runtime_python"):$PATH"
export TSR_RUN_DIR="$run" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Shanghai
export HF_HOME="/mnt/sn-007/zaninzeng/tsr-reproduction/.runtime/hf-cache"
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export TSR_RANK_CACHE_ROOT="$run/runtime/datasets"
export TORCH_EXTENSIONS_DIR="$run/runtime/torch-extensions" TRITON_CACHE_DIR="$run/runtime/triton"
export VLLM_CACHE_ROOT="$run/runtime/vllm" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$run/bundle/alignment-handbook/src:$run/bundle"
trap '"$runtime_python" "$run/bundle/scripts/validate_best.py" failure "$run"' ERR
mkdir -p "$run/bootstrap" "$run/runtime"
exec > >(tee -a "$run/bootstrap/job.log") 2>&1
"$runtime_python" -m pip freeze > "$run/bootstrap/pip-freeze.txt"
"$runtime_python" - <<'PY'
from pathlib import Path
import hashlib,json,os,torch
run=Path(os.environ['TSR_RUN_DIR'])
if torch.cuda.device_count()!=8:
    raise RuntimeError('Expected exactly eight visible GPUs')
for rel,expected in json.loads((run/'source-sha256.json').read_text()).items():
    if hashlib.sha256((run/'bundle'/rel).read_bytes()).hexdigest()!=expected:
        raise RuntimeError('Frozen official source differs: '+rel)
import importlib.util
native_alignment=importlib.util.find_spec('alignment').origin
if not Path(native_alignment).resolve().is_relative_to(run/'bundle'):
    raise RuntimeError('Imported another checkout of alignment-handbook')
(run/'gpu-validated.json').write_text(json.dumps({'gpus':[torch.cuda.get_device_name(i) for i in range(8)],'native_alignment':native_alignment}))
PY
(
    while true; do
        nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader >> "$run/bootstrap/gpu.csv" || true
        sleep 30
    done
) &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
bash "$run/bundle/scripts/run_best.sh"
