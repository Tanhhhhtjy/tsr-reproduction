#!/usr/bin/env bash
# Install the verified best-profile environment in this official checkout.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -x "$root/.venv/bin/python" ]]; then
    "${PYTHON:-python3.11}" -m venv "$root/.venv"
fi
py="$root/.venv/bin/python"
"$py" -m pip install -U pip wheel
"$py" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# Preserve the executed historical stack; its vLLM/NumPy metadata cannot be resolved together.
"$py" -m pip install --no-deps -r "$root/requirements-best-lock.txt"
"$py" -m pip install --no-deps 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
"$py" -m pip install --no-deps -e "$root/alignment-handbook"
"$py" - <<'PY'
import importlib.metadata as metadata
import torch,transformers,trl,deepspeed,vllm,flash_attn,alpaca_eval
for name in ['torch','transformers','trl','deepspeed','vllm','flash-attn','alpaca_eval']:
    print(name,metadata.version(name))
print('CUDA available:',torch.cuda.is_available())
PY

"$py" "$root/scripts/download_data.py"
