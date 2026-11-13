#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TSR_PROFILE=best TSR_PERFORMANCE_PROFILE=h200-fast TSR_DROP_INVALID_SCORES=1
export TSR_SELF_JUDGE_TEMPERATURE=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export TSR_RUN_DIR=${TSR_RUN_DIR:-$PROJECT_ROOT/work/runs/best-$(date -u +%Y%m%d-%H%M%S)}
export SFT_BASE_MODEL=${SFT_BASE_MODEL:?Set SFT_BASE_MODEL to Llama-3.1-8B Base}
export PYTHONPATH="$PROJECT_ROOT/alignment-handbook/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TSR_RUN_DIR/trial/logs" "$TSR_RUN_DIR/trial/metrics"
if ! mkdir "$TSR_RUN_DIR/.started"; then
    echo "Run already started; use a fresh run directory" >&2; exit 1
fi
trap 'python "$PROJECT_ROOT/scripts/validate_best.py" failure "$TSR_RUN_DIR"' ERR
python "$PROJECT_ROOT/scripts/validate_best.py" status "$TSR_RUN_DIR" sft
bash "$PROJECT_ROOT/scripts/sft.sh" > "$TSR_RUN_DIR/trial/logs/train_M0.log" 2>&1
python "$PROJECT_ROOT/scripts/validate_best.py" model "$TSR_RUN_DIR" M0
bash "$PROJECT_ROOT/scripts/tsr.sh"
python "$PROJECT_ROOT/scripts/validate_best.py" status "$TSR_RUN_DIR" evaluation_generation
final="$TSR_RUN_DIR/trial/models/Llama-3.1-8B-mix-sft-epoch3-temporal-iterjudge-iter1"
output="$TSR_RUN_DIR/trial/alpaca_eval/M2_future_tie_charcap_distinct/model_outputs.json"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*} python "$PROJECT_ROOT/tsr/gen_alpaca.py" \
    --model "$final" --out "$output" --temperature 0 --seed 0 \
    --reference_outputs "$TSR_RUN_DIR/inputs/alpaca_reference.json" \
    > "$TSR_RUN_DIR/trial/logs/eval_generation.log" 2>&1
python "$PROJECT_ROOT/scripts/validate_best.py" status "$TSR_RUN_DIR" training_and_generation_completed
python - "$TSR_RUN_DIR" <<'PY'
from pathlib import Path
import json,sys,time
run=Path(sys.argv[1]); rows=json.loads((run/'trial/alpaca_eval/M2_future_tie_charcap_distinct/model_outputs.json').read_text())
reference=json.loads((run/'inputs/alpaca_reference.json').read_text())
if len(rows)!=805 or [r['instruction'] for r in rows]!=[r['instruction'] for r in reference]:
    raise ValueError('Incomplete evaluation generation')
(run/'TRAINING_AND_GENERATION_COMPLETED.json').write_text(json.dumps({'completed_unix':time.time(),'native_official_pipeline':True,'questions':805}))
PY
