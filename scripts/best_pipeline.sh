#!/usr/bin/env bash
# Best-profile execution using this checkout's official generators, formatters and trainers.
best_status() {
    python "$PROJECT_ROOT/scripts/validate_best.py" status "$TSR_RUN_DIR" "$1"
}

best_infer() {
    local mode=$1 model=$2 input=$3 out=$4 extra_in=${5:-} extra_out=${6:-}
    local -a devices pids=() parts=() extra_parts=() extra=()
    IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
    local i=0 name; name=$(basename "$out" .json)
    mkdir -p "$(dirname "$out")" "$TSR_RUN_DIR/trial/logs" "$TSR_RUN_DIR/trial/metrics/inference"
    for device in "${devices[@]}"; do
        extra=()
        if [[ -n "$extra_in" ]]; then
            extra=(--extra-judge-input "$extra_in" --extra-judge-out "$extra_out.part$i")
            extra_parts+=("$extra_out.part$i")
        fi
        CUDA_VISIBLE_DEVICES=$device python "$PROJECT_ROOT/tsr/infer.py" "$mode" \
            --model "$model" --input "$input" --out "$out.part$i" --seed 42 \
            --vllm_seed "$((42+i))" --shard "$i" --nshards "$WORLD_SIZE" "${extra[@]}" \
            > "$TSR_RUN_DIR/trial/logs/$name.part$i.log" 2>&1 &
        pids+=("$!"); parts+=("$out.part$i"); i=$((i+1))
    done
    local result=0
    for pid in "${pids[@]}"; do wait "$pid" || result=1; done
    if [[ "$result" != 0 ]]; then echo "Native inference failed: $name; see per-shard logs" >&2; return 1; fi
    python "$PROJECT_ROOT/tsr/infer.py" merge --out "$out" "${parts[@]}"
    if [[ -n "$extra_out" ]]; then
        python "$PROJECT_ROOT/tsr/infer.py" merge --out "$extra_out" "${extra_parts[@]}"
        rm -- "${extra_parts[@]}"
    fi
    for ((i=0; i<WORLD_SIZE; i++)); do
        mv "$out.part$i.metrics.json" "$TSR_RUN_DIR/trial/metrics/inference/$name.part$i.json"
    done
    rm -- "${parts[@]}"
}

best_pairs() {
    local mode=$1 output=$2 current=$3 past=$4 future=${5:-}
    local -a extra=()
    [[ -n "$future" ]] && extra=(--input_path_future "$future")
    mkdir -p "$output"
    python "$PROJECT_ROOT/settings/tsr/reformat2DPO_$mode.py" \
        --input_path_chosen "$current" --input_path_rejected "$past" "${extra[@]}" \
        --output_path "$output/train.json" --best --max-chosen-characters 2500
    python "$PROJECT_ROOT/scripts/validate_best.py" pairs "$output"
}

best_train() {
    local stage=$1 template=$2
    best_status "$stage"
    launch_training dpo "$PROJECT_ROOT/settings/tsr/config/dpo/$template.yaml" \
        > "$TSR_RUN_DIR/trial/logs/train_$stage.log" 2>&1
    python "$PROJECT_ROOT/scripts/validate_best.py" model "$TSR_RUN_DIR" "$stage"
}

best_run_tsr() {
    local base="$TSR_RUN_DIR/trial/models/Llama-3.1-8B-mix-sft-epoch3"
    local stem="$base-temporal-iterjudge-iter"
    local qa raw current past future
    for iter in 0 1; do
        qa="$PROJECT_ROOT/data/SelfPlayBeginData/qa$((iter+1)).jsonl"
        raw="$TSR_RUN_DIR/trial/candidates/iter$iter"
        current="$raw/${iter}_chosen_judgement.json"
        past="$raw/${iter}_rejected_judgement.json"
        future="$raw/${iter}_future_judgement.json"
        best_status "iter${iter}_current_generation_and_judging"
        if [[ "$iter" == 0 ]]; then
            best_infer genjudge "$base" "$qa" "$current"
            past="$current"  # Identical M0 source; no duplicate generation or cache.
            current_model="$base"
        else
            current_model="${stem}0"
            best_infer gen "$base" "$qa" "$raw/1_rejected_response.json"
            best_infer genjudge "$current_model" "$qa" "$current" "$raw/1_rejected_response.json" "$past"
        fi
        best_pairs past "$TSR_RUN_DIR/trial/data/iter${iter}_tmp" "$current" "$past"
        best_train "Mf$iter" "config_full_iter${iter}_tmp"
        best_status "iter${iter}_future_generation_and_judging"
        best_infer gen "${stem}${iter}_tmp" "$qa" "$raw/${iter}_future_response.json"
        best_infer judge "$current_model" "$raw/${iter}_future_response.json" "$future"
        best_pairs future "$TSR_RUN_DIR/trial/data/iter${iter}" "$current" "$past" "$future"
        best_train "M$((iter+1))" "config_full_iter${iter}"
        python "$PROJECT_ROOT/scripts/validate_best.py" cleanup "$TSR_RUN_DIR" "iter$iter"
    done
}
