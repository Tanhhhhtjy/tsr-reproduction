# Best completed TSR run: cap2500, self-judge temperature 0

Run: `tsr-llama31-8b-cap2500-judge0-1113a`. Final model: `M2_future_tie_charcap_distinct`.

## Result

Full AlpacaEval 2.0 (805 questions): **LC Win 29.27982054398985%, Win 27.1869097989354%**, mean 1,761.7553 characters. Two answers reached the generation limit (0.24845%). Training and scoring completed successfully. Only final M2 was evaluated in this run; no intermediate evaluation curve is claimed.

This is the highest LC Win among the completed variants inspected at publication. It is a single-seed modified TSR result, not a controlled multi-seed comparison with the paper.

## Training

| Setting | Value |
|---|---|
| Base | `meta-llama/Llama-3.1-8B` |
| Initialization | fresh Base → SFT M0 → Mf0 → M1 → Mf1 → M2 |
| Hardware | one SN5 node, 8 H200 |
| Training/data seed | 42 |
| Global batch | 32, all stages |
| SFT | 3 epochs, lr 2e-6, max sequence 2048; 8 × micro 2 × accumulation 2 |
| DPO | 1 epoch, lr 5e-7, beta 0.1, warmup 0.1, cosine; max sequence 2048, max prompt 512 |
| DPO batching | 8 × micro 4 × accumulation 1; drop incomplete last global batch |
| Execution | BF16, ZeRO-2, communication overlap disabled, original AdamW; DPO FlashAttention-2, gradient checkpointing off |
| Candidate generation | K=7, temperature 1, top_p 1, max output 1024 |
| Penalties | repetition 1; presence 0; frequency 0 |
| Local self-judge | temperature 0, one judgment per response, `explicit-score-v1` |
| Inference | 8 vLLM DP shards; seeds 42 + shard index, max sequences 512, batched-token budget 16384, prefix cache and chunked prefill |
| Data | original SFT data; qa1 for iter0 and qa2 for iter1, 5,000 prompts each |

The raw score parser removes invalid candidates before computing extrema; it never turns a parse failure into score zero. Every retained preference has a strictly positive score gap. Future wins chosen source ties; current wins rejected source ties. After fixing those pools, the first distinct-text pair with chosen Unicode character length ≤2500 is selected. No source/score fallback or response truncation is allowed. The cap applies to DPO chosen text only, not SFT, rejected, raw candidates or evaluation answers.

| Dataset / trained model | Retained pairs | Equal-score prompts removed | No eligible chosen ≤2500 | Invalid candidates removed | Max chosen characters |
|---|---:|---:|---:|---:|---:|
| iter0/D1 → Mf0 | 4,256 | 223 | 521 | 1 | 2,500 |
| iter0/D2 → M1 | 4,598 | 93 | 309 | 1 | 2,500 |
| iter1/D1 → Mf1 | 4,137 | 705 | 158 | 2 | 2,500 |
| iter1/D2 → M2 | 4,160 | 705 | 135 | 2 | 2,500 |

Retained equal-score pairs, identical-text pairs and over-cap chosen pairs are all zero. There is no shortest-answer preference, length matching or chosen-longer restriction. The frozen plan's `length_matching.order` field is inactive (`enabled=false`); the active `chosen_character_cap` and pair statistics specify the applicable selection order.

## Evaluation

Answer generation: temperature **0**, seed **0**, top_p **1**, max output **1024**, context **4096**, EOS **128001** and stop strings `<|user|>`, `<|system|>`. The training chat template and IFT system prompt are used with BOS exactly once. The fixed 805-question reference is included.

External evaluator: `gpt-4o-2024-08-06`, reference `gpt4_1106_preview`, native weighted AlpacaEval annotator, judge temperature **1**, request concurrency **16**. `Win` is the official `win_rate`, aggregating continuous preferences; the leaderboard's `discrete_win_rate` (27.8882%) is a different column. LC Win is the evaluator's length-controlled estimate. Neither is training reward.

On the original SN5 installation, after preparing the run as shown in the root README, review the manifest and submit:

```bash
kubectl apply -f work/runs/tsr-best-cap2500-judge0-rerun/job.json
```

The GPU job trains and generates evaluation answers, then exits. Scoring uses the configured Python environment and a private JSON file (do not commit a populated copy):

```json
{
  "JUDGE_MODEL": "gpt-4o-2024-08-06",
  "JUDGE_BASE_URL": "https://api.openai.com/v1",
  "JUDGE_API_KEY": ""
}
```

```bash
chmod 600 /private/path/judge.json
.venv/bin/python scripts/score_when_ready.py \
  --run work/runs/tsr-best-cap2500-judge0-rerun \
  --credentials /private/path/judge.json --api-concurrency 16
```

Only `EVALUATION_COMPLETED.json` confirms completion of requested scoring. Preparing a bundle or verifying committed records makes no API calls.

## Provenance and storage

The archive includes original job manifests, input/model hashes, frozen source, actual stage configs, scalar training histories, performance metrics, final evaluation answers and annotations, plus file hashes. Original absolute paths document the executed run; they are not portable configuration defaults.

Runnable source incorporates later offline regression checks; the frozen `source_snapshot/` remains byte-identical to the executed run. This branch also preserves the target repository's `drop_ties` variant. Existing historical `results/` were not replaced.

Training generates no per-update checkpoints or optimizer dumps. The driver deletes consumed raw candidate/judgment caches and isolated duplicate DPO inputs, retaining compact metrics, canonical preference data and final stage models locally. Weights and canonical training pairs are not uploaded to GitHub.
