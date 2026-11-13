# TSR reproduction: best completed run

This branch publishes the highest **AlpacaEval 2.0 length-controlled win rate (LC Win)** among our completed TSR variants. It contains the runnable pipeline and the records of **one** new experiment, `tsr-llama31-8b-cap2500-judge0-1113a`. Existing results from the repository's `main` branch remain historical records.

| Model | Questions | LC Win (%) | Win (%) | Mean characters |
|---|---:|---:|---:|---:|
| Llama-3.1-8B, final M2 | 805 / 805 | **29.28** | **27.19** | **1,762** |

“Best” means highest LC Win, not highest unadjusted Win. These are single-run results (training seed 42), not a multi-seed average. This is a modified TSR experiment, not an exact reproduction of every paper setting. AlpacaEval uses `gpt-4o-2024-08-06` against `gpt4_1106_preview`. `Win` is the evaluator's native `win_rate`; the archived leaderboard also contains the separate discrete win rate. See [the experiment records](experiments/tsr-best-cap2500-judge0-20261113/README.md).

## Selection rules

All four DPO datasets, including those training the future models, follow these rules:

1. Remove candidates whose raw self-judgment has a missing, ambiguous, non-finite or out-of-range score. An explicitly parsed score of zero is valid.
2. Require `chosen_score > rejected_score`: remove equal-score pairs at every stage.
3. Chosen comes from the maximum-scoring current/future source; future wins equal maxima. Rejected comes from the minimum-scoring current/past source; current wins equal minima. D1 has no future source, so chosen is current-best.
4. Within those fixed source/score pools, take the first pair in candidate order with different stripped texts and **`len(chosen_text) <= 2500`**. This counts Unicode characters in the original response, not tokens or bytes. If none qualifies, discard the prompt without relaxing source/score priorities or truncating text.

There is no length-gap minimization, chosen-longer requirement, or rejected-length cap. Training self-generation uses **temperature 1**, local self-judging **temperature 0**, and final evaluation answer generation **temperature 0**. The external GPT-4o evaluator retains temperature 1.

## Reproduce

The pipeline starts from verified **Llama-3.1-8B Base → fresh SFT M0 → Mf0 → M1 → Mf1 → M2**. Upstream is [TemporalSelfRewarding/TSR](https://github.com/TemporalSelfRewarding/TSR) at `020266aa9ef3cc35bf81384c9b59f98cff62aa1c` plus `patches/`.

The run used one SN5 node with **8 H200**, global batch **32**, SFT 3 epochs at `2e-6`, and each DPO stage 1 epoch at `5e-7`, beta `0.1`. The H200 profile uses ZeRO-2 and eight data-parallel vLLM shards. Candidate sampling uses K=7, top_p=1, max output 1024 tokens, repetition penalty 1, presence/frequency penalties 0.

After configuring `config.env` and running `bash setup.sh` (Python 3.11, CUDA 12.4+, gated base-model access required), prepare a **fresh** run:

```bash
.venv/bin/python scripts/prepare_sn5.py \
  --run-name tsr-best-cap2500-judge0-rerun \
  --gpus 8 --node any-sn5 \
  --variant future_tie_charcap_distinct --max-chosen-characters 2500 \
  --seed 42 --self-judge-temperature 0 --drop-invalid-scores \
  --eval-temperature 0 --eval-final-only --skip-smoke \
  --performance-profile h200-fast --zero-stage 2 --dpo-micro-batch 4
```

Preparation writes an immutable bundle and `job.json`; it does not submit a job. The SN5 launcher contains the original account, shared-storage path, PVC and node-label settings; adapt these deployment details before using another account or cluster. Actual scientific settings and stage configs are in [BEST_RUN.md](docs/BEST_RUN.md) and the archive. Do not reuse older SFT weights, pair datasets or model directories for this fresh-run protocol.

Final evaluation generation uses 805 frozen questions, seed 0, context 4096 and max output 1024. Scoring runs separately using `scripts/score_when_ready.py` and a private, mode-0600 credentials file; no credentials are included. See [scoring instructions](docs/BEST_RUN.md#evaluation).

## Records and validation

- [Complete results and training curves](experiments/tsr-best-cap2500-judge0-20261113/README.md)
- [Exact configuration, pair counts and deployment instructions](docs/BEST_RUN.md)
- [Original repository reproduction report](docs/ORIGINAL_REPRODUCTION.md), retained for historical context

```bash
python3 scripts/verify_best_run.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -q
```

The archive contains all 805 answers and GPT-4o annotations, the fixed reference, five stages' scalar training histories, four pair-filtering summaries, original configs, frozen source and SHA-256 manifests. It excludes model weights, optimizer state, raw training-generation caches and API credentials. Final weights and canonical pair datasets remain on the original training storage; this Git repository is a code/results archive.

Consumed raw training caches are removed by the driver; metrics, configs, canonical preference data and final stage models are retained locally. No per-update model or optimizer checkpoints are saved. Inference scheduling/sharding can change sampled answers even with the same seed; bitwise replay is not claimed.

## License

MIT. The patches modify MIT-licensed upstream code (© 2025 Yidong Wang & Xin Wang). Llama-generated outputs are subject to the Llama 3.1 Community License.
