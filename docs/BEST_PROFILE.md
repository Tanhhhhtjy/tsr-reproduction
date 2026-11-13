# Best profile on the official TSR source

This modification runs the official TSR generators, formatters, SFT entry and DPO entry from this checkout. It does not import or execute the separate `tsr-reproduction` pipeline. The existing validated Python environment may be shared as a dependency installation.

The reference configuration is the completed `tsr-llama31-8b-cap2500-judge0-1113a` experiment (805 questions, LC Win 29.28%, Win 27.19%). **Those are the earlier reproduction's scores, not scores for this new official-source run.** A new run must finish generation and GPT-4o scoring before any new result is claimed.

## Training and selection

Fresh Llama-3.1-8B Base → three-epoch SFT M0 → Mf0 → M1 → Mf1 → M2. Initializations are fixed:

| Stage | Starts from | Preference data |
|---|---|---|
| M0 | Base | original IFT + EFT SFT data |
| Mf0 | M0 | iter0 D1, qa1 |
| M1 | M0 | iter0 D2, qa1 |
| Mf1 | M1 | iter1 D1, qa2 |
| M2 | **M1** | iter1 D2, qa2 |

Future models provide candidate responses; final M2 does not initialize from a future model. The old `020266a` checkout incorrectly initialized **Mf1 from Mf0**, while its M2 config already specified M1. Upstream `a9a379d82b632e6aa9cb0a350662eca11da16cc8` fixes Mf1. This version retains that fix and rejects templates that try to initialize either Mf1 or M2 from a future model.

All four preference datasets use the same ordered rules:

1. Reparse raw self-judgments with `explicit-score-v1`. Remove candidates with missing, ambiguous, non-finite, malformed or out-of-range scores **before** extrema selection. An explicit score zero remains valid.
2. Require `chosen_score > rejected_score`, including when training Mf0 and Mf1.
3. Maximize chosen score over current/future candidates (future wins equal maxima). Minimize rejected score over current/past candidates (current wins equal minima). D1 has only current candidates for chosen.
4. Within those fixed source/score pools, take the first distinct stripped-text combination in candidate order whose **chosen Unicode character length is ≤2,500**, inclusive. Count the original text with Python `len()`. Do not truncate, minimize length differences, prefer shorter text, or relax source/score priorities. Drop a prompt if no eligible pair exists.

| Setting | Best profile |
|---|---|
| Training hardware | 8 H200 on one SN5 node |
| Training and data seed | 42 |
| Global batch | 32 in all stages |
| SFT | lr 2e-6, 3 epochs, max sequence 2048, micro 2 × accumulation 2 × 8 GPUs |
| DPO | lr 5e-7, 1 epoch, beta 0.1, warmup 0.1, cosine, max sequence 2048, max prompt 512 |
| DPO batching | micro 4 × accumulation 1 × 8 GPUs; drop incomplete final global batch |
| Training execution | ZeRO-2, BF16, original PyTorch AdamW, no communication overlap; DPO FlashAttention-2, no DPO gradient checkpointing |
| Candidate generation | K=7, temperature 1, top_p 1, max output 1024, repetition penalty 1, presence/frequency penalties 0 |
| Local self-judge | **temperature 0**, one judgment per candidate |
| Inference | 8 data-parallel vLLM engines, seeds 42 + shard index, token budget 16384, max sequences 512, prefix cache and chunked prefill |
| Final evaluation | only M2, all 805 frozen questions; answer temperature 0, seed 0, context 4096, max output 1024 |
| External evaluator | GPT-4o-2024-08-06, reference gpt4_1106_preview, official continuous-preference/LC scoring, judge temperature 1, concurrency 16 |

The best entry explicitly exports `TSR_SELF_JUDGE_TEMPERATURE=0`; standalone reference entry points retain temperature 1 when that variable is unset. Candidate generation and self-judging use independent, newly constructed `SamplingParams`. This avoids stale vLLM greedy/sampling state when changing temperature. Fused calls and sharding can change sampled outputs even at the same seed; bitwise replay is not claimed.

## Running

Use Python 3.11 and the verified torch 2.5.1+cu124, Transformers 4.46.3, TRL 0.12.2, DeepSpeed 0.15.4, vLLM 0.6.4.post1 environment. `requirements-best-lock.txt` records the existing full stack, including evaluation dependencies. It is a historical lock installed with dependency resolution disabled; the older vLLM package declares a NumPy constraint incompatible with the recorded NumPy 2.0.2, despite that being the executed environment. Do not silently replace packages when comparing this profile.

For a fresh CUDA 12.4+ Linux host, run `PYTHON=python3.11 bash scripts/setup_best.sh`. It creates `.venv`, installs the locked stack, builds the native handbook as editable, and checks imports. The original official training data is fetched at a pinned commit and checked against `data/manifest.json`; it is ignored by Git and explicitly included when freezing a run. Base weights are supplied separately; no model-weight download or training is started by setup.

On the original installation the configured environment is `/mnt/sn-007/zaninzeng/tsr-reproduction/.venv`; runtime code is still imported from this official checkout's frozen bundle, verified at startup.

Prepare a fresh immutable run (preparation does **not** submit jobs):

```bash
python scripts/prepare_best_run.py \
  --run-name tsr-official-best-1113a \
  --base-model /path/to/Llama-3.1-8B-Base \
  --reference /path/to/frozen/alpaca_reference.json \
  --python /path/to/verified/.venv/bin/python \
  --credentials /private/path/judge.json
```

Preparation requires a clean committed source checkout. It freezes source/data, file hashes, the base-model manifest, run plan, ZeRO/Accelerate settings and deployment manifests under `work/runs/<name>/`. The Kubernetes manifests use the original SN5 namespace, PVC and node labels; adapt deployment fields for another installation. A private mode-0600 credentials file contains `JUDGE_MODEL`, `JUDGE_BASE_URL`, and `JUDGE_API_KEY`; it is referenced at runtime and never copied into the source bundle.

```bash
kubectl apply -f work/runs/tsr-official-best-1113a/job.json
kubectl apply -f work/runs/tsr-official-best-1113a/score-job.json
```

The eight-GPU job runs native training and evaluation answer generation, then exits. A CPU worker waits for the complete 805 answers and scores them with GPT-4o. Only `EVALUATION_COMPLETED.json` confirms the final evaluation. `TRAINING_AND_GENERATION_COMPLETED.json` alone does not imply scoring completed.

To run the same frozen pipeline directly on an eight-GPU training host instead of Kubernetes, use its prepared paths and environment:

```bash
export PATH=/path/to/verified/.venv/bin:$PATH
export TSR_RUN_DIR=/absolute/path/to/work/runs/tsr-official-best-1113a
export SFT_BASE_MODEL=/path/to/Llama-3.1-8B-Base
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash "$TSR_RUN_DIR/bundle/scripts/run_best.sh"
```

The legacy reference behavior of `scripts/sft.sh` and `scripts/tsr.sh` remains available when invoked without the `run_best.sh` entry. The best entry explicitly selects `TSR_PROFILE=best`, dispatching within those official launchers. Native formatters expose `--best`; legacy direct formatter calls retain upstream rules.

## Fixes and checks

| Finding | Action |
|---|---|
| Mf1 init error in old upstream; M2 must use M1 | Keep current upstream correction; add explicit Mf1/M2 initialization guard and provenance per stage |
| Score parser could return out-of-range values or turn missing scores into zero | Use validated explicit score parsing; preserve invalid values as invalid; remove candidates before pairing |
| Generation/judging shared one temperature | Independent fresh judge sampling parameters; verify T=0 while generation remains T=1 |
| Judging assumed every prompt had the first prompt's candidate count | Group by each prompt's actual candidate count; handle empty inputs |
| `--seed` did not explicitly seed the vLLM engine | Pass the engine seed and record shard seeds/sampling settings |
| Logging three random examples crashes when a filtered dataset has fewer than three rows | Bound sample count by actual dataset size |
| NaN/Inf might be hidden by logging filters and exported as a model | Disable NaN filtering in configs and abort DPO on non-finite logged metrics |
| TensorBoard append on FSx can fail with `OSError: Invalid argument` | Write event files on node-local storage and sync them to the run after each stage |
| Shared preprocessing cache can stall distributed ranks | Set a private dataset cache per training rank before importing datasets |
| Silent overwrite/reuse can mix experiments | Require a fresh run and empty stage output directories; freeze source and actual configs |

Existing upstream fixes retained: DPO BOS handling, explicit single-file local DPO loading, original multi-file SFT loading, corrected SFT base ID/model-card API, full batch-32 accumulation and fail-fast launchers. Strict gap, character cap, source tie priorities, distinct-text filtering and judge T=0 are requested **experimental settings**, not claims that the original upstream policy was buggy.

Offline validation:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -q
```

This includes real tiny-model CPU SFT/DPO optimizer steps and native formatter/init/temperature regressions. Tests do not download models, allocate GPUs or call the external judge.

## Retention and audit

Keep scalar histories, final native weights for each stage, actual configs and initialization provenance, canonical preference pairs and filtering stats, inference timing/token counts and sampling settings, GPU utilization samples, and final evaluation outputs/annotations. Never save per-update model/optimizer checkpoints or full training batches/tensor dumps. Remove consumed raw candidate/judgment caches after their corresponding DPO stages validate successfully; retain compact hash/count manifests.

Paths within each run: `status.json`, `bootstrap/`, `trial/configs/`, `trial/metrics/`, `trial/logs/`, `trial/data/`, `trial/models/`, and `trial/alpaca_eval/`. Stage metrics verify positive optimizer steps, finite histories and complete saved weight shards. Native training data loading uses only `train.json`, so adjacent `stats.json` or nested verification copies cannot multiply DPO rows.
