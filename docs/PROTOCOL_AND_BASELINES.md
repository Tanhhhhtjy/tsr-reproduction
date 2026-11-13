# Standardized Evaluation Protocol & Baseline Reference

This document freezes the exact evaluation protocol, updated TSR pairing rules (`cap2500 + judge T=0 + eval T=0`), and baseline numbers established across our reproduction (`Tanhhhhtjy/tsr-reproduction`) and the updated official TSR protocol (`tsr-llama31-8b-cap2500-judge0-1113a`, PR #1). All downstream continuation experiments use this protocol as the canonical reference.

---

## 1. Frozen AlpacaEval 2.0 Evaluation Protocol

| Item | Specification |
|---|---|
| Benchmark | AlpacaEval 2.0 (`805` instructions) |
| Reference outputs | Official `gpt4_1106_preview` (`alpaca_eval_gpt4_baseline`), SHA-256 `f90dac29878557296f0f2d79adbd25eeef69bc09478a2af1e708a8ed2016950d` |
| External judge | `gpt-4o-2024-08-06` |
| Annotator | `alpaca_eval_clf_gpt4_turbo` (`alpaca_eval==0.6.6`), logprob-weighted over `{"m", "M"}` (`max_tokens=1`, `temperature=1`, `top_logprobs=5`) |
| Primary metrics | **LC Win (%)** (`length_controlled_v1` GLM) AND **Raw Win (%)** (`win_rate` continuous logprob preference), reported alongside **Mean Length (Unicode chars)** |
| Answer generation (updated standard) | **Temperature `0.0`**, `seed=0`, `top_p=1.0`, `max_tokens=1024`, IFT training system prompt + chat template (`BOS` prepended once), stop token `<|end_of_text|>` (`128001`) and strings `<|user|>`, `<|system|>` |
| Answer generation (legacy runs) | Same template and stop rules, with `temperature=1.0` |

> **Why both LC Win and Raw Win are mandatory:** On our M2 checkpoint family, AlpacaEval 2.0's GLM satisfies `LC ≈ 14.04 + 1.3245 · Win − 0.01110 · length` (`R² = 0.856`, residual `sd = 0.74`), shifting LC by **+1.11 points per −100 characters** at fixed raw win rate. Any method claim must improve Raw Win at controlled length (or beat the iso-quality curve), rather than trading lower Raw Win for shorter outputs.

---

## 2. Updated Official TSR Training & Pairing Protocol (`cap2500-judge0`)

Because the original paper's early randomly sampled subsets on internal servers were not archived, the authors' updated official GitHub protocol adopts the following 8 rules on the released public splits (`data/sft_data`, `qa1.jsonl` for iter0, `qa2.jsonl` for iter1; `qa3.jsonl` / `qa4.jsonl` available for extended SR iterations):

1. **Initialization:** Train end-to-end from `meta-llama/Llama-3.1-8B` → `SFT (M0)` → `Mf0` → `M1` → `Mf1` → `M2` (`Mf1` initialized from `M1`, BOS prepended in DPO, global batch size 32, bf16 + ZeRO-2/3 with fp32 master weights).
2. **Candidate generation (`T = 1.0`):** Sample `K = 7` responses per prompt with `temperature = 1.0`, `top_p = 1.0`, `max_tokens = 1024`.
3. **Self-judging (`T = 0.0`):** Score each candidate once (`num_judgments = 1`) using the additive 1–5 rubric at **`temperature = 0.0`**.
4. **Invalid score removal:** Drop individual candidate responses whose score fails to parse or falls outside `[0, 5]` *before* computing pool extrema (never map parse failure to score `0`).
5. **Strict score gap (`--filter`):** Require `score(chosen) > score(rejected)` at every DPO stage (`D1` and `D2` in all iterations); drop equal-score prompts.
6. **Source tie-breaking:** Among maximum-score candidates for `chosen`, prefer the **future** model (`Mf_i`) over the current model (`M_i`) on score ties; among minimum-score candidates for `rejected`, prefer the **current** model (`M_i`) over `M0` on score ties.
7. **Distinct text & `chosen ≤ 2500` character cap (following Meta-Rewarding, Wu et al., 2024):** Enumerate candidate pairs satisfying Rules 4–6 in original candidate order and select the first pair where `chosen.strip() != rejected.strip()` and **`len(chosen) <= 2500`** Unicode characters (no truncation, no source/score fallback; drop the prompt if no top-score `chosen` is `≤ 2500` chars).
8. **Deterministic evaluation (`T = 0.0`):** Decode AlpacaEval 2.0 responses at `temperature = 0.0`.

---

## 3. Master Baseline & Decomposition Table

All rows below are scored with `gpt-4o-2024-08-06` against the identical `gpt4_1106_preview` reference (`n = 805`).

### 3.1 Full Pipeline Baselines (`eval T = 1.0` vs `eval T = 0.0` vs Updated `cap2500-judge0`)

| Model / Run | Training Protocol | Eval Temp | LC Win (%) | Raw Win (%) | Win SE | Mean Chars |
|---|---|---:|---:|---:|---:|---:|
| Paper reported `M0` / `M1` / `M2` | Original unarchived subsets | — | 8.73 / 20.48 / 27.94 | 5.96 / 19.07 / 29.44 | — | 1,324 / 1,820 / 2,063 |
| Paper reported `SR iter0` / `SR iter1` | Self-Rewarding | — | 13.29 / 17.00 | 10.99 / 15.71 | — | 1,567 / 1,789 |
| Paper reported `w/o Future iter0` / `iter1` | Past-anchored only | — | 14.35 / 20.96 | 11.61 / 19.69 | — | — |
| **Ours `M0`** (mean of 3 seeds) | Released upstream scripts (no cap) | 1.0 | 7.26 | 5.73 | — | 1,367 |
| **Ours `Mf0`** (`= SR iter0 = w/o Future iter0`) | Released upstream scripts (no cap) | 1.0 | 13.81 | 11.43 | 0.97 | 1,597 |
| **Ours `SR1`** (`SR iter1`) | Released upstream scripts (no cap) | 1.0 | 16.78 | 16.09 | 1.14 | 1,827 |
| **Ours `W1`** (`w/o Future iter1`) | Released upstream scripts (no cap) | 1.0 | 22.45 | 23.37 | 1.32 | 2,035 |
| **Ours `M1`** (`TSR iter0`) | Released upstream scripts (no cap) | 1.0 | 20.93 | 19.32 | 1.23 | 1,810 |
| **Ours `Mf1`** (`iter1 D1`, past-only from `M1`) | Released upstream scripts (no cap) | 1.0 | 22.30 | 22.87 | 1.31 | 2,012 |
| **Ours `M2`** (`TSR iter1`, seed 42) | Released upstream scripts (no cap) | 1.0 | 21.78 | 28.38 | 1.44 | 2,656 |
| **Ours `M2_seed43`** (`TSR iter1`, seed 43) | Released upstream scripts (no cap) | 1.0 | 23.01 | 27.51 | 1.42 | 2,462 |
| **Ours `M2_drop_ties`** (`--filter` on iter1 D2) | Released upstream + drop equal-score | 1.0 | 23.51 | 27.89 | 1.42 | 2,595 |
| **Ours `M1_temp0`** (`TSR iter0`, same weights as `M1`) | Released upstream scripts (no cap) | **0.0** | **23.19** | **19.31** | 1.23 | **1,680** |
| **Ours `M2_temp0`** (`TSR iter1`, same weights as `M2`) | Released upstream scripts (no cap) | **0.0** | **24.74** | **31.44** | 1.48 | **2,581** |
| **Updated TSR `M2_future_tie_charcap_distinct`** (PR #1) | **8-rule `cap2500 + self-judge T=0`** | **0.0** | **29.28** | **27.19** | 1.44 | **1,762** |

### 3.2 Factor Decomposition of `21.78% → 29.28%` LC Win

1. **Decoding temperature (`eval T=1.0 → T=0.0` on identical `M2` weights):**
   - `LC Win`: `21.78% → 24.74%` (**`+2.96`** points)
   - `Raw Win`: `28.38% → 31.44%` (**`+3.06`** points)
   - `Mean length`: `2,656 → 2,581` (`−75` chars)
2. **Training character cap & filtering (`M2_temp0` no-cap → PR #1 `cap2500 + self-judge T=0`, both evaluated at `T=0.0`):**
   - `LC Win`: `24.74% → 29.28%` (**`+4.54`** points)
   - `Raw Win`: `31.44% → 27.19%` (**`−4.25`** points)
   - `Mean length`: `2,581 → 1,762` (**`−819`** chars; training `iter1 D2` chosen mean length compressed from `1,993` to `1,487` chars)
3. **Self-judge saturation under `self-judge T=0.0` (from PR #1 `iter1_future_tie_charcap_distinct/D2/stats.json`):**
   - `group_max_is_5_rate`: **`99.76%`** on `M1` (`31,610 / 35,000` responses scored 5) and **`99.86%`** on `Mf1` (`32,374 / 35,000` scored 5).
   - Retained pairs: `4,160` (`705` equal-score ties dropped; `135` prompts had no top-score candidate `≤ 2,500` chars; `4,046` over-cap combinations excluded).
   - Chosen source: `4,155 / 4,160` (`99.88%`) from `Mf1`, of which **`4,145 / 4,160` (`99.64%`) were selected by `5 == 5` score tie** and only **`10` (`0.24%`)** had a strictly higher future score.

---

## 4. Baseline Policy for Continuation Experiments

1. **Primary baseline convention:** Adopt the updated official GitHub protocol (`chosen ≤ 2500`, strict score gap `--filter`, `self-judge T=0.0`, `eval T=0.0`, public `qa1`–`qa4` splits) across **all** compared methods so length capping is held strictly constant:
   - `SFT (M0)` evaluated at `T = 0.0`
   - `Self-Rewarding (SR)` with identical `chosen ≤ 2500 + --filter + judge T=0 + eval T=0` (including equal-DPO-budget `SR iter3` on `qa1`–`qa4` = 4 DPO stages = `TSR iter1`)
   - `TSR w/o Future` with identical `chosen ≤ 2500 + --filter + judge T=0 + eval T=0`
   - `TSR Official Updated` (`M1` and `M2` under `cap2500-judge0`, `LC = 29.28% / Win = 27.19% / 1,762 chars`)
2. **Paper attribution footnote:** Note once that because the early randomly sampled subsets used in the original paper are no longer archived, all baselines and ablations are executed under the authors' updated official repository protocol (incorporating the `≤ 2,500`-character chosen cap following *Meta-Rewarding*, Wu et al., 2024, and greedy `T=0` evaluation) for strictly matched comparison.
3. **Success criterion for proposed methods:** Under matched `chosen ≤ 2500` and `eval T=0`, improve **both LC Win and Raw Win** over matched `SR`, `w/o Future`, and `TSR Official Updated` without relying on additional length compression.
