# Native official-source best-profile run

Active run: **`tsr-official-best-1113b`**, eight H200 in SN5, fresh Base → SFT M0 → Mf0 → M1 → Mf1 → M2. Frozen native source: `28144212105c648113ca7e158f3d711961a5fc3f`. See `submission.json` and `launch-status.json` for timestamps and placement.

**This native run is still in progress. Fresh SFT completed 180 optimizer updates and the pipeline advanced to iter0 generation/self-judging. No new final LC/Win result is claimed.** See `progress-snapshot.json` and `training-metrics/M0.json` for the timestamped live SFT check. The earlier reproduction's full 805-question LC Win 29.28% / Win 27.19% remains separately archived under `tsr-best-cap2500-judge0-20261113`.

Attempt `1113a` reached SFT optimization, then failed on TensorBoard append to FSx (`OSError: Invalid argument`). Both jobs exited. The fix writes events locally and synchronizes closed files after each training stage. Run `1113b` restarts from Base; no failed-run weights or training candidates are reused. See `attempt-1113a-failure.json`.

Both Mf1 and M2 initialize from M1. All four DPO datasets require strict positive score gaps, future/current source tie priorities, distinct text, a 2500-character chosen cap, and invalid-score removal. Candidate generation T=1, self-judge T=0, evaluation answer generation T=0. [Full configuration and fixes](../../docs/BEST_PROFILE.md).

The GPU job trains and generates all 805 final evaluation answers. A CPU worker performs the authorized GPT-4o scoring after answers are ready. Live artifacts: `/mnt/sn-007/zaninzeng/TSR-official-best/work/runs/tsr-official-best-1113b`.

Included: actual run plan, source hashes, base-weight hashes (weights excluded), submission record, deployment manifest and available actual SFT configuration/initialization proof. No credentials, raw training-generation caches, optimizer states or model weights are included.

The completed native SFT state, final scalar metrics and results are under `training/M0/` and `training-metrics/M0.json`. Model weights remain on training storage.

The first native preference set retained 4,163/5,000 pairs; retained score ties, identical texts and over-cap chosen texts are all zero. `inference-sampling-validation.json` confirms 35,000 candidate generations at T=1 and 35,000 self-judgments at T=0. Mf0 DPO produced finite training metrics; progress is timestamped in `progress-snapshot.json`.
