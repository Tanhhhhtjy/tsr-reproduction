# Console logs for the LC 29.28% run

Run: `tsr-llama31-8b-cap2500-judge0-1113a`. These are the original saved console logs from the completed run already archived here, not a new training or evaluation run.

- `pipeline/`: 65 logs covering five training stages, sharded candidate generation and self-judging, pair construction, final answer generation, and the pipeline driver.
- `bootstrap/`: the job startup log.
- `scoring/`: the final M2 AlpacaEval scoring log.
- `publication-manifest.json`: original and published hashes, byte counts, and redaction counts for all 67 log files.

Credentials and private judge-provider URLs are redacted where present. All other text, carriage-return progress bars, and terminal escape sequences are preserved. Raw console logs may contain sampled training examples and historical runtime paths; full candidate datasets are not included. Model weights, checkpoints, optimizer state, API request caches, and TensorBoard event files are not included.
