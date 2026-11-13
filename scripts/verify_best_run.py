#!/usr/bin/env python3
"""Verify the committed best-run records offline, using only the standard library."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "experiments/tsr-best-cap2500-judge0-20261113"
RUN = "tsr-llama31-8b-cap2500-judge0-1113a"
MODEL = "M2_future_tie_charcap_distinct"


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    check(not any(p.is_symlink() for p in root.rglob("*")), "Unexpected archive symlink")
    check(actual == set(manifest["files"]) | {"manifest.json"}, "Archive file coverage differs")
    for name, item in manifest["files"].items():
        path = root / name
        check(path.resolve().is_relative_to(root), "Unsafe archive path")
        content = path.read_bytes()
        check(len(content) == item["bytes"], f"Size differs: {name}")
        check(hashlib.sha256(content).hexdigest() == item["sha256"], f"Hash differs: {name}")
    run = root / "runs" / RUN
    directory = run / "evaluation" / MODEL
    marker = json.loads((directory / "GPT4O_SCORE_COMPLETE.json").read_text())
    raw_outputs = gzip.decompress((directory / "model_outputs.json.gz").read_bytes())
    outputs = json.loads(raw_outputs)
    reference_raw = (root / "inputs/alpaca_reference.json").read_bytes()
    reference = json.loads(reference_raw)
    annotations = json.loads(gzip.decompress((directory / "annotations.json.gz").read_bytes()))
    check(len(outputs) == len(reference) == len(annotations) == marker["n_total"] == 805, "Incomplete evaluation")
    check(hashlib.sha256(raw_outputs).hexdigest() == marker["input_sha256"], "Scored input hash differs")
    check(hashlib.sha256(reference_raw).hexdigest() == marker["reference_sha256"], "Reference hash differs")
    check([r["instruction"] for r in outputs] == [r["instruction"] for r in reference], "Question order differs")
    output_map = {row["instruction"]: row["output"] for row in outputs}
    reference_map = {row["instruction"]: row["output"] for row in reference}
    check(len(output_map) == len(reference_map) == 805, "Duplicate question")
    check({r["instruction"] for r in annotations} == set(output_map), "Annotation coverage differs")
    for row in annotations:
        check(row["output_2"] == output_map[row["instruction"]], "Annotation candidate text differs")
        check(row["output_1"] == reference_map[row["instruction"]], "Annotation reference text differs")
        check(math.isfinite(row["preference"]) and 1 <= row["preference"] <= 2, "Invalid preference")
    with (directory / "leaderboard.csv").open() as stream:
        board = list(csv.DictReader(stream))
    check(len(board) == 1, "Unexpected leaderboard rows")
    for key in ["win_rate", "length_controlled_winrate"]:
        check(math.isclose(float(board[0][key]), marker[key], abs_tol=1e-9), f"Metric differs: {key}")
    diag = json.loads((directory / "model_outputs.diag.json").read_text())
    check(diag["temperature"] == 0 and diag["seed"] == 0, "Evaluation sampling differs")
    check(math.isclose(sum(len(r["output"]) for r in outputs) / 805, diag["mean_chars"]), "Length differs")
    expected = {"iter0/D1":4256, "iter0/D2":4598, "iter1/D1":4137,
                "iter1_future_tie_charcap_distinct/D2":4160}
    for dataset, count in expected.items():
        item = json.loads((run / "pair_statistics" / dataset / "stats.json").read_text())
        stats = item["stats"]
        check(stats["n_pairs"] == count and stats["gap_min"] > 0, f"Pair count/gap differs: {dataset}")
        check(stats["retained_tie_pairs"] == stats["identical_pair"] == stats["chosen_over_character_limit_pairs"] == 0,
              f"Invalid retained pair: {dataset}")
        check(stats["chosen_length_max"] <= stats["max_chosen_characters"] == 2500, "Character cap differs")
        check(stats["drop_invalid_scores"] and stats["model_source_priority"], "Pair rules differ")
    temps = json.loads((run / "training_metrics/self-judge-temperature-validation.json").read_text())
    check(temps["self_judge_temperature"] == 0 and temps["response_temperature"] == 1, "Training sampling differs")
    return {"verified_files":len(manifest["files"]), "questions":805,
            "lc_win":marker["length_controlled_winrate"], "win":marker["win_rate"],
            "pair_datasets":len(expected), "external_api_calls":0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT)
    print(json.dumps(verify(parser.parse_args().archive), indent=2))
