#!/usr/bin/env python3
"""Plot scalar histories of the best run; requires requirements-reporting.txt only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "experiments/tsr-best-cap2500-judge0-20261113"
RUN = ROOT / "runs/tsr-llama31-8b-cap2500-judge0-1113a"
STAGES = ["M0", "Mf0", "M1", "Mf1", "M2_future_tie_charcap_distinct"]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), constrained_layout=True)
    for ax, stage in zip(axes.flat, STAGES):
        rows = json.loads((RUN / "training" / stage / "trainer_state.json").read_text())["log_history"]
        points = [r for r in rows if "step" in r and "loss" in r]
        ax.plot([r["step"] for r in points], [r["loss"] for r in points], color="#276aaf", linewidth=1.2)
        ax.set(title="M2" if stage.startswith("M2_") else stage,
               xlabel="Optimizer step", ylabel="SFT loss" if stage == "M0" else "DPO loss")
        ax.grid(alpha=0.2)
    axes.flat[5].axis("off")
    axes.flat[5].text(0.02, 0.95,
        "Final M2: 805/805 questions\n\nLC Win: 29.28%\nWin: 27.19%\nMean characters: 1,762\n\nGenerate T=1; self-judge T=0\nEvaluation answers T=0\nTraining seed: 42",
        va="top", fontsize=11, transform=axes.flat[5].transAxes)
    fig.suptitle("Best TSR run: chosen <= 2,500 characters, self-judge T=0", fontsize=14)
    destination = ROOT / "figures"
    destination.mkdir(exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(destination / f"training-loss.{extension}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
