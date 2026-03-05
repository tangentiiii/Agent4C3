"""
Analyse simulation results: creator rewards across rounds.

Outputs
-------
- analysis/figures/creator_rewards.xlsx   – one row per round, one column per creator
- analysis/figures/fig_creator_rewards.png – line chart of rewards over rounds

Usage
-----
    # auto-detect latest run
    python -m analysis.sim_reward_analysis

    # specify a run directory
    python -m analysis.sim_reward_analysis results/run_20260305_143000
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _latest_run_dir() -> Path:
    run_dirs = sorted(RESULTS_DIR.glob("run_*"))
    if not run_dirs:
        print("No simulation runs found in results/. Run `python main.py simulate` first.")
        sys.exit(1)
    return run_dirs[-1]


def _load_rewards(run_dir: Path) -> pd.DataFrame:
    """Return a DataFrame: index=round, columns=creator labels, values=reward."""
    round_files = sorted(run_dir.glob("round_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    rows: list[dict] = []
    for rf in round_files:
        with open(rf, encoding="utf-8") as f:
            data = json.load(f)
        round_num = data["round"]
        round_rewards: dict[str, float | None] = {}
        for post in data["posts"]:
            label = f"Creator {post['creator_id']}"
            round_rewards[label] = post.get("reward")
        round_rewards["Round"] = round_num
        rows.append(round_rewards)

    df = pd.DataFrame(rows).set_index("Round").sort_index()
    creator_cols = sorted(df.columns, key=lambda c: int(c.split()[-1]))
    return df[creator_cols]


def analyse(run_dir: Path | None = None):
    if run_dir is None:
        run_dir = _latest_run_dir()
    print(f"Analysing run: {run_dir.name}")

    df = _load_rewards(run_dir)
    print(df.to_string())

    # ── Excel ──
    excel_path = FIG_DIR / "creator_rewards.xlsx"
    df.to_excel(excel_path, sheet_name="Rewards")
    print(f"\n[Excel] Saved {excel_path}")

    # ── Line chart (subplot grid: one panel per creator) ──
    n_creators = len(df.columns)
    ncols = min(n_creators, 4)
    nrows = (n_creators + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    colors = plt.cm.tab10.colors
    y_min = df.min().min()
    y_max = df.max().max()
    y_pad = max((y_max - y_min) * 0.1, 0.5)

    for idx, col in enumerate(df.columns):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]
        ax.plot(df.index, df[col], marker="o", linewidth=1.8, markersize=4,
                color=colors[idx % len(colors)])
        ax.set_title(col, fontsize=12, fontweight="bold")
        ax.set_xlabel("Round", fontsize=10)
        ax.set_ylabel("Reward", fontsize=10)
        ax.set_xticks(df.index)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, alpha=0.3)

    for idx in range(n_creators, nrows * ncols):
        row_i, col_i = divmod(idx, ncols)
        axes[row_i][col_i].set_visible(False)

    fig.suptitle("Content Creator Rewards per Round", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    fig_path = FIG_DIR / "fig_creator_rewards.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved {fig_path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if target and not target.is_absolute():
        target = ROOT / target
    analyse(target)
