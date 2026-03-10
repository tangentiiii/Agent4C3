"""
User activity analysis on the full Reddit Corpus (by subreddit).

Downloads the corpus via ConvoKit, applies the same text filtering as
data_preparation.py, then produces:
  - Excel: 3 sheets (top-level comments / replies / total per user, sorted desc)
  - Bar charts for each category
  - Percentile tables + chart (p0–p100, step 5)

NOTE: In this corpus, submissions (posts) are NOT stored as utterances.
All utterances are comments. We categorise them as:
  - "top-level comment" (reply_to == conversation_id) — direct reply to a submission
  - "reply" (reply_to != conversation_id) — reply to another comment

All outputs go to analysis_reddit_corpus/figures/.
"""

import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from convokit import Corpus, download
from tqdm import tqdm

matplotlib.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_CACHE = os.path.join(SCRIPT_DIR, "user_data.json")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Filtering (same logic as data_preparation.py) ────────────────────────────

MEANINGLESS_PATTERNS = re.compile(
    r"^\[deleted]$|^\[removed]$|^\[removed by reddit]$",
    re.IGNORECASE,
)
MIN_TEXT_LENGTH = 5
MIN_INTERACTIONS = 10


def is_meaningful_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    if MEANINGLESS_PATTERNS.match(cleaned):
        return False
    if len(cleaned) < MIN_TEXT_LENGTH:
        return False
    return True


# ── Step 0: Download corpus & extract filtered user data ─────────────────────

def extract_and_cache() -> list[dict]:
    """Download, filter, and cache user data. Reuses cache if it exists."""
    if os.path.exists(DATA_CACHE):
        print(f"Found cached user data at {DATA_CACHE}, loading...")
        with open(DATA_CACHE, encoding="utf-8") as f:
            return json.load(f)

    print("Downloading Reddit Corpus (full)...")
    corpus = Corpus(filename=download("reddit-corpus"))
    corpus.print_summary_stats()

    user_top_level: dict[str, int] = defaultdict(int)
    user_replies: dict[str, int] = defaultdict(int)

    print("Extracting user data from utterances...")
    for utt in tqdm(corpus.iter_utterances(), desc="Processing utterances"):
        if not is_meaningful_text(utt.text):
            continue
        speaker = utt.speaker.id
        if utt.reply_to == utt.conversation_id:
            user_top_level[speaker] += 1
        else:
            user_replies[speaker] += 1

    all_speakers = set(user_top_level.keys()) | set(user_replies.keys())
    users = []
    for name in tqdm(sorted(all_speakers), desc="Building user records"):
        n_top = user_top_level.get(name, 0)
        n_rep = user_replies.get(name, 0)
        if n_top + n_rep < MIN_INTERACTIONS:
            continue
        users.append({
            "user_name": name,
            "num_top_level_comments": n_top,
            "num_replies": n_rep,
        })

    print(f"Extracted {len(users)} users with >= {MIN_INTERACTIONS} interactions")

    with open(DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    print(f"Cached user data to {DATA_CACHE}")
    return users


# ── Main analysis ────────────────────────────────────────────────────────────

def main():
    users = extract_and_cache()

    records = []
    for u in users:
        nt = u["num_top_level_comments"]
        nr = u["num_replies"]
        records.append({
            "user_name": u["user_name"],
            "num_top_level_comments": nt,
            "num_replies": nr,
            "num_total": nt + nr,
        })

    df = pd.DataFrame(records)
    print(f"\nTotal users after filtering: {len(df)}")

    # ── 1. Excel (3 sheets, sorted descending) ───────────────────────────────

    excel_path = os.path.join(FIG_DIR, "user_activity.xlsx")

    df_top = (
        df[["user_name", "num_top_level_comments"]]
        .sort_values("num_top_level_comments", ascending=False)
        .reset_index(drop=True)
    )
    df_rep = (
        df[["user_name", "num_replies"]]
        .sort_values("num_replies", ascending=False)
        .reset_index(drop=True)
    )
    df_total = (
        df[["user_name", "num_total"]]
        .sort_values("num_total", ascending=False)
        .reset_index(drop=True)
    )

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_top.to_excel(writer, sheet_name="Top-level Comments", index=False)
        df_rep.to_excel(writer, sheet_name="Replies", index=False)
        df_total.to_excel(writer, sheet_name="Total", index=False)

    print(f"[1] Saved {excel_path}  |  {len(df)} users")

    # ── 2. Bar charts ────────────────────────────────────────────────────────

    categories = [
        ("num_top_level_comments", "Top-level Comments per User", "fig_top_level_per_user.png", "#4C72B0"),
        ("num_replies", "Replies per User", "fig_replies_per_user.png", "#55A868"),
        ("num_total", "Total Comments per User", "fig_total_per_user.png", "#DD8452"),
    ]

    for col, title, fname, color in categories:
        values = sorted(df[col].tolist(), reverse=True)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(values)), values, color=color, width=1.0)
        if max(values) > 0:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
        ax.set_xlabel("User (sorted by count)", fontsize=12)
        ax.set_ylabel("Count (log scale)", fontsize=12)
        ax.set_title(f"{title} — Reddit Corpus Full (n={len(values)})", fontsize=14)
        ax.set_xlim(-0.5, len(values) - 0.5)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        plt.tight_layout()
        fig_path = os.path.join(FIG_DIR, fname)
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[2] Saved {fname}  |  mean={np.mean(values):.1f}, "
              f"median={np.median(values):.1f}, max={max(values)}, min={min(values)}")

    # ── 3. Percentiles (p0–p100, step 5) ─────────────────────────────────────

    percentiles = list(range(0, 101, 5))
    percentile_data = {"percentile": [f"p{p}" for p in percentiles]}

    for col, label in [
        ("num_top_level_comments", "top_level_comments"),
        ("num_replies", "replies"),
        ("num_total", "total"),
    ]:
        percentile_data[label] = [np.percentile(df[col], p) for p in percentiles]

    df_pct = pd.DataFrame(percentile_data)

    pct_excel_path = os.path.join(FIG_DIR, "user_activity_percentiles.xlsx")
    df_pct.to_excel(pct_excel_path, index=False, sheet_name="Percentiles")

    print(f"\n[3] Percentile table (p0–p100, step 5):")
    print(df_pct.to_string(index=False))
    print(f"\n    Saved {pct_excel_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(percentiles, df_pct["top_level_comments"], marker="o",
            label="Top-level Comments", color="#4C72B0")
    ax.plot(percentiles, df_pct["replies"], marker="s",
            label="Replies", color="#55A868")
    ax.plot(percentiles, df_pct["total"], marker="^",
            label="Total", color="#DD8452")
    ax.set_xlabel("Percentile", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("User Activity Percentiles — Reddit Corpus Full", fontsize=14)
    ax.set_xticks(percentiles)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pct_fig_path = os.path.join(FIG_DIR, "fig_percentiles.png")
    plt.savefig(pct_fig_path, dpi=150)
    plt.close()
    print(f"    Saved {pct_fig_path}")


if __name__ == "__main__":
    main()
