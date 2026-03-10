"""
Post-level and user-reply analysis on the Reddit Corpus (small) dataset.

Downloads the corpus via ConvoKit (uses cache if available), applies the same
text filtering as data_preparation.py, then produces:

  1. Comments per post — how many (meaningful) comments each post received
  2. Distinct posts replied to per user — how many unique posts each user commented on

Outputs:
  - Excel file with 2 sheets (comments_per_post / posts_replied_per_user)
  - Bar charts + percentile charts for both metrics
  - Summary statistics printed to console

All outputs go to analysis/figures_post_comment/.
"""

import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from convokit import Corpus, download
from tqdm import tqdm

matplotlib.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures_post_comment")
os.makedirs(FIG_DIR, exist_ok=True)

MEANINGLESS_PATTERNS = re.compile(
    r"^\[deleted]$|^\[removed]$|^\[removed by reddit]$",
    re.IGNORECASE,
)
MIN_TEXT_LENGTH = 5


def is_meaningful_text(text: str) -> bool:
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    if MEANINGLESS_PATTERNS.match(cleaned):
        return False
    if len(cleaned) < MIN_TEXT_LENGTH:
        return False
    return True


def make_bar_chart(values, title, xlabel, ylabel, fname, color):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(values)), values, color=color, width=1.0)
    if len(values) > 0 and max(values) > 0:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, fname)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def main():
    print("Loading Reddit Corpus (small)...")
    corpus = Corpus(filename=download("reddit-corpus-small"))
    corpus.print_summary_stats()

    # conversation_id -> number of meaningful comments
    comments_per_post: dict[str, int] = defaultdict(int)
    # post_id -> post text (for reference)
    post_texts: dict[str, str] = {}
    # user -> set of conversation_ids they commented on
    user_replied_posts: dict[str, set] = defaultdict(set)

    print("Scanning utterances...")
    for utt in tqdm(corpus.iter_utterances(), desc="Processing"):
        if utt.reply_to is None:
            # Root utterance = post
            if is_meaningful_text(utt.text):
                post_texts[utt.id] = utt.text[:120]
        else:
            if not is_meaningful_text(utt.text):
                continue
            conv_id = utt.conversation_id
            comments_per_post[conv_id] += 1
            user_replied_posts[utt.speaker.id].add(conv_id)

    # ── Metric 1: Comments per post ──────────────────────────────────────────

    all_conv_ids = set()
    for conv in corpus.iter_conversations():
        all_conv_ids.add(conv.id)

    post_records = []
    for conv_id in sorted(all_conv_ids):
        n_comments = comments_per_post.get(conv_id, 0)
        preview = post_texts.get(conv_id, "(no meaningful text)")
        post_records.append({
            "post_id": conv_id,
            "num_comments": n_comments,
            "post_preview": preview,
        })

    df_posts = pd.DataFrame(post_records).sort_values(
        "num_comments", ascending=False
    ).reset_index(drop=True)

    total_posts = len(df_posts)
    posts_with_comments = (df_posts["num_comments"] > 0).sum()
    comment_vals = df_posts["num_comments"].tolist()

    print(f"\n{'='*60}")
    print(f"Metric 1: Comments per post")
    print(f"{'='*60}")
    print(f"  Total posts (conversations):  {total_posts}")
    print(f"  Posts with >= 1 comment:      {posts_with_comments}")
    print(f"  Mean comments per post:       {np.mean(comment_vals):.2f}")
    print(f"  Median comments per post:     {np.median(comment_vals):.1f}")
    print(f"  Max comments on a single post:{max(comment_vals)}")
    print(f"  Min comments:                 {min(comment_vals)}")
    print(f"\n  Top 20 posts by comment count:")
    for i, row in df_posts.head(20).iterrows():
        print(f"    {row['num_comments']:>5}  {row['post_id'][:30]:<30}  "
              f"{row['post_preview'][:60]}")

    # ── Metric 2: Distinct posts replied to, per user ────────────────────────

    user_records = []
    for user, conv_set in user_replied_posts.items():
        user_records.append({
            "user_name": user,
            "num_posts_replied": len(conv_set),
        })

    df_users = pd.DataFrame(user_records).sort_values(
        "num_posts_replied", ascending=False
    ).reset_index(drop=True)

    user_vals = df_users["num_posts_replied"].tolist()

    print(f"\n{'='*60}")
    print(f"Metric 2: Distinct posts replied to, per user")
    print(f"{'='*60}")
    print(f"  Total users who commented:    {len(df_users)}")
    print(f"  Mean posts replied per user:  {np.mean(user_vals):.2f}")
    print(f"  Median posts replied per user:{np.median(user_vals):.1f}")
    print(f"  Max posts replied:            {max(user_vals)}")
    print(f"  Min posts replied:            {min(user_vals)}")
    print(f"\n  Top 20 users by posts replied:")
    for i, row in df_users.head(20).iterrows():
        print(f"    {row['num_posts_replied']:>5}  {row['user_name']}")

    # ── Excel ────────────────────────────────────────────────────────────────

    excel_path = os.path.join(FIG_DIR, "post_comment_stats.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_posts[["post_id", "num_comments", "post_preview"]].to_excel(
            writer, sheet_name="Comments per Post", index=False
        )
        df_users.to_excel(
            writer, sheet_name="Posts Replied per User", index=False
        )
    print(f"\n[Excel] Saved {excel_path}")

    # ── Bar charts ───────────────────────────────────────────────────────────

    sorted_comments = sorted(comment_vals, reverse=True)
    p = make_bar_chart(
        sorted_comments,
        f"Comments per Post — Reddit Corpus Small (n={total_posts})",
        "Post (sorted by comment count)", "Comment count (log scale)",
        "fig_comments_per_post.png", "#4C72B0",
    )
    print(f"[Chart] Saved {p}")

    sorted_user_vals = sorted(user_vals, reverse=True)
    p = make_bar_chart(
        sorted_user_vals,
        f"Distinct Posts Replied per User — Reddit Corpus Small (n={len(df_users)})",
        "User (sorted by count)", "Posts replied (log scale)",
        "fig_posts_replied_per_user.png", "#55A868",
    )
    print(f"[Chart] Saved {p}")

    # ── Percentiles ──────────────────────────────────────────────────────────

    percentiles = list(range(0, 101, 5))
    pct_data = {
        "percentile": [f"p{p}" for p in percentiles],
        "comments_per_post": [np.percentile(comment_vals, p) for p in percentiles],
        "posts_replied_per_user": [np.percentile(user_vals, p) for p in percentiles],
    }
    df_pct = pd.DataFrame(pct_data)

    pct_path = os.path.join(FIG_DIR, "percentiles.xlsx")
    df_pct.to_excel(pct_path, index=False, sheet_name="Percentiles")

    print(f"\n[Percentile Table]")
    print(df_pct.to_string(index=False))
    print(f"\n    Saved {pct_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(percentiles, df_pct["comments_per_post"], marker="o", color="#4C72B0")
    axes[0].set_xlabel("Percentile", fontsize=12)
    axes[0].set_ylabel("Comments per Post", fontsize=12)
    axes[0].set_title("Comments per Post — Percentiles", fontsize=13)
    axes[0].set_xticks(percentiles)
    axes[0].tick_params(axis="x", rotation=45, labelsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(percentiles, df_pct["posts_replied_per_user"], marker="s", color="#55A868")
    axes[1].set_xlabel("Percentile", fontsize=12)
    axes[1].set_ylabel("Posts Replied", fontsize=12)
    axes[1].set_title("Posts Replied per User — Percentiles", fontsize=13)
    axes[1].set_xticks(percentiles)
    axes[1].tick_params(axis="x", rotation=45, labelsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    pct_fig = os.path.join(FIG_DIR, "fig_percentiles.png")
    plt.savefig(pct_fig, dpi=150)
    plt.close()
    print(f"[Chart] Saved {pct_fig}")

    # ── Histogram (for a different view) ─────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].hist(comment_vals, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Number of Comments", fontsize=12)
    axes[0].set_ylabel("Number of Posts", fontsize=12)
    axes[0].set_title("Distribution of Comments per Post", fontsize=13)
    axes[0].axvline(np.mean(comment_vals), color="red", linestyle="--",
                    label=f"Mean = {np.mean(comment_vals):.1f}")
    axes[0].axvline(np.median(comment_vals), color="orange", linestyle="--",
                    label=f"Median = {np.median(comment_vals):.1f}")
    axes[0].legend(fontsize=10)

    axes[1].hist(user_vals, bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Number of Distinct Posts Replied", fontsize=12)
    axes[1].set_ylabel("Number of Users", fontsize=12)
    axes[1].set_title("Distribution of Posts Replied per User", fontsize=13)
    axes[1].axvline(np.mean(user_vals), color="red", linestyle="--",
                    label=f"Mean = {np.mean(user_vals):.1f}")
    axes[1].axvline(np.median(user_vals), color="orange", linestyle="--",
                    label=f"Median = {np.median(user_vals):.1f}")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    hist_fig = os.path.join(FIG_DIR, "fig_histograms.png")
    plt.savefig(hist_fig, dpi=150)
    plt.close()
    print(f"[Chart] Saved {hist_fig}")

    print(f"\nDone! All outputs in {FIG_DIR}")


if __name__ == "__main__":
    main()
