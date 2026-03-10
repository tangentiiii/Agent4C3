"""
User activity analysis on the filtered Reddit Corpus (small) dataset.

Reads data/processed/user_data.json (output of data_preparation.py) and produces:
  - Excel file with 3 sheets: posts / comments / posts+comments per user
  - Bar charts for each category
  - Percentile tables (p0–p100, step 5)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.family"] = "DejaVu Sans"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Load filtered user data ──────────────────────────────────────────────────

with open(os.path.join(DATA_DIR, "processed", "user_data.json"), encoding="utf-8") as f:
    user_data = json.load(f)

records = []
for u in user_data:
    n_posts = len(u["posts"])
    n_comments = len(u["comments"])
    records.append({
        "user_name": u["user_name"],
        "num_posts": n_posts,
        "num_comments": n_comments,
        "num_posts_and_comments": n_posts + n_comments,
    })

df = pd.DataFrame(records)

# ── 1. Write to Excel (3 sheets, sorted descending) ─────────────────────────

excel_path = os.path.join(FIG_DIR, "user_activity.xlsx")

df_posts = (
    df[["user_name", "num_posts"]]
    .sort_values("num_posts", ascending=False)
    .reset_index(drop=True)
)
df_comments = (
    df[["user_name", "num_comments"]]
    .sort_values("num_comments", ascending=False)
    .reset_index(drop=True)
)
df_total = (
    df[["user_name", "num_posts_and_comments"]]
    .sort_values("num_posts_and_comments", ascending=False)
    .reset_index(drop=True)
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_posts.to_excel(writer, sheet_name="Posts", index=False)
    df_comments.to_excel(writer, sheet_name="Comments", index=False)
    df_total.to_excel(writer, sheet_name="Posts and Comments", index=False)

print(f"[1] Saved {excel_path}  |  {len(df)} users")

# ── 2. Bar charts ────────────────────────────────────────────────────────────

categories = [
    ("num_posts", "Posts per User", "fig_posts_per_user.png", "#4C72B0"),
    ("num_comments", "Comments per User", "fig_comments_per_user.png", "#55A868"),
    ("num_posts_and_comments", "Posts + Comments per User", "fig_total_per_user.png", "#DD8452"),
]

for col, title, fname, color in categories:
    values = sorted(df[col].tolist(), reverse=True)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(values)), values, color=color, width=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("User (sorted by count)", fontsize=12)
    ax.set_ylabel("Count (log scale)", fontsize=12)
    ax.set_title(f"{title} (n={len(values)})", fontsize=14)
    ax.set_xlim(-0.5, len(values) - 0.5)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    plt.tight_layout()
    fig_path = os.path.join(FIG_DIR, fname)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[2] Saved {fname}  |  mean={np.mean(values):.1f}, "
          f"median={np.median(values):.1f}, max={max(values)}, min={min(values)}")

# ── 3. Percentiles (p0–p100, step 5) ────────────────────────────────────────

percentiles = list(range(0, 101, 5))
percentile_data = {"percentile": [f"p{p}" for p in percentiles]}

for col, label in [
    ("num_posts", "posts"),
    ("num_comments", "comments"),
    ("num_posts_and_comments", "posts_and_comments"),
]:
    percentile_data[label] = [np.percentile(df[col], p) for p in percentiles]

df_pct = pd.DataFrame(percentile_data)

pct_excel_path = os.path.join(FIG_DIR, "user_activity_percentiles.xlsx")
df_pct.to_excel(pct_excel_path, index=False, sheet_name="Percentiles")

print(f"\n[3] Percentile table (p0–p100, step 5):")
print(df_pct.to_string(index=False))
print(f"\n    Saved {pct_excel_path}")

# percentile line chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(percentiles, df_pct["posts"], marker="o", label="Posts", color="#4C72B0")
ax.plot(percentiles, df_pct["comments"], marker="s", label="Comments", color="#55A868")
ax.plot(percentiles, df_pct["posts_and_comments"], marker="^", label="Posts + Comments", color="#DD8452")
ax.set_xlabel("Percentile", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("User Activity Percentiles", fontsize=14)
ax.set_xticks(percentiles)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
pct_fig_path = os.path.join(FIG_DIR, "fig_percentiles.png")
plt.savefig(pct_fig_path, dpi=150)
plt.close()
print(f"    Saved {pct_fig_path}")
