import json
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
FIG_DIR = os.path.join(ROOT, "analysis", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. user_data.json: posts + comments per user histogram ──

with open(os.path.join(DATA_DIR, "processed", "user_data.json")) as f:
    user_data = json.load(f)

totals = [len(u["posts"]) + len(u["comments"]) for u in user_data]

fig, ax = plt.subplots(figsize=(12, 5))
sorted_totals = sorted(totals, reverse=True)
ax.bar(range(len(sorted_totals)), sorted_totals, color="#4C72B0", width=1.0)
ax.set_yscale("log")
ax.set_xlabel("User (sorted by activity)", fontsize=12)
ax.set_ylabel("Posts + Comments Count (log scale)", fontsize=12)
ax.set_title(f"Posts + Comments per User (n={len(totals)})", fontsize=14)
ax.set_xlim(-0.5, len(sorted_totals) - 0.5)
ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_user_activity.png"), dpi=150)
plt.close()
print(f"[1] Saved figures/fig_user_activity.png  |  {len(totals)} users, "
      f"mean={np.mean(totals):.1f}, median={np.median(totals):.1f}, "
      f"max={max(totals)}, min={min(totals)}")

# ── 2. personas.json: Big Five tier distribution ──

with open(os.path.join(DATA_DIR, "personas", "personas.json")) as f:
    personas = json.load(f)

tier_map = {"Low": -1, "Medium": 0, "High": 1}
counts = {-1: 0, 0: 0, 1: 0}

for p in personas:
    bf = p["big_five"]
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        tier_val = tier_map[bf[trait]["tier"]]
        counts[tier_val] += 1

labels = ["Low", "Medium", "High"]
values = [counts[-1], counts[0], counts[1]]
colors = ["#DD8452", "#55A868", "#4C72B0"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_xlabel("Big Five Tier", fontsize=12)
ax.set_ylabel("Count (across all users × 5 traits)", fontsize=12)
ax.set_title(f"Big Five Personality Tier Distribution (n={len(personas)} users)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_big_five_distribution.png"), dpi=150)
plt.close()
print(f"[2] Saved figures/fig_big_five_distribution.png  |  Low={counts[-1]}, Medium={counts[0]}, High={counts[1]}")

# ── 3. click_like_data.json: click / no-click / like pie chart ──

with open(os.path.join(DATA_DIR, "synthetic", "click_like_data.json")) as f:
    click_data = json.load(f)

total_click = 0
total_no_click = 0
total_like = 0

for user, items in click_data.items():
    for item in items:
        if item["click"] == 1:
            total_click += 1
            if item.get("like", 0) == 1:
                total_like += 1
        else:
            total_no_click += 1

click_no_like = total_click - total_like

pie_labels = ["Clicked & Liked", "Clicked & Not Liked", "Not Clicked"]
pie_values = [total_like, click_no_like, total_no_click]
pie_colors = ["#55A868", "#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    pie_values, labels=pie_labels, colors=pie_colors,
    autopct=lambda p: f'{p:.1f}%\n({int(round(p * sum(pie_values) / 100))})',
    startangle=90, textprops={"fontsize": 12},
    wedgeprops={"edgecolor": "white", "linewidth": 2}
)
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight("bold")
ax.set_title(f"Click & Like Distribution (total={sum(pie_values)} items, {len(click_data)} users)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_click_like_pie.png"), dpi=150)
plt.close()
print(f"[3] Saved figures/fig_click_like_pie.png  |  Clicked&Liked={total_like}, "
      f"Clicked&NotLiked={click_no_like}, NotClicked={total_no_click}")
