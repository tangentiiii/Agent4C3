"""
Build evaluation dataset: split conversations into train/test,
filter comment-only users, and produce eval_users.json + eval_split.json.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

import yaml


def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    seed = config.get("seed", 42)
    random.seed(seed)

    eval_cfg = config.get("evaluation", {})
    test_ratio = eval_cfg.get("test_ratio", 0.2)
    min_comments_after_split = eval_cfg.get("min_comments_after_split", 5)
    max_eval_users = eval_cfg.get("max_eval_users", 50)
    n = eval_cfg.get("n", 3)

    base_dir = Path(__file__).parent.parent.parent / "data"
    eval_dir = base_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    with open(base_dir / "processed" / "user_data.json", "r") as f:
        all_users = json.load(f)

    with open(base_dir / "processed" / "all_posts_meta.json", "r") as f:
        all_posts_meta = json.load(f)

    # --- 2a: Filter comment-only users ---
    comment_only_users = [u for u in all_users if len(u["posts"]) == 0]
    print(f"Total users: {len(all_users)}")
    print(f"Comment-only users: {len(comment_only_users)}")

    # --- 2b: Split conversations into train/test ---
    all_conv_ids = list({p["conversation_id"] for p in all_posts_meta})
    random.shuffle(all_conv_ids)
    split_idx = int(len(all_conv_ids) * (1 - test_ratio))
    train_conv_ids = set(all_conv_ids[:split_idx])
    test_conv_ids = set(all_conv_ids[split_idx:])

    eval_split = {
        "train_conversation_ids": sorted(train_conv_ids),
        "test_conversation_ids": sorted(test_conv_ids),
    }
    with open(eval_dir / "eval_split.json", "w", encoding="utf-8") as f:
        json.dump(eval_split, f, ensure_ascii=False, indent=2)
    print(f"Conversation split: {len(train_conv_ids)} train / {len(test_conv_ids)} test")

    # --- 2c: Build per-user evaluation records ---
    eligible_users = []
    for user in comment_only_users:
        user_conv_ids = {c["conversation_id"] for c in user["comments"]}
        test_positive_conv_ids = user_conv_ids & test_conv_ids

        if len(test_positive_conv_ids) < n:
            continue

        train_comments = [
            c for c in user["comments"]
            if c["conversation_id"] not in test_conv_ids
        ]

        if len(train_comments) < min_comments_after_split:
            continue

        num_removed = len(user["comments"]) - len(train_comments)

        eligible_users.append({
            "user_name": user["user_name"],
            "posts": [],
            "comments": [
                {"comment_text": c["comment_text"], "parent_text": c["parent_text"]}
                for c in train_comments
            ],
            "test_positive_conversation_ids": sorted(test_positive_conv_ids),
            "num_removed_comments": num_removed,
            "num_remaining_comments": len(train_comments),
        })

    print(f"Eligible users (>= {min_comments_after_split} remaining comments "
          f"AND >= {n} test positives): {len(eligible_users)}")

    # --- 2d: Randomly sample eval users ---
    if len(eligible_users) > max_eval_users:
        sampled_users = random.sample(eligible_users, max_eval_users)
        print(f"Sampled {max_eval_users} eval users from {len(eligible_users)} eligible")
    else:
        sampled_users = eligible_users
        print(f"Using all {len(sampled_users)} eligible users (pool <= max_eval_users)")

    with open(eval_dir / "eval_users.json", "w", encoding="utf-8") as f:
        json.dump(sampled_users, f, ensure_ascii=False, indent=2)

    # --- 2e: Print summary statistics ---
    total_test_positives = sum(len(u["test_positive_conversation_ids"]) for u in sampled_users)
    avg_removed = (
        sum(u["num_removed_comments"] for u in sampled_users) / len(sampled_users)
        if sampled_users else 0
    )
    pct_ge5 = (
        sum(1 for u in sampled_users if u["num_remaining_comments"] >= 5)
        / len(sampled_users) * 100
        if sampled_users else 0
    )

    print("\n--- Evaluation Dataset Summary ---")
    print(f"  Total comment-only users:           {len(comment_only_users)}")
    print(f"  Eligible users:                     {len(eligible_users)}")
    print(f"  Sampled eval users:                 {len(sampled_users)}")
    print(f"  Total test-positive pairs:          {total_test_positives}")
    print(f"  Avg comments removed per user:      {avg_removed:.1f}")
    print(f"  % users retaining >= 5 comments:    {pct_ge5:.1f}%")
    print(f"  Saved to: {eval_dir / 'eval_users.json'}")


if __name__ == "__main__":
    main()
