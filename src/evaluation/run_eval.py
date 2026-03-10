"""
Run K-choose-N post identification evaluation.

For each eval user, sample N positive posts (conversations the user actually
participated in) and K-N random negative posts, then ask the LLM to identify
all N posts the user would engage with.
"""
import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

from src.llm_client import load_prompt, call_llm_json


def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _build_posts_index(all_posts_meta: list[dict]) -> dict:
    """Build conv_id -> post index."""
    return {p["conversation_id"]: p for p in all_posts_meta}


def sample_negatives(
    positive_conv_ids: set[str],
    user_participated_convs: set[str],
    posts_by_conv: dict,
    num_negatives: int,
) -> list[dict]:
    """Sample random negative posts, excluding user's participated conversations."""
    excluded = user_participated_convs | positive_conv_ids
    pool = list(set(posts_by_conv.keys()) - excluded)
    sampled_ids = random.sample(pool, min(num_negatives, len(pool)))
    return [posts_by_conv[cid] for cid in sampled_ids]


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"Post {i}:\n{c['text']}\n")
    return "\n".join(lines)


def _format_persona_for_prompt(persona: dict) -> str:
    return persona.get("raw", json.dumps(persona, ensure_ascii=False))


def _format_comment_history(comments: list[dict], max_items: int = 15) -> str:
    """Format a user's training comments as browsing/engagement history."""
    if not comments:
        return "(no comment history)"
    lines = []
    for c in comments[-max_items:]:
        parent = c.get("parent_text", "")
        snippet = parent[:200].replace("\n", " ").strip()
        if len(parent) > 200:
            snippet += "…"
        lines.append(f'- Commented on: "{snippet}"')
    return "\n".join(lines)


def _run_single_query(
    user_name: str,
    persona: dict,
    positive_posts: list[dict],
    negatives: list[dict],
    comment_history: list[dict],
    system_prompt: str,
    user_template: str,
    model: str,
    temperature: float,
    k: int,
    n: int,
) -> dict:
    """Run a single evaluation query and return the result record."""
    candidates = positive_posts + negatives
    random.shuffle(candidates)

    gold_conv_ids = {p["conversation_id"] for p in positive_posts}
    gold_indices = [
        i for i, c in enumerate(candidates)
        if c["conversation_id"] in gold_conv_ids
    ]

    persona_str = _format_persona_for_prompt(persona)
    candidates_str = _format_candidates(candidates)
    history_str = _format_comment_history(comment_history)
    user_prompt = (
        user_template
        .replace("{persona}", persona_str)
        .replace("{history}", history_str)
        .replace("{k}", str(k))
        .replace("{n}", str(n))
        .replace("{candidates}", candidates_str)
    )

    try:
        response = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )
        choices = response.get("choices", [])
        chosen_indices = [c - 1 for c in choices if isinstance(c, int)]
        reason = response.get("reason", "")
    except Exception as e:
        print(f"  LLM error for {user_name}: {e}")
        chosen_indices = []
        reason = f"ERROR: {e}"

    gold_set = set(gold_indices)
    chosen_set = set(chosen_indices)
    tp = len(gold_set & chosen_set)

    precision = tp / len(chosen_set) if chosen_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    exact_match = gold_set == chosen_set

    return {
        "user_name": user_name,
        "positive_conversation_ids": sorted(gold_conv_ids),
        "k": k,
        "n": n,
        "gold_indices": sorted(gold_indices),
        "chosen_indices": sorted(chosen_indices),
        "precision": precision,
        "recall": recall,
        "exact_match": exact_match,
        "reason": reason,
    }


def main():
    config = load_config()
    seed = config.get("seed", 42)
    random.seed(seed)

    eval_cfg = config.get("evaluation", {})
    k = eval_cfg.get("k", 10)
    n = eval_cfg.get("n", 3)

    model = config["model"]["name"]
    temperature = config["model"]["temperature"]
    max_workers = config.get("concurrency", {}).get("max_workers", 10)

    base_dir = Path(__file__).parent.parent.parent / "data"
    eval_dir = base_dir / "evaluation"

    with open(eval_dir / "eval_users.json", "r") as f:
        eval_users = json.load(f)

    with open(eval_dir / "eval_personas.json", "r") as f:
        eval_personas_list = json.load(f)
    personas_by_name = {p["user_name"]: p for p in eval_personas_list}

    with open(base_dir / "processed" / "all_posts_meta.json", "r") as f:
        all_posts_meta = json.load(f)

    with open(base_dir / "processed" / "user_data.json", "r") as f:
        all_users_data = json.load(f)

    posts_by_conv = _build_posts_index(all_posts_meta)

    user_full_convs = {}
    for u in all_users_data:
        convs = set()
        for c in u.get("comments", []):
            if "conversation_id" in c:
                convs.add(c["conversation_id"])
        user_full_convs[u["user_name"]] = convs

    prompt_data = load_prompt("eval_identify_post")
    system_prompt = prompt_data["eval_identify_post"]["system"]
    user_template = prompt_data["eval_identify_post"]["user"]

    tasks = []
    skipped = 0
    for user in eval_users:
        persona = personas_by_name.get(user["user_name"])
        if not persona:
            print(f"  WARNING: No persona for {user['user_name']}, skipping")
            continue

        test_pos_ids = user["test_positive_conversation_ids"]
        valid_pos_ids = [cid for cid in test_pos_ids if cid in posts_by_conv]
        if len(valid_pos_ids) < n:
            skipped += 1
            continue

        sampled_pos_ids = random.sample(valid_pos_ids, n)
        positive_posts = [posts_by_conv[cid] for cid in sampled_pos_ids]

        participated = user_full_convs.get(user["user_name"], set())
        negatives = sample_negatives(
            set(sampled_pos_ids), participated, posts_by_conv, k - n,
        )

        comment_history = user.get("comments", [])
        tasks.append((
            user["user_name"], persona,
            positive_posts, negatives, comment_history,
            system_prompt, user_template, model, temperature, k, n,
        ))

    if skipped:
        print(f"  Skipped {skipped} users with < {n} valid test positives")

    print(f"Running {len(tasks)} evaluation queries (K={k}, N={n})...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(_run_single_query, *t): t for t in tasks
        }
        for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Evaluating"):
            results.append(future.result())

    output_path = eval_dir / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {output_path}")

    if results:
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        exact_matches = sum(1 for r in results if r["exact_match"])
        print(f"  Precision:    {avg_precision:.3f}")
        print(f"  Recall:       {avg_recall:.3f}")
        print(f"  Exact Match:  {exact_matches}/{len(results)} = {exact_matches/len(results):.3f}")
        print(f"  Random baseline precision: {n/k:.3f}")


if __name__ == "__main__":
    main()
