import json
import random
from pathlib import Path

import yaml
from tqdm import tqdm

from src.llm_client import load_prompt, call_llm_json


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_persona(persona: dict) -> str:
    """Format a persona dict into a readable string for the prompt."""
    lines = []
    if "conversational_style" in persona:
        lines.append(f"Conversational Style: {persona['conversational_style']}")
    if "core_interests" in persona:
        lines.append(f"Core Interests: {persona['core_interests']}")
    if "activity" in persona and isinstance(persona["activity"], dict):
        activity = persona["activity"]
        lines.append(
            f"Activity: [{activity.get('tier', 'Unknown')}] {activity.get('description', '')}".strip()
        )
    if "diversity" in persona and isinstance(persona["diversity"], dict):
        diversity = persona["diversity"]
        lines.append(
            f"Diversity: [{diversity.get('tier', 'Unknown')}] {diversity.get('description', '')}".strip()
        )
    if "big_five" in persona:
        lines.append("Big Five Personality:")
        for trait, info in persona["big_five"].items():
            lines.append(f"  - {trait.capitalize()}: [{info['tier']}] {info['description']}")
    return "\n".join(lines)


def format_posts_for_prompt(posts: list[str]) -> str:
    lines = []
    for i, post in enumerate(posts, 1):
        lines.append(f"{i}. {post[:300]}")
    return "\n".join(lines)


def synthesize_for_user(
    persona: dict,
    all_posts: list[str],
    sample_size: int,
    config: dict,
) -> list[dict]:
    prompt_data = load_prompt("synthesize_click_like")
    system_prompt = prompt_data["synthesize_click_like"]["system"]
    user_template = prompt_data["synthesize_click_like"]["user"]

    sampled = random.sample(all_posts, min(sample_size, len(all_posts)))

    persona_str = format_persona(persona)
    posts_str = format_posts_for_prompt(sampled)

    user_prompt = user_template.replace("{persona}", persona_str).replace("{posts}", posts_str)

    response = call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=config["model"]["name"],
        temperature=config["model"]["temperature"],
    )

    results = response.get("results", [])
    return results


def main():
    config = load_config()
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "personas" / "personas.json", "r") as f:
        personas = json.load(f)

    with open(data_dir / "processed" / "all_posts.json", "r") as f:
        all_posts = json.load(f)

    sample_size = config["data"]["synthetic_sample_size"]
    all_synthetic = {}

    for persona in tqdm(personas, desc="Synthesizing click/like data"):
        user_name = persona["user_name"]
        results = synthesize_for_user(persona, all_posts, sample_size, config)
        all_synthetic[user_name] = results

    output_path = data_dir / "synthetic" / "click_like_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_synthetic, f, ensure_ascii=False, indent=2)
    print(f"Saved synthetic data for {len(all_synthetic)} users to {output_path}")


if __name__ == "__main__":
    main()
