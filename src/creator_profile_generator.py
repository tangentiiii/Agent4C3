import json
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from tqdm import tqdm

from src.llm_client import load_prompt, call_llm_json
from src.persona_generator import format_user_history


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_eligible_users(config: dict) -> list[dict]:
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "processed" / "user_data.json", "r") as f:
        users = json.load(f)

    min_interactions = config["data"].get("min_creator_interactions", 10)
    eligible = [
        u for u in users
        if len(u.get("posts", [])) + len(u.get("comments", [])) >= min_interactions
    ]
    print(f"Eligible users (>= {min_interactions} interactions): {len(eligible)} / {len(users)}")
    return eligible


def sample_users(eligible_users: list[dict], num_creators: int) -> list[dict]:
    if len(eligible_users) < num_creators:
        print(
            f"Warning: only {len(eligible_users)} eligible users available, "
            f"but {num_creators} requested. Using all eligible users."
        )
        return eligible_users
    return random.sample(eligible_users, num_creators)


def _generate_single_profile(
    user: dict, system_prompt: str, user_template: str, model: str, temperature: float,
) -> dict:
    history_str = format_user_history(user)
    user_prompt = user_template.replace("{user_history_data}", history_str)

    response = call_llm_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        temperature=temperature,
    )

    return {
        "tone_style": str(response.get("tone_style", "")).strip(),
        "core_interests": response.get("core_interests", []),
        "content_approach": str(response.get("content_approach", "")).strip(),
        "distinctive_traits": str(response.get("distinctive_traits", "")).strip(),
        "source_user": user["user_name"],
    }


def generate_creator_profiles(users: list[dict], config: dict) -> list[dict]:
    prompt_data = load_prompt("generate_creator_profile")
    system_prompt = prompt_data["generate_creator_profile"]["system"]
    user_template = prompt_data["generate_creator_profile"]["user"]

    model = config["model"]["name"]
    temperature = config["model"]["temperature"]
    max_workers = config.get("concurrency", {}).get("max_workers", 10)

    profiles = [None] * len(users)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _generate_single_profile, user, system_prompt, user_template, model, temperature,
            ): i
            for i, user in enumerate(users)
        }
        for future in tqdm(
            as_completed(future_to_idx), total=len(users), desc="Generating creator profiles",
        ):
            idx = future_to_idx[future]
            profiles[idx] = future.result()

    return profiles


def main():
    config = load_config()
    data_dir = Path(__file__).parent.parent / "data"

    random.seed(config.get("seed", 42))

    eligible_users = load_eligible_users(config)
    num_creators = config["simulation"]["num_creators"]
    sampled = sample_users(eligible_users, num_creators)
    print(f"Sampled {len(sampled)} users for creator profiles")

    profiles = generate_creator_profiles(sampled, config)

    output_path = data_dir / "creator_profiles" / "creator_profiles.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(profiles)} creator profiles to {output_path}")


if __name__ == "__main__":
    main()
