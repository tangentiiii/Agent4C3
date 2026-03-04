import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm

from src.llm_client import load_prompt, call_llm


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_user_history(user: dict, max_posts: int = 10, max_comments: int = 20) -> str:
    """Format a user's history from JSON/dict into a string for the LLM prompt."""
    lines = []
    posts = user["posts"][:max_posts]
    if posts:
        lines.append("Posts:")
        for i, p in enumerate(posts, 1):
            lines.append(f"  {i}. {p[:300]}")

    comments = user["comments"][:max_comments]
    if comments:
        lines.append("\nComments (comment_text -> in reply to parent_text):")
        for i, c in enumerate(comments, 1):
            parent = c["parent_text"][:200] if c["parent_text"] else "(no parent)"
            comment = c["comment_text"][:200]
            lines.append(f"  {i}. comment_text: \"{comment}\"")
            lines.append(f"     parent_text: \"{parent}\"")

    return "\n".join(lines)


def parse_persona(raw_text: str) -> dict:
    """Parse the LLM's persona output into a structured dict."""
    persona = {"raw": raw_text}

    style_match = re.search(r"CONVERSATIONAL_STYLE:\s*(.+)", raw_text)
    if style_match:
        persona["conversational_style"] = style_match.group(1).strip()

    interests_match = re.search(r"CORE_INTERESTS:\s*(.+)", raw_text)
    if interests_match:
        persona["core_interests"] = interests_match.group(1).strip()

    reasons = re.findall(r"REASON:\s*(.+)", raw_text)
    if len(reasons) >= 2:
        persona["style_reason"] = reasons[0].strip()
        persona["interests_reason"] = reasons[1].strip()

    activity_match = re.search(r"ACTIVITY:\s*\[(\w+)\]\s*-?\s*(.*)", raw_text)
    if activity_match:
        persona["activity"] = {
            "tier": activity_match.group(1).strip(),
            "description": activity_match.group(2).strip(),
        }

    diversity_match = re.search(r"DIVERSITY:\s*\[(\w+)\]\s*-?\s*(.*)", raw_text)
    if diversity_match:
        persona["diversity"] = {
            "tier": diversity_match.group(1).strip(),
            "description": diversity_match.group(2).strip(),
        }

    big_five = {}
    for trait in ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]:
        match = re.search(rf"-\s*{trait}:\s*\[(\w+)\]\s*-?\s*(.*)", raw_text)
        if match:
            big_five[trait.lower()] = {
                "tier": match.group(1).strip(),
                "description": match.group(2).strip(),
            }
    persona["big_five"] = big_five

    return persona


def generate_personas(users: list[dict], config: dict) -> list[dict]:
    prompt_data = load_prompt("generate_persona")
    system_prompt = prompt_data["generate_persona"]["system"]
    user_template = prompt_data["generate_persona"]["user"]

    model = config["model"]["name"]
    temperature = config["model"]["temperature"]

    personas = []
    for user in tqdm(users, desc="Generating personas"):
        history_str = format_user_history(user)
        user_prompt = user_template.replace("{user_history_data}", history_str)

        raw_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
        )

        persona = parse_persona(raw_response)
        persona["user_name"] = user["user_name"]
        personas.append(persona)

    return personas


def main():
    config = load_config()
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "processed" / "user_data.json", "r") as f:
        users = json.load(f)

    max_users = config["data"]["max_users"]
    if len(users) > max_users:
        users = users[:max_users]
        print(f"Limited to {max_users} users")

    personas = generate_personas(users, config)

    output_path = data_dir / "personas" / "personas.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(personas)} personas to {output_path}")


if __name__ == "__main__":
    main()
