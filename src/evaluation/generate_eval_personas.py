"""
Generate personas for evaluation users by reusing the existing persona generation pipeline.
"""
import json
from pathlib import Path

import yaml

from src.persona_generator import generate_personas


def load_config() -> dict:
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    base_dir = Path(__file__).parent.parent.parent / "data"
    eval_dir = base_dir / "evaluation"

    with open(eval_dir / "eval_users.json", "r") as f:
        eval_users = json.load(f)

    print(f"Generating personas for {len(eval_users)} eval users...")
    personas = generate_personas(eval_users, config)

    output_path = eval_dir / "eval_personas.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(personas)} eval personas to {output_path}")


if __name__ == "__main__":
    main()
