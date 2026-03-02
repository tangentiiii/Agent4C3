import yaml
from pathlib import Path

from src.llm_client import load_prompt, call_llm_json


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ContentCreator:
    def __init__(self, creator_id: int):
        self.creator_id = creator_id
        self.history: list[dict] = []  # [{post: {title, abstract}, reward: float}]
        self._config = _load_config()

    def _format_history(self) -> str:
        lines = []
        for i, entry in enumerate(self.history):
            post = entry["post"]
            reward = entry["reward"]
            lines.append(f"Round {i}:")
            lines.append(f"  Title: {post['title']}")
            lines.append(f"  Abstract: {post['abstract']}")
            lines.append(f"  Reward (likes): {reward}")
            lines.append("")
        return "\n".join(lines)

    def create_post(self) -> dict:
        """Generate a new post. Returns {title: str, abstract: str}."""
        prompt_data = load_prompt("creator_post")
        system_prompt = prompt_data["creator_post"]["system"]

        title_limit = self._config["content"]["title_word_limit"]
        abstract_limit = self._config["content"]["abstract_word_limit"]

        if not self.history:
            template = prompt_data["creator_post"]["user_initial"]
        else:
            template = prompt_data["creator_post"]["user_with_history"]

        user_prompt = (
            template
            .replace("{creator_id}", str(self.creator_id))
            .replace("{title_word_limit}", str(title_limit))
            .replace("{abstract_word_limit}", str(abstract_limit))
            .replace("{history}", self._format_history())
        )

        response = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._config["model"]["name"],
            temperature=self._config["model"]["temperature"],
        )

        post = {
            "title": response.get("title", "Untitled"),
            "abstract": response.get("abstract", ""),
        }
        return post

    def record_reward(self, post: dict, reward: float):
        """Record a post and its reward in the creator's history."""
        self.history.append({"post": post, "reward": reward})
