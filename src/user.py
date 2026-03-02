import json

import yaml
from pathlib import Path

from src.llm_client import load_prompt, call_llm_json
from src.synthetic_data import format_persona


def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _format_synthetic_data(synthetic: list[dict], max_items: int = 10) -> str:
    if not synthetic:
        return "(no prior data)"
    lines = []
    for item in synthetic[:max_items]:
        click = item.get("click", 0)
        if click:
            like = item.get("like", 0)
            lines.append(f"- \"{item.get('text', '')}\" -> clicked, {'liked' if like else 'not liked'}")
        else:
            lines.append(f"- \"{item.get('text', '')}\" -> not clicked")
    return "\n".join(lines)


def _format_history(history: list[dict], max_items: int = 15) -> str:
    if not history:
        return "(no browsing history yet)"
    lines = []
    for item in history[-max_items:]:
        click = item.get("click", 0)
        if click:
            like = item.get("like", 0)
            lines.append(f"- \"{item['title']}\" -> clicked, {'liked' if like else 'not liked'}")
        else:
            lines.append(f"- \"{item['title']}\" -> not clicked")
    return "\n".join(lines)


class User:
    def __init__(self, persona: dict, synthetic_data: list[dict]):
        self.persona = persona
        self.user_name = persona["user_name"]
        self.synthetic_data = synthetic_data
        self.history: list[dict] = []
        self._config = _load_config()

    def click(self, posts: list[dict]) -> list[int]:
        """
        Decide which posts to click given a list of posts with titles.
        Returns list of indices (0-based) that the user clicks on.
        """
        prompt_data = load_prompt("user_click")
        system_prompt = prompt_data["user_click"]["system"]
        user_template = prompt_data["user_click"]["user"]

        title_lines = []
        for i, post in enumerate(posts):
            title_lines.append(f"{i}. {post['title']}")

        user_prompt = (
            user_template
            .replace("{persona}", format_persona(self.persona))
            .replace("{synthetic_data}", _format_synthetic_data(self.synthetic_data))
            .replace("{history}", _format_history(self.history))
            .replace("{post_titles}", "\n".join(title_lines))
        )

        response = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._config["model"]["name"],
            temperature=self._config["model"]["temperature"],
        )

        clicked_indices = response.get("clicks", [])
        valid_indices = [i for i in clicked_indices if 0 <= i < len(posts)]
        return valid_indices

    def like(self, title: str, abstract: str) -> bool:
        """Decide whether to like a post after reading its abstract."""
        prompt_data = load_prompt("user_like")
        system_prompt = prompt_data["user_like"]["system"]
        user_template = prompt_data["user_like"]["user"]

        user_prompt = (
            user_template
            .replace("{persona}", format_persona(self.persona))
            .replace("{synthetic_data}", _format_synthetic_data(self.synthetic_data))
            .replace("{title}", title)
            .replace("{abstract}", abstract)
        )

        response = call_llm_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self._config["model"]["name"],
            temperature=self._config["model"]["temperature"],
        )

        return bool(response.get("like", 0))

    def process_posts(self, posts: list[dict]) -> list[dict]:
        """
        Full user action cycle: observe titles, click, then like/dislike.
        Returns interaction records for this round.
        """
        clicked_indices = self.click(posts)
        interactions = []

        for i, post in enumerate(posts):
            if i in clicked_indices:
                liked = self.like(post["title"], post["abstract"])
                record = {
                    "title": post["title"],
                    "creator_id": post["creator_id"],
                    "click": 1,
                    "like": 1 if liked else 0,
                }
            else:
                record = {
                    "title": post["title"],
                    "creator_id": post["creator_id"],
                    "click": 0,
                }
            interactions.append(record)

        self.history.extend(interactions)
        return interactions
