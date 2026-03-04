import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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

    def click(self, posts: list[dict]) -> tuple[list[int], dict[int, str]]:
        """
        Decide which posts to click given a list of posts with titles.
        Returns (list of clicked indices, dict mapping index -> reason).
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

        valid_indices = []
        reasons: dict[int, str] = {}
        raw_decisions = response.get("decisions", [])
        if isinstance(raw_decisions, list) and raw_decisions:
            for item in raw_decisions:
                if not isinstance(item, dict):
                    continue
                idx = item.get("post")
                click = item.get("click", 0)
                reason = item.get("reason", "")
                if isinstance(idx, int) and 0 <= idx < len(posts):
                    reasons[idx] = reason
                    if click:
                        valid_indices.append(idx)
        else:
            # Backward compatibility with previous output schema.
            raw_clicks = response.get("clicks", [])
            for item in raw_clicks:
                if isinstance(item, dict):
                    idx = item.get("post")
                    reason = item.get("reason", "")
                else:
                    idx = item
                    reason = ""
                if isinstance(idx, int) and 0 <= idx < len(posts):
                    valid_indices.append(idx)
                    reasons[idx] = reason
        return valid_indices, reasons

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

    def process_posts(self, posts: list[dict], max_workers: int = 10) -> list[dict]:
        """
        Full user action cycle: observe titles, click, then like/dislike.
        Like calls for clicked posts are issued concurrently.
        Returns interaction records for this round.
        """
        clicked_indices_list, click_reasons = self.click(posts)
        clicked_indices = set(clicked_indices_list)

        like_results: dict[int, bool] = {}
        clicked_posts = [(i, posts[i]) for i in range(len(posts)) if i in clicked_indices]

        if clicked_posts:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.like, p["title"], p["abstract"]): i
                    for i, p in clicked_posts
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    like_results[idx] = future.result()

        interactions = []
        for i, post in enumerate(posts):
            if i in clicked_indices:
                interactions.append({
                    "title": post["title"],
                    "creator_id": post["creator_id"],
                    "click": 1,
                    "click_reason": click_reasons.get(i, ""),
                    "like": 1 if like_results[i] else 0,
                })
            else:
                interactions.append({
                    "title": post["title"],
                    "creator_id": post["creator_id"],
                    "click": 0,
                    "click_reason": click_reasons.get(i, ""),
                })

        self.history.extend(interactions)
        return interactions
