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
        self.history: list[dict] = []  # [{post: {...}, reward: float}]
        self._config = _load_config()
        self._reward_tiers = self._build_reward_tiers()

    @staticmethod
    def _default_reward_tiers() -> list[dict]:
        return [
            {
                "name": "very_low",
                "min": 0.0,
                "max": 0.0,
                "description": "No traction. Your recent content is not resonating at all.",
            },
            {
                "name": "low",
                "min": 1.0,
                "max": 2.0,
                "description": "Weak traction. Consider stronger hooks or a meaningful topic shift.",
            },
            {
                "name": "medium",
                "min": 3.0,
                "max": 5.0,
                "description": "Moderate traction. Keep improving what works with clear refinements.",
            },
            {
                "name": "high",
                "min": 6.0,
                "max": None,
                "description": "Strong traction. Double down on successful patterns and quality.",
            },
        ]

    @staticmethod
    def _safe_float(value, fallback):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _build_reward_tiers(self) -> list[dict]:
        raw_tiers = self._config.get("content", {}).get("reward_tiers", [])
        if not isinstance(raw_tiers, list) or not raw_tiers:
            return self._default_reward_tiers()

        normalized = []
        for i, tier in enumerate(raw_tiers):
            if not isinstance(tier, dict):
                continue
            name = str(tier.get("name", f"tier_{i}")).strip() or f"tier_{i}"
            min_v = self._safe_float(tier.get("min"), None)
            max_raw = tier.get("max")
            max_v = None if max_raw is None else self._safe_float(max_raw, None)
            if min_v is None:
                min_v = float("-inf")
            if max_v is not None and min_v > max_v:
                continue
            description = str(tier.get("description", "")).strip()
            if not description:
                description = f"Reward range for {name}."
            normalized.append({
                "name": name,
                "min": min_v,
                "max": max_v,
                "description": description,
            })

        if not normalized:
            return self._default_reward_tiers()
        return sorted(normalized, key=lambda t: t["min"])

    def _reward_tier_for(self, reward) -> dict:
        reward_value = self._safe_float(reward, 0.0)
        for tier in self._reward_tiers:
            min_v = tier["min"]
            max_v = tier["max"]
            if reward_value >= min_v and (max_v is None or reward_value <= max_v):
                return tier
        if reward_value < self._reward_tiers[0]["min"]:
            return self._reward_tiers[0]
        return self._reward_tiers[-1]

    @staticmethod
    def _truncate_to_word_limit(text: str, word_limit: int) -> str:
        if not isinstance(text, str):
            text = str(text)
        words = text.strip().split()
        if len(words) <= word_limit:
            return " ".join(words)
        return " ".join(words[:word_limit])

    def _format_history(self) -> str:
        lines = []
        for i, entry in enumerate(self.history):
            post = entry["post"]
            reward = entry["reward"]
            tier = self._reward_tier_for(reward)
            lines.append(f"Round {i}:")
            lines.append(f"  Title: {post['title']}")
            lines.append(f"  Abstract: {post['abstract']}")
            topic = str(post.get("topic", "")).strip()
            strategy = str(post.get("next_strategy", "")).strip()
            if topic:
                lines.append(f"  Topic: {topic}")
            if strategy:
                lines.append(f"  Strategy used: {strategy}")
            lines.append(f"  Reward (likes): {reward}")
            lines.append(f"  Reward tier: {tier['name']}")
            lines.append(f"  Reward meaning: {tier['description']}")
            lines.append("")
        return "\n".join(lines)

    def create_post(self) -> dict:
        """Generate a new post."""
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

        title = self._truncate_to_word_limit(response.get("title", "Untitled"), title_limit)
        abstract = self._truncate_to_word_limit(
            response.get("abstract", ""),
            abstract_limit,
        )
        post = {
            "next_strategy": str(response.get("next_strategy", "")).strip()
            or "Run a focused strategy to improve engagement.",
            "topic": str(response.get("topic", "")).strip() or "general",
            "title": title if title else "Untitled",
            "abstract": abstract,
        }
        return post

    def record_reward(self, post: dict, reward: float):
        """Record a post and its reward in the creator's history."""
        self.history.append({"post": post, "reward": reward})
