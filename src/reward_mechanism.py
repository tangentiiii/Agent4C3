from abc import ABC, abstractmethod
from collections import defaultdict


class RewardMechanism(ABC):
    """Base class for rewarding mechanisms."""

    @abstractmethod
    def compute_rewards(
        self,
        interactions: list[dict],
        creator_ids: list[int],
    ) -> dict[int, float]:
        """
        Compute rewards for each content creator based on user interactions.

        Args:
            interactions: flat list of all user interaction records for this round.
                Each record has: {title, creator_id, click: 0/1, like?: 0/1}
            creator_ids: list of all creator IDs.

        Returns:
            dict mapping creator_id -> reward value
        """
        pass


class DefaultMechanism(RewardMechanism):
    """Reward = total number of likes received."""

    def compute_rewards(
        self,
        interactions: list[dict],
        creator_ids: list[int],
    ) -> dict[int, float]:
        likes = defaultdict(int)
        for record in interactions:
            if record.get("click") == 1 and record.get("like") == 1:
                likes[record["creator_id"]] += 1

        return {cid: float(likes.get(cid, 0)) for cid in creator_ids}


MECHANISMS = {
    "default": DefaultMechanism,
}


def get_mechanism(name: str) -> RewardMechanism:
    cls = MECHANISMS.get(name)
    if cls is None:
        raise ValueError(f"Unknown mechanism: {name}. Available: {list(MECHANISMS.keys())}")
    return cls()
