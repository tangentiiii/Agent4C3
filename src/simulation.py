import json
from pathlib import Path
from datetime import datetime

import yaml
from tqdm import tqdm

from src.user import User
from src.content_creator import ContentCreator
from src.reward_mechanism import get_mechanism


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_users(config: dict) -> list[User]:
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "personas" / "personas.json", "r") as f:
        personas = json.load(f)

    with open(data_dir / "synthetic" / "click_like_data.json", "r") as f:
        synthetic_data = json.load(f)

    max_users = config["data"]["max_users"]
    personas = personas[:max_users]

    users = []
    for persona in personas:
        user_name = persona["user_name"]
        syn = synthetic_data.get(user_name, [])
        users.append(User(persona=persona, synthetic_data=syn))

    return users


def create_creators(config: dict) -> list[ContentCreator]:
    num_creators = config["simulation"]["num_creators"]
    return [ContentCreator(creator_id=i) for i in range(num_creators)]


def run_simulation(config: dict):
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run_{timestamp}"
    run_dir.mkdir()

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    users = load_users(config)
    creators = create_creators(config)
    mechanism = get_mechanism(config["simulation"]["reward_mechanism"])
    num_rounds = config["simulation"]["num_rounds"]
    creator_ids = [c.creator_id for c in creators]

    print(f"Simulation: {len(users)} users, {len(creators)} creators, {num_rounds} rounds")
    print(f"Reward mechanism: {config['simulation']['reward_mechanism']}")

    # Round 0: initial posts
    print("\n--- Round 0: Initial posts ---")
    current_posts = []
    for creator in tqdm(creators, desc="Generating initial posts"):
        post = creator.create_post()
        post["creator_id"] = creator.creator_id
        current_posts.append(post)

    _save_round(run_dir, 0, current_posts, [], {})

    for r in range(1, num_rounds + 1):
        print(f"\n--- Round {r}/{num_rounds} ---")

        # Users observe and interact with posts
        all_interactions = []
        for user in tqdm(users, desc=f"Round {r}: Users acting"):
            interactions = user.process_posts(current_posts)
            for interaction in interactions:
                interaction["user_name"] = user.user_name
            all_interactions.extend(interactions)

        # Compute rewards
        rewards = mechanism.compute_rewards(all_interactions, creator_ids)
        print(f"Rewards: {rewards}")

        # Record rewards for each creator
        for creator in creators:
            post = next(p for p in current_posts if p["creator_id"] == creator.creator_id)
            creator.record_reward(
                {"title": post["title"], "abstract": post["abstract"]},
                rewards[creator.creator_id],
            )

        _save_round(run_dir, r, current_posts, all_interactions, rewards)

        # Creators generate new posts for next round
        if r < num_rounds:
            current_posts = []
            for creator in tqdm(creators, desc=f"Round {r}: Creators posting"):
                post = creator.create_post()
                post["creator_id"] = creator.creator_id
                current_posts.append(post)

    print(f"\nSimulation complete. Results saved to {run_dir}")
    return run_dir


def _save_round(
    run_dir: Path,
    round_num: int,
    posts: list[dict],
    interactions: list[dict],
    rewards: dict,
):
    round_data = {
        "round": round_num,
        "posts": posts,
        "interactions": interactions,
        "rewards": {str(k): v for k, v in rewards.items()},
    }
    with open(run_dir / f"round_{round_num}.json", "w", encoding="utf-8") as f:
        json.dump(round_data, f, ensure_ascii=False, indent=2)
