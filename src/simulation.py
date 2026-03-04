import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _generate_post(creator: ContentCreator) -> dict:
    post = creator.create_post()
    post["creator_id"] = creator.creator_id
    return post


def _generate_posts_concurrent(
    creators: list[ContentCreator], max_workers: int, desc: str,
) -> list[dict]:
    posts = [None] * len(creators)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_generate_post, c): i for i, c in enumerate(creators)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(creators), desc=desc):
            idx = future_to_idx[future]
            posts[idx] = future.result()
    return posts


def _process_user(user: User, posts: list[dict], max_workers: int) -> list[dict]:
    interactions = user.process_posts(posts, max_workers=max_workers)
    for interaction in interactions:
        interaction["user_name"] = user.user_name
    return interactions


def _process_users_concurrent(
    users: list[User],
    posts: list[dict],
    max_workers: int,
    desc: str,
) -> list[dict]:
    all_interactions = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_user, u, posts, max_workers): i
            for i, u in enumerate(users)
        }
        results = [None] * len(users)
        for future in tqdm(as_completed(future_to_idx), total=len(users), desc=desc):
            idx = future_to_idx[future]
            results[idx] = future.result()
    for result in results:
        all_interactions.extend(result)
    return all_interactions


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
    mechanism_params = config["simulation"].get("mechanism_params")
    mechanism = get_mechanism(config["simulation"]["reward_mechanism"], mechanism_params)
    num_rounds = config["simulation"]["num_rounds"]
    max_workers = config.get("concurrency", {}).get("max_workers", 10)
    creator_ids = [c.creator_id for c in creators]
    user_names = [u.user_name for u in users]

    print(f"Simulation: {len(users)} users, {len(creators)} creators, {num_rounds} rounds")
    print(f"Reward mechanism: {config['simulation']['reward_mechanism']}")
    print(f"Max concurrent API workers: {max_workers}")

    # Round 0: initial posts (no interactions yet)
    print("\n--- Round 0: Initial posts ---")
    current_posts = _generate_posts_concurrent(creators, max_workers, "Generating initial posts")

    _save_round(run_dir, 0, current_posts, [], {}, user_names)

    for r in range(1, num_rounds + 1):
        print(f"\n--- Round {r}/{num_rounds} ---")

        all_interactions = _process_users_concurrent(
            users, current_posts, max_workers, f"Round {r}: Users acting",
        )

        rewards = mechanism.compute_rewards(all_interactions, creator_ids)
        print(f"Rewards: {rewards}")

        for creator in creators:
            post = next(p for p in current_posts if p["creator_id"] == creator.creator_id)
            creator.record_reward(
                {"title": post["title"], "abstract": post["abstract"]},
                rewards[creator.creator_id],
            )

        _save_round(run_dir, r, current_posts, all_interactions, rewards, user_names)

        if r < num_rounds:
            current_posts = _generate_posts_concurrent(
                creators, max_workers, f"Round {r}: Creators posting",
            )

    print(f"\nSimulation complete. Results saved to {run_dir}")
    return run_dir


def _save_round(
    run_dir: Path,
    round_num: int,
    posts: list[dict],
    interactions: list[dict],
    rewards: dict,
    user_names: list[str],
):
    structured_posts = []
    for post in posts:
        cid = post["creator_id"]

        clicks = {}
        likes = {}
        click_reasons = {}
        for uname in user_names:
            record = next(
                (i for i in interactions
                 if i["creator_id"] == cid and i["user_name"] == uname),
                None,
            )
            if record and record.get("click") == 1:
                clicks[uname] = 1
                likes[uname] = record.get("like", 0)
                click_reasons[uname] = record.get("click_reason", "")
            else:
                clicks[uname] = 0
                likes[uname] = 0

        structured_posts.append({
            "creator_id": cid,
            "title": post["title"],
            "abstract": post["abstract"],
            "clicks": clicks,
            "click_reasons": click_reasons,
            "likes": likes,
            "reward": rewards.get(cid),
        })

    round_data = {
        "round": round_num,
        "posts": structured_posts,
    }
    with open(run_dir / f"round_{round_num}.json", "w", encoding="utf-8") as f:
        json.dump(round_data, f, ensure_ascii=False, indent=2)
