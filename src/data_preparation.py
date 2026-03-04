import json
import re
from pathlib import Path
from collections import defaultdict

import yaml
from convokit import Corpus, download
from tqdm import tqdm

MEANINGLESS_PATTERNS = re.compile(
    r"^\[deleted]$|^\[removed]$|^\[removed by reddit]$",
    re.IGNORECASE,
)
MIN_TEXT_LENGTH = 5


def is_meaningful_text(text: str) -> bool:
    """Return True if text carries real content worth keeping."""
    if not text or not text.strip():
        return False
    cleaned = text.strip()
    if MEANINGLESS_PATTERNS.match(cleaned):
        return False
    if len(cleaned) < MIN_TEXT_LENGTH:
        return False
    return True


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_corpus() -> Corpus:
    print("Loading Reddit Corpus (small)...")
    corpus = Corpus(filename=download("reddit-corpus-small"))
    corpus.print_summary_stats()
    return corpus


def extract_user_data(corpus: Corpus, min_interactions: int = 5) -> list[dict]:
    """Extract posts and comments for each user from the corpus."""
    user_posts = defaultdict(list)
    user_comments = defaultdict(list)

    utterance_map = {utt.id: utt for utt in corpus.iter_utterances()}

    print("Extracting user data from utterances...")
    for utt in tqdm(corpus.iter_utterances(), desc="Processing utterances"):
        speaker_name = utt.speaker.id
        if utt.reply_to is None:
            if is_meaningful_text(utt.text):
                user_posts[speaker_name].append(utt.text)
        else:
            if not is_meaningful_text(utt.text):
                continue
            parent = utterance_map.get(utt.reply_to)
            parent_text = parent.text if parent else ""
            user_comments[speaker_name].append({
                "comment_text": utt.text,
                "parent_text": parent_text,
            })

    all_speakers = set(user_posts.keys()) | set(user_comments.keys())
    users = []
    for name in tqdm(sorted(all_speakers), desc="Building user records"):
        posts = user_posts.get(name, [])
        comments = user_comments.get(name, [])
        if len(posts) + len(comments) < min_interactions:
            continue
        users.append({
            "user_name": name,
            "posts": posts,
            "comments": comments,
        })

    print(f"Extracted {len(users)} users with >= {min_interactions} interactions")
    return users


def save_user_data(users: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    print(f"Saved user data to {output_path}")


def collect_all_posts(corpus: Corpus) -> list[str]:
    """Collect all post titles/texts for use in synthetic data generation."""
    posts = []
    for utt in corpus.iter_utterances():
        if utt.reply_to is None and is_meaningful_text(utt.text):
            posts.append(utt.text)
    return posts


def main():
    config = load_config()
    corpus = download_corpus()

    min_interactions = config["data"]["min_user_interactions"]
    users = extract_user_data(corpus, min_interactions=min_interactions)

    base_dir = Path(__file__).parent.parent / "data"
    save_user_data(users, base_dir / "processed" / "user_data.json")

    total_raw_posts = sum(1 for utt in corpus.iter_utterances() if utt.reply_to is None)
    all_posts = collect_all_posts(corpus)
    print(f"Filtered posts: {total_raw_posts} raw -> {len(all_posts)} meaningful "
          f"({total_raw_posts - len(all_posts)} removed)")
    with open(base_dir / "processed" / "all_posts.json", "w", encoding="utf-8") as f:
        json.dump(all_posts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_posts)} posts to all_posts.json")


if __name__ == "__main__":
    main()
