# Agent4C3 — Minimum Prototype

A multi-agent social media simulation framework that studies creator–user dynamics and content-creator-competition(C3) under various reward mechanisms. LLM-powered agents act as content creators and users, enabling controlled experiments on how different incentive structures shape content creation behavior.

## Overview

This project simulates a simplified social media platform where:

- **Users** are LLM agents whose personas are derived from real Reddit interaction histories (Big Five personality traits, interests, conversational style, etc.).
- **Content Creators** are LLM agents that generate posts (title + abstract) and adapt their strategies based on reward feedback.
- **Reward Mechanisms** translate user engagement (clicks and likes) into creator rewards using a variety of algorithms from the literature (M³, BRCM, and baselines).

The simulation runs over multiple rounds: creators post content, users decide what to click and like, rewards are computed, and creators adjust their strategies accordingly.

## Project Structure

```
minimum_prototype/
├── main.py                          # CLI entry point
├── config.yaml                      # Global configuration
├── requirements.txt                 # Python dependencies
├── .env                             # API key (not tracked)
│
├── src/
│   ├── data_preparation.py          # Download & process Reddit corpus (via convokit)
│   ├── persona_generator.py         # Generate user personas from post history
│   ├── creator_profile_generator.py # Generate creator profiles from user data
│   ├── synthetic_data.py            # Synthesize click/like behavior per persona
│   ├── simulation.py                # Main simulation loop
│   ├── user.py                      # User agent (click & like decisions)
│   ├── content_creator.py           # Creator agent (post generation & adaptation)
│   ├── llm_client.py                # OpenRouter LLM client & prompt loader
│   └── reward_mechanism.py          # Reward mechanism implementations
│
├── prompts/
│   ├── generate_persona.yml         # Persona extraction from user history
│   ├── generate_creator_profile.yml # Creator profile generation
│   ├── synthesize_click_like.yml    # Synthetic engagement data generation
│   ├── user_click.yml               # User click decision prompt
│   ├── user_like.yml                # User like decision prompt
│   └── creator_post.yml             # Creator content generation prompt
│
├── data/                            # Generated data (gitignored)
│   ├── raw/                         # Raw Reddit corpus
│   ├── processed/                   # Cleaned user data & posts
│   ├── personas/                    # LLM-generated user personas
│   ├── creator_profiles/            # LLM-generated creator profiles
│   └── synthetic/                   # Synthetic click/like data
│
├── results/                         # Simulation outputs (gitignored)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── config.json              # Snapshot of config used
│       └── round_N.json             # Per-round results
│
└── analysis/
    ├── data_stats.py                # Visualization scripts
    └── figures/                     # Generated plots
```

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenRouter](https://openrouter.ai/) API key

### Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

### Running the Pipeline

The pipeline consists of five sequential steps. You can run them individually or all at once:

```bash
# Run all steps end-to-end
python main.py all

# Or run each step separately
python main.py prepare    # Step 1: Download & process Reddit data via convokit
python main.py personas   # Step 2: Generate user personas via LLM
python main.py profiles   # Step 3: Generate creator profiles via LLM
python main.py synthetic  # Step 4: Synthesize click/like behavior data
python main.py simulate   # Step 5: Run the multi-round simulation
```

Steps 1–4 are data preparation; their outputs are cached in `data/` and only need to be run once. Step 5 (simulation) can be re-run with different configurations.

## Configuration

All parameters are controlled via `config.yaml`:

| Section | Key Parameters | Description |
|---------|---------------|-------------|
| `model` | `name`, `temperature` | LLM model and sampling settings |
| `data` | `max_users`, `min_user_interactions` | Data filtering and user pool size |
| `simulation` | `num_creators`, `num_rounds`, `reward_mechanism` | Simulation scale and reward algorithm |
| `concurrency` | `max_workers` | Parallel LLM API call limit |
| `content` | `title_word_limit`, `abstract_word_limit`, `reward_tiers` | Post constraints and reward tier definitions |

## Reward Mechanisms

The following reward mechanisms are implemented:

| Mechanism | Description | Parameters |
|-----------|-------------|------------|
| `default` | Baseline — reward equals total like count | — |
| `M3_0` | M³(0) — per-user matching score | — |
| `M3_expo` | M³(expo.) — exposure-based softmax allocation | `K`, `beta` |
| `M3_enga` | M³(enga.) — engagement-weighted softmax allocation | `K`, `beta` |
| `BRCM_opt` | BRCM_opt — backward rewarding with uniform f-vector | — |
| `BRCM_star` | BRCM* — theoretically optimal backward rewarding | — |
| `BRCM_1` | BRCM_1 — backward rewarding with harmonic f-vector | — |

To switch mechanisms, update `reward_mechanism` and `mechanism_params` in `config.yaml`.

## Output Format

Each simulation run creates a timestamped directory under `results/`. Each round produces a JSON file containing:

- **Posts**: creator-generated content with strategy, topic, title, and abstract
- **Interactions**: per-user click/like decisions with reasoning
- **Rewards**: computed reward value per creator under the selected mechanism

## Analysis

Generate visualization figures from the data:

```bash
python analysis/data_stats.py
```

This produces plots for user activity distributions, Big Five personality trait breakdowns, and click/like behavior summaries, saved to `analysis/figures/`.
