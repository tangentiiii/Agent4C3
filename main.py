#!/usr/bin/env python3
"""
Twitter for C3 - Minimum Prototype
Entry point for the simulation pipeline.

Usage:
    python main.py prepare    # Step 1: Download & process Reddit data
    python main.py personas   # Step 2: Generate user personas via LLM
    python main.py profiles   # Step 3: Generate creator profiles via LLM
    python main.py synthetic  # Step 4: Generate synthetic click/like data
    python main.py simulate   # Step 5: Run the simulation
    python main.py all        # Run all steps sequentially
"""
import sys

import yaml
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml" 
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def step_prepare():
    print("=" * 60)
    print("STEP 1: Data Preparation")
    print("=" * 60)
    from src.data_preparation import main as prepare_main
    prepare_main()


def step_personas():
    print("=" * 60)
    print("STEP 2: Persona Generation")
    print("=" * 60)
    from src.persona_generator import main as persona_main
    persona_main()


def step_profiles():
    print("=" * 60)
    print("STEP 3: Creator Profile Generation")
    print("=" * 60)
    from src.creator_profile_generator import main as profile_main
    profile_main()


def step_synthetic():
    print("=" * 60)
    print("STEP 4: Synthetic Click/Like Data")
    print("=" * 60)
    from src.synthetic_data import main as synthetic_main
    synthetic_main()


def step_simulate():
    print("=" * 60)
    print("STEP 5: Simulation")
    print("=" * 60)
    from src.simulation import load_config, run_simulation
    config = load_config()
    run_simulation(config)


STEPS = {
    "prepare": step_prepare,
    "personas": step_personas,
    "profiles": step_profiles,
    "synthetic": step_synthetic,
    "simulate": step_simulate,
}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "all":
        for name, func in STEPS.items():
            func()
            print()
    elif command in STEPS:
        STEPS[command]()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(STEPS.keys())}, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
