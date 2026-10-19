#!/usr/bin/env python3
"""
Master reproduction script: Runs all experiments from scratch according to experiment_config.json.

Usage:
    python scripts/run_experiments.py --config config/experiment_config.json
    python scripts/run_experiments.py --config config/experiment_config.json --quick
"""

import argparse
from pathlib import Path
import sys

# Ensure src module is importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.experiments import load_config, run_all_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run 1D-BPP Optimization Experiments (GA, PSO, FFD)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment_config.json",
        help="Path to experiment configuration JSON file"
    )
    parser.add_argument(
        "--quick",
        "--smoke-test",
        action="store_true",
        help="Run in quick smoke-test mode with reduced seeds and budget"
    )

    args = parser.parse_args()

    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    run_all_experiments(config=config, repo_root=repo_root, quick_mode=args.quick)


if __name__ == "__main__":
    main()
