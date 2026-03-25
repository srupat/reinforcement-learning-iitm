from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from environment_wrapper import make_four_room_env, make_standard_grid_env
from experiment_configs import experiment_to_dict, get_experiment_spec, list_experiment_specs
from td_learning import TrainingResult, train_q_learning, train_sarsa
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_WANDB_PROJECT = "rl-programming-assignment-2"


def _format_value(value: float | int | str) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def training_config_slug(training_config) -> str:
    parts = [
        f"episodes_{training_config.episodes}",
        f"alpha_{_format_value(training_config.alpha)}",
        f"gamma_{_format_value(training_config.gamma)}",
        f"exploration_{training_config.exploration_strategy}",
    ]
    if training_config.exploration_strategy == "epsilon_greedy":
        parts.append(f"epsilon_{_format_value(training_config.epsilon)}")
    elif training_config.exploration_strategy == "softmax":
        parts.append(f"temperature_{_format_value(training_config.temperature)}")
    return "__".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one named reinforcement learning experiment.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name. Use --list to see available options.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the seed for this run.")
    parser.add_argument("--episodes", type=int, default=None, help="Override the number of episodes.")
    parser.add_argument("--alpha", type=float, default=None, help="Override the learning rate.")
    parser.add_argument("--gamma", type=float, default=None, help="Override the discount factor.")
    parser.add_argument(
        "--exploration-strategy",
        type=str,
        default=None,
        choices=["epsilon_greedy", "softmax"],
        help="Override the exploration strategy.",
    )
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon for epsilon-greedy runs.")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature for softmax runs.")
    parser.add_argument("--wandb", action="store_true", help="Log this run to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline for local testing before logging in.",
    )
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B run group.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available experiment names and exit.",
    )
    return parser.parse_args()


def build_env(spec):
    if spec.env_name == "standard_grid":
        return make_standard_grid_env(spec.env_config)
    if spec.env_name == "four_room":
        return make_four_room_env(spec.env_config)
    raise ValueError(f"Unsupported environment: {spec.env_name}")


def run_spec(spec, episode_logger=None) -> TrainingResult:
    env = build_env(spec)
    if spec.algorithm == "q_learning":
        return train_q_learning(env=env, config=spec.training_config, episode_logger=episode_logger)
    if spec.algorithm == "sarsa":
        return train_sarsa(env=env, config=spec.training_config, episode_logger=episode_logger)
    raise ValueError(f"Unsupported algorithm: {spec.algorithm}")


def save_result(spec, result: TrainingResult) -> Path:
    run_dir = result_dir_for_spec(spec)
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = experiment_to_dict(spec)
    metadata["summary"] = {
        "mean_reward": float(result.episode_rewards.mean()),
        "mean_steps": float(result.episode_steps.mean()),
        "success_rate": float(result.episode_successes.mean()),
    }

    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    np.savez_compressed(
        run_dir / "metrics.npz",
        q_values=result.q_values,
        episode_rewards=result.episode_rewards,
        episode_steps=result.episode_steps,
        episode_successes=result.episode_successes,
        state_visit_counts=result.state_visit_counts,
    )
    return run_dir


def result_dir_for_spec(spec) -> Path:
    return RESULTS_DIR / spec.name / training_config_slug(spec.training_config) / f"seed_{spec.training_config.seed}"


def result_exists(spec) -> bool:
    run_dir = result_dir_for_spec(spec)
    return (run_dir / "metadata.json").exists() and (run_dir / "metrics.npz").exists()


def print_summary(spec, result: TrainingResult, run_dir: Path) -> None:
    print(f"Experiment: {spec.name}")
    print(f"Algorithm: {spec.algorithm}")
    print(f"Environment: {spec.env_name}")
    print(f"Seed: {spec.training_config.seed}")
    print(f"Episodes: {spec.training_config.episodes}")
    print(f"Run label: {training_config_slug(spec.training_config)}")
    print(f"Mean reward: {result.episode_rewards.mean():.2f}")
    print(f"Mean steps: {result.episode_steps.mean():.2f}")
    print(f"Success rate: {result.episode_successes.mean():.3f}")
    print(f"Saved results to: {run_dir}")


def main() -> None:
    args = parse_args()

    if args.list:
        for name in list_experiment_specs():
            print(name)
        return
    if args.experiment is None:
        raise ValueError("Please provide --experiment or use --list to inspect the available names.")

    spec = get_experiment_spec(args.experiment)
    training_config = spec.training_config
    if args.seed is not None:
        training_config = replace(training_config, seed=args.seed)
    if args.episodes is not None:
        training_config = replace(training_config, episodes=args.episodes)
    if args.alpha is not None:
        training_config = replace(training_config, alpha=args.alpha)
    if args.gamma is not None:
        training_config = replace(training_config, gamma=args.gamma)
    if args.exploration_strategy is not None:
        training_config = replace(training_config, exploration_strategy=args.exploration_strategy)
    if args.epsilon is not None:
        training_config = replace(training_config, epsilon=args.epsilon)
    if args.temperature is not None:
        training_config = replace(training_config, temperature=args.temperature)
    spec = replace(spec, training_config=training_config)

    wandb_run = None
    episode_logger = None
    if args.wandb:
        from wandb_utils import init_wandb_run, log_result_summary, spec_to_wandb_config

        tags = [spec.algorithm, spec.env_name, spec.training_config.exploration_strategy]
        wandb_run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            job_type="train",
            run_name=f"{spec.name}__seed_{spec.training_config.seed}__{training_config_slug(spec.training_config)}",
            group=args.wandb_group,
            tags=tags,
            config=spec_to_wandb_config(spec),
        )
        episode_logger = wandb_run.log

    result = run_spec(spec, episode_logger=episode_logger)
    run_dir = save_result(spec, result)
    if wandb_run is not None:
        log_result_summary(wandb_run, result)
        wandb_run.summary["result_dir"] = str(run_dir)
        wandb_run.finish()
    print_summary(spec, result, run_dir)


if __name__ == "__main__":
    main()
