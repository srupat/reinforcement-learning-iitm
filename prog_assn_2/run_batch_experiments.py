from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from experiment_configs import get_experiment_spec, list_experiment_specs
from run_experiment import RESULTS_DIR, result_dir_for_spec, result_exists, run_spec, save_result, training_config_slug


DEFAULT_WANDB_PROJECT = "rl-programming-assignment-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment experiment sweeps across seeds and hyperparameters.")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names, or 'all' for the full assignment set.",
    )
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes per run.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes to use.")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seed list.")
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated alpha values.",
    )
    parser.add_argument(
        "--gammas",
        type=str,
        default="0.7,0.8,0.9,1.0",
        help="Comma-separated gamma values.",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0.001,0.01,0.05,0.1",
        help="Comma-separated epsilon values for epsilon-greedy experiments.",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.01,0.1,1.0,2.0",
        help="Comma-separated temperature values for softmax experiments.",
    )
    parser.add_argument(
        "--batch-name",
        type=str,
        default=None,
        help="Optional folder name for the batch summary output.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log each run in the batch to Weights & Biases.")
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT, help="W&B project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode. Use offline for local testing before logging in.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the available experiment names and exit.",
    )
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def resolve_experiments(raw: str) -> list[str]:
    if raw == "all":
        return list_experiment_specs()
    return [item.strip() for item in raw.split(",") if item.strip()]


def sweep_values(spec, alphas: list[float], gammas: list[float], epsilons: list[float], temperatures: list[float]) -> list[dict]:
    values: list[dict] = []
    if spec.training_config.exploration_strategy == "epsilon_greedy":
        for alpha in alphas:
            for gamma in gammas:
                for epsilon in epsilons:
                    values.append(
                        {
                            "alpha": alpha,
                            "gamma": gamma,
                            "epsilon": epsilon,
                            "temperature": spec.training_config.temperature,
                        }
                    )
    elif spec.training_config.exploration_strategy == "softmax":
        for alpha in alphas:
            for gamma in gammas:
                for temperature in temperatures:
                    values.append(
                        {
                            "alpha": alpha,
                            "gamma": gamma,
                            "epsilon": spec.training_config.epsilon,
                            "temperature": temperature,
                        }
                    )
    else:
        raise ValueError(f"Unsupported exploration strategy: {spec.training_config.exploration_strategy}")
    return values


def batch_output_dir(batch_name: str | None) -> Path:
    label = batch_name or datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / "batches" / label
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_batch_outputs(output_dir: Path, rows: list[dict], batch_metadata: dict) -> None:
    with (output_dir / "batch_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(batch_metadata, handle, indent=2)

    fieldnames = [
        "experiment",
        "algorithm",
        "environment",
        "seed",
        "episodes",
        "alpha",
        "gamma",
        "exploration_strategy",
        "epsilon",
        "temperature",
        "mean_reward",
        "mean_steps",
        "success_rate",
        "result_dir",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_from_existing_run(spec) -> dict:
    run_dir = result_dir_for_spec(spec)
    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "experiment": spec.name,
        "algorithm": spec.algorithm,
        "environment": spec.env_name,
        "seed": spec.training_config.seed,
        "episodes": spec.training_config.episodes,
        "alpha": spec.training_config.alpha,
        "gamma": spec.training_config.gamma,
        "exploration_strategy": spec.training_config.exploration_strategy,
        "epsilon": spec.training_config.epsilon,
        "temperature": spec.training_config.temperature,
        "mean_reward": float(metadata["summary"]["mean_reward"]),
        "mean_steps": float(metadata["summary"]["mean_steps"]),
        "success_rate": float(metadata["summary"]["success_rate"]),
        "result_dir": str(run_dir),
    }


def row_from_result(spec, result, run_dir: Path) -> dict:
    return {
        "experiment": spec.name,
        "algorithm": spec.algorithm,
        "environment": spec.env_name,
        "seed": spec.training_config.seed,
        "episodes": spec.training_config.episodes,
        "alpha": spec.training_config.alpha,
        "gamma": spec.training_config.gamma,
        "exploration_strategy": spec.training_config.exploration_strategy,
        "epsilon": spec.training_config.epsilon,
        "temperature": spec.training_config.temperature,
        "mean_reward": float(result.episode_rewards.mean()),
        "mean_steps": float(result.episode_steps.mean()),
        "success_rate": float(result.episode_successes.mean()),
        "result_dir": str(run_dir),
    }


def execute_spec(spec) -> dict:
    result = run_spec(spec, episode_logger=None)
    run_dir = save_result(spec, result)
    return row_from_result(spec, result, run_dir)


def execute_spec_subprocess(spec) -> dict:
    run_experiment_path = Path(__file__).resolve().parent / "run_experiment.py"
    command = [
        sys.executable,
        str(run_experiment_path),
        "--experiment",
        spec.name,
        "--seed",
        str(spec.training_config.seed),
        "--episodes",
        str(spec.training_config.episodes),
        "--alpha",
        str(spec.training_config.alpha),
        "--gamma",
        str(spec.training_config.gamma),
        "--exploration-strategy",
        spec.training_config.exploration_strategy,
        "--epsilon",
        str(spec.training_config.epsilon),
        "--temperature",
        str(spec.training_config.temperature),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Parallel worker failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )
    if not result_exists(spec):
        raise RuntimeError(
            "Parallel worker exited successfully but did not create the expected result files.\n"
            f"Command: {' '.join(command)}\n"
            f"Stdout:\n{completed.stdout}\n"
            f"Stderr:\n{completed.stderr}"
        )
    return row_from_existing_run(spec)


def main() -> None:
    args = parse_args()

    if args.list:
        for name in list_experiment_specs():
            print(name)
        return

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.jobs > 1 and args.wandb:
        raise ValueError("Parallel runs with live W&B logging are not supported. Use --jobs > 1 without --wandb.")

    experiments = resolve_experiments(args.experiments)
    seeds = parse_int_list(args.seeds)
    alphas = parse_float_list(args.alphas)
    gammas = parse_float_list(args.gammas)
    epsilons = parse_float_list(args.epsilons)
    temperatures = parse_float_list(args.temperatures)

    rows: list[dict] = []
    output_dir = batch_output_dir(args.batch_name)

    total_runs = 0
    for experiment_name in experiments:
        spec = get_experiment_spec(experiment_name)
        total_runs += len(seeds) * len(sweep_values(spec, alphas, gammas, epsilons, temperatures))

    batch_metadata = {
        "experiments": experiments,
        "episodes": args.episodes,
        "jobs": args.jobs,
        "seeds": seeds,
        "alphas": alphas,
        "gammas": gammas,
        "epsilons": epsilons,
        "temperatures": temperatures,
        "total_runs": total_runs,
    }

    print(f"Planned runs: {total_runs}")
    print(f"Batch summary directory: {output_dir}")
    print(f"Worker processes: {args.jobs}")

    pending_specs = []

    for experiment_name in experiments:
        base_spec = get_experiment_spec(experiment_name)
        overrides = sweep_values(base_spec, alphas, gammas, epsilons, temperatures)

        for override in overrides:
            for seed in seeds:
                training_config = replace(
                    base_spec.training_config,
                    episodes=args.episodes,
                    seed=seed,
                    alpha=override["alpha"],
                    gamma=override["gamma"],
                    epsilon=override["epsilon"],
                    temperature=override["temperature"],
                )
                spec = replace(base_spec, training_config=training_config)
                if result_exists(spec):
                    rows.append(row_from_existing_run(spec))
                    print(
                        f"[skip {len(rows)}/{total_runs}] "
                        f"{spec.name} | {training_config_slug(training_config)} | seed={seed}"
                    )
                    write_batch_outputs(output_dir, rows, batch_metadata)
                    continue

                if args.jobs > 1:
                    pending_specs.append(spec)
                    continue

                wandb_run = None
                episode_logger = None
                if args.wandb:
                    from wandb_utils import init_wandb_run, log_result_summary, spec_to_wandb_config

                    try:
                        wandb_run = init_wandb_run(
                            project=args.wandb_project,
                            entity=args.wandb_entity,
                            mode=args.wandb_mode,
                            job_type="train",
                            run_name=f"{spec.name}__seed_{seed}__{training_config_slug(training_config)}",
                            group=args.batch_name or "batch_runs",
                            tags=[spec.algorithm, spec.env_name, training_config.exploration_strategy, "batch"],
                            config=spec_to_wandb_config(spec),
                        )
                        episode_logger = wandb_run.log
                    except Exception as exc:
                        print(
                            f"W&B init failed for {spec.name} | {training_config_slug(training_config)} | "
                            f"seed={seed}. Continuing without W&B for this run. Error: {exc}"
                        )

                result = run_spec(spec, episode_logger=episode_logger)
                run_dir = save_result(spec, result)
                if wandb_run is not None:
                    try:
                        log_result_summary(wandb_run, result)
                        wandb_run.summary["result_dir"] = str(run_dir)
                        wandb_run.finish()
                    except Exception as exc:
                        print(f"W&B finalize failed for {spec.name} | seed={seed}. Error: {exc}")

                row = row_from_result(spec, result, run_dir)
                rows.append(row)
                print(
                    f"[{len(rows)}/{total_runs}] "
                    f"{spec.name} | {training_config_slug(training_config)} | seed={seed}"
                )
                write_batch_outputs(output_dir, rows, batch_metadata)

    if pending_specs:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_spec = {executor.submit(execute_spec_subprocess, spec): spec for spec in pending_specs}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                row = future.result()
                rows.append(row)
                print(
                    f"[{len(rows)}/{total_runs}] "
                    f"{spec.name} | {training_config_slug(spec.training_config)} | seed={spec.training_config.seed}"
                )
                write_batch_outputs(output_dir, rows, batch_metadata)

    print(f"Completed runs: {len(rows)}")
    print(f"Wrote batch summary to: {output_dir}")


if __name__ == "__main__":
    main()
