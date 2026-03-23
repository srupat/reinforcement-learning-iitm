from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import wandb

from experiment_configs import get_experiment_spec, list_experiment_specs
from run_experiment import RESULTS_DIR, build_env, training_config_slug
from wandb_utils import (
    DEFAULT_WANDB_PROJECT,
    curve_table,
    heatmap_image,
    init_wandb_run,
    policy_heatmap_image,
    spec_to_wandb_config,
    training_config_from_dict,
    tuning_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Log assignment tuning plots and final analysis to Weights & Biases.")
    parser.add_argument(
        "--report-type",
        type=str,
        required=True,
        choices=["tuning", "final"],
        help="Whether to log hyperparameter tuning views or final averaged plots.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names, or 'all' for the full assignment set.",
    )
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
        "--run-label",
        type=str,
        default="best",
        help="For final reports, use 'best' or pass an explicit run label.",
    )
    return parser.parse_args()


def resolve_experiments(raw: str) -> list[str]:
    if raw == "all":
        return list_experiment_specs()
    return [item.strip() for item in raw.split(",") if item.strip()]


def iter_result_records(experiments: list[str]) -> list[dict]:
    records: list[dict] = []
    for experiment_name in experiments:
        experiment_dir = RESULTS_DIR / experiment_name
        if not experiment_dir.exists():
            continue
        for metadata_path in experiment_dir.rglob("metadata.json"):
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            metrics_path = metadata_path.with_name("metrics.npz")
            if not metrics_path.exists():
                continue
            training_config = training_config_from_dict(metadata["training_config"])
            records.append(
                {
                    "experiment": experiment_name,
                    "metadata": metadata,
                    "metrics_path": metrics_path,
                    "run_dir": metadata_path.parent,
                    "run_label": training_config_slug(training_config),
                    "training_config": training_config,
                }
            )
    return records


def grouped_tuning_rows(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["experiment"], record["run_label"])].append(record)

    rows: list[dict] = []
    for (experiment, run_label), items in grouped.items():
        rewards = [item["metadata"]["summary"]["mean_reward"] for item in items]
        steps = [item["metadata"]["summary"]["mean_steps"] for item in items]
        success = [item["metadata"]["summary"]["success_rate"] for item in items]
        config = items[0]["training_config"]
        rows.append(
            {
                "experiment": experiment,
                "run_label": run_label,
                "alpha": config.alpha,
                "gamma": config.gamma,
                "exploration_strategy": config.exploration_strategy,
                "epsilon": config.epsilon,
                "temperature": config.temperature,
                "num_seeds": len(items),
                "mean_reward": float(np.mean(rewards)),
                "mean_steps": float(np.mean(steps)),
                "success_rate": float(np.mean(success)),
            }
        )
    rows.sort(key=lambda row: (row["experiment"], -row["mean_reward"], row["mean_steps"]))
    return rows


def log_tuning_report(experiments: list[str], args: argparse.Namespace) -> None:
    records = iter_result_records(experiments)
    rows = grouped_tuning_rows(records)
    run = init_wandb_run(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        job_type="analysis",
        run_name=f"tuning_report__{'all' if args.experiments == 'all' else 'selected'}",
        group="assignment_reports",
        tags=["tuning", "analysis"],
        config={"experiments": experiments, "report_type": "tuning"},
    )

    overall_table = tuning_table(rows)
    run.log({"tuning/overall_table": overall_table})

    for experiment in experiments:
        experiment_rows = [row for row in rows if row["experiment"] == experiment]
        if not experiment_rows:
            continue

        table = tuning_table(experiment_rows)
        run.log({f"tuning/{experiment}/table": table})
        run.log(
            {
                f"tuning/{experiment}/reward_vs_steps": wandb.plot.scatter(
                    table,
                    "mean_steps",
                    "mean_reward",
                    title=f"{experiment}: reward vs steps",
                )
            }
        )

    run.summary["num_experiments"] = len(experiments)
    run.summary["num_rows"] = len(rows)
    run.finish()


def average_metrics(records: list[dict]) -> dict:
    reward_curves = []
    step_curves = []
    success_curves = []
    visit_counts = []
    q_values = []

    for record in records:
        data = np.load(record["metrics_path"])
        reward_curves.append(data["episode_rewards"])
        step_curves.append(data["episode_steps"])
        success_curves.append(data["episode_successes"].astype(float))
        visit_counts.append(data["state_visit_counts"])
        q_values.append(data["q_values"])

    return {
        "avg_reward_curve": np.mean(np.stack(reward_curves), axis=0),
        "avg_step_curve": np.mean(np.stack(step_curves), axis=0),
        "avg_success_curve": np.mean(np.stack(success_curves), axis=0),
        "avg_visit_counts": np.mean(np.stack(visit_counts), axis=0),
        "avg_q_values": np.mean(np.stack(q_values), axis=0),
    }


def best_run_label(records: list[dict]) -> str:
    rows = grouped_tuning_rows(records)
    if not rows:
        raise ValueError("No saved runs were found for this experiment.")
    best = sorted(rows, key=lambda row: (-row["mean_reward"], row["mean_steps"], -row["success_rate"]))[0]
    return best["run_label"]


def log_final_report(experiments: list[str], args: argparse.Namespace) -> None:
    all_records = iter_result_records(experiments)
    for experiment in experiments:
        records = [record for record in all_records if record["experiment"] == experiment]
        if not records:
            continue

        chosen_run_label = args.run_label if args.run_label != "best" else best_run_label(records)
        selected_records = [record for record in records if record["run_label"] == chosen_run_label]
        if not selected_records:
            continue

        spec = get_experiment_spec(experiment)
        training_config = selected_records[0]["training_config"]
        spec = replace(spec, training_config=training_config)
        env = build_env(spec)
        aggregated = average_metrics(selected_records)

        num_rows = env.grid_world.num_rows
        num_cols = env.grid_world.num_cols
        visit_grid = aggregated["avg_visit_counts"].reshape(num_rows, num_cols)
        max_q_grid = aggregated["avg_q_values"].max(axis=1).reshape(num_rows, num_cols)
        policy_grid = aggregated["avg_q_values"].argmax(axis=1).reshape(num_rows, num_cols)
        blocked_cells = set()
        if env.grid_world.obs_states is not None:
            blocked_cells = {tuple(cell) for cell in env.grid_world.obs_states.tolist()}
        goal_cells = {tuple(cell) for cell in env.grid_world.goal_states.tolist()}

        run = init_wandb_run(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            job_type="analysis",
            run_name=f"final_report__{experiment}__{chosen_run_label}",
            group="assignment_reports",
            tags=["final", "analysis", spec.algorithm, spec.env_name],
            config={
                "report_type": "final",
                "experiment_name": experiment,
                "selected_run_label": chosen_run_label,
                "num_runs": len(selected_records),
                **spec_to_wandb_config(spec),
            },
        )

        episodes = np.arange(1, len(aggregated["avg_reward_curve"]) + 1)
        reward_table = curve_table(episodes, aggregated["avg_reward_curve"], "average_reward")
        step_table = curve_table(episodes, aggregated["avg_step_curve"], "average_steps")
        success_table = curve_table(episodes, aggregated["avg_success_curve"], "average_success")

        run.log(
            {
                "final/average_reward_curve": wandb.plot.line(
                    reward_table, "episode", "average_reward", title=f"{experiment}: average reward per episode"
                ),
                "final/average_steps_curve": wandb.plot.line(
                    step_table, "episode", "average_steps", title=f"{experiment}: average steps per episode"
                ),
                "final/average_success_curve": wandb.plot.line(
                    success_table, "episode", "average_success", title=f"{experiment}: success rate per episode"
                ),
                "final/state_visit_heatmap": heatmap_image(
                    visit_grid,
                    title=f"{experiment}: average state visits",
                    cmap="magma",
                ),
                "final/q_value_policy_heatmap": policy_heatmap_image(
                    max_q_grid,
                    policy_grid,
                    title=f"{experiment}: average max Q-value and policy",
                    blocked_cells=blocked_cells,
                    goal_cells=goal_cells,
                ),
            }
        )

        run.summary["num_runs"] = len(selected_records)
        run.summary["selected_run_label"] = chosen_run_label
        run.summary["mean_reward"] = float(np.mean(aggregated["avg_reward_curve"]))
        run.summary["mean_steps"] = float(np.mean(aggregated["avg_step_curve"]))
        run.summary["final_success_rate"] = float(np.mean(aggregated["avg_success_curve"]))
        run.finish()


def main() -> None:
    args = parse_args()
    experiments = resolve_experiments(args.experiments)

    if args.report_type == "tuning":
        log_tuning_report(experiments, args)
        return
    log_final_report(experiments, args)


if __name__ == "__main__":
    main()
