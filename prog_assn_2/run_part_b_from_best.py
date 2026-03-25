from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from experiment_configs import get_experiment_spec, list_experiment_specs
from run_batch_experiments import batch_output_dir, execute_spec, execute_spec_subprocess, row_from_existing_run, write_batch_outputs
from run_experiment import RESULTS_DIR, result_exists, training_config_slug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Part B using the best hyperparameters selected from a completed Part A batch."
    )
    parser.add_argument(
        "--source-batch",
        type=str,
        default="part_a_full",
        help="Batch folder name under results/batches containing the completed Part A summary.csv.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated experiment names, or 'all' for the full assignment set.",
    )
    parser.add_argument("--episodes", type=int, default=300, help="Number of episodes per Part B run.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel worker processes to use.")
    parser.add_argument("--num-seeds", type=int, default=100, help="Number of seeds to run per best configuration.")
    parser.add_argument("--seed-offset", type=int, default=0, help="Starting seed value for the Part B range.")
    parser.add_argument(
        "--batch-name",
        type=str,
        default="part_b_full",
        help="Folder name for the Part B batch summary output.",
    )
    return parser.parse_args()


def resolve_experiments(raw: str) -> list[str]:
    if raw == "all":
        return list_experiment_specs()
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_batch_rows(source_batch: str) -> list[dict]:
    summary_path = RESULTS_DIR / "batches" / source_batch / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find Part A summary file: {summary_path}")

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def best_rows_by_experiment(rows: list[dict], experiments: list[str]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["experiment"] not in experiments:
            continue
        run_label = (
            f"episodes_{row['episodes'].replace('-', 'm').replace('.', 'p')}"
            f"__alpha_{row['alpha'].replace('-', 'm').replace('.', 'p')}"
            f"__gamma_{row['gamma'].replace('-', 'm').replace('.', 'p')}"
            f"__exploration_{row['exploration_strategy']}"
        )
        if row["exploration_strategy"] == "epsilon_greedy":
            run_label += f"__epsilon_{row['epsilon'].replace('-', 'm').replace('.', 'p')}"
        else:
            run_label += f"__temperature_{row['temperature'].replace('-', 'm').replace('.', 'p')}"
        grouped[(row["experiment"], run_label)].append(row)

    winners: list[dict] = []
    per_experiment: dict[str, list[dict]] = defaultdict(list)
    for (experiment, run_label), items in grouped.items():
        per_experiment[experiment].append(
            {
                "experiment": experiment,
                "run_label": run_label,
                "episodes": int(items[0]["episodes"]),
                "alpha": float(items[0]["alpha"]),
                "gamma": float(items[0]["gamma"]),
                "exploration_strategy": items[0]["exploration_strategy"],
                "epsilon": float(items[0]["epsilon"]),
                "temperature": float(items[0]["temperature"]),
                "num_seeds": len(items),
                "mean_reward": sum(float(item["mean_reward"]) for item in items) / len(items),
                "mean_steps": sum(float(item["mean_steps"]) for item in items) / len(items),
                "success_rate": sum(float(item["success_rate"]) for item in items) / len(items),
            }
        )

    for experiment in experiments:
        candidates = per_experiment.get(experiment, [])
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda row: (-row["mean_reward"], row["mean_steps"], -row["success_rate"]),
        )[0]
        winners.append(best)

    return winners


def write_best_configs(output_dir: Path, winners: list[dict]) -> None:
    with (output_dir / "best_configs.json").open("w", encoding="utf-8") as handle:
        json.dump(winners, handle, indent=2)

    if not winners:
        return

    fieldnames = [
        "experiment",
        "run_label",
        "episodes",
        "alpha",
        "gamma",
        "exploration_strategy",
        "epsilon",
        "temperature",
        "num_seeds",
        "mean_reward",
        "mean_steps",
        "success_rate",
    ]
    with (output_dir / "best_configs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(winners)


def main() -> None:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1.")

    experiments = resolve_experiments(args.experiments)
    rows = load_batch_rows(args.source_batch)
    winners = best_rows_by_experiment(rows, experiments)
    if not winners:
        raise ValueError("No best configurations were found for the requested experiments.")

    output_dir = batch_output_dir(args.batch_name)
    write_best_configs(output_dir, winners)

    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))
    total_runs = len(winners) * len(seeds)
    batch_metadata = {
        "source_batch": args.source_batch,
        "experiments": experiments,
        "episodes": args.episodes,
        "jobs": args.jobs,
        "num_seeds": args.num_seeds,
        "seed_offset": args.seed_offset,
        "total_runs": total_runs,
        "best_configs": winners,
    }

    print(f"Selected best configs for {len(winners)} experiments.")
    print(f"Planned runs: {total_runs}")
    print(f"Batch summary directory: {output_dir}")
    print(f"Worker processes: {args.jobs}")

    rows_out: list[dict] = []
    pending_specs = []

    for winner in winners:
        base_spec = get_experiment_spec(winner["experiment"])
        for seed in seeds:
            training_config = replace(
                base_spec.training_config,
                episodes=args.episodes,
                seed=seed,
                alpha=winner["alpha"],
                gamma=winner["gamma"],
                exploration_strategy=winner["exploration_strategy"],
                epsilon=winner["epsilon"],
                temperature=winner["temperature"],
            )
            spec = replace(base_spec, training_config=training_config)

            if result_exists(spec):
                rows_out.append(row_from_existing_run(spec))
                print(
                    f"[skip {len(rows_out)}/{total_runs}] "
                    f"{spec.name} | {training_config_slug(training_config)} | seed={seed}"
                )
                write_batch_outputs(output_dir, rows_out, batch_metadata)
                continue

            if args.jobs > 1:
                pending_specs.append(spec)
                continue

            row = execute_spec(spec)
            rows_out.append(row)
            print(
                f"[{len(rows_out)}/{total_runs}] "
                f"{spec.name} | {training_config_slug(training_config)} | seed={seed}"
            )
            write_batch_outputs(output_dir, rows_out, batch_metadata)

    if pending_specs:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_spec = {executor.submit(execute_spec_subprocess, spec): spec for spec in pending_specs}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                row = future.result()
                rows_out.append(row)
                print(
                    f"[{len(rows_out)}/{total_runs}] "
                    f"{spec.name} | {training_config_slug(spec.training_config)} | seed={spec.training_config.seed}"
                )
                write_batch_outputs(output_dir, rows_out, batch_metadata)

    print(f"Completed runs: {len(rows_out)}")
    print(f"Wrote batch summary to: {output_dir}")


if __name__ == "__main__":
    main()
