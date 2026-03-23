from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import wandb

from td_learning import TrainingConfig


DEFAULT_WANDB_PROJECT = "rl-programming-assignment-2"
WANDB_TEMP_DIR = Path(__file__).resolve().parent / "wandb_tmp"


def training_config_from_dict(data: dict) -> TrainingConfig:
    return TrainingConfig(
        episodes=int(data["episodes"]),
        alpha=float(data["alpha"]),
        gamma=float(data["gamma"]),
        seed=int(data["seed"]),
        exploration_strategy=str(data["exploration_strategy"]),
        epsilon=float(data["epsilon"]),
        temperature=float(data["temperature"]),
    )


def init_wandb_run(
    *,
    project: str,
    entity: str | None,
    mode: str,
    job_type: str,
    run_name: str,
    group: str | None,
    tags: list[str],
    config: dict,
):
    WANDB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(WANDB_TEMP_DIR)
    os.environ["TEMP"] = str(WANDB_TEMP_DIR)
    tempfile.tempdir = str(WANDB_TEMP_DIR)
    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        job_type=job_type,
        name=run_name,
        group=group,
        tags=tags,
        config=config,
        reinit="finish_previous",
    )


def log_result_summary(run, result) -> None:
    run.summary["mean_reward"] = float(result.episode_rewards.mean())
    run.summary["mean_steps"] = float(result.episode_steps.mean())
    run.summary["success_rate"] = float(result.episode_successes.mean())


def curve_table(episodes: np.ndarray, values: np.ndarray, y_name: str) -> wandb.Table:
    table = wandb.Table(columns=["episode", y_name])
    for episode, value in zip(episodes.tolist(), values.tolist()):
        table.add_data(int(episode), float(value))
    return table


def tuning_table(rows: list[dict]) -> wandb.Table:
    columns = [
        "experiment",
        "run_label",
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
    table = wandb.Table(columns=columns)
    for row in rows:
        table.add_data(
            row["experiment"],
            row["run_label"],
            float(row["alpha"]),
            float(row["gamma"]),
            row["exploration_strategy"],
            float(row["epsilon"]),
            float(row["temperature"]),
            int(row["num_seeds"]),
            float(row["mean_reward"]),
            float(row["mean_steps"]),
            float(row["success_rate"]),
        )
    return table


def heatmap_image(
    grid: np.ndarray,
    *,
    title: str,
    cmap: str,
    overlay_text: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(grid, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(grid.shape[1]))
    ax.set_yticks(range(grid.shape[0]))
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if overlay_text is not None:
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                if overlay_text[row, col]:
                    ax.text(col, row, overlay_text[row, col], ha="center", va="center", color="black")

    fig.tight_layout()
    logged = wandb.Image(fig)
    plt.close(fig)
    return logged


def policy_heatmap_image(
    value_grid: np.ndarray,
    policy_grid: np.ndarray,
    *,
    title: str,
    blocked_cells: set[tuple[int, int]] | None = None,
    goal_cells: set[tuple[int, int]] | None = None,
):
    arrows = {0: "U", 1: "D", 2: "L", 3: "R"}
    overlay = np.full(value_grid.shape, "", dtype=object)
    blocked_cells = blocked_cells or set()
    goal_cells = goal_cells or set()

    for row in range(value_grid.shape[0]):
        for col in range(value_grid.shape[1]):
            cell = (row, col)
            if cell in blocked_cells:
                overlay[row, col] = "X"
            elif cell in goal_cells:
                overlay[row, col] = "G"
            else:
                overlay[row, col] = arrows.get(int(policy_grid[row, col]), "")

    return heatmap_image(value_grid, title=title, cmap="viridis", overlay_text=overlay)


def spec_to_wandb_config(spec) -> dict:
    return {
        "experiment_name": spec.name,
        "environment_name": spec.env_name,
        "algorithm": spec.algorithm,
        "env_config": asdict(spec.env_config),
        "training_config": asdict(spec.training_config),
    }
