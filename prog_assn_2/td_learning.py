from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 300
    alpha: float = 0.1
    gamma: float = 0.9
    seed: int = 0
    exploration_strategy: str = "epsilon_greedy"
    epsilon: float = 0.1
    temperature: float = 1.0


@dataclass
class TrainingResult:
    q_values: np.ndarray
    episode_rewards: np.ndarray
    episode_steps: np.ndarray
    episode_successes: np.ndarray
    state_visit_counts: np.ndarray


def epsilon_greedy_action(
    q_values: np.ndarray,
    state: int,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_values.shape[1]))

    state_values = q_values[state]
    max_value = np.max(state_values)
    # Random tie-breaking avoids always selecting the first max-valued action.
    greedy_actions = np.flatnonzero(np.isclose(state_values, max_value))
    return int(rng.choice(greedy_actions))


def softmax_action(
    q_values: np.ndarray,
    state: int,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0:
        raise ValueError("Softmax temperature must be positive.")

    preferences = q_values[state] - np.max(q_values[state])
    probabilities = np.exp(preferences / temperature)
    probabilities /= probabilities.sum()
    return int(rng.choice(q_values.shape[1], p=probabilities))


def select_action(
    q_values: np.ndarray,
    state: int,
    config: TrainingConfig,
    rng: np.random.Generator,
) -> int:
    if config.exploration_strategy == "epsilon_greedy":
        return epsilon_greedy_action(q_values, state, config.epsilon, rng)
    if config.exploration_strategy == "softmax":
        return softmax_action(q_values, state, config.temperature, rng)
    raise ValueError(f"Unsupported exploration strategy: {config.exploration_strategy}")


def _initialize_training(
    env,
    config: TrainingConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.random.Generator]:
    q_values = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_rewards = np.zeros(config.episodes, dtype=float)
    episode_steps = np.zeros(config.episodes, dtype=int)
    episode_successes = np.zeros(config.episodes, dtype=bool)
    state_visit_counts = np.zeros(env.n_states, dtype=int)
    rng = np.random.default_rng(config.seed)
    return q_values, episode_rewards, episode_steps, episode_successes, state_visit_counts, rng


def train_q_learning(
    env,
    config: TrainingConfig,
    episode_logger: Callable[[dict], None] | None = None,
) -> TrainingResult:
    q_values, episode_rewards, episode_steps, episode_successes, state_visit_counts, rng = _initialize_training(
        env,
        config,
    )

    for episode in range(config.episodes):
        state = env.reset()
        state_visit_counts[state] += 1
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = select_action(q_values, state, config, rng)
            next_state, reward, done = env.step(state, action)
            state_visit_counts[next_state] += 1

            td_target = reward
            if not done:
                # Q-learning bootstraps from the greedy value at the next state.
                td_target += config.gamma * np.max(q_values[next_state])

            td_error = td_target - q_values[state, action]
            q_values[state, action] += config.alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards[episode] = total_reward
        episode_steps[episode] = steps
        episode_successes[episode] = state in env.goal_states
        if episode_logger is not None:
            episode_logger(
                {
                    "episode": episode + 1,
                    "episode_reward": total_reward,
                    "episode_steps": steps,
                    "episode_success": float(episode_successes[episode]),
                }
            )

    return TrainingResult(
        q_values=q_values,
        episode_rewards=episode_rewards,
        episode_steps=episode_steps,
        episode_successes=episode_successes,
        state_visit_counts=state_visit_counts,
    )


def train_sarsa(
    env,
    config: TrainingConfig,
    episode_logger: Callable[[dict], None] | None = None,
) -> TrainingResult:
    q_values, episode_rewards, episode_steps, episode_successes, state_visit_counts, rng = _initialize_training(
        env,
        config,
    )

    for episode in range(config.episodes):
        state = env.reset()
        state_visit_counts[state] += 1
        action = select_action(q_values, state, config, rng)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            next_state, reward, done = env.step(state, action)
            state_visit_counts[next_state] += 1

            td_target = reward
            if not done:
                next_action = select_action(q_values, next_state, config, rng)
                # SARSA bootstraps from the next action chosen by the behavior policy.
                td_target += config.gamma * q_values[next_state, next_action]
            else:
                next_action = None

            td_error = td_target - q_values[state, action]
            q_values[state, action] += config.alpha * td_error

            state = next_state
            if next_action is not None:
                action = next_action
            total_reward += reward
            steps += 1

        episode_rewards[episode] = total_reward
        episode_steps[episode] = steps
        episode_successes[episode] = state in env.goal_states
        if episode_logger is not None:
            episode_logger(
                {
                    "episode": episode + 1,
                    "episode_reward": total_reward,
                    "episode_steps": steps,
                    "episode_success": float(episode_successes[episode]),
                }
            )

    return TrainingResult(
        q_values=q_values,
        episode_rewards=episode_rewards,
        episode_steps=episode_steps,
        episode_successes=episode_successes,
        state_visit_counts=state_visit_counts,
    )


def summarize_training(result: TrainingResult) -> None:
    first_window = min(20, len(result.episode_rewards))
    last_window = min(20, len(result.episode_rewards))

    print(f"Episodes: {len(result.episode_rewards)}")
    print(
        f"Average reward over first {first_window} episodes: "
        f"{result.episode_rewards[:first_window].mean():.2f}"
    )
    print(
        f"Average reward over last {last_window} episodes: "
        f"{result.episode_rewards[-last_window:].mean():.2f}"
    )
    print(
        f"Average steps over first {first_window} episodes: "
        f"{result.episode_steps[:first_window].mean():.2f}"
    )
    print(
        f"Average steps over last {last_window} episodes: "
        f"{result.episode_steps[-last_window:].mean():.2f}"
    )
