from __future__ import annotations

from environment_wrapper import StandardGridConfig, make_standard_grid_env
from td_learning import TrainingConfig, summarize_training, train_sarsa


if __name__ == "__main__":
    # Use the same simple setup as Q-learning so the two algorithms are easy to compare.
    env_config = StandardGridConfig(
        start_state=(0, 4),
        transition_prob=1.0,
        wind=False,
        max_steps=100,
    )
    train_config = TrainingConfig(
        episodes=300,
        alpha=0.1,
        gamma=0.9,
        exploration_strategy="epsilon_greedy",
        epsilon=0.1,
        seed=0,
    )

    env = make_standard_grid_env(env_config)
    result = train_sarsa(env=env, config=train_config)

    print("Basic SARSA run complete.")
    summarize_training(result)
