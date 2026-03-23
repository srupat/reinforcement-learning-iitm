from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GRID_WORLD_DIR = Path(__file__).resolve().parent / "Grid-World-Environment"
# Import the provided environment code without modifying the instructor files.
if str(GRID_WORLD_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_WORLD_DIR))

from env import create_four_room, create_standard_grid


@dataclass(frozen=True)
class StandardGridConfig:
    start_state: tuple[int, int] = (0, 4)
    transition_prob: float = 1.0
    wind: bool = False
    max_steps: int = 100


@dataclass(frozen=True)
class FourRoomConfig:
    start_state: tuple[int, int] = (8, 0)
    goal_change: bool = True
    transition_prob: float = 1.0
    max_steps: int = 100


class AssignmentGridWorld:
    """Thin wrapper that gives the provided grid world a training-friendly API."""

    def __init__(self, grid_world, max_steps: int = 100):
        self.grid_world = grid_world
        self.max_steps = max_steps
        self.steps_taken = 0

        self.n_states = int(grid_world.num_states)
        self.n_actions = int(grid_world.num_actions)
        self.start_state = int(np.asarray(grid_world.start_state_seq).reshape(-1)[0])
        self.goal_states = set()
        self._refresh_goal_states()

    def _refresh_goal_states(self) -> None:
        self.start_state = int(np.asarray(self.grid_world.start_state_seq).reshape(-1)[0])
        self.goal_states = {
            int(state) for state in np.asarray(self.grid_world.goal_states_seq).flatten()
        }

    def reset(self) -> int:
        self.steps_taken = 0
        if getattr(self.grid_world, "goal_change", False) and getattr(self.grid_world, "env", "") == "four_room":
            # Preserve the dynamic-goal behavior, but return a plain integer state.
            self.grid_world.goal_states = self.grid_world._random_goal_state()
            self.grid_world.create_gridworld()
            self._refresh_goal_states()
        return self.start_state

    def step(self, state: int, action: int) -> tuple[int, float, bool]:
        next_state, reward = self.grid_world.step(int(state), int(action))
        self.steps_taken += 1

        next_state = int(next_state)
        reward_value = float(np.asarray(reward).reshape(-1)[0])
        done = next_state in self.goal_states or self.steps_taken >= self.max_steps
        return next_state, reward_value, done


def make_standard_grid_env(config: StandardGridConfig | None = None) -> AssignmentGridWorld:
    config = config or StandardGridConfig()
    start_state = np.array([[config.start_state[0], config.start_state[1]]], dtype=int)
    grid_world = create_standard_grid(
        start_state=start_state,
        transition_prob=config.transition_prob,
        wind=config.wind,
    )
    return AssignmentGridWorld(grid_world=grid_world, max_steps=config.max_steps)


def make_four_room_env(config: FourRoomConfig | None = None) -> AssignmentGridWorld:
    config = config or FourRoomConfig()
    start_state = np.array([[config.start_state[0], config.start_state[1]]], dtype=int)
    grid_world = create_four_room(
        start_state=start_state,
        goal_change=config.goal_change,
        transition_prob=config.transition_prob,
    )
    return AssignmentGridWorld(grid_world=grid_world, max_steps=config.max_steps)
