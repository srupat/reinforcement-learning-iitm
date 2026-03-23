from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from environment_wrapper import FourRoomConfig, StandardGridConfig
from td_learning import TrainingConfig


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    env_name: str
    algorithm: str
    env_config: StandardGridConfig | FourRoomConfig
    training_config: TrainingConfig


def _base_training_config() -> TrainingConfig:
    return TrainingConfig(
        episodes=300,
        alpha=0.1,
        gamma=0.9,
        seed=0,
        exploration_strategy="epsilon_greedy",
        epsilon=0.1,
        temperature=1.0,
    )


def _coord_label(state: tuple[int, int]) -> str:
    return f"start{state[0]}_{state[1]}"


def _float_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _standard_q_learning_specs() -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {}
    for transition_prob in (0.7, 1.0):
        for start_state in ((0, 4), (3, 6)):
            for exploration_strategy in ("epsilon_greedy", "softmax"):
                name = (
                    "standard_q_learning_"
                    f"tp{_float_label(transition_prob)}_"
                    f"{_coord_label(start_state)}_"
                    f"{exploration_strategy}"
                )
                specs[name] = ExperimentSpec(
                    name=name,
                    env_name="standard_grid",
                    algorithm="q_learning",
                    env_config=StandardGridConfig(
                        start_state=start_state,
                        transition_prob=transition_prob,
                        wind=False,
                        max_steps=100,
                    ),
                    training_config=replace(
                        _base_training_config(),
                        exploration_strategy=exploration_strategy,
                    ),
                )
    return specs


def _standard_sarsa_specs() -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {}
    for wind in (False, True):
        for start_state in ((0, 4), (3, 6)):
            for exploration_strategy in ("epsilon_greedy", "softmax"):
                name = (
                    "standard_sarsa_"
                    f"wind_{str(wind).lower()}_"
                    f"{_coord_label(start_state)}_"
                    f"{exploration_strategy}"
                )
                specs[name] = ExperimentSpec(
                    name=name,
                    env_name="standard_grid",
                    algorithm="sarsa",
                    env_config=StandardGridConfig(
                        start_state=start_state,
                        transition_prob=1.0,
                        wind=wind,
                        max_steps=100,
                    ),
                    training_config=replace(
                        _base_training_config(),
                        exploration_strategy=exploration_strategy,
                    ),
                )
    return specs


def _four_room_specs() -> dict[str, ExperimentSpec]:
    specs: dict[str, ExperimentSpec] = {}
    for algorithm in ("q_learning", "sarsa"):
        for goal_change in (False, True):
            name = f"four_room_{algorithm}_goal_change_{str(goal_change).lower()}"
            specs[name] = ExperimentSpec(
                name=name,
                env_name="four_room",
                algorithm=algorithm,
                env_config=FourRoomConfig(
                    start_state=(8, 0),
                    goal_change=goal_change,
                    transition_prob=1.0,
                    max_steps=100,
                ),
                training_config=_base_training_config(),
            )
    return specs


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    **_standard_q_learning_specs(),
    **_standard_sarsa_specs(),
    **_four_room_specs(),
}


def get_experiment_spec(name: str) -> ExperimentSpec:
    if name not in EXPERIMENT_SPECS:
        available = ", ".join(sorted(EXPERIMENT_SPECS))
        raise KeyError(f"Unknown experiment '{name}'. Available experiments: {available}")
    return EXPERIMENT_SPECS[name]


def list_experiment_specs() -> list[str]:
    return sorted(EXPERIMENT_SPECS)


def experiment_to_dict(spec: ExperimentSpec) -> dict:
    return {
        "name": spec.name,
        "env_name": spec.env_name,
        "algorithm": spec.algorithm,
        "env_config": asdict(spec.env_config),
        "training_config": asdict(spec.training_config),
    }
