"""Research-oriented dashboard views built from existing result objects."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import streamlit as st

from ..types import BenchmarkResult, EpisodeResult, SweepResult, to_serializable
from .components import build_actions_figure, build_entropy_figure, build_hidden_state_figure
from .formatting import action_label, format_float, format_percent, observation_label
from .i18n import t


_NOT_AVAILABLE = "Not available"


def render_research(result, config, model) -> None:
    """Render research analysis using only existing result/config/model objects."""

    _render_experiment_overview(result, config, model)
    st.divider()

    if isinstance(result, EpisodeResult):
        _render_episode_analysis(result)
    elif isinstance(result, (BenchmarkResult, SweepResult)):
        st.info("Timeline is available for episode-level results only.")
        _render_result_summary(result)
    else:
        st.info(_NOT_AVAILABLE)

    st.divider()
    _render_technical_details(config, model)


def _render_experiment_overview(result: object, config: object, model: object) -> None:
    """Render run metadata and model dimensions."""

    st.header("Experiment Overview")
    mode = _result_mode(result)
    simulation = getattr(config, "simulation", None)
    model_config = getattr(getattr(config, "env", None), "model", None)
    episodes = getattr(simulation, "episodes", None)
    episode_length = getattr(simulation, "episode_length", None)
    seed = getattr(simulation, "seed", None)

    overview = [
        ("Simulation mode", mode),
        (
            "Number of steps",
            _format_value(
                _episode_step_count(result)
                if isinstance(result, EpisodeResult)
                else _product_or_unavailable(episodes, episode_length)
            ),
        ),
        ("Number of episodes", _format_value(episodes)),
        ("Model states", _dimension(model, model_config, "num_states", 1)),
        ("Model observations", _dimension(model, model_config, "num_observations", 0)),
        ("Model actions", _dimension(model, model_config, "num_actions", 0)),
        ("Random seed", _format_value(seed)),
        ("Execution time", _NOT_AVAILABLE),
    ]
    columns = st.columns(4)
    for index, (label, value) in enumerate(overview):
        with columns[index % len(columns)]:
            st.metric(label, value)

    st.subheader("Current Configuration")
    if config is None:
        st.write(_NOT_AVAILABLE)
    else:
        st.json(to_serializable(config))


def _render_episode_analysis(episode: EpisodeResult) -> None:
    """Render belief, reward, policy, and timestep analysis for one episode."""

    st.header("Belief State Analysis")
    if episode.beliefs:
        chart_columns = st.columns(3)
        with chart_columns[0]:
            st.pyplot(build_hidden_state_figure(episode), width="stretch")
        with chart_columns[1]:
            st.pyplot(build_entropy_figure(episode), width="stretch")
        with chart_columns[2]:
            confidence = [float(max(belief)) for belief in episode.beliefs]
            st.line_chart({"Confidence": confidence}, width="stretch")

        entropy_values = [float(value) for value in episode.posterior_entropies]
        uncertainty_reduction = (
            entropy_values[0] - entropy_values[-1] if entropy_values else None
        )
        metric_columns = st.columns(2)
        with metric_columns[0]:
            st.metric("Initial entropy", _format_value(entropy_values[0] if entropy_values else None))
        with metric_columns[1]:
            st.metric("Uncertainty reduction", _format_value(uncertainty_reduction))
    else:
        st.info(_NOT_AVAILABLE)

    st.divider()
    _render_reward_analysis(episode)
    st.divider()
    _render_policy_analysis(episode)
    st.divider()
    _render_timeline(episode)


def _render_reward_analysis(episode: EpisodeResult) -> None:
    """Render reward and cumulative reward from the episode trajectory."""

    st.header("Reward Analysis")
    rewards = [float(value) for value in episode.rewards]
    if not rewards:
        st.info(_NOT_AVAILABLE)
        return

    cumulative = np.cumsum(rewards).tolist()
    st.line_chart({"Reward": rewards, "Cumulative reward": cumulative}, width="stretch")
    columns = st.columns(2)
    with columns[0]:
        st.metric("Total reward", format_float(float(episode.total_reward)))
    with columns[1]:
        st.metric("Average reward", format_float(float(np.mean(rewards))))


def _render_policy_analysis(episode: EpisodeResult) -> None:
    """Render selected actions, frequencies, and average action probabilities."""

    st.header("Policy / Decision Analysis")
    if not episode.actions:
        st.info(_NOT_AVAILABLE)
        return

    action_counts = Counter(int(action) for action in episode.actions)
    frequency_data = {
        action_label(action): count for action, count in sorted(action_counts.items())
    }
    st.subheader("Action frequency")
    st.bar_chart(frequency_data, width="stretch")

    st.subheader("Selected actions")
    st.write(" → ".join(action_label(int(action)) for action in episode.actions))

    if episode.action_probabilities:
        probabilities = np.asarray(episode.action_probabilities, dtype=np.float64)
        averages = np.mean(probabilities, axis=0)
        st.subheader("Average action probabilities")
        st.bar_chart(
            {action_label(index): float(value) for index, value in enumerate(averages)},
            width="stretch",
        )
    else:
        st.info(_NOT_AVAILABLE)


def _render_timeline(episode: EpisodeResult) -> None:
    """Render an interactive inspection of one existing episode timestep."""

    st.header("Simulation Timeline")
    step_count = _episode_step_count(episode)
    if step_count == 0:
        st.info(_NOT_AVAILABLE)
        return

    step_index = st.slider("Step", min_value=0, max_value=step_count - 1, value=0, step=1)
    columns = st.columns(4)
    with columns[0]:
        st.metric("Step", str(step_index))
    with columns[1]:
        st.metric("Observation", _observation_at(episode, step_index))
    with columns[2]:
        st.metric("Action", _action_at(episode, step_index))
    with columns[3]:
        st.metric("Reward", _format_value(_value_at(episode.rewards, step_index)))

    belief = _value_at(episode.beliefs, step_index)
    if belief is not None:
        st.write("Belief state")
        st.bar_chart(
            {f"State {index}": float(value) for index, value in enumerate(belief)},
            width="stretch",
        )
    else:
        st.info(_NOT_AVAILABLE)


def _render_result_summary(result: BenchmarkResult | SweepResult) -> None:
    """Render aggregate result metrics when episode trajectories are unavailable."""

    st.header("Reward Analysis")
    if isinstance(result, BenchmarkResult):
        summaries = result.summaries
        if not summaries:
            st.info(_NOT_AVAILABLE)
            return
        st.dataframe(
            [
                {
                    "Agent": summary.agent_name,
                    "Mean reward": float(summary.mean_reward),
                    "Success rate": format_percent(summary.success_rate),
                    "Mean entropy": float(summary.mean_posterior_entropy),
                    "Information gain": float(summary.mean_information_gain),
                }
                for summary in summaries
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        if not result.mean_reward:
            st.info(_NOT_AVAILABLE)
            return
        st.line_chart(
            {
                agent_name: [
                    float(values[index])
                    for values in result.mean_reward
                ]
                for index, agent_name in enumerate(result.agent_names)
            },
            width="stretch",
        )


def _render_technical_details(config: object, model: object) -> None:
    """Render technical configuration and available model dimensions."""

    with st.expander(t("research_technical_details")):
        if config is None:
            st.write(_NOT_AVAILABLE)
            return
        environment = getattr(config, "env", None)
        st.json(
            {
                "ModelConfig": to_serializable(getattr(environment, "model", {})),
                "Environment": to_serializable(environment),
                "Inference": to_serializable(getattr(config, "inference", {})),
                "ActionSelection": to_serializable(getattr(config, "selection", {})),
                "Algorithms": ["Bayesian inference", "Expected Free Energy", "Policy selection"],
                "Model dimensions": {
                    "states": _dimension(model, getattr(environment, "model", None), "num_states", 1),
                    "observations": _dimension(model, getattr(environment, "model", None), "num_observations", 0),
                    "actions": _dimension(model, getattr(environment, "model", None), "num_actions", 0),
                },
            }
        )


def _result_mode(result: object) -> str:
    if isinstance(result, EpisodeResult):
        return "Demo"
    if isinstance(result, SweepResult):
        return "Sweep"
    if isinstance(result, BenchmarkResult):
        return "Ablation" if "no_epistemic" in result.agent_names else "Benchmark"
    return _NOT_AVAILABLE


def _episode_step_count(result: EpisodeResult) -> int:
    return max(len(result.beliefs), len(result.rewards), len(result.observations))


def _product_or_unavailable(left: Any, right: Any) -> int | str:
    if left is None or right is None:
        return _NOT_AVAILABLE
    return int(left) * int(right)


def _dimension(model: object, model_config: object, name: str, axis: int) -> int | str:
    configured = getattr(model_config, name, None)
    if configured is not None:
        return int(configured)
    matrix = getattr(model, "A", None) if name == "num_observations" else None
    if matrix is not None:
        try:
            return int(matrix.shape[axis])
        except (AttributeError, IndexError, TypeError):
            pass
    return _NOT_AVAILABLE


def _format_value(value: object) -> str:
    if value is None:
        return _NOT_AVAILABLE
    return str(value)


def _value_at(values: list[Any], index: int) -> Any:
    return values[index] if 0 <= index < len(values) else None


def _observation_at(episode: EpisodeResult, index: int) -> str:
    observation = _value_at(episode.observations, index)
    return observation_label(int(observation)) if observation is not None else _NOT_AVAILABLE


def _action_at(episode: EpisodeResult, index: int) -> str:
    action = _value_at(episode.actions, index)
    return action_label(int(action)) if action is not None else _NOT_AVAILABLE
