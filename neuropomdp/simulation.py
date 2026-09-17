"""Simulation utilities for running episodes and collecting traces."""

from __future__ import annotations

from typing import Callable

import jax
import numpy as np

from .agent import ActiveInferenceAgent, BayesianUtilityAgent, PragmaticOnlyAgent, RandomAgent
from .config import EnvironmentConfig, SimulationConfig
from .environment import NeuroPOMDPEnvironment
from .types import ActionID, AgentSummary, BenchmarkResult, EpisodeResult
from .types import to_serializable


AgentLike = ActiveInferenceAgent | BayesianUtilityAgent | PragmaticOnlyAgent | RandomAgent


def _make_agent_key(seed: int, episode_index: int) -> jax.Array:
    return jax.random.PRNGKey(seed + episode_index * 9973)


def run_episode(agent: AgentLike, env_config: EnvironmentConfig, key: jax.Array) -> EpisodeResult:
    """Run one episode and record a full trajectory."""

    env = NeuroPOMDPEnvironment(env_config)
    env_key, key = jax.random.split(key)
    obs, info = env.reset(env_key)
    agent.reset()
    inference = agent.update(obs, action=None)

    observations = [int(obs)]
    actions: list[int] = []
    rewards: list[float] = [0.0]
    hidden_states = [int(info["hidden_state"])]
    beliefs = [np.asarray(agent.belief).tolist()]
    posterior_entropies = [float(inference.posterior_entropy)]
    free_energies = [float(inference.free_energy)]
    pragmatic_values: list[list[float]] = []
    epistemic_values: list[list[float]] = []
    efe_values: list[list[float]] = []
    action_probabilities: list[list[float]] = []
    selected_action_efe: list[float] = []
    info_gains: list[float] = []
    pragmatic_costs: list[float] = []

    done = False
    step_index = 0
    while not done and step_index < env_config.episode_length:
        key, action_key = jax.random.split(key)
        action, diagnostics = agent.select_action(action_key)
        obs, reward, done, info = env.step(action)
        inference = agent.update(obs, action=action)

        actions.append(int(action))
        observations.append(int(obs))
        rewards.append(float(reward))
        hidden_states.append(int(info["hidden_state"]))
        beliefs.append(np.asarray(agent.belief).tolist())
        posterior_entropies.append(float(inference.posterior_entropy))
        free_energies.append(float(inference.free_energy))
        pragmatic_values.append(np.asarray(diagnostics.pragmatic_costs).tolist())
        epistemic_values.append(np.asarray(diagnostics.epistemic_values).tolist())
        efe_values.append(np.asarray(diagnostics.efe_values).tolist())
        action_probabilities.append(np.asarray(diagnostics.action_probabilities).tolist())
        selected_action_efe.append(float(diagnostics.efe_values[action]))
        info_gains.append(float(diagnostics.epistemic_values[action]))
        pragmatic_costs.append(float(diagnostics.pragmatic_costs[action]))
        step_index += 1

    posterior_entropies = [float(x) for x in posterior_entropies]
    belief_array = np.asarray(beliefs)
    total_reward = float(np.sum(rewards))
    success = bool(rewards[-1] > 0.0)
    mean_information_gain = float(np.mean(info_gains)) if info_gains else 0.0
    mean_pragmatic_cost = float(np.mean(pragmatic_costs)) if pragmatic_costs else 0.0
    return EpisodeResult(
        observations=observations,
        actions=actions,
        rewards=rewards,
        hidden_states=hidden_states,
        beliefs=belief_array.tolist(),
        posterior_entropies=posterior_entropies,
        free_energies=free_energies,
        pragmatic_values=pragmatic_values,
        epistemic_values=epistemic_values,
        efe_values=efe_values,
        action_probabilities=action_probabilities,
        selected_action_efe=selected_action_efe,
        total_reward=total_reward,
        success=success,
        mean_information_gain=mean_information_gain,
        mean_pragmatic_cost=mean_pragmatic_cost,
    )


def run_benchmark(
    agent_factories: list[tuple[str, Callable[[], AgentLike]]],
    sim_config: SimulationConfig,
    env_config: EnvironmentConfig,
) -> BenchmarkResult:
    """Run multiple agents under identical seeds and environment settings."""

    episodes_by_agent: dict[str, list[EpisodeResult]] = {}
    summaries: list[AgentSummary] = []

    for agent_name, factory in agent_factories:
        episodes: list[EpisodeResult] = []
        for episode_index in range(sim_config.episodes):
            episode_key = _make_agent_key(sim_config.seed, episode_index)
            agent = factory()
            episodes.append(run_episode(agent, env_config, episode_key))
        episodes_by_agent[agent_name] = episodes
        summaries.append(summarize_agent(agent_name, episodes))

    return BenchmarkResult(
        seed=sim_config.seed,
        config={"simulation": to_serializable(sim_config), "environment": to_serializable(env_config)},
        agent_names=[name for name, _ in agent_factories],
        summaries=summaries,
        episodes_by_agent=episodes_by_agent,
    )


def summarize_agent(agent_name: str, episodes: list[EpisodeResult]) -> AgentSummary:
    """Aggregate benchmark metrics for one agent."""

    rewards = np.asarray([episode.total_reward for episode in episodes], dtype=np.float64)
    successes = np.asarray([episode.success for episode in episodes], dtype=np.float64)
    entropies = np.asarray([np.mean(episode.posterior_entropies) for episode in episodes], dtype=np.float64)
    info_gains = np.asarray([episode.mean_information_gain for episode in episodes], dtype=np.float64)
    episode_count = len(episodes)
    reward_std = float(rewards.std(ddof=0)) if episode_count else 0.0
    mean_reward = float(rewards.mean()) if episode_count else 0.0
    success_rate = float(successes.mean()) if episode_count else 0.0
    reward_margin = 1.96 * reward_std / np.sqrt(episode_count) if episode_count else 0.0
    success_margin = 1.96 * np.sqrt(success_rate * (1.0 - success_rate) / episode_count) if episode_count else 0.0
    info_action_count = sum(action == int(ActionID.INSPECT) for episode in episodes for action in episode.actions)
    total_actions = sum(len(episode.actions) for episode in episodes)
    action_counts = {
        "inspect": int(info_action_count),
        "exploit_left": int(sum(action == int(ActionID.EXPLOIT_LEFT) for episode in episodes for action in episode.actions)),
        "exploit_right": int(sum(action == int(ActionID.EXPLOIT_RIGHT) for episode in episodes for action in episode.actions)),
    }
    return AgentSummary(
        agent_name=agent_name,
        episode_count=episode_count,
        mean_reward=mean_reward,
        reward_std=reward_std,
        reward_ci95_low=mean_reward - reward_margin,
        reward_ci95_high=mean_reward + reward_margin,
        success_rate=success_rate,
        success_ci95_low=max(0.0, success_rate - success_margin),
        success_ci95_high=min(1.0, success_rate + success_margin),
        mean_posterior_entropy=float(entropies.mean()) if len(entropies) else 0.0,
        mean_information_gain=float(info_gains.mean()) if len(info_gains) else 0.0,
        info_action_frequency=float(info_action_count / total_actions) if total_actions else 0.0,
        action_counts=action_counts,
    )
