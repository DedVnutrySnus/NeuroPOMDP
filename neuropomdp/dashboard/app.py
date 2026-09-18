"""Streamlit app for the NeuroPOMDP research dashboard."""

from __future__ import annotations

from dataclasses import replace
import traceback

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from .. import __version__
from ..config import BenchmarkConfig, ModelConfig
from ..model import build_generative_model
from .about import render_about_page
from .components import (
    build_action_probability_figure,
    build_actions_figure,
    build_efe_figure,
    build_entropy_figure,
    build_hidden_state_figure,
    build_model_matrix_figure,
    build_model_vector_figure,
    build_value_comparison_figure,
    figure_to_png_bytes,
)
from .export import metrics_csv_bytes, result_json_bytes, timestep_csv_bytes
from .formatting import (
    ACTION_LABELS,
    OBSERVATION_LABELS,
    STATE_LABELS,
    StepDiagnostics,
    action_label,
    agent_label,
    benchmark_rows,
    benchmark_delta,
    extract_step_diagnostics,
    format_float,
    format_probability_bar,
    format_percent,
    generate_action_explanation,
    rows_to_markdown,
    sweep_rows,
)
from .i18n import current_language, set_language, t
from .history import render_history_page, save_run
from .instructions import render_instructions_page
from .research import render_research
from .state import (
    build_dashboard_config,
    get_default_output_dir,
    run_ablation_dashboard,
    run_benchmark_dashboard,
    run_demo,
    run_sweep_dashboard,
)


st.set_page_config(
    page_title="NeuroPOMDP",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_model(model_config: ModelConfig):
    """Cache the generative model for the current configuration."""

    return build_generative_model(model_config)


def main() -> None:
    """Render the dashboard."""

    _apply_minimal_style()
    sidebar = _render_sidebar()

    st.title("NeuroPOMDP Dashboard")
    st.subheader(t("dashboard_subtitle"))
    st.caption(t("version", version=__version__))
    st.caption(t("configure_experiment"))

    current_config = build_dashboard_config(**sidebar["config_kwargs"])
    model = get_model(current_config.env.model)

    result = st.session_state.get("last_result")
    last_mode = st.session_state.get("last_mode")
    display_config = st.session_state.get("last_config", current_config)
    has_result = result is not None and last_mode == sidebar["mode"]
    _render_status_block(sidebar["mode"], current_config, model, has_result)

    simulation_tab, analysis_tab, instructions_tab, research_tab, history_tab, about_tab = st.tabs(
        [
            t("tab_simulation"),
            t("tab_analysis"),
            t("tab_instructions"),
            t("tab_research"),
            t("tab_history"),
            t("tab_about"),
        ]
    )
    with simulation_tab:
        if has_result:
            if display_config != current_config:
                st.info(t("sidebar_changed"))
            _render_result_panel(
                mode=sidebar["mode"],
                result=result,
                config=display_config,
                model=get_model(display_config.env.model),
                show_plots=sidebar["show_plots"],
                sweep_controls=sidebar["sweep_controls"],
            )
        else:
            st.info(t("run_to_populate"))

    with analysis_tab:
        if has_result:
            _render_analysis_panel(result, sidebar["mode"])
        else:
            st.info(t("analysis_after_run"))

    with instructions_tab:
        render_instructions_page()

    with research_tab:
        if has_result:
            render_research(result, display_config, get_model(display_config.env.model))
        else:
            st.info(t("research_after_run"))

    with history_tab:
        render_history_page()

    with about_tab:
        render_about_page()

    if sidebar["run_clicked"]:
        _run_selected_experiment(sidebar, current_config)


def _render_status_block(mode: str, config: BenchmarkConfig, model, has_result: bool) -> None:
    """Render current dashboard state without changing experiment data."""

    columns = st.columns(3)
    with columns[0]:
        st.metric(t("current_mode"), t(mode.lower().replace(" ", "_")))
    with columns[1]:
        status_key = "run_complete" if has_result else "model_ready"
        st.metric(t("model_status"), t(status_key))
    with columns[2]:
        steps = int(config.simulation.episodes) * int(config.simulation.episode_length)
        st.metric(t("simulation_steps"), f"{steps:,}")
    st.markdown('<div class="dashboard-divider"></div>', unsafe_allow_html=True)


def _render_analysis_panel(result, mode: str) -> None:
    """Render statistics and comparisons without duplicating analysis plots."""

    if mode == "Demo":
        st.info(t("analysis_demo_message"))
    elif mode == "Benchmark":
        _render_benchmark(result, show_plots=False)
    elif mode == "Ablation":
        _render_ablation(result, show_plots=False)
    elif mode == "Parameter Sweep":
        _render_sweep(result, show_plots=False)


def _apply_minimal_style() -> None:
    """Apply restrained dashboard styling."""

    st.markdown(
        """
        <style>
        .block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: #18324b; font-weight: 650; letter-spacing: 0; }
        h1 { font-size: 2.35rem; margin-bottom: 0.25rem; }
        h2 { font-size: 1.45rem; margin-top: 1.5rem; }
        h3 { font-size: 1.15rem; margin-top: 1.25rem; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 0.45rem;
            padding: 0.8rem 1rem;
            box-shadow: 0 1px 2px rgba(24, 50, 75, 0.05);
        }
        [data-testid="stMetricLabel"] { color: #587087; }
        [data-testid="stMetricValue"] { color: #18324b; }
        [data-testid="stTabs"] button { font-weight: 600; }
        [data-testid="stExpander"] { border-color: #d9e2ec; border-radius: 0.45rem; }
        .dashboard-divider { border-top: 1px solid #d9e2ec; margin: 1.5rem 0; }
        .stDownloadButton button { width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar() -> dict[str, object]:
    """Collect dashboard inputs from the sidebar."""

    language_label = st.sidebar.radio(
        "Язык / Language",
        options=["English", "Русский"],
        index=0 if current_language() == "en" else 1,
        key="dashboard_language_selector",
    )
    set_language("ru" if language_label == "Русский" else "en")

    st.sidebar.header(t("experiment"))
    mode = st.sidebar.radio(
        t("mode"),
        options=[t("demo"), t("benchmark"), t("ablation"), t("parameter_sweep")],
        index=0,
    )
    mode_keys = ["Demo", "Benchmark", "Ablation", "Parameter Sweep"]
    mode_labels = [t(key.lower().replace(" ", "_")) for key in mode_keys]
    mode = mode_keys[mode_labels.index(mode)]

    with st.sidebar.expander(t("general"), expanded=True):
        seed = st.number_input(t("random_seed"), min_value=0, max_value=1_000_000, value=0, step=1)
        episodes = st.number_input(t("episodes"), min_value=1, max_value=512, value=16, step=1)
        episode_length = st.number_input(t("episode_length"), min_value=1, max_value=20, value=2, step=1)
        show_plots = st.checkbox(t("enable_plots"), value=True)

    with st.sidebar.expander(t("environment"), expanded=True):
        observation_noise = st.slider(t("observation_noise"), 0.0, 1.0, 0.15, 0.01)
        information_quality = st.slider(t("information_quality"), 0.0, 1.0, 0.95, 0.01)
        transition_stochasticity = st.slider(t("transition_stochasticity"), 0.0, 1.0, 0.05, 0.01)
        initial_uncertainty = st.slider(t("initial_uncertainty"), 0.0, 1.0, 0.5, 0.01)

    with st.sidebar.expander(t("agent"), expanded=True):
        precision = st.slider(t("action_precision"), 0.1, 20.0, 8.0, 0.1)
        selection_mode = st.radio(t("action_selection"), options=[t("stochastic"), t("deterministic")], index=0)

    with st.sidebar.expander(t("advanced_settings")):
        output_dir = st.text_input(t("output_directory"), value=get_default_output_dir())
        inference_eps = st.number_input(t("inference_epsilon"), min_value=1e-12, max_value=1e-3, value=1e-8, format="%.8f")
        sweep_min = st.slider(t("minimum_observation_noise"), 0.0, 1.0, 0.05, 0.01)
        sweep_max = st.slider(t("maximum_observation_noise"), 0.0, 1.0, 0.45, 0.01)
        sweep_points = st.number_input(t("sweep_points"), min_value=1, max_value=20, value=4, step=1)
        episodes_per_point = st.number_input(t("episodes_per_point"), min_value=1, max_value=128, value=8, step=1)

    button_label = t("run_sweep") if mode == "Parameter Sweep" else t("run_experiment")
    run_clicked = st.sidebar.button(button_label, type="primary")

    return {
        "mode": mode,
        "run_clicked": run_clicked,
        "show_plots": show_plots,
        "config_kwargs": {
            "seed": int(seed),
            "episodes": int(episodes),
            "episode_length": int(episode_length),
            "output_dir": output_dir,
            "show_plots": show_plots,
            "observation_noise": float(observation_noise),
            "information_quality": float(information_quality),
            "transition_stochasticity": float(transition_stochasticity),
            "initial_uncertainty": float(initial_uncertainty),
            "precision": float(precision),
            "deterministic": selection_mode == "Deterministic",
            "inference_eps": float(inference_eps),
        },
        "sweep_controls": {
            "noise_min": float(sweep_min),
            "noise_max": float(sweep_max),
            "sweep_points": int(sweep_points),
            "episodes_per_point": int(episodes_per_point),
        },
    }


def _run_selected_experiment(sidebar: dict[str, object], config: BenchmarkConfig) -> None:
    """Execute the selected experiment and store the result."""

    progress_bar = st.sidebar.progress(0)
    progress_text = st.sidebar.empty()

    def update_progress(done: int, total: int, message: str) -> None:
        total = max(int(total), 1)
        progress_bar.progress(min(int(round(done / total * 100)), 100))
        progress_text.caption(message)

    try:
        mode = str(sidebar["mode"])
        if mode == "Demo":
            result = run_demo(config)
        elif mode == "Benchmark":
            result = run_benchmark_dashboard(config, progress_callback=update_progress)
        elif mode == "Ablation":
            result = run_ablation_dashboard(config)
        elif mode == "Parameter Sweep":
            sweep = sidebar["sweep_controls"]
            result = run_sweep_dashboard(
                config,
                noise_min=float(sweep["noise_min"]),
                noise_max=float(sweep["noise_max"]),
                sweep_points=int(sweep["sweep_points"]),
                episodes_per_point=int(sweep["episodes_per_point"]),
                progress_callback=update_progress,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
    except Exception as exc:  # noqa: BLE001
        _render_failure(exc)
        progress_bar.empty()
        progress_text.empty()
        return

    try:
        save_run(mode, config, result)
    except (OSError, TypeError, ValueError) as exc:
        st.warning(t("history_save_failed", error=str(exc)))

    st.session_state["last_result"] = result
    st.session_state["last_mode"] = mode
    st.session_state["last_config"] = config
    progress_bar.progress(100)
    progress_text.success(t("experiment_complete"))
    st.rerun()


def _render_failure(exc: Exception) -> None:
    """Display a compact failure panel."""

    st.error(t("experiment_failed"))
    st.write(str(exc))
    with st.expander(t("technical_details")):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def _render_placeholder(config: BenchmarkConfig, model, show_plots: bool) -> None:
    """Render a small placeholder before the first run."""

    st.info(t("run_to_populate"))
    _render_math_model()
    _render_model_inspection(config, model)


def _render_export_buttons(result, config: BenchmarkConfig) -> None:
    """Render downloads for metrics, timestep data, and the complete result."""

    columns = st.columns(3)
    with columns[0]:
        st.download_button(
            t("download_metrics_csv"),
            metrics_csv_bytes(result),
            "metrics.csv",
            mime="text/csv",
        )
    with columns[1]:
        st.download_button(
            t("download_timestep_csv"),
            timestep_csv_bytes(result),
            "timestep_data.csv",
            mime="text/csv",
        )
    with columns[2]:
        st.download_button(
            t("download_results_json"),
            result_json_bytes(result, config),
            "results.json",
            mime="application/json",
        )


def _render_result_panel(
    *,
    mode: str,
    result,
    config: BenchmarkConfig,
    model,
    show_plots: bool,
    sweep_controls: dict[str, object],
) -> None:
    """Render the active experiment result."""

    if mode == "Demo":
        _render_demo(result, config, model, show_plots)
    elif mode == "Benchmark":
        _render_benchmark(result, show_plots)
    elif mode == "Ablation":
        _render_ablation(result, show_plots)
    elif mode == "Parameter Sweep":
        _render_sweep(result, show_plots)
    else:
        st.warning(t("unsupported_mode", mode=mode))

    _render_math_model()
    _render_model_inspection(config, model)
    _render_export_buttons(result, config)


def _demo_step_index(episode) -> int:
    """Return a valid demo timestep without creating a zero-width slider."""

    max_step = len(episode.actions) - 1
    if max_step <= 0:
        return 0
    return st.slider(t("inspect_timestep"), min_value=0, max_value=max_step, value=0, step=1)


def _render_demo(result, config: BenchmarkConfig, model, show_plots: bool) -> None:
    """Render the demo episode view."""

    st.header(t("demo_heading"))
    episode = result
    step_index = _demo_step_index(episode)

    step = extract_step_diagnostics(episode, step_index)

    metric_values = [
        (t("observation"), observation_display(step.observation)),
        (t("selected_action"), action_display(step.selected_action)),
        (t("most_likely_hidden_state"), state_display(int(np.argmax(step.belief)))),
        (t("posterior_confidence"), format_percent(float(np.max(step.belief)))),
        (t("free_energy"), format_float(step.free_energy)),
        (t("expected_free_energy"), format_float(float(step.efe_values[step.selected_action]) if step.selected_action is not None else float("nan"))),
    ]
    _render_metrics(metric_values)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown(t("why_action"))
        st.markdown(generate_action_explanation(step))
        st.markdown(t("action_comparison"))
        st.markdown(_localized_rows_to_markdown(_action_comparison_rows(step), ["Action", "Pragmatic", "Epistemic", "Total EFE", "Policy Probability"]))
        st.markdown(f"**{t('current_belief').rstrip(':')}**")
        st.markdown(_belief_text(step))
        st.markdown(f"**{t('action_probabilities').rstrip(':')}**")
        st.markdown(_probability_text(step))
    with right:
        if show_plots:
            fig = build_value_comparison_figure(step)
            st.pyplot(fig, clear_figure=False, width="stretch")
            st.download_button(t("download_value_figure"), figure_to_png_bytes(fig), "demo_values.png", mime="image/png")
            efe_fig = build_efe_figure(step)
            st.pyplot(efe_fig, clear_figure=False, width="stretch")
            st.download_button(t("download_efe_figure"), figure_to_png_bytes(efe_fig), "demo_efe.png", mime="image/png")
            probability_fig = build_action_probability_figure(step)
            st.pyplot(probability_fig, clear_figure=False, width="stretch")
            st.download_button(t("download_probability_figure"), figure_to_png_bytes(probability_fig), "demo_action_probabilities.png", mime="image/png")

    if show_plots:
        st.divider()
        st.subheader(t("trajectory_charts"))
        belief_fig = build_hidden_state_figure(episode)
        entropy_fig = build_entropy_figure(episode)
        actions_fig = build_actions_figure(episode)
        chart_cols = st.columns(3)
        with chart_cols[0]:
            st.pyplot(belief_fig, width="stretch")
        with chart_cols[1]:
            st.pyplot(entropy_fig, width="stretch")
        with chart_cols[2]:
            st.pyplot(actions_fig, width="stretch")
        download_cols = st.columns(3)
        with download_cols[0]:
            st.download_button(t("download_beliefs_figure"), figure_to_png_bytes(belief_fig), "demo_beliefs.png", mime="image/png")
        with download_cols[1]:
            st.download_button(t("download_entropy_figure"), figure_to_png_bytes(entropy_fig), "demo_entropy.png", mime="image/png")
        with download_cols[2]:
            st.download_button(t("download_actions_figure"), figure_to_png_bytes(actions_fig), "demo_actions.png", mime="image/png")

def _render_benchmark(result, show_plots: bool) -> None:
    """Render the benchmark comparison view."""

    st.header(t("benchmark_heading"))
    rows = benchmark_rows(result, ["random", "pragmatic_only", "active_inference"])
    summary_map = {summary.agent_name: summary for summary in result.summaries}
    active = summary_map["active_inference"]
    metrics = [
        (t("mean_reward"), format_float(active.mean_reward)),
        (t("success_rate"), format_percent(active.success_rate)),
        (t("mean_entropy"), format_float(active.mean_posterior_entropy)),
        (t("mean_information_gain"), format_float(active.mean_information_gain)),
    ]
    _render_metrics(metrics)

    st.markdown(t("benchmark_table"))
    st.markdown(_localized_rows_to_markdown(rows, ["Agent", "Mean Reward", "Success Rate", "Mean Entropy", "Mean Information Gain", "Info-Seeking Frequency"]))

    if show_plots:
        st.divider()
        st.subheader(t("comparison_charts"))
        reward_fig = _benchmark_reward_figure(result)
        action_fig = _benchmark_action_distribution_figure(result)
        entropy_fig = _benchmark_entropy_figure(result)
        info_fig = _benchmark_info_figure(result)
        top_cols = st.columns(2)
        with top_cols[0]:
            st.pyplot(reward_fig, width="stretch")
        with top_cols[1]:
            st.pyplot(action_fig, width="stretch")
        bottom_cols = st.columns(2)
        with bottom_cols[0]:
            st.pyplot(entropy_fig, width="stretch")
        with bottom_cols[1]:
            st.pyplot(info_fig, width="stretch")
        download_cols = st.columns(4)
        with download_cols[0]:
            st.download_button(t("reward_figure"), figure_to_png_bytes(reward_fig), "benchmark_reward.png", mime="image/png")
        with download_cols[1]:
            st.download_button(t("action_figure"), figure_to_png_bytes(action_fig), "benchmark_actions.png", mime="image/png")
        with download_cols[2]:
            st.download_button(t("entropy_figure"), figure_to_png_bytes(entropy_fig), "benchmark_entropy.png", mime="image/png")
        with download_cols[3]:
            st.download_button(t("info_figure"), figure_to_png_bytes(info_fig), "benchmark_information_gain.png", mime="image/png")

def _render_ablation(result, show_plots: bool) -> None:
    """Render the ablation comparison view."""

    st.header(t("ablation_heading"))
    rows = benchmark_rows(result, ["active_inference", "no_epistemic"])
    summary_map = {summary.agent_name: summary for summary in result.summaries}
    delta = benchmark_delta(result, "no_epistemic", "active_inference")
    active = summary_map["active_inference"]
    no_epistemic = summary_map["no_epistemic"]
    metrics = [
        (t("reward_delta"), format_float(delta["reward"])),
        (t("success_delta"), format_percent(delta["success_rate"])),
        (t("entropy_delta"), format_float(delta["entropy"])),
        (t("info_gain_delta"), format_float(delta["information_gain"])),
    ]
    _render_metrics(metrics)

    st.markdown(t("effect_removing_epistemic"))
    st.write(
        t(
            "removing_epistemic_message",
            frequency=format_percent(delta["info_frequency"]),
            reward=format_float(delta["reward"]),
        )
    )
    st.markdown(_localized_rows_to_markdown(rows, ["Agent", "Mean Reward", "Success Rate", "Mean Entropy", "Mean Information Gain", "Info-Seeking Frequency"]))

    if show_plots:
        st.divider()
        fig = _ablation_figure(result)
        chart_col = st.columns([0.12, 0.76, 0.12])[1]
        with chart_col:
            st.pyplot(fig, width="stretch")
        st.download_button(t("download_ablation_figure"), figure_to_png_bytes(fig), "ablation_comparison.png", mime="image/png")

def _render_sweep(result, show_plots: bool) -> None:
    """Render the parameter sweep view."""

    st.header(t("parameter_sweep_heading"))
    rows = sweep_rows(result)
    st.markdown(_localized_rows_to_markdown(rows, ["Observation Noise", "Agent", "Mean Reward", "Success Rate", "Mean Entropy", "Info-Seeking Frequency"]))

    if show_plots:
        st.divider()
        st.subheader(t("sweep_charts"))
        reward_fig = _sweep_reward_figure(result)
        info_fig = _sweep_info_figure(result)
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.pyplot(reward_fig, width="stretch")
        with chart_cols[1]:
            st.pyplot(info_fig, width="stretch")
        download_cols = st.columns(2)
        with download_cols[0]:
            st.download_button(t("download_sweep_reward_figure"), figure_to_png_bytes(reward_fig), "sweep_reward.png", mime="image/png")
        with download_cols[1]:
            st.download_button(t("download_sweep_info_figure"), figure_to_png_bytes(info_fig), "sweep_information_gain.png", mime="image/png")

def _render_math_model() -> None:
    """Render the short mathematical model overview."""

    with st.expander(t("mathematical_model")):
        st.markdown(
            """
            - `A[o, s] = P(o|s)`
            - `B[a, s', s] = P(s'|s, a)`
            - `D[s] = P(s)`
            - `q(s|o) \u221d A[o,s]D[s]`
            - `F = KL(q||D) - E_q[log A[o,s]]`
            - `G = pragmatic - epistemic`
            """
        )


def _render_model_inspection(config: BenchmarkConfig, model) -> None:
    """Render the generative model inspection panel."""

    with st.expander(t("generative_model")):
        st.markdown(t("a_matrix"))
        fig_a = build_model_matrix_figure(
            np.asarray(model.A),
            [observation_display(index) for index in range(model.A.shape[0])],
            [state_display(index) for index in range(model.A.shape[1])],
            t("chart_likelihood_matrix"),
        )
        st.pyplot(fig_a, width="stretch")
        st.markdown(t("c_preferences"))
        fig_c = build_model_vector_figure(np.asarray(model.C), [observation_display(index) for index in range(model.C.shape[0])], t("chart_preference_vector"), t("chart_preference"))
        st.pyplot(fig_c, width="stretch")
        st.markdown(t("d_prior"))
        fig_d = build_model_vector_figure(np.asarray(model.D), [state_display(index) for index in range(model.D.shape[0])], t("chart_prior_vector"), t("chart_probability"))
        st.pyplot(fig_d, width="stretch")

        action_index = st.selectbox(
            t("transition_matrix_for_action"),
            options=list(range(model.B.shape[0])),
            format_func=lambda index: action_display(index),
        )
        fig_b = build_model_matrix_figure(
            np.asarray(model.B[action_index]),
            [state_display(index) for index in range(model.B.shape[1])],
            [state_display(index) for index in range(model.B.shape[2])],
            t("transition_matrix_b_for_action", action=action_display(action_index)),
        )
        st.pyplot(fig_b, width="stretch")


def _render_metrics(metrics: list[tuple[str, str]]) -> None:
    """Render a compact metric grid."""

    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics, strict=False):
        with column:
            st.metric(label, value)


_TABLE_LABEL_KEYS = {
    "Action": "action",
    "Pragmatic": "pragmatic",
    "Epistemic": "epistemic",
    "Total EFE": "total_efe",
    "Policy Probability": "policy_probability",
    "Agent": "agent_column",
    "Mean Reward": "mean_reward",
    "Success Rate": "success_rate",
    "Mean Entropy": "mean_entropy",
    "Mean Information Gain": "mean_information_gain",
    "Info-Seeking Frequency": "info_seeking_frequency",
    "Observation Noise": "observation_noise_column",
}


def _localized_rows_to_markdown(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    """Render result rows with localized headers while preserving their keys."""

    display_fieldnames = [t(_TABLE_LABEL_KEYS.get(fieldname, fieldname)) for fieldname in fieldnames]
    display_rows = [
        {display_name: row.get(fieldname, "") for fieldname, display_name in zip(fieldnames, display_fieldnames, strict=False)}
        for row in rows
    ]
    return rows_to_markdown(display_rows, display_fieldnames)


def observation_display(index: int) -> str:
    """Return a display label for an observation."""

    return t(f"observation_{index}") if 0 <= index < len(OBSERVATION_LABELS) else f"Observation {index}"


def state_display(index: int) -> str:
    """Return a display label for a state."""

    return t(f"state_{index}") if 0 <= index < len(STATE_LABELS) else f"State {index}"


def action_display(index: int | None) -> str:
    """Return a display label for an action."""

    if index is None:
        return "None"
    return t(f"action_{index}") if 0 <= index < len(ACTION_LABELS) else f"Action {index}"


def _agent_display(name: str) -> str:
    """Return a localized display label for an agent identifier."""

    return t(f"agent_{name}")


def _belief_text(step: StepDiagnostics) -> str:
    """Render current belief as compact text."""

    lines = ["Current belief:"]
    for index, value in enumerate(step.belief):
        lines.append(f"{state_display(index):<16} {format_probability_bar(value)} {format_float(value, 2)}")
    return "\n".join(lines)


def _probability_text(step: StepDiagnostics) -> str:
    """Render action probabilities as compact text."""

    lines = ["Action probabilities:"]
    for index, value in enumerate(step.action_probabilities):
        lines.append(f"{action_display(index):<16} {format_float(value, 2)}")
    return "\n".join(lines)


def _action_comparison_rows(step: StepDiagnostics) -> list[dict[str, object]]:
    rows = []
    for index in range(len(step.efe_values)):
        rows.append(
            {
                "Action": action_display(index),
                "Pragmatic": float(step.pragmatic_values[index]),
                "Epistemic": float(step.epistemic_values[index]),
                "Total EFE": float(step.efe_values[index]),
                "Policy Probability": float(step.action_probabilities[index]),
            }
        )
    return rows


def _benchmark_reward_figure(result) -> plt.Figure:
    """Render the benchmark reward comparison."""

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    names = [summary.agent_name for summary in result.summaries]
    values = [summary.mean_reward for summary in result.summaries]
    ax.bar([_agent_display(name) for name in names], values, color="#4c72b0")
    ax.set_ylabel(t("chart_mean_reward"))
    ax.set_title(t("chart_reward_comparison"))
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def _benchmark_action_distribution_figure(result) -> plt.Figure:
    """Render action distribution comparison."""

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    agent_names = [summary.agent_name for summary in result.summaries]
    action_keys = ["inspect", "exploit_left", "exploit_right"]
    bottom = np.zeros(len(agent_names))
    x = np.arange(len(agent_names))
    for key, color, label_key in zip(action_keys, ["#4c72b0", "#dd8452", "#55a868"], ["action_0", "action_1", "action_2"], strict=False):
        values = []
        for summary in result.summaries:
            total = sum(summary.action_counts.values()) or 1
            values.append(summary.action_counts.get(key, 0) / total)
        ax.bar(x, values, bottom=bottom, color=color, label=t(label_key))
        bottom = bottom + np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels([_agent_display(name) for name in agent_names])
    ax.set_ylabel(t("chart_action_share"))
    ax.set_title(t("chart_action_distribution"))
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _benchmark_entropy_figure(result) -> plt.Figure:
    """Render the benchmark entropy comparison."""

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(
        [_agent_display(summary.agent_name) for summary in result.summaries],
        [summary.mean_posterior_entropy for summary in result.summaries],
        color="#c44e52",
    )
    ax.set_ylabel(t("chart_mean_posterior_entropy"))
    ax.set_title(t("chart_entropy_comparison"))
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def _benchmark_info_figure(result) -> plt.Figure:
    """Render the benchmark information-seeking comparison."""

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(
        [_agent_display(summary.agent_name) for summary in result.summaries],
        [summary.info_action_frequency for summary in result.summaries],
        color="#55a868",
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(t("chart_info_seeking_frequency"))
    ax.set_title(t("chart_information_seeking_behavior"))
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def _ablation_figure(result) -> plt.Figure:
    """Render the ablation comparison figure."""

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for summary in result.summaries:
        ax.bar(_agent_display(summary.agent_name), summary.mean_reward, color="#4c72b0" if summary.agent_name == "active_inference" else "#dd8452")
    ax.set_ylabel(t("chart_mean_reward"))
    ax.set_title(t("chart_effect_removing_epistemic"))
    ax.grid(True, axis="y", alpha=0.15)
    fig.tight_layout()
    return fig


def _sweep_reward_figure(result) -> plt.Figure:
    """Render the sweep reward chart with error bars when multi-seed summaries are available."""

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    noise = np.asarray(result.noise_values, dtype=np.float64)
    rewards = np.asarray(result.mean_reward, dtype=np.float64)
    ci = None
    if hasattr(result, "per_seed_results") and result.per_seed_results:
        all_seed_rewards = np.asarray(
            [np.asarray(seed_result.mean_reward, dtype=np.float64) for seed_result in result.per_seed_results],
            dtype=np.float64,
        )
        if all_seed_rewards.ndim == 3 and all_seed_rewards.shape[0] > 1:
            seed_means = all_seed_rewards[:, :, :]
            ci = 1.96 * np.std(seed_means, axis=0, ddof=0) / np.sqrt(seed_means.shape[0])
    for index, agent_name in enumerate(result.agent_names):
        y = rewards[:, index]
        if ci is not None:
            ax.errorbar(noise, y, yerr=ci[:, index], fmt="-o", capsize=3, label=_agent_display(agent_name), alpha=0.8)
        else:
            ax.plot(noise, y, marker="o", linewidth=2.0, label=_agent_display(agent_name))
    ax.set_xlabel(t("chart_observation_noise"))
    ax.set_ylabel(t("chart_mean_reward"))
    ax.set_title(t("chart_performance_vs_noise"))
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _sweep_success_figure(result) -> plt.Figure:
    """Render the sweep success-rate chart with error bars when multi-seed summaries are available."""

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    noise = np.asarray(result.noise_values, dtype=np.float64)
    success = np.asarray(result.success_rate, dtype=np.float64)
    ci = None
    if hasattr(result, "per_seed_results") and result.per_seed_results:
        all_seed_success = np.asarray(
            [np.asarray(seed_result.success_rate, dtype=np.float64) for seed_result in result.per_seed_results],
            dtype=np.float64,
        )
        if all_seed_success.ndim == 3 and all_seed_success.shape[0] > 1:
            ci = 1.96 * np.std(all_seed_success, axis=0, ddof=0) / np.sqrt(all_seed_success.shape[0])
    for index, agent_name in enumerate(result.agent_names):
        y = success[:, index]
        if ci is not None:
            ax.errorbar(noise, y, yerr=ci[:, index], fmt="-o", capsize=3, label=_agent_display(agent_name), alpha=0.8)
        else:
            ax.plot(noise, y, marker="o", linewidth=2.0, label=_agent_display(agent_name))
    ax.set_xlabel(t("chart_observation_noise"))
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs Observation Noise")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _sweep_info_figure(result) -> plt.Figure:
    """Render the sweep information-seeking chart."""

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    noise = np.asarray(result.noise_values, dtype=np.float64)
    info = np.asarray(result.info_action_frequency, dtype=np.float64)
    ci = None
    if hasattr(result, "per_seed_results") and result.per_seed_results:
        all_seed_info = np.asarray(
            [np.asarray(seed_result.info_action_frequency, dtype=np.float64) for seed_result in result.per_seed_results],
            dtype=np.float64,
        )
        if all_seed_info.ndim == 3 and all_seed_info.shape[0] > 1:
            ci = 1.96 * np.std(all_seed_info, axis=0, ddof=0) / np.sqrt(all_seed_info.shape[0])
    for index, agent_name in enumerate(result.agent_names):
        y = info[:, index]
        if ci is not None:
            ax.errorbar(noise, y, yerr=ci[:, index], fmt="-o", capsize=3, label=_agent_display(agent_name), alpha=0.8)
        else:
            ax.plot(noise, y, marker="o", linewidth=2.0, label=_agent_display(agent_name))
    ax.set_xlabel(t("chart_observation_noise"))
    ax.set_ylabel(t("chart_info_seeking_frequency"))
    ax.set_title(t("chart_info_frequency_vs_noise"))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


if __name__ == "__main__":  # pragma: no cover
    main()
