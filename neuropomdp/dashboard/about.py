"""Research overview page for the NeuroPOMDP dashboard."""

from __future__ import annotations

import streamlit as st
from .i18n import current_language


def render_about_page() -> None:
    """Render the English research overview for NeuroPOMDP."""

    if current_language() == "ru":
        _render_russian_about_page()
        return

    st.header("NeuroPOMDP")
    st.write(
        "NeuroPOMDP is a simulation and analysis framework for studying decision-making "
        "in partially observable environments. It provides tools for experimenting with "
        "POMDP-based models, evaluating agent behavior, and visualizing simulation results."
    )

    st.subheader("What is a POMDP?")
    st.write(
        "A Partially Observable Markov Decision Process extends a Markov Decision Process "
        "by introducing incomplete information. The agent cannot observe the true state "
        "of the environment directly; it receives observations and uses them to select actions."
    )
    st.markdown(
        """
        - **State (S):** The hidden true state of the environment.
        - **Observation (O):** The information available to the agent.
        - **Action (A):** The decision made by the agent.
        - **Reward (R):** The feedback used to evaluate decisions.
        - **Policy (π):** The strategy that determines actions.
        """
    )

    st.subheader("Belief State")
    st.write(
        "Since the true state is unavailable, the agent maintains a belief state: a probability "
        "distribution over possible states. Belief updates allow the agent to reason under "
        "uncertainty and improve decision-making over time."
    )

    st.subheader("Project Workflow")
    st.code(
        "Environment\n"
        "     ↓\n"
        "Observation\n"
        "     ↓\n"
        "Belief Update\n"
        "     ↓\n"
        "Policy Selection\n"
        "     ↓\n"
        "Action\n"
        "     ↓\n"
        "Reward Evaluation",
        language="text",
    )

    st.subheader("Research Motivation")
    st.write(
        "POMDP-based methods are important in settings where complete information is unavailable "
        "and decisions must be made from uncertain evidence. Relevant application areas include:"
    )
    st.markdown(
        """
        - Robotics
        - Autonomous systems
        - Artificial intelligence
        - Planning under uncertainty
        - Decision-making systems
        """
    )

    st.subheader("Technologies")
    st.markdown(
        """
        - Python
        - JAX
        - NumPy
        - SciPy
        - Streamlit
        - Matplotlib
        """
    )

    st.subheader("Project Components")
    st.markdown(
        """
        **Core**
        - Environment simulation
        - Decision models
        - Algorithms

        **Dashboard**
        - Interactive visualization
        - Experiment control
        - Result analysis

        **Research Tools**
        - Benchmark experiments
        - Ablation studies
        - Simulation analysis
        """
    )

    st.subheader("Future Development")
    st.write("Potential extensions:")
    st.markdown(
        """
        - Advanced visualization
        - Larger experiment management
        - Comparison of different policies
        - Additional POMDP algorithms
        """
    )


def _render_russian_about_page() -> None:
    """Render the Russian research overview for NeuroPOMDP."""

    st.header("NeuroPOMDP")
    st.write(
        "NeuroPOMDP — исследовательский программный проект для моделирования и анализа "
        "задач принятия решений в условиях частично наблюдаемой информации. Проект "
        "предоставляет инструменты для экспериментов с POMDP-моделями, оценки поведения "
        "агентов и визуализации результатов симуляций."
    )

    st.subheader("Что такое POMDP?")
    st.write(
        "Partially Observable Markov Decision Process (POMDP) — это расширение Markov "
        "Decision Process, в котором агент не имеет полного доступа к состоянию системы. "
        "Он получает наблюдения и использует их для выбора действий."
    )
    st.markdown(
        """
        - **State (S), состояние:** реальное скрытое состояние окружающей среды.
        - **Observation (O), наблюдение:** информация, доступная агенту.
        - **Action (A), действие:** решение, принятое агентом.
        - **Reward (R), вознаграждение:** обратная связь для оценки качества решения.
        - **Policy (π), политика:** стратегия выбора действий.
        """
    )

    st.subheader("Belief State")
    st.write(
        "Так как агент не знает настоящее состояние среды, он поддерживает belief state — "
        "распределение вероятностей возможных состояний. Это внутреннее представление "
        "позволяет принимать решения даже при неполной информации и обновлять оценку "
        "неопределённости по мере поступления новых наблюдений."
    )

    st.subheader("Simulation")
    st.write(
        "NeuroPOMDP позволяет запускать симуляции, анализировать поведение агента и "
        "визуализировать результаты экспериментов."
    )

    st.subheader("Project Workflow")
    st.code(
        "Environment\n"
        "     ↓\n"
        "Observation\n"
        "     ↓\n"
        "Belief Update\n"
        "     ↓\n"
        "Policy Decision\n"
        "     ↓\n"
        "Action\n"
        "     ↓\n"
        "Reward Evaluation",
        language="text",
    )

    st.subheader("Research Motivation")
    st.write("Системы с частично наблюдаемой информацией встречаются в следующих областях:")
    st.markdown(
        """
        - робототехника;
        - автономные системы;
        - управление;
        - искусственный интеллект;
        - планирование действий.
        """
    )

    st.subheader("Использованные технологии")
    st.markdown(
        """
        - Python
        - JAX
        - NumPy
        - SciPy
        - Streamlit
        - Matplotlib
        """
    )

    st.subheader("Структура проекта")
    st.markdown(
        """
        **Core**
        - модели среды;
        - алгоритмы;
        - симуляция.

        **Dashboard**
        - визуализация;
        - анализ результатов;
        - управление экспериментами.

        **Research Features**
        - benchmark;
        - ablation analysis;
        - comparison of strategies.
        """
    )
