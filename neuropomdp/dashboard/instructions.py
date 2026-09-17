"""User-facing instructions for the NeuroPOMDP dashboard."""

from __future__ import annotations

import streamlit as st

from .i18n import current_language


def render_instructions_page() -> None:
    """Explain the experiment workflow and how to interpret dashboard results."""

    if current_language() == "ru":
        _render_russian_instructions()
        return
    _render_english_instructions()


def _render_english_instructions() -> None:
    st.header("How NeuroPOMDP works")
    st.write(
        "NeuroPOMDP studies decisions when the true environment state is hidden. "
        "The agent receives observations, maintains a probability distribution over possible states, "
        "and chooses actions using Expected Free Energy (EFE)."
    )

    st.subheader("One decision step")
    st.code(
        "Hidden state\n"
        "    ↓\n"
        "Observation\n"
        "    ↓\n"
        "Belief update: q(s | o)\n"
        "    ↓\n"
        "Evaluate actions: pragmatic cost + epistemic value\n"
        "    ↓\n"
        "Choose Inspect or Exploit\n"
        "    ↓\n"
        "Transition and reward",
        language="text",
    )
    st.markdown(
        """
        1. **Observation:** the environment emits an observation that may be ambiguous.
        2. **Belief update:** Bayesian inference estimates the probability of each hidden state.
        3. **Action evaluation:** the agent predicts consequences of each available action.
        4. **Selection:** the lowest expected free energy is preferred.
        5. **Learning through action:** the selected action changes the state and produces a reward.
        """
    )

    st.subheader("Pragmatic and epistemic value")
    st.write(
        "Pragmatic value asks whether an action is likely to produce a preferred outcome now. "
        "Epistemic value asks how much the action is expected to reduce uncertainty and improve later decisions."
    )
    st.latex(r"G(a) = G_{pragmatic}(a) - G_{epistemic}(a)")
    st.write(
        "Inspect may have an immediate cost, but it can reveal which exploit action will succeed next. "
        "That future decision advantage is the reason information-seeking can outperform greedy exploitation."
    )

    st.subheader("What the experiment modes mean")
    st.markdown(
        """
        - **Demo:** inspect one episode step by step, including beliefs, entropy, actions, and EFE.
        - **Benchmark:** compare Random, Pragmatic-only, Active Inference, and Bayesian Utility agents.
        - **Ablation:** compare Active Inference with the same setup after removing epistemic value.
        - **Parameter Sweep:** vary observation noise and measure when information-seeking helps.
        """
    )

    st.subheader("How to read multi-seed results")
    st.markdown(
        """
        - **Mean reward:** average total reward across episodes and seeds.
        - **Success rate:** fraction of episodes ending in a successful terminal outcome.
        - **95% CI:** uncertainty around the mean across independent seeds.
        - **Reliable advantage:** the lower CI bound is above zero, so the estimated advantage is consistently positive across seeds.
        """
    )
    st.info(
        "Use at least three independent seeds for exploratory analysis. Treat a positive mean without a confidence interval above zero as suggestive, not conclusive."
    )

    st.subheader("Recommended workflow")
    st.markdown(
        """
        1. Start with **Demo** to inspect one decision.
        2. Run **Ablation** to isolate the contribution of epistemic value.
        3. Run a **Parameter Sweep** with multiple seeds.
        4. Use the Research tab and downloaded CSV/JSON files for reporting.
        """
    )


def _render_russian_instructions() -> None:
    st.header("Как работает NeuroPOMDP")
    st.write(
        "NeuroPOMDP изучает принятие решений, когда настоящее состояние среды скрыто. "
        "Агент получает наблюдения, поддерживает распределение вероятностей по возможным состояниям "
        "и выбирает действия с помощью ожидаемой свободной энергии (EFE)."
    )

    st.subheader("Один шаг принятия решения")
    st.code(
        "Скрытое состояние\n"
        "    ↓\n"
        "Наблюдение\n"
        "    ↓\n"
        "Обновление belief state: q(s | o)\n"
        "    ↓\n"
        "Оценка действий: прагматическая стоимость + эпистемическая ценность\n"
        "    ↓\n"
        "Выбор Inspect или Exploit\n"
        "    ↓\n"
        "Переход и награда",
        language="text",
    )
    st.markdown(
        """
        1. **Наблюдение:** среда выдаёт наблюдение, которое может быть неоднозначным.
        2. **Обновление belief state:** байесовский вывод оценивает вероятность каждого скрытого состояния.
        3. **Оценка действий:** агент прогнозирует последствия каждого доступного действия.
        4. **Выбор:** предпочтение получает действие с наименьшей ожидаемой свободной энергией.
        5. **Действие и результат:** действие меняет состояние среды и создаёт награду.
        """
    )

    st.subheader("Прагматическая и эпистемическая ценность")
    st.write(
        "Прагматическая ценность показывает, насколько действие полезно для результата сейчас. "
        "Эпистемическая ценность показывает, насколько действие уменьшит неопределённость и улучшит будущие решения."
    )
    st.latex(r"G(a) = G_{pragmatic}(a) - G_{epistemic}(a)")
    st.write(
        "Inspect может иметь непосредственную стоимость, но раскрывает, какое exploit-действие будет успешным. "
        "Именно это преимущество будущего решения делает поиск информации полезным."
    )

    st.subheader("Режимы эксперимента")
    st.markdown(
        """
        - **Demo:** пошаговый просмотр одного эпизода, belief state, энтропии, действий и EFE.
        - **Benchmark:** сравнение Random, Pragmatic-only, Active Inference и Bayesian Utility.
        - **Ablation:** сравнение Active Inference с вариантом без эпистемической ценности.
        - **Parameter Sweep:** изменение шума наблюдений и поиск режимов, где информация помогает.
        """
    )

    st.subheader("Как читать multi-seed результаты")
    st.markdown(
        """
        - **Средняя награда:** средняя суммарная награда по эпизодам и seed.
        - **Доля успеха:** доля эпизодов с успешным терминальным исходом.
        - **95% CI:** неопределённость средней оценки по независимым seed.
        - **Устойчивое преимущество:** нижняя граница CI выше нуля, поэтому преимущество положительно на уровне seed-агрегации.
        """
    )
    st.info(
        "Для исследовательского анализа используйте минимум три независимых seed. Положительное среднее без CI выше нуля является признаком, но не окончательным доказательством."
    )

    st.subheader("Рекомендуемый порядок работы")
    st.markdown(
        """
        1. Начните с **Demo**, чтобы увидеть одно решение.
        2. Запустите **Ablation**, чтобы отделить вклад эпистемической ценности.
        3. Запустите **Parameter Sweep** с несколькими seed.
        4. Используйте Research tab и CSV/JSON-файлы для отчёта.
        """
    )
