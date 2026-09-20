# Experiments

The command line interface is available through:

```bash
python -m neuropomdp.main <command>
```

All experiment commands support common configuration flags such as `--seed`, `--episodes`, `--episode-length`, `--output-dir`, `--no-plots`, and environment parameters.

## Demo

```bash
python -m neuropomdp.main demo
```

Runs one episode and writes `demo_episode.json`. When plotting is enabled, it also writes belief, entropy, action, value, and expected free energy figures.

## Benchmark

```bash
python -m neuropomdp.main benchmark
```

Compares the available agents and writes `benchmark.json`.

For repeated runs:

```bash
python -m neuropomdp.main benchmark --seeds 0 1 2
```

This writes `benchmark_multi.json`.

## Ablation

```bash
python -m neuropomdp.main ablation
```

Compares the full Active Inference agent with a no-epistemic baseline.

For repeated runs:

```bash
python -m neuropomdp.main ablation --seeds 0 1 2
```

This writes `ablation_multi.json` and `ablation_multi_summary.json`.

## Observation-Noise Sweep

```bash
python -m neuropomdp.main sweep --sweep-noise-values 0.05 0.15 0.30 0.45
```

Runs the active agent and no-epistemic baseline at each observation-noise value.

For repeated runs:

```bash
python -m neuropomdp.main sweep \
    --seeds 0 1 2 \
    --episodes 32 \
    --sweep-noise-values 0.05 0.15 0.30 0.45 \
    --output-dir outputs/research_sweep \
    --no-plots
```

This writes `sweep_multi.json`, `sweep_multi_summary.json`, and `sweep_multi_effect_summary.json`.

## Metrics

The main reported metrics are cumulative reward, success rate, posterior entropy, information-action frequency, and confidence intervals for multi-seed runs.
