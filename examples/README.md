# Examples

This directory contains small command examples for running NeuroPOMDP experiments.

## Quick Demo

```bash
python -m neuropomdp.main demo --no-plots
```

## Multi-Seed Benchmark

```bash
python -m neuropomdp.main benchmark --seeds 0 1 2 --episodes 32 --no-plots
```

## Research Sweep

```bash
python -m neuropomdp.main sweep \
    --seeds 0 1 2 \
    --episodes 32 \
    --sweep-noise-values 0.05 0.15 0.30 0.45 \
    --output-dir outputs/research_sweep \
    --no-plots
```
