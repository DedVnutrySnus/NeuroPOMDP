# NeuroPOMDP 0.1.0

## Release status

This release marks the first Windows desktop-ready build of NeuroPOMDP, packaging the research dashboard and benchmark tooling into a single runnable distribution.

## Highlights

- Discrete Active Inference benchmark with pragmatic and epistemic action values
- Reward-aware planning and no-epistemic ablation mode
- Benchmark, ablation, and observation-noise sweep experiments
- Multi-seed aggregation with 95% confidence intervals
- Reliable-advantage detection for regimes where the lower confidence bound remains above zero
- Streamlit research dashboard with Demo, Benchmark, Ablation, Parameter Sweep, Research, History, Instructions, and About tabs
- Windows desktop bundle generated with PyInstaller

## What is included

- Full Python project source for research and experiments
- Desktop launcher for local dashboard startup on Windows
- Benchmark CLI commands for demo, benchmark, ablation, and sweep runs
- JSON output exports for experiment summaries and aggregate metrics
- A packaged Windows release archive ready for local use

## Validation

The release has been validated with fresh project checks:

- Full automated test suite passed: 43 tests passing
- Windows packaged executable launches successfully in diagnostic mode
- Dashboard runtime loads without import failures
- Bootstrap/runtime dependencies are resolved in the frozen application bundle

## Windows usage

1. Extract the archive.
2. Run `NeuroPOMDP.exe` from the extracted `NeuroPOMDP` folder.
3. The executable starts a local dashboard in the default browser.

For source development or local debugging:

```bash
python -m neuropomdp.launcher
```

## Research notes

Use explicit seeds for repeatable experiments. For reporting, prefer multi-seed runs and inspect confidence intervals rather than relying on a single run.

## Project goals

NeuroPOMDP is designed as a compact, CPU-friendly Active Inference benchmark focused on the epistemic value of information. The project is intended to make the exploration/exploitation tradeoff legible and measurable in a small, interpretable POMDP setting.
