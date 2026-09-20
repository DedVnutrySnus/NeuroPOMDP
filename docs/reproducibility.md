# Reproducibility

NeuroPOMDP experiments are controlled by explicit configuration values and random seeds.

## Environment

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

For an isolated environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Tests

Run the automated tests with:

```bash
python -m pytest tests -q
```

## Repeated Experiments

Use explicit seeds for repeated experiments:

```bash
python -m neuropomdp.main benchmark --seeds 0 1 2 --no-plots
```

The same seed and configuration are intended to reproduce the same sampled episodes.

## Outputs

Experiment outputs are written to the configured output directory, which defaults to `outputs`.

Recommended practice is to keep the command, seed list, dependency versions, and generated JSON outputs together when comparing results.
