# Contributing

Thanks for improving Self-Driving Car Simulation. Keep changes reproducible, narrowly scoped, and honest about what was measured.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install `.[training]` or `.[simulator]` only when working on those paths.

## Before A Pull Request

```bash
ruff check src tests scripts drive.py
ruff format --check src tests scripts drive.py
pytest -q
python -m json.tool notebooks/behavioral_cloning.ipynb >/dev/null
```

- Add tests for changed preprocessing, telemetry, control, data, or extraction behavior.
- Never commit downloaded datasets, simulator builds, model checkpoints, or secrets.
- Update the model card when intended use, data assumptions, or limitations change.
- Include exact evaluation commands and distinguish notebook observations from new results.
- Regenerate `docs/demo-preview.webp` when its source or script changes.

## Research Claims

Do not describe validation MSE as autonomous-driving success. Closed-loop claims need a documented track, simulator version, number of runs, intervention definition, and aggregate results.

By participating, you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
