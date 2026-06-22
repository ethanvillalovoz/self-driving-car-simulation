# Contributing

Thanks for helping improve Self-Driving Car Simulation. This repository contains a behavioral cloning demo with bundled data, model, and simulator artifacts, so changes should preserve reproducibility and avoid committing unnecessary generated files.

## Development Setup

For lightweight code checks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

For full notebook training or simulator inference, use the full dependency set:

```bash
conda create -n self-driving-car python=3.10 -y
conda activate self-driving-car
pip install -r requirements.txt
```

## Verification

Run these checks before opening a pull request:

```bash
python -m py_compile drive.py
pytest
python -m json.tool notebooks/behavioral_cloning.ipynb > /tmp/behavioral_cloning.ipynb
```

If your change affects training behavior, also rerun the notebook and include the resulting loss/behavior notes in the pull request.

## Contribution Guidelines

- Keep core model behavior and simulator protocol changes small and well explained.
- Do not commit local virtual environments, notebook checkpoints, alternate model checkpoints, or regenerated datasets unless they are intentionally promoted as public artifacts.
- Add or update tests for preprocessing, telemetry, or control logic changes.
- Update the README or docs when setup, commands, artifact paths, or simulator assumptions change.
- Preserve the existing trained model and dataset paths unless the migration is intentional and documented.

## Pull Requests

Please include:

- A short summary of the change.
- The affected area: data, notebook, inference server, model artifact, simulator, docs, or CI.
- Commands used to verify the change.
- Notes about any remaining limitations or follow-up work.

## Conduct

Be respectful, constructive, and specific when discussing changes or issues.
