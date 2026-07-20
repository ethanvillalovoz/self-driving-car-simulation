# Reproducibility

## Environment

- Python 3.11 or newer for maintained source and tests.
- TensorFlow 2.16 to 2.19 for training or model loading.
- Legacy `python-socketio==4.6.1` and `python-engineio==3.13.2` for simulator protocol compatibility.

Use separate extras so core tests do not require TensorFlow:

```bash
pip install -e ".[dev]"
pip install -e ".[training]"
pip install -e ".[simulator]"
```

## Deterministic Controls

`self-driving-train` defaults to seed 42. The seed controls steering-bin subsampling, train/validation assignment, sample selection, translation, brightness, and horizontal flips. TensorFlow kernels can still vary across hardware; record the platform and framework version for exact comparisons.

## Original Run

The historical notebook recorded:

```text
parameters: 252,219
epochs: 10
training time: 789.10 seconds
best validation MSE: 0.0707725 at epoch 4
final training MSE: 0.0731621
final validation MSE: 0.0708167
```

Those values came from the original notebook environment and stochastic generator. Version 1.1 does not relabel them as results from the refactored trainer.

The same values are committed as machine-readable JSON in [`examples/original-run-metrics.json`](../examples/original-run-metrics.json).

## Public Figure

The README overview is generated from the committed replay poster, simulator frame, historical
metrics JSON, and maintained implementation paths:

```bash
python scripts/render_public_figures.py
```

The command rewrites `behavioral-cloning-overview.{svg,pdf,png}` in `docs/media/`. Its figure
contract and input/output hashes are stored in `docs/figures/behavioral-cloning-overview/`.

## Stronger Evaluation Protocol

For a future closed-loop claim, record:

1. simulator build and track;
2. model and data release checksum;
3. initial condition and speed limit;
4. at least five runs per condition;
5. completion, interventions, lane departures, and recovery time;
6. aggregate mean, spread, and every failed run.

That evidence should be added as a versioned machine-readable artifact, not only a README sentence.
