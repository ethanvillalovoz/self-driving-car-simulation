# Data And Artifacts

This repository keeps source code, tests, documentation, and lightweight visuals in Git. Larger runtime and training artifacts are hosted as GitHub Release assets so the repository remains practical to clone.

## Release Artifacts

Run the downloader from the repository root:

```bash
python scripts/download_artifacts.py --all
```

The release contains:

- `self-driving-car-model-v1.tar.gz` - trained Keras model restored to `model/model.h5`.
- `self-driving-car-data-v1.tar.gz` - simulator telemetry log and camera frames restored to `data/`.
- `self-driving-car-simulator-linux-v1.tar.gz` - Linux build of the Udacity self-driving car simulator restored to `simulator-linux/`.

## Artifact Checksums

```text
a4a551864792d24c78af9b31efcea1173207787bd72c939325a4c3795d0fc483  self-driving-car-model-v1.tar.gz
5ebb75a78a11ca35f05b285674240e5f42f534c9094b183f80fa58a38162f845  self-driving-car-data-v1.tar.gz
d8e5d79e7757e8cea4e26cfd6287313cdbdd108a3a972d6e88f0723e5cd2baa8  self-driving-car-simulator-linux-v1.tar.gz
```

The repository itself keeps:

- `docs/simulator_screenshot.png` - simulator screenshot for the README.
- `docs/loss_curve.png` - training/validation loss curve from the notebook run.

## Reproducibility Notes

- The notebook expects `data/driving_log.csv` and `data/IMG/` to exist relative to `notebooks/`.
- The inference server expects the trained model at `model/model.h5`.
- The Linux simulator artifact is provided for convenience; users on macOS or Windows should download the matching simulator build from the Udacity simulator repository linked in the README.
- Large regenerated files, alternate model checkpoints, downloaded simulator builds, and local notebook executions should stay out of version control unless they are intentionally promoted as public release artifacts.
