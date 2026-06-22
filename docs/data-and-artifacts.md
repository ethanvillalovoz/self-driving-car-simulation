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
77fa26f38499fe3e8e47cdd67be831720a1b836d24545de8f2fe54b80e1a02eb  self-driving-car-model-v1.tar.gz
dbf05e94f21bbfcbf1c5066a9e199aebc6bc843df77886b7284afe6509f61e16  self-driving-car-data-v1.tar.gz
b3b8cbef851bee3f8529a3bc4870fbb20de23741fdc3b700bb328c45e69ef1d2  self-driving-car-simulator-linux-v1.tar.gz
```

The repository itself keeps:

- `docs/simulator_screenshot.png` - simulator screenshot for the README.
- `docs/loss_curve.png` - training/validation loss curve from the notebook run.

## Reproducibility Notes

- The notebook expects `data/driving_log.csv` and `data/IMG/` to exist relative to `notebooks/`.
- The inference server expects the trained model at `model/model.h5`.
- The Linux simulator artifact is provided for convenience; users on macOS or Windows should download the matching simulator build from the Udacity simulator repository linked in the README.
- Large regenerated files, alternate model checkpoints, downloaded simulator builds, and local notebook executions should stay out of version control unless they are intentionally promoted as public release artifacts.
