# Data And Artifacts

This repository includes the artifacts needed to inspect and rerun the behavioral cloning demo without hunting through external links.

## Included Artifacts

- `data/driving_log.csv` - simulator telemetry log with image paths, steering angles, throttle, brake/reverse, and speed.
- `data/IMG/` - captured center, left, and right camera frames from the Udacity simulator.
- `model/model.h5` - trained Keras model used by `drive.py` for autonomous mode.
- `simulator-linux/` - Linux build of the Udacity self-driving car simulator.
- `docs/simulator_screenshot.png` - simulator screenshot for the README.
- `docs/loss_curve.png` - training/validation loss curve from the notebook run.

## Reproducibility Notes

- The notebook expects `data/driving_log.csv` and `data/IMG/` to exist relative to `notebooks/`.
- The inference server expects the trained model at `model/model.h5`.
- The Linux simulator binary is tracked for convenience; users on macOS or Windows should download the matching simulator build from the Udacity simulator repository linked in the README.
- Large regenerated files, alternate model checkpoints, and local notebook executions should stay out of version control unless they are intentionally promoted as public artifacts.
