# Model Card

## Model

An NVIDIA-inspired convolutional regressor with 252,219 trainable parameters. It maps one preprocessed front-camera frame to a scalar steering command.

## Intended Use

- Reproduce behavioral cloning in the Udacity self-driving car simulator.
- Study image preprocessing, steering-data imbalance, augmentation, and closed-loop inference integration.
- Provide a compact educational baseline for simulator experiments.

## Out Of Scope

- Real vehicles or physical safety decisions.
- Public-road, pedestrian, weather, or multi-camera perception claims.
- Driver assistance, obstacle avoidance, speed planning, or localization.
- Using offline MSE as evidence of safe closed-loop behavior.

## Inputs And Outputs

Input frames are cropped from rows 60 through 134, converted from RGB to YUV, Gaussian blurred, resized to `200 x 66`, and normalized to `[0, 1]`. The output is one steering value, clamped to `[-1, 1]` by default. Throttle is a separate fixed proportional controller.

## Training Data

The release contains simulator telemetry collected for this project. Records are balanced across 25 steering bins with a default cap of 400 center-camera records per bin. Side cameras add fixed `+0.15` and `-0.15` steering corrections. The maintained trainer uses a deterministic seed and 80/20 sample split.

No demographic or human-subject attributes are used. The primary distribution risk is environmental: one simulator, limited tracks, limited lighting and road geometry, and one collection procedure.

## Recorded Evaluation

The original notebook records a best validation MSE of `0.0707725` at epoch 4 across a 10-epoch run. Exact text-output values extracted before notebook cleanup are preserved in [`original-run-metrics.json`](../examples/original-run-metrics.json).

No closed-loop completion rate, lane-departure count, intervention count, or multi-seed uncertainty was recorded. Those measurements are required before making stronger behavior claims.

## Failure Modes

- Distribution shift in scenery, brightness, road curvature, camera position, or speed.
- Compounding errors after the vehicle leaves the training trajectory.
- Ambiguous frame-only situations that require temporal context.
- Fixed throttle behavior that ignores curvature and model uncertainty.
- Steering saturation hiding an out-of-distribution model prediction.

## Versioning

The `v1.0.0` release preserves the original `.h5` model. Source version 1.1 adds validated inference and deterministic training code without claiming the model was retrained.
