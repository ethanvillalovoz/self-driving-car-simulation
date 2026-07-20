# Behavioral-Cloning Overview Figure Contract

## Claim

This figure should allow a skeptical reviewer to conclude that the repository separates
deterministic training-data preparation, shared image preprocessing and steering regression,
bounded simulator control, and offline replay diagnostics because it shows each path with its
committed inputs, implementation assumptions, and evidence boundary.

## Role and size

- Role: README system overview and paper-width method/evidence figure.
- Final size: `7.16 x 4.45` inches.
- Editable source: `src/behavioral_cloning/visualization.py`.
- Regeneration entry point: `scripts/render_public_figures.py`.
- Exports: vector SVG/PDF and a 300 dpi PNG.

## Evidence and encodings

- The camera strip, preprocessed model input, and steering trace are crops from the committed
  `docs/media/self-driving-offline-replay.webp` poster.
- The simulator deployment context is the committed `docs/simulator-frame.webp` capture.
- Training preparation, preprocessing dimensions, CNN layers, steering bounds, and throttle
  rule are read from the maintained source paths referenced by the provenance manifest.
- Historical parameter count, epoch count, and validation MSE come from
  `examples/original-run-metrics.json`.
- The composition uses a camera-first visual grammar rather than a dashboard of interchangeable
  cards: three training views establish the observation space, one continuous model spine links
  fixed preprocessing to the learned regressor and runtime command rules, and the offline steering
  trace is the dominant evidence region.
- Blue denotes learned image-to-steering computation, amber denotes recorded telemetry, green
  denotes bounded runtime control, and gray denotes fixed or deterministic processing. Labels
  preserve the distinctions in grayscale.

## Crop and selection record

The fixed replay poster is `1600 x 900` pixels. The renderer uses these immutable pixel crops:

- left camera: `(37, 104, 529, 350)`;
- center camera: `(550, 104, 1041, 350)`;
- right camera: `(1062, 104, 1554, 350)`;
- processed model input: `(1128, 451, 1564, 596)`;
- steering trace: `(37, 470, 1095, 745)`;
- simulator context: `(0, 125, 1600, 775)` from the committed simulator frame.

The renderer does not search the video or select a favorable outcome. It reorganizes regions
from the already committed poster so they remain legible at paper width.

## Conditions and boundary

- Three simulator cameras with side-camera steering corrections of `+0.15` and `-0.15`.
- Steering balancing uses 25 bins with a default cap of 400 center-camera records per bin.
- Maintained training uses seed 42, an 80/20 split, augmentation, and an NVIDIA-style CNN.
- Runtime input is one center-camera frame plus speed; steering is clamped to `[-1, 1]` and
  throttle uses a separate proportional rule clamped to `[0, 1]`.
- The offline replay contains 180 consecutive recorded simulator frames.
- The historical notebook recorded one 10-epoch run with no complete seed or hardware record.

The figure does not establish lane keeping, recovery, track completion, intervention rate,
closed-loop stability, real-world driving, multi-seed uncertainty, or safety.
