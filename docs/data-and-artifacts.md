# Data And Artifact Manifest

Large runtime assets live in the `v1.0.0` GitHub Release so a source clone stays lightweight. The `self-driving-artifacts` command downloads each archive to a temporary directory, verifies its SHA-256 digest, rejects links/devices/path traversal, and then extracts regular files.

## Restore

```bash
self-driving-artifacts --model
self-driving-artifacts --data
self-driving-artifacts --simulator
self-driving-artifacts --all
```

The default destination is the current directory. Existing expected paths are not replaced unless `--force` is provided.

## Manifest

| Asset | Restored path | SHA-256 |
| --- | --- | --- |
| `self-driving-car-model-v1.tar.gz` | `model/model.h5` | `a4a551864792d24c78af9b31efcea1173207787bd72c939325a4c3795d0fc483` |
| `self-driving-car-data-v1.tar.gz` | `data/driving_log.csv`, `data/IMG/` | `5ebb75a78a11ca35f05b285674240e5f42f534c9094b183f80fa58a38162f845` |
| `self-driving-car-simulator-linux-v1.tar.gz` | `simulator-linux/` | `d8e5d79e7757e8cea4e26cfd6287313cdbdd108a3a972d6e88f0723e5cd2baa8` |

## Data Shape

The driving log contains center, left, and right camera paths plus steering, throttle, reverse, and speed fields. The maintained trainer uses the three camera paths and steering target:

- center camera: recorded steering;
- left camera: steering plus `0.15`;
- right camera: steering minus `0.15`.

The correction is an explicit training assumption, not a calibrated camera model.

## Source-Control Policy

Do not commit the downloaded archives, extracted camera frames, simulator binaries, local models, or checkpoints. Promote a replacement artifact only with a new release tag, checksum, model-card update, and exact regeneration command.
