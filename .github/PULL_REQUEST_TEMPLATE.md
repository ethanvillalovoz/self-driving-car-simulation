## Summary

What changes, and why?

## Scope

- Affected path: training / inference / simulator / artifacts / docs
- Public API or artifact changes:
- Related issue:

## Evidence

List exact verification commands and results. If this changes model behavior, include the evaluation protocol and every relevant failure, not only a successful clip.

## Checklist

- [ ] `ruff check src tests scripts drive.py`
- [ ] `ruff format --check src tests scripts drive.py`
- [ ] `pytest -q`
- [ ] Notebook and JSON artifacts still parse
- [ ] Tests cover the changed behavior
- [ ] Documentation and model limitations are current
- [ ] No datasets, checkpoints, simulator binaries, or credentials are committed
