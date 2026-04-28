# UV Environment Guide

This project uses `uv` as the primary environment and dependency manager.

## 1) Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2) Create and sync environment

Runtime-only:

```bash
uv sync
```

Full robotics stack (recommended):

```bash
uv sync --extra full --extra dev
```

Minimal kinematics stack:

```bash
uv sync --extra kinematics
```

## 3) Validate critical dependencies

```bash
uv run python -c "import motorbridge; print('motorbridge OK')"
uv run python -c "import pinocchio as pin; print('pinocchio OK', pin.__version__)"
uv run pytest -q
uv run python scripts/verify_env.py --require-pinocchio
```

## 4) Dependency maintenance rules

- Keep `motorbridge` as the only motor/bus control substrate.
- Add new packages via `pyproject.toml` only.
- Commit `uv.lock` together with any dependency change.
- When changing dependency versions, rerun:

```bash
uv lock
uv sync --extra full --extra dev
uv run pytest -q
uv run python scripts/verify_env.py --require-pinocchio
```

## 5) Recommended daily commands

```bash
uv sync --extra full --extra dev
uv run pytest -q
```
