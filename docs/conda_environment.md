# Conda Environment Guide

This project supports Conda as an alternative to `uv`.

## 1) Install Miniconda or Mambaforge

- Miniconda: https://docs.conda.io/en/latest/miniconda.html
- Mambaforge (recommended for faster solve): https://github.com/conda-forge/miniforge

## 2) Create environment (runtime-only)

```bash
conda env create -f conda_environments/environment.runtime.yml
conda activate motorbridge-arm-runtime
```

## 3) Create environment (full robotics stack)

```bash
conda env create -f conda_environments/environment.full.yml
conda activate motorbridge-arm-full
```

## 4) Validate environment

```bash
python scripts/verify_env.py --require-pinocchio
pytest -q
```

## 5) Update dependencies

1. Update `pyproject.toml` first.
2. Update corresponding `conda_environments/*.yml`.
3. Recreate and verify:

```bash
conda env update -f conda_environments/environment.full.yml --prune
python scripts/verify_env.py --require-pinocchio
pytest -q
```

## 6) Optional lockfile workflow (recommended)

For fully reproducible conda installs, use `conda-lock`:

```bash
pip install conda-lock
conda-lock -f conda_environments/environment.full.yml -p linux-64
```

Commit the generated `conda-lock.yml` when your team adopts this flow.
