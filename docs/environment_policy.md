# Environment Policy

This repository supports two officially maintained environment managers:

- `uv` (primary)
- `conda` (fully supported alternative)

## Mandatory rules

1. Do not install random ad-hoc packages into active envs.
2. Dependency source of truth is `pyproject.toml`.
3. `uv` users must update and commit `uv.lock`.
4. `conda` users must update `conda_environments/*.yml`.
5. Any dependency change requires:
   - env sync/update
   - `python scripts/verify_env.py --require-pinocchio`
   - `pytest -q`

## Required runtime capabilities

- `motorbridge` import must succeed.
- `pinocchio` import must succeed for full FK/IK/dynamics path.
- URDF default path must exist and be readable.

If `pinocchio` is missing, code falls back to simplified kinematics, which is not accepted for production runs.
