from __future__ import annotations

import subprocess


def test_no_private_kin_access():
    """Ensure nothing in motorbridge_arm_sdk accesses private Kinematics attributes."""
    sdk_dir = "motorbridge_arm_sdk/motorbridge_arm_sdk"
    patterns = ["_kin._model", "_kin._frame_id"]
    for pat in patterns:
        result = subprocess.run(
            ["grep", "-R", "--include=*.py", "--exclude-dir=__pycache__", "-c", pat, sdk_dir],
            capture_output=True,
            text=True,
        )
        # grep -c returns lines with "0" for no matches, or "file:N" for matches.
        # Exit code 1 means no matches at all (what we want).
        # Any non-zero exit from grep means no match in that file.
        lines = [l for l in result.stdout.strip().splitlines() if l]
        for line in lines:
            _, count = line.rsplit(":", 1)
            assert int(count) == 0, f"Found private access pattern '{pat}' in source"
