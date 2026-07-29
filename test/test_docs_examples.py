import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "docs" / "examples"

EXAMPLE_SCRIPTS = sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda p: p.stem)
def test_docs_example(script_path: Path, tmp_path: Path):
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # to avoid displaying plots

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"{script_path} failed with exit code {result.returncode}\n\n"
        f"stdout:\n{result.stdout}\n\n"
        f"stderr:\n{result.stderr}"
    )
