import subprocess
import sys
from pathlib import Path


def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/build_demo_data.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "build site/data/demo.json" in result.stdout.lower()
