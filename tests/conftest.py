"""Shared pytest fixtures for the PETGEM test suite.

Adds the repo root to sys.path so `import petgem` resolves the `utils`
package without requiring an install. Exposes helper fixtures pointing at
the repository layout (REPO_ROOT, CASES_DIR, FM_CSEM_BIN).
"""
import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Make `import petgem` work from anywhere the tests run.  The repo layout
# has the package source at utils/, which pyproject.toml maps to the
# distribution name `petgem` at install time.  For local pytest runs we
# expose the same alias via sys.modules so tests work both with and
# without `pip install -e .`.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    import petgem  # noqa: F401 — works when the package is installed
except ImportError:
    import utils as _petgem_pkg
    sys.modules["petgem"] = _petgem_pkg


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def cases_dir(repo_root) -> Path:
    """Absolute path to tests/cases/."""
    return repo_root / "tests" / "cases"


@pytest.fixture(scope="session")
def fm_csem_binary(repo_root) -> Path:
    """Absolute path to the fm.csem binary, if built.

    Resolution order:
      1. $PETGEM_FM_CSEM environment variable.
      2. <repo_root>/fm.csem
      3. <repo_root>/build/fm.csem
      4. shutil.which("fm.csem") — PATH lookup.

    Returns the first hit. Tests that need this fixture should `pytest.skip`
    when the file does not exist (so the suite passes on CI machines that
    haven't built the C kernels).
    """
    env = os.environ.get("PETGEM_FM_CSEM")
    if env:
        return Path(env)

    for candidate in (repo_root / "fm.csem", repo_root / "build" / "fm.csem"):
        if candidate.exists():
            return candidate

    which = shutil.which("fm.csem")
    if which:
        return Path(which)

    # Return the conventional location so test failure messages are helpful;
    # individual tests must guard with `pytest.skip` when this doesn't exist.
    return repo_root / "fm.csem"
