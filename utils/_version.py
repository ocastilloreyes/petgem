import re
from pathlib import Path

def _read_version_from_header():
    # Go from utils/ to project root
    root = Path(__file__).resolve().parents[1]
    header = root / "include" / "version.h"

    content = header.read_text()

    major = re.search(r"#define VERSION_MAJOR (\d+)", content).group(1)
    minor = re.search(r"#define VERSION_MINOR (\d+)", content).group(1)
    patch = re.search(r"#define VERSION_PATCH (\d+)", content).group(1)

    return f"{major}.{minor}.{patch}"

__version__ = _read_version_from_header()