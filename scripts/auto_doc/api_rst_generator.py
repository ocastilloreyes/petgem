"""Write the Sphinx/Breathe pages of the C API reference, plus an index.

Public headers in include/ are grouped into the pages listed in PAGES.
Private headers (*_internal.h) are skipped. A public header missing from
PAGES is an error, so a new module must be placed in a page.

Output: docs/source/api/<page>.rst and docs/source/api/index.rst.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER_DIR = REPO_ROOT / "include"
API_RST_DIR = REPO_ROOT / "docs" / "source" / "api"
DOXYGEN_PROJECT = "PETGEM"

# (page, title, description, headers)
PAGES = [
    ("kernels", "Kernels and runtime",
     "Kernel entry points and the run scaffolding they share.",
     ["kernels.h", "runtime.h"]),
    ("io", "Input, output and common utilities",
     "Common parameters, bundle loading, output paths and provenance, report "
     "printing, constants and version.",
     ["io.h", "common.h", "constants.h", "version.h"]),
    ("discretization", "Discretization",
     "Nédélec and nodal bases, DMPlex grid setup, element matrices and "
     "quadrature.",
     ["fe_nedelec.h", "fe_nodal.h", "grid.h", "fem.h"]),
    ("numerics", "Assembly, solver and receivers",
     "Global operator and discrete gradient, linear solve and BDDC setup, "
     "interpolation at the receivers.",
     ["assembly.h", "solver.h", "receiver_interp.h"]),
    ("physics", "CSEM and MT",
     "Transmitters and dipole sources (CSEM); 1D boundary field, boundary "
     "right-hand side and transfer functions (MT).",
     ["csem.h", "mt.h"]),
    ("inversion", "Inversion",
     "im.csem parameters, objective and gradient, and the L-BFGS optimizer.",
     ["inversion.h", "lbfgs.h"]),
]


def main():
    headers = {h.name for h in HEADER_DIR.glob("*.h") if not h.name.endswith("_internal.h")}
    if not headers:
        raise SystemExit(f"ERROR: no public headers found in {HEADER_DIR}")
    listed = {h for _, _, _, hs in PAGES for h in hs}
    if headers - listed:
        raise SystemExit(f"ERROR: headers not assigned to an API page: {sorted(headers - listed)}")
    if listed - headers:
        raise SystemExit(f"ERROR: API pages list missing headers: {sorted(listed - headers)}")

    API_RST_DIR.mkdir(parents=True, exist_ok=True)
    for page, title, description, hs in PAGES:
        text = f".. _api-{page}:\n\n{title}\n{'=' * len(title)}\n\n{description}\n\n"
        for h in hs:
            text += (f"``{h}``\n{'-' * (len(h) + 4)}\n\n"
                     f".. doxygenfile:: {h}\n   :project: {DOXYGEN_PROJECT}\n\n")
        out = API_RST_DIR / f"{page}.rst"
        out.write_text(text)
        print(f"Generated: {out}")

    index = API_RST_DIR / "index.rst"
    index.write_text("C API reference\n===============\n\n"
                     "Generated from the Doxygen comments of the public headers in ``include/``.\n\n"
                     ".. toctree::\n   :maxdepth: 1\n\n"
                     + "".join(f"   {page}\n" for page, *_ in PAGES))
    print(f"Generated: {index}")


if __name__ == "__main__":
    main()
