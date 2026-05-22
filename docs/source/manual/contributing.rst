============
Contributing
============

This page collects the conventions for working on **PETGEM**'s code and
documentation.

Building
--------
``make`` builds all three binaries; see :doc:`install` for build options
(``USE_INTEL``, ``USE_EXTRAE``, ``V=1`` for verbose command lines, ``NO_COLOR``).
After changing a header, rebuild from clean so stale object files do not cause
ABI skew:

.. code-block:: bash

   make clean && make

Useful Makefile targets (run ``make help`` for the full list):

- ``make`` / ``make all`` - build ``fm.csem``, ``im.csem``, and ``petgem``
- ``make docs`` - build the documentation (Doxygen + Sphinx)
- ``make clean`` / ``make clean_doc`` - remove build / documentation artifacts

Code conventions
----------------
- **Do not modify the shared forward assembly/solver code when adding inverse
  features.** The forward kernel (``fm.csem``) must stay byte-for-byte
  unaffected; add new functions rather than changing shared paths.
- Keep the file-header comment block consistent across ``.c``/``.h`` files.
- The compiler is invoked with ``-Wall -Wextra`` (plus ``-Wpedantic`` on gcc);
  keep new code warning-clean.

Adding a Nédélec order
**********************
The high-order basis is dispatched through a per-order ops table
(``NedelecOps`` in ``include/hvfem.h``). To add an order: implement the
per-order static helpers in ``hvfem.c``, build a ``NedelecOps`` instance for
them, and register it in ``nedelecOpsForOrder()``. The rest of the assembly
calls through the table, so no hot-path ``switch (nord)`` is needed.

Documentation
-------------
The documentation lives under ``docs/source`` and is built with Sphinx; the C
API reference is generated from the headers with Doxygen + Breathe, and the
Python API with autodoc. Build it locally with:

.. code-block:: bash

   make docs        # output: docs/build/html/index.html

The clean + Doxygen + API-stub generation is shared with Read the Docs via
``scripts/auto_doc/prepare_docs.sh`` (single source of truth, so the local and
hosted builds stay in lockstep). Aim for a warning-free build.

Continuous integration
----------------------
The GitHub Actions workflow builds the binaries (with and without Extrae),
runs the pytest suite, exercises a forward smoke pipeline and one Extrae
profiling run at ``nord=1``, and builds the documentation. The integration
tests run inside the project Docker image, which provides PETSc (complex
scalars), HDF5, and the Python stack.

Testing
-------
Run the suite before submitting changes - see :doc:`testing`. New behavior
should come with a unit or integration test; numerical changes to the forward
kernel must keep the reference NRMSD within tolerance.
