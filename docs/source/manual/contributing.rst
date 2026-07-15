============
Contributing
============

Conventions for working on **PETGEM**'s code and documentation.

Building
--------
``make`` builds all three binaries; see :doc:`install` for the build options
(``USE_EXTRAE``, ``USE_INTEL``, ``V=1``, ``NO_COLOR``). Object files are not
tracked against header changes, so rebuild from clean after editing anything in
``include/``:

.. code-block:: bash

   make clean && make

Run ``make help`` for the target list.

Source layout
-------------
- ``include/`` - public headers. These are what the C API reference is generated
  from; ``*_internal.h`` headers are private to ``src/`` and excluded.
- ``src/`` - translation units. The forward and inverse kernel cores
  (``fm_csem.c``, ``im_csem.c``) expose ``runForward`` / ``runInverse``; the
  ``*_main.c`` files are thin ``main()`` wrappers for the single-purpose
  binaries, and ``petgem.c`` is the dispatcher. Inversion-only code lives in
  ``inversion.c``, ``inversion_smoother.c``, and ``lbfgs.c``.
- The finite-element core is ``fe_nedelec.c`` (Nédélec basis), ``fe_nodal.c``
  (:math:`H^1` basis), ``fem.c`` (element matrices), and ``assembly.c`` (global
  assembly, including the discrete gradient and the MMS right-hand side).
- ``utils/`` - the Python pre/post-processing package (imported as ``petgem``).

The interface layer
*******************
Input and output handling is shared, and lives in ``io.c``:

- ``petgemParams`` is the parameter base **common to both kernels** (input
  bundle, output directory and stem, polynomial order, MPI task count). It is
  parsed by ``readPetgemParams()``. ``fm.csem`` uses it directly; ``im.csem``
  embeds one as ``imParams.common`` and extends it with the inversion-only
  controls, which ``readimParams()`` parses.
- Every inversion option is ``-im_*``. The retired ``-inv_*`` spelling is
  rejected with an error naming its replacement.
- Both kernels compose output paths with ``buildOutputPath()`` and write the
  same root provenance block with ``writeRunProvenance()`` (see :doc:`formats`).
  Add product-specific attributes after that call rather than writing a second,
  parallel provenance block.

``scripts/auto_doc/check_doc_drift.sh`` guards the retired names, so an old
spelling reintroduced in ``src/`` or ``include/`` fails CI.

Code conventions
----------------
- The compiler runs with ``-Wall -Wextra`` (plus ``-Wpedantic`` on gcc). Keep new
  code warning-clean.
- Keep the file-header comment block consistent across ``.c``/``.h`` files, and
  keep Doxygen ``@param`` names in step with the signatures - CI enforces the
  latter (see below).

Documentation
-------------
The documentation lives under ``docs/source`` and is built with Sphinx. The C
API reference is generated from the headers in ``include/`` with Doxygen and
Breathe; the Python API with autodoc.

.. code-block:: bash

   make docs        # output: docs/build/html/index.html

``make docs`` runs ``scripts/auto_doc/prepare_docs.sh`` (clean, Doxygen XML,
API ``.rst`` generation) and then Sphinx. Read the Docs runs the same script as
its ``pre_build`` job, so the local and hosted builds stay in step. The
generated ``docs/source/api/`` pages are not committed.

Aim for a warning-free build, and keep the pages consistent with the code: every
documented option, path, and API should exist in the repository.

Continuous integration
----------------------
``.github/workflows/ci-develop.yml`` gates code and documentation jobs
independently:

- **Code**: compiles ``fm.csem`` and runs the forward test suite via the
  reusable ``tests-fm-csem.yml`` workflow (see :doc:`testing`), inside the
  project CI image.
- **Docs**: builds the documentation (``make docs``) and runs
  ``scripts/auto_doc/check_doc_drift.sh``, a drift guard that fails if a retired
  code token reappears in ``src/``/``include/``, or if a Doxygen ``@param`` no
  longer matches its function signature.

The CI image itself is built separately by ``image.yml``, only when ``docker/**``
changes.

Testing
-------
Run the suite before submitting changes - see :doc:`testing`. New behavior
should come with a test; a change that alters the forward result must be
reconciled against the golden references, which are regenerated deliberately
and reviewed as part of the diff.
