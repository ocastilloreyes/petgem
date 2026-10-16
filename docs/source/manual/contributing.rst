============
Contributing
============

Conventions for working on **PETGEM**'s code and documentation.

Building
--------
``make`` builds the five binaries; see :doc:`install` for the build options
(``USE_EXTRAE``, ``USE_INTEL``, ``V=1``, ``NO_COLOR``). Objects are rebuilt when
a header they include changes. Run ``make help`` for the target list.

Source layout
-------------
``include/`` holds the public headers (the C API reference is generated from
them; ``*_internal.h`` headers are private to ``src/``), and ``src/`` the
translation units. The modules form layers; a module only depends on its own
layer or on lower ones:

.. list-table::
   :header-rows: 1
   :widths: 6 18 76

   * - Layer
     - Role
     - Modules
   * - 0
     - Base
     - ``constants``, ``common`` (report printing, timers), ``version``
   * - 1
     - Generic I/O
     - ``io``: common parameters, the model part of the bundle, output paths,
       run provenance
   * - 2
     - Discretization
     - ``fe_nedelec`` (Nédélec basis), ``fe_nodal`` (:math:`H^1` basis),
       ``grid`` (DMPlex sections, cells, boundary faces), ``fem`` (element
       matrices, quadrature)
   * - 3
     - Numerics
     - ``assembly`` (Maxwell operator, discrete gradient, mass refill,
       right-hand-side containers), ``solver`` (KSP, PCBDDC, primal vertices),
       ``receiver_interp``
   * - 4
     - Physics
     - ``csem`` (transmitters, PEC grid, dipole right-hand side, receiver
       fields), ``mt`` (1D boundary field, boundary right-hand side, impedance
       and tipper)
   * - 5
     - Inversion
     - ``inversion``, ``inversion_io``, ``inversion_smoother``, ``lbfgs``
   * - 6
     - Application
     - ``runtime`` (PETSc start-up, timed phases, Extrae events, common report
       sections), the kernels ``fm_csem``, ``fm_mt``, ``im_csem``, ``td_fm_csem``, the
       dispatcher ``petgem`` and the ``*_main`` entry points

``csem`` and ``mt`` are independent of each other. ``tests/unit/test_layers.py``
checks these rules on every ``#include``, and fails when a new module has not
been placed in a layer.

``utils/`` is the Python pre/post-processing package (installable as
``petgem``; ``utils/preprocess.py`` and ``utils/make_observed.py`` always use
the code next to them).

Shared parameters and output
****************************
Generic input and output lives in ``io.c``; the physics-specific part lives in
``csem.c``, ``mt.c`` and ``inversion_io.c``:

- ``PetgemParams`` holds the parameters common to all kernels (input bundle,
  output directory and stem, polynomial order). It is parsed by
  ``readPetgemParams()``. ``fm.csem`` and ``fm.mt`` use it directly (``fm.mt``
  adds ``MtParams``, parsed by ``readMtParams()``; ``fm.tdcsem`` adds the
  ``-tdcsem_*`` options); ``im.csem`` embeds one as
  ``ImParams.common`` and extends it with the inversion-only controls, which
  ``readImParams()`` parses.
- Every inversion option is ``-im_*``.
- Every kernel composes output paths with ``buildOutputPath()`` and writes the
  same root provenance block with ``writeRunProvenance()`` (see :doc:`formats`).
  Add product-specific attributes after that call rather than writing a second,
  parallel provenance block.

Code conventions
----------------
- The compiler runs with ``-Wall -Wextra`` (plus ``-Wpedantic`` on gcc). Keep new
  code warning-clean.
- Format C sources with ``clang-format -i`` and the repository's
  ``.clang-format``.
- Functions and variables are lowerCamelCase, types are PascalCase, macros and
  enum constants are UPPER_CASE. Acronyms are written as words
  (``assembleCsemRhs``, ``MtParams``, ``createCellScalarDm``); PETSc names keep
  PETSc's spelling.
- Parameter structs are named ``params`` (``PetgemParams``), ``imParams``
  (``ImParams``) and ``mtParams`` (``MtParams``).
- Use American spelling in code, comments and documentation.
- Keep the file-header comment block of ``.c``/``.h`` files, and document
  public functions in their header with Doxygen (``@brief``, ``@param``,
  ``@return``).

Documentation
-------------
The documentation lives under ``docs/source`` and is built with Sphinx. The C
API reference is generated from the headers in ``include/`` with Doxygen and
Breathe; the Python API with autodoc.

.. code-block:: bash

   # Build the documentation
   make docs

``make docs`` runs ``scripts/auto_doc/prepare_docs.sh`` (clean, Doxygen XML,
API ``.rst`` generation) and then Sphinx. Read the Docs runs the same script as
its ``pre_build`` job, so the local and hosted builds stay in step. The
generated ``docs/source/api/`` pages are not committed.

Every option, path and function named in the pages must exist in the code.

Continuous integration
----------------------
``.github/workflows/ci-develop.yml`` runs two independent groups of jobs. The
code jobs build every binary and run the tests (see :doc:`testing`). The docs
jobs run ``make docs`` and ``scripts/auto_doc/check_doc_drift.sh``, which
fails when a retired name reappears in ``src/`` or ``include/``, when a
Doxygen ``@param`` no longer matches its function, or when a prototype in
``include/`` names its parameters differently from the definition. The CI
image is built by ``image.yml`` when ``docker/`` changes.

Testing
-------
Run the suite before submitting changes (see :doc:`testing`). New behavior
should come with a test. A change of the results needs regenerated goldens,
committed and reviewed with the change.
