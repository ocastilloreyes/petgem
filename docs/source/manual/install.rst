============
Installation
============

**PETGEM** is built with ``make`` against an existing PETSc installation. It
targets Linux; a Docker image with the full dependency stack is provided.

Requirements
------------
- `PETSc <https://petsc.org/release/>`_ built with MPI and **complex** scalars.
  The configuration used by the project image is:

  .. code-block:: bash

     ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran \
                 --download-mpich --download-fblaslapack \
                 --with-scalar-type=complex --download-mumps --download-scalapack \
                 --download-ptscotch --download-cmake --with-debugging=1 \
                 --download-hdf5 --download-triangle

- ``petsc4py`` (used by the Python preprocessing).
- `Gmsh <http://gmsh.info/>`_ - mesh generation.
- Python 3 with ``numpy``, ``h5py``, ``meshio``; ``matplotlib`` for the example
  post-processing scripts, ``pytest`` for the test suite.
- `Extrae <https://tools.bsc.es/extrae>`_ - optional, for performance tracing
  (``make USE_EXTRAE=1``).
- ``doxygen`` plus the packages in ``docs/requirements.txt`` - optional, to
  build the documentation.

``PETSC_DIR`` and ``PETSC_ARCH`` must be set; the Makefile stops with an
explicit error if ``PETSC_DIR`` does not point at a PETSc installation.

Docker
------
``docker/Dockerfile`` builds the image: PETSc (complex scalars), Gmsh,
Extrae, and the Python stack. CI publishes it as
``ghcr.io/ocastilloreyes/petgem-ci-env:latest``. The repository is mounted,
not copied:

.. code-block:: bash

   git clone https://github.com/ocastilloreyes/petgem.git
   cd petgem

   docker pull ghcr.io/ocastilloreyes/petgem-ci-env:latest
   docker tag ghcr.io/ocastilloreyes/petgem-ci-env:latest petgem-env
   docker run --rm -it -u $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace petgem-env bash

   # inside the container
   make

To build the image locally instead of pulling it:

.. code-block:: bash

   docker build -t petgem-env -f docker/Dockerfile .

See :doc:`quickstart` for a first run.

Building
--------
``make`` builds all five binaries into ``build/``:

- ``build/fm.csem`` - CSEM forward kernel
- ``build/im.csem`` - CSEM inverse kernel
- ``build/fm.mt`` - MT forward kernel
- ``build/fm.tdcsem`` - time-domain CSEM forward kernel
- ``build/petgem`` - dispatcher

With ``USE_EXTRAE=1`` the binaries are suffixed ``.extrae``
(``build/fm.csem.extrae``, ...). A single binary can be built on its own, e.g.
``make build/fm.csem``.

Makefile targets
****************

.. list-table::
   :header-rows: 1

   * - Target
     - Description
   * - ``all`` (default)
     - Build ``fm.csem``, ``im.csem``, ``fm.mt``, ``fm.tdcsem``, and ``petgem``
   * - ``clean``
     - Remove ``build/`` and object files
   * - ``docs``
     - Build the documentation (Doxygen XML + API stubs + Sphinx HTML)
   * - ``clean_doc``
     - Remove generated documentation artifacts
   * - ``help``
     - List targets and build options

Build options
*************
Set as ``make <target> OPTION=1``:

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``USE_EXTRAE=1``
     - Build with Extrae instrumentation (requires ``EXTRAE_HOME``)
   * - ``USE_INTEL=1``
     - Force Intel compiler flags. Auto-detected from PETSc's ``$(PCC)``, so it
       is normally not needed; ``USE_INTEL=0`` forces gcc-style flags.
   * - ``V=1``
     - Echo full compiler and linker command lines
   * - ``NO_COLOR=1``
     - Disable colored build output

Python helpers
--------------
The scripts under ``utils/`` run directly from a clone and always use the code
next to them, so no install step is required:

.. code-block:: bash

   python3 utils/preprocess.py -mode fm.csem ...

To import the package as ``petgem`` from your own scripts:

.. code-block:: bash

   pip install -e .
