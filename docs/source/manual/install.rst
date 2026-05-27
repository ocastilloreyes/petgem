============
Installation
============
This section guides you through installing **PETGEM**. You will find instructions for setting up the environment, building the code, running tests, and generating documentation. **PETGEM** supports Linux-based systems and can be run natively or via Docker for a ready-to-use environment.

Requirements
------------
Before installing **PETGEM**, ensure the following dependencies are available:

- `PETSc <https://petsc.org/release/>`_ compiled with MPI and support for complex numbers. A recommended configuration is:

  .. code-block:: bash

     ./configure --with-cc=gcc --with-cxx=g++ --with-fc=gfortran \
                 --download-mpich --download-fblaslapack \
                 --with-scalar-type=complex --download-mumps --download-scalapack \
                 --download-ptscotch --download-cmake --with-debugging=1 \
                 --download-hdf5 --download-triangle
- `Gmsh <http://gmsh.info/>`_
- Python 3.10+ with packages: `numpy`, `h5py`, `meshio`, `matplotlib`, `sphinx`
- `Extrae <https://tools.bsc.es/extrae>`_ for performance tracing (optional)

Docker installation
-------------------
**PETGEM** provides a ready-to-use Docker environment for easy setup:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/ocastilloreyes/petgem.git
   cd petgem

   # Build Docker image
   docker build -t petgem-env -f docker/dockerfile.release .

   # Run Docker
   docker run --rm -it -v $(pwd):/workspace -w /workspace petgem-env

   # Compile PETGEM (inside container)
   make USE_EXTRAE=0

   # Run a test model
   gmsh tests/csem_model/mesh_p1.geo -3
   python3 tests/csem_model/generate_resistivity_model.py 1
   python3 tests/csem_model/generate_params_file.py 1 
   mpirun -n 4 build/fm.csem -options_file tests/csem_model/params_nord1.txt
   python3 tests/csem_model/compare_responses.py 1 

Makefile usage
--------------
**PETGEM** provides a Makefile to simplify building the code and generating documentation.

**Common makefile targets:**

.. list-table::
   :header-rows: 1

   * - Target
     - Description
   * - all
     - Build the PETGEM kernels (default)
   * - clean
     - Remove object files and executables
   * - docs
     - Generate all documentation
   * - clean_doc
     - Clean documentation
   * - help
     - Show Makefile help message

**Optional build options:**

You can set optional flags when invoking `make`: `make <target> OPTION=1`

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - USE_INTEL=1
     - Use Intel MPI compiler (`mpiicc`) instead of PETSc default
   * - USE_EXTRAE=1
     - Enable Extrae instrumentation for performance tracing

Python helpers
--------------
The pre- and post-processing scripts under ``utils/`` (e.g.
``utils/preprocess.py``) run directly from a clone - they add the in-tree
package to the path automatically, so no installation step is required:

.. code-block:: bash

   python3 utils/preprocess.py -mode forward ...

Optionally, install the Python layer so it is importable as ``petgem`` from
your own scripts (e.g. ``import petgem; petgem.readResponses(...)``):

.. code-block:: bash

   pip install -e .

This requires ``numpy``, ``meshio``, and ``petsc4py`` (already present in the
Docker image).