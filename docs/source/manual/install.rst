Installation
============

Requirements
------------
- Ubuntu 22.04 (tested)
- PETSc with MPI, HDF5, MUMPS, petsc4py 
- GMSH
- Python 3.10+ with numpy, h5py, meshio, matplotlib, sphinx
- Optional: Extrae for performance tracing

Docker installation
-------------------
PETGEM provides a ready-to-use Docker environment:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/ocastilloreyes/petgem.git
   cd petgem

   # Build Docker image
   docker build -t petgem-env -f docker/dockerfile.release .

   # Run PETGEM inside container
   docker run --rm -it -v $(pwd):/workspace -w /workspace petgem-env bash

   # Compile code
   make USE_EXTRAE=0

   # Run a test model
   gmsh tests/canonical_model/mesh.geo -3
   python3 tests/canonical_model/generate_resistivity_model.py
   python3 tests/canonical_model/generate_params_file.py


Makefile options
----------------
- Build PETGEM kernels:

.. code-block:: bash

    make all

- Clean build:

.. code-block:: bash

    make clean

- Optional build flags:

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - USE_INTEL=1
     - Use Intel MPI compiler instead of PETSc default
   * - USE_EXTRAE=1
     - Enable Extrae instrumentation for performance tracing
