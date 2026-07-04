.. raw:: html

   <div style="display: flex; align-items: center; margin-bottom: 20px;">
       <img src="docs/source/_static/petgem_logo.png" alt="PETGEM Logo" width="100px" style="margin-right: 20px;">
       <div>
           <h1>Parallel Edge-element Toolkit for General Electromagnetic Modeling</h1>
       </div>
   </div>

.. image:: https://img.shields.io/github/actions/workflow/status/ocastilloreyes/petgem/ci-develop.yml?branch=develop
   :target: https://github.com/ocastilloreyes/petgem/actions
   :alt: CI Status

.. image:: https://readthedocs.org/projects/petgem/badge/?version=latest
   :target: https://petgem.readthedocs.io/en/latest/index.html
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/ocastilloreyes/petgem
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License: BSD 3-Clause

.. image:: https://img.shields.io/github/v/release/ocastilloreyes/petgem
   :target: https://github.com/ocastilloreyes/petgem/releases
   :alt: Latest release

.. image:: https://img.shields.io/github/v/tag/ocastilloreyes/petgem?label=docker%20image&logo=docker&sort=semver
   :target: https://github.com/ocastilloreyes/petgem/pkgs/container/petgem-ci-env
   :alt: Docker image version

**PETGEM** (Parallel Edge-element Toolkit for General Electromagnetic Modeling) is a high-performance open-source software designed for 
the simulation of electromagnetic (EM) fields. It is developed and maintained by researchers at the `Universitat Politècnica de Catalunya (UPC) <https://www.ac.upc.edu/en?set_language=en>`_ and the `Barcelona Supercomputing Center (BSC) <https://www.bsc.es/es/discover-bsc/organisation/scientific-structure/geophysical-applications>`_.

Key features include:

- Parallel and scalable solver for large-scale 3D EM forward and inverse modeling
- Support for tetrahedral meshes and high-order edge finite element formulations
- Optimized for **HPC clusters and exascale architectures**
- Flexible **C kernels** for performance

Dependencies
------------

PETGEM requires the following main dependencies:

- `PETSc <https://petsc.org/>`_ (with MPI, BLAS/LAPACK, MUMPS, HDF5, petsc4py)

- `Gmsh <http://gmsh.info/>`_

- Python 3.x packages
   - numpy
   - matplotlib
   - h5py
   - meshio
   - cython 
   - setuptools 
   - wheel
   - sphinx (for documentation)
   - pytest (for testing)

For a fully reproducible environment, a Docker image is provided (see below).

Docker support
--------------

You can build and run PETGEM inside Docker for a consistent development and testing environment.

.. code-block:: bash

   # Clone repository
   git clone https://github.com/ocastilloreyes/petgem.git
   cd petgem

   # Build Docker image
   docker build -t petgem-env -f docker/dockerfile.release .

   # Run PETGEM inside container
   docker run --rm -it -v $(pwd):/workspace -w /workspace petgem-env bash

   # Compile PETGEM
   make

   # Setup environment
   export MODEL_DIR=examples/canonical_model
   export ORDER=1

   # Mesh generation (specific to order=1)
   gmsh ${MODEL_DIR}/mesh.geo -3

   # Generate input data (mesh, params file)
   python3 utils/preprocess.py \
        -mode forward \
        -order ${ORDER} \
        -case_dir  ${MODEL_DIR} \
        -mesh_filename mesh.msh \
        -source_filename sources.txt \
        -receiver_filename receivers.txt \
        -sigma_file sigmas.txt \
        -input_filename input_p${ORDER}.h5 \
        -params_filename params_p${ORDER}.txt \
        -output_vtk model.vtu

   # Forward modeling
   mpirun -n 14 build/petgem modeling -options_file ${MODEL_DIR}/params_p${ORDER}.txt

   # Postprocess output
   python3 ${MODEL_DIR}/postprocess.py \
        -responses_filename responses_p${ORDER}.h5 \
        -case_dir ${MODEL_DIR} \
        -input_filename input_p${ORDER}.h5 \
        -reference_filename reference.h5 \
        -figure_filename fields_p${ORDER}.png \
        -tolerance 0.03


Documentation
-------------

Full user and developer documentation is available at:

📖 https://petgem.readthedocs.io/en/latest/index.html

Citing PETGEM
-------------

If you use **PETGEM** in your research, please cite the following articles:

- Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J.M. (2019).
  *Parallel 3-D marine controlled-source electromagnetic modelling using high-order
  tetrahedral Nédélec elements*. Geophysical Journal International, Volume 219,
  Issue 1, October 2019, Pages 39–65, https://doi.org/10.1093/gji/ggz285

- Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
  code for 3D CSEM forward modeling using edge finite elements*. Computers &
  Geosciences, vol 119: 123-136. ISSN 0098-3004,  Elsevier.
  https://doi.org/10.1016/j.cageo.2018.07.005

For additional publications, you may consult:

- `Google Scholar profile <https://scholar.google.es/citations?user=ifjbBssAAAAJ&hl=es&oi=ao>`_  

License
-------

This project is distributed under the **BSD 3-Clause License**.  
See the `License <LICENSE.rst>`_ for details.