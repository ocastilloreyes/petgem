.. image:: docs/source/_static/petgem_logo.png
   :alt: PETGEM Logo
   :align: center
   :width: 300px

==================================
PETGEM: Parallel Exascale Toolkit for Geophysical Electromagnetic Modeling
==================================

.. image:: https://img.shields.io/github/actions/workflow/status/ocastilloreyes/petgem/ci.yml?branch=develop
   :target: https://github.com/ocastilloreyes/petgem/actions
   :alt: CI Status

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://petgem.readthedocs.io/en/latest/index.html
   :alt: Documentation

.. image:: https://img.shields.io/github/license/ocastilloreyes/petgem
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License: BSD 3-Clause

.. image:: https://img.shields.io/github/v/release/ocastilloreyes/petgem
   :target: https://github.com/ocastilloreyes/petgem/releases
   :alt: Latest Release


Summary
-------

**PETGEM** (Parallel Exascale Toolkit for Geophysical Electromagnetic Modeling) is a high-performance open-source software designed for 
the simulation of electromagnetic (EM) fields in geophysical exploration.  
It is developed and maintained by researchers at the **Universitat Politècnica de Catalunya (UPC)** and the **Barcelona Supercomputing Center (BSC)**.

Key features include:

* Parallel and scalable solver for large-scale 3D EM forward modeling.
* Support for tetrahedral meshes and high-order edge finite element formulations.
* Optimized for **HPC clusters and exascale architectures**.
* Flexible **C kernel** for performance.


Dependencies
------------

PETGEM requires the following main dependencies:

* `PETSc <https://petsc.org/>`_ (with MPI, BLAS/LAPACK, MUMPS, HDF5)
* `Gmsh <http://gmsh.info/>`_
* Python 3.x packages:
  - numpy
  - matplotlib
  - h5py
  - meshio
  - sphinx (for documentation)

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

   # Compile code
   make USE_EXTRAE=0

   # Run a test model
   gmsh tests/canonical_model/mesh.geo -3
   python3 tests/canonical_model/generate_resistivity_model.py
   python3 tests/canonical_model/generate_params_file.py


Documentation
-------------

Full user and developer documentation is available at:

📖 https://petgem.readthedocs.io/en/latest/index.html


Citing PETGEM
-------------

If you use **PETGEM** in your research, please cite the following articles:

* `Castillo-Reyes, O. et al. (2019) <https://doi.org/10.1093/gji/ggz285>`_:

  Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J.M. (2019).
  *Parallel 3-D marine controlled-source electromagnetic modelling using high-order
  tetrahedral Nédélec elements*. Geophysical Journal International, Volume 219,
  Issue 1, October 2019, Pages 39–65, https://doi.org/10.1093/gji/ggz285

* `Castillo-Reyes, O. et al. (2018) <https://doi.org/10.1016/j.cageo.2018.07.005>`_:

  Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
  code for 3D CSEM forward modeling using edge finite elements*. Computers &
  Geosciences, vol 119: 123-136. ISSN 0098-3004,  Elsevier.
  https://doi.org/10.1016/j.cageo.2018.07.005

For additional publications, you may consult:

* `Google Scholar profile <https://scholar.google.es/citations?user=ifjbBssAAAAJ&hl=es&oi=ao>`_  
* `Publication list <docs/source/publications.rst>`_ (detailed bibliography)


License
-------

This project is distributed under the **BSD 3-Clause License**.  
See the `LICENSE <LICENSE>`_ file for details.




**petgem** is developed as open-source under BSD-3 license at Computer Applications
in Science & Engineering of the Barcelona Supercomputing Center - Centro Nacional
de Supercomputación. Please, see the CONDITIONS OF USE described in the LICENSE.rst file.
