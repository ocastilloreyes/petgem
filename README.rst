.. raw:: html

   <div style="display: flex; align-items: center; margin-bottom: 20px;">
       <img src="docs/source/_static/petgem_logo.png" alt="PETGEM Logo" width="100px" style="margin-right: 20px;">
       <div>
           <h1>Parallel Edge-element Toolkit for General Electromagnetic Modeling</h1>
       </div>
   </div>

.. image:: https://img.shields.io/github/actions/workflow/status/ocastilloreyes/petgem/ci-develop.yml?branch=develop&label=CI
   :target: https://github.com/ocastilloreyes/petgem/actions/workflows/ci-develop.yml
   :alt: CI Status

.. image:: https://readthedocs.org/projects/petgem/badge/?version=latest
   :target: https://petgem.readthedocs.io/en/latest/index.html
   :alt: Documentation Status

.. image:: https://img.shields.io/github/license/ocastilloreyes/petgem
   :target: https://github.com/ocastilloreyes/petgem/blob/develop/LICENSE.rst
   :alt: License: BSD 3-Clause

.. image:: https://img.shields.io/github/v/release/ocastilloreyes/petgem
   :target: https://github.com/ocastilloreyes/petgem/releases
   :alt: Latest release

.. image:: https://img.shields.io/badge/docker-petgem--ci--env%3Alatest-blue?logo=docker
   :target: https://github.com/ocastilloreyes/petgem/pkgs/container/petgem-ci-env
   :alt: Docker image

**PETGEM** is a parallel code for 3D frequency-domain electromagnetic
modeling with high-order Nédélec (edge) finite elements, orders 1 to 6, on
unstructured tetrahedral meshes. It is written in C on top of
`PETSc <https://petsc.org/>`_ and MPI, and developed at the
`Universitat Politècnica de Catalunya (UPC) <https://www.ac.upc.edu/en?set_language=en>`_
and the `Barcelona Supercomputing Center (BSC) <https://www.bsc.es/es/discover-bsc/organisation/scientific-structure/geophysical-applications>`_.

It provides four kernels:

- ``fm.csem`` - CSEM forward modeling (frecuency domain)
- ``fm.tdcsem`` - CSEM forward modeling (time domain)
- ``fm.mt`` - MT forward modeling
- ``im.csem`` - CSEM inverse modeling (frequency domain, L-BFGS, adjoint-state gradients)

and a ``petgem`` dispatcher that runs any of them. A small Python package in
``utils/`` builds the HDF5 input files and reads the results.

Dependencies
------------
- `PETSc <https://petsc.org/>`_ with **complex scalars**, MPI, MUMPS and HDF5, plus ``petsc4py``
- `Gmsh <http://gmsh.info/>`_
- Python 3 with ``numpy``, ``h5py`` and ``meshio``; ``matplotlib`` for the
  example scripts, ``pytest`` for the tests

The Docker image built from ``docker/Dockerfile`` contains all of them.

Quick start
-----------
.. code-block:: bash

   git clone https://github.com/ocastilloreyes/petgem.git
   cd petgem

   docker pull ghcr.io/ocastilloreyes/petgem-ci-env:latest
   docker tag ghcr.io/ocastilloreyes/petgem-ci-env:latest petgem-env
   docker run --rm -it -u $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace petgem-env bash

   # inside the container
   make
   bash examples/fm_csem_marine/scripts/build_bundles.sh 1
   (cd examples/fm_csem_marine && mpirun -n 4 ../../build/fm.csem -options_file configs/params.txt)
   python3 examples/fm_csem_marine/scripts/postprocess.py

This runs the marine CSEM example and compares it with its reference. The
other cases are in ``examples/``.

Documentation
-------------
https://petgem.readthedocs.io/en/latest/index.html

Citing PETGEM
-------------
If you use **PETGEM** for CSEM forward modeling in your research, please cite:

- Castillo-Reyes, O., de la Puente, J., García-Castillo, L. E., Cela, J.M. (2019).
  *Parallel 3-D marine controlled-source electromagnetic modelling using high-order
  tetrahedral Nédélec elements*. Geophysical Journal International, 219(1), 39–65.
  https://doi.org/10.1093/gji/ggz285

- Castillo-Reyes, O., de la Puente, J., Cela, J. M. (2018). *PETGEM: A parallel
  code for 3D CSEM forward modeling using edge finite elements*. Computers &
  Geosciences, 119, 123–136. https://doi.org/10.1016/j.cageo.2018.07.005

If you use **PETGEM** for MT forward modeling in your research, please cite:

- Castillo-Reyes, O., Modesto, D., Queralt, P., Marcuello, A., Ledo, J.,
  Amor-Martin, A., de la Puente, J., García-Castillo, L.E. (2022). *3D
  magnetotelluric modeling using high-order tetrahedral Nédélec elements on
  massively parallel computing platforms*. Computers & Geosciences, 160, 105030.
  https://doi.org/10.1016/j.cageo.2021.105030

More publications are listed in the documentation and on
`Google Scholar <https://scholar.google.es/citations?user=ifjbBssAAAAJ&hl=es&oi=ao>`_.

License
-------
BSD 3-Clause. See `LICENSE.rst <LICENSE.rst>`_.
