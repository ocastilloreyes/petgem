.. PETGEM documentation master file.

######################
PETGEM's documentation
######################

**PETGEM** (Parallel Edge-element Toolkit for General Electromagnetic Modeling)
is an open-source code for 3D forward and inverse modeling in the frequency
domain. It discretizes the electric field with high-order Nédélec (edge) finite
elements of polynomial order 1 to 6 on unstructured tetrahedral meshes, and is
written in C on top of `PETSc <https://petsc.org/release/>`_ and MPI.

The repository builds two kernels and a dispatcher:

- ``fm.csem`` - forward modeling: computes the CSEM response of a given
  conductivity model.
- ``im.csem`` - inverse modeling: recovers a per-material conductivity model
  from observed data with L-BFGS and adjoint-state gradients.
- ``petgem`` - a dispatcher that selects either kernel
  (``petgem fm`` / ``petgem im``).

Pre- and post-processing are handled by a small Python package under ``utils/``
(importable as ``petgem``), which assembles the HDF5 input bundle the kernels
read.

More information
----------------

- `PETGEM GitHub repository <https://github.com/ocastilloreyes/petgem/>`_
- See our publication list at :ref:`publications`

.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   manual/install
   manual/quickstart
   manual/overview

.. toctree::
   :maxdepth: 2
   :caption: Forward modeling:

   manual/forward_modeling
   manual/examples

.. toctree::
   :maxdepth: 2
   :caption: Inverse modeling:

   manual/inverse_modeling
   manual/inverse_examples

.. toctree::
   :maxdepth: 2
   :caption: Reference:

   manual/method
   manual/formats
   manual/meshing
   manual/solver
   manual/troubleshooting
   manual/python_api
   api/index

.. toctree::
   :maxdepth: 2
   :caption: Development:

   manual/testing
   manual/contributing

.. toctree::
   :maxdepth: 1
   :caption: About:

   manual/publications
   manual/contact


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
