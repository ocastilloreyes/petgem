.. PETGEM documentation master file.

######################
PETGEM's documentation
######################

**PETGEM** (Parallel Edge-element Toolkit for General Electromagnetic Modeling)
is an open-source code for 3D electromagnetic modeling. It solves the CSEM
forward problem in the frequency and time domains, the MT forward problem and
the frequency-domain CSEM inverse problem. The electric field is discretized
with high-order Nédélec (edge) finite elements of polynomial order 1 to 6 on
unstructured tetrahedral meshes. **PETGEM** is written in C on top of
`PETSc <https://petsc.org/release/>`_ and MPI.

The repository provides four kernels and a dispatcher:

- ``fm.csem`` - Frequency-domain CSEM forward modeling:
  computes the electric and magnetic fields of dipole sources for a given
  conductivity model.

- ``fm.tdcsem`` - Time-domain CSEM forward modeling:
  computes the transient response of a given conductivity model to a source
  waveform.

- ``fm.mt`` - Magnetotelluric (MT) forward modeling:
  computes the MT response of a given conductivity model: impedance, apparent
  resistivity, phase and tipper.

- ``im.csem`` - CSEM inverse modeling:
  estimates the subsurface conductivity from observed data with L-BFGS and
  adjoint-state gradients.

- ``petgem`` - Dispatcher:
  runs any kernel through one command (``petgem fm.csem``,
  ``petgem fm.tdcsem``, ``petgem fm.mt`` or ``petgem im.csem``).

A small Python package under ``utils/`` (importable as ``petgem``) prepares the
HDF5 input bundle that the kernels read and reads their results.

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
   manual/td_modeling
   manual/td_examples
   manual/mt_modeling
   manual/mt_examples

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
