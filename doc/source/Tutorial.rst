.. _Tutorial:

Tutorial
========

The PETGEM tutorial contains a small collection of programs which demonstrate
main aspects of the PETGEM work-flow. Each example has the
following structure:

#. ``Readme.txt``. Short description about what the example does.
#. ``params.yaml``. Physical parameters for the 3D CSEM survey.
#. ``petsc.opts``. Parameters for `PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers.
#. Data for modeling (mesh, conductivity model and receivers list)

The data for simple cases modelling are freely available in the
:ref:`Download` section.

.. _Basic notions:

Basic notions
-------------
The use of PETGEM can be summarized in three steps: pre-processing, modelling and
post-processing. The pre-processing and modelling phases requires parameter files
where the physical conditions of the 3D CSEM model are defined. In the
``params.yaml`` the 3D CSEM survey is described using several keywords
that allow one to define the main physical parameters and necessary file
locations. In sake of simplicity and in order to avoid a specific parser, the
``params.yaml`` file is defined as Python
dictionaries. Furthermore, the syntax of the ``params.yaml`` file is very simple yet powerful. As consequence, the
dictionary names and his key names MUST NOT BE changed. See Preprocessing
parameters file description and Modelling parameters file description
in :ref:`Manual` section for a full explanation of those keywords.

For a general 3D CSEM survey, the PETGEM work-flow can be summarize as follows:

#. Following the contents of the ``params.yaml`` file, a set of data are preprocessed (mesh, conductivity model and receivers positions) by the ``kernel.py``
#. A problem instance is created
#. The problem sets up its domain, sub-domains, source, solver. This stage include the computation of the main data structures
#. Parallel assembling of :math:`Ax=b`.
#. The solution is obtained in parallel by calling a ``ksp()`` `PETSc <https://www.mcs.anl.gov/petsc/>`__ object.
#. Interpolation of electromagnetic responses & post-processing parallel stage
#. Finally the solution can be stored by calling ``postprocess()`` method. Current version support hdf5.

Based on previous work-flow, any 3D CSEM modelling requires the following
input files:

#. A mesh file (current version supports Gmsh meshes)
#. A conductivity model associated with the materials defined in the mesh file
#. A list of receivers positions in hdf5 format for the electromagnetic responses post-processing
#. A ``params.yaml`` file where are defined the 3D CSEM parameters
#. A ``petsc.opts`` file where are defined options for `PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers
#. A ``kernel.py`` script which manage the pre-processing and modelling tasks respectively

.. _Running a simulation-Tutorial:

Running a simulation
--------------------

This section introduces the basics of running PETGEM on the command
line. Following command should be run in the top-level directory of the PETGEM
source tree.

PETGEM kernel is invoked as follows:

.. code-block:: bash

  $ mpirun -n MPI_tasks python3 kernel.py -options_file path/petsc.opts path/params.yaml

where ``MPI_tasks`` is the number of MPI parallel tasks, ``kernel.py`` is
the script that manages the PETGEM work-flow, ``petsc.opts`` is the
`PETSc <https://www.mcs.anl.gov/petsc/>`_ options file and ``params.yaml``
is the modelling parameters file for PETGEM.

A template of this file is included in ``examples/``
of the PETGEM source tree. Additionally, a freely available copy of this file
is located at :ref:`Download` section.

See :ref:`Model parameters file` section for more details about
``params.yaml`` file.

.. _Model parameters file:

Model parameters file
---------------------

By definition, any 3D CSEM survey should include: physical parameters, a mesh
file, source and receivers files. These data are included in the
``params.yanl`` file. Additionally, options for
`PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers are defined in a
``petsc.opts`` file.

In order to avoid a specific parser, ``params.yaml`` file is imported by
PETGEM as a Python dictionary. As consequence, the dictionary name and his key names
MUST NOT BE changed.

A glance of ``params.yaml`` file is the following:

.. code-block:: python

    ###############################################################################
    # PETGEM parameters file
    ###############################################################################
    # Model parameters
    model:
      mesh_file: examples/DIPOLE1D.msh      # Mesh file (gmsh format v2)
      basis_order: 1                        # Vector basis order (1,2,3,4,5,6)
      frequency: 2.0                        # Frequency
      src_position: [1750., 1750., -975.]   # Source position (xyz)
      src_azimuth: 0                        # Source rotation in xy plane
      src_dip: 0                            # Source rotation in xz plane
      src_current: 1.                       # Source current
      src_length: 1.                        # Source length
      sigma_horizontal: [1., 0.01, 1., 3.3333]   # Horizontal conductivity for each material
      sigma_vertical: [1., 0.01, 1., 3.3333]     # Vertical conductivity for each material
      receivers_file: examples/receiver_pos.h5 # Receiver positions file (xyz)

    # Execution parameters
    run:
      cuda: False                           # Flag to activate/deactivate cuda support

    # Output parameters
    output:
      directory: examples/out               # Directory for output (results)
      directory_scratch: examples/tmp       # Directory for temporal files

A template of this file is included in ``examples/``
of the PETGEM source tree. Additionally, a freely available copy of this file
is located at :ref:`Download` section. Furthermore, in
:ref:`Running a simulation-Manual` section of the PETGEM Manual is included
a deep description about this file.

.. _Visualization of results-Tutorial:

Visualization of results
------------------------
Once a solution of a 3D CSEM survey has been obtained, it should be
post-processed by using a visualization program. PETGEM does not do the
visualization by itself, but it generates output file (hdf5 format is supported)
with the electromagnetic responses (Ex, Ey, Ez, Hx, Hy, Hz) at receivers positions. It also gives timing values
in order to evaluate the performance.
