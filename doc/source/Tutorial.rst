.. _Tutorial:

Tutorial
========

The PETGEM tutorial contains a small collection of programs which demonstrate
main aspects of the PETGEM work-flow. Each example has the
following structure:

#. ``Readme.txt``. Short description about what the example does.
#. ``Input_preprocessing``. The input data of the 3D CSEM modelling of the program (file mesh and receivers positions file).
#. ``preprocessingParams.py``. Parameters for pre-processing stage.
#. ``modelParams.py``. Physical parameters for the 3D CSEM survey.
#. ``petsc.opts``. Parameters for `PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers.

The data for simple cases modelling are freely available in the
:ref:`Download` section.

.. _Basic notions:

Basic notions
-------------
The use of PETGEM can be summarized in three steps: pre-processing, modelling and
post-processing. The pre-processing and modelling phases requires parameter files
where the physical conditions of the 3D CSEM model are defined. In such files,
also referred to as ``preprocessingParams.py`` (pre-processing) and
``modelParams.py`` (modelling), the 3D CSEM survey is described using several keywords
that allow one to define the main physical parameters and necessary file
locations. In sake of simplicity and in order to avoid a specific parser, the
``preprocessingParams.py`` and ``modelParams.py`` files are defined as Python
dictionaries. Furthermore, the syntax of the ``preprocessingParams.py`` and
``modelParams.py`` files is very simple yet powerful. As consequence, the
dictionary names and his key names MUST NOT BE changed. See Preprocessing
parameters file description and Modelling parameters file description
in :ref:`Manual` section for a full explanation of those keywords.

For a general 3D CSEM survey, the PETGEM work-flow can be summarize as follows:

#. Following the contents of the ``preprocessingParams.py`` file, a set of data are preprocessed (mesh, conductivity model and receivers positions)
#. The kernel (``kernel.py``) reads a ``modelParams.py``
#. Following the contents of the ``modelParams.py``, a problem instance is created
#. The problem sets up its domain, sub-domains, source, solver. This stage include the computation of the main data structures
#. Parallel assembling of :math:`Ax=b`. See :ref:`CSEM problem` and :ref:`Edge finite element formulation` sections for a detail mathematical background of this equation system
#. The solution is obtained in parallel by calling a ``ksp()`` `PETSc <https://www.mcs.anl.gov/petsc/>`__ object.
#. Interpolation of electromagnetic responses & post-processing parallel stage
#. Finally the solution can be stored by calling ``postProcessingFields()`` function. Current version support `PETSc <https://www.mcs.anl.gov/petsc/>`_, Matlab and ASCII formats.

Based on previous work-flow, any 3D CSEM modelling requires the following
input files:

#. A mesh file (current version supports Gmsh meshes)
#. A conductivity model associated with the materials defined in the mesh file
#. A list of receivers positions in ASCII format for the electric responses post-processing
#. A ``preprocessingParams.py`` file where are defined the pre-processing parameters
#. A ``modelParams.py`` file where are defined the 3D CSEM parameters
#. A ``petsc.opts`` file where are defined options for `PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers
#. A ``run_preprocessing.py`` script and a ``kernel.py`` script which manage the pre-processing and modelling tasks respectively

.. _Preprocessing-Tutorial:

Pre-processing
--------------
The ``run_preprocessing.py`` script provides functions to change input file formats
into a representation that is more suitable for PETGEM. It transforms mesh files
with its conductivity model and receivers positions into a `PETSc <https://www.mcs.anl.gov/petsc/>`_ binary format, which allow
parallel computations in a simple way. Therefore, this step is mandatory
for any modelling. ``run_preprocessing.py`` script is included in the top-level
directory of the PETGEM source tree.

The ``run_preprocessing.py`` script is invoked has follows:

.. code-block:: bash

  $ python3 run_preprocessing.py path/preprocessingParams.py

A template of ``run_preprocessing.py`` script is included in ``examples/``
of the PETGEM source tree. Additionally, a freely available copy of this file
is located at :ref:`Download` section. Please, see
:ref:`Preprocessing parameters file` section for more details about
``preprocessingParams.py`` file.

.. _Preprocessing parameters file:

Pre-processing parameters file
------------------------------

As already said, the first step for a 3D CSEM modelling using PETGEM is the
pre-processing phase. Here, a mesh file with its conductivity model and receivers
positions are exported to `PETSc <https://www.mcs.anl.gov/petsc/>`_ binary files.

A glance of this file is the following:

.. code-block:: python

   # Parameters file template for PETGEM preprocessing. Here a mesh file, conductivity model and receivers file are mandatory.
   # In order to avoid a specific parser, this file is imported by PETGEM as a Python dictionary. As consequence, the dictionary
   # name and his key names MUST NOT BE changed. All file paths should consider as absolute.

   preprocessing = {
    # ---------- Mesh file ----------
    'MESH_FILE': 'DIPOLE1D/Input_preprocessing/DIPOLE1D.msh',

    # ---------- Material conductivities ----------
    'MATERIAL_CONDUCTIVITIES': [1.0, 1./100., 1., 1./.3],

    # ---------- Receivers positions file ----------
    'RECEIVERS_FILE': 'DIPOLE1D/Input_preprocessing/RECEIVER_POSITIONS.txt',

    # ---------- Path for Output ----------
    'OUT_DIR': 'DIPOLE1D/Input_model/',
    }

A template of this file is included in ``examples/``
of the PETGEM source tree. Additionally, a freely available copy of this file
is located at :ref:`Download` section. Furthermore, in
:ref:`Preprocessing-Manual` section of the PETGEM Manual is included a
deep explanation about this file.

.. _Running a simulation-Tutorial:

Running a simulation
--------------------

This section introduces the basics of running PETGEM on the command
line. Following command should be run in the top-level directory of the PETGEM
source tree.

PETGEM kernel is invoked as follows:

.. code-block:: bash

  $ mpirun -n MPI_tasks python3 kernel.py -options_file path/petsc.opts path/modelParams.py

where ``MPI_tasks`` is the number of MPI parallel tasks, ``kernel.py`` is
the script that manages the PETGEM work-flow, ``petsc.opts`` is the
`PETSc <https://www.mcs.anl.gov/petsc/>`_ options file and ``modelParams.py``
is the modelling parameters file for PETGEM.

A template of this file is included in ``examples/``
of the PETGEM source tree. Additionally, a freely available copy of this file
is located at :ref:`Download` section.

See :ref:`Model parameters file` section for more details about
``modelParams.py`` file.

.. _Model parameters file:

Model parameters file
---------------------

By definition, any 3D CSEM survey should include: physical parameters, a mesh
file, source and receivers files. These data are included in the
``modelParams.py`` file. Additionally, options for
`PETSc <https://www.mcs.anl.gov/petsc/>`_ solvers are defined in a
``petsc.opts`` file.

In order to avoid a specific parser, ``modelParams.py`` file is imported by
PETGEM as a Python dictionary. As consequence, the dictionary name and his key names
MUST NOT BE changed.

A glance of ``modelParams.py`` file is the following:

.. code-block:: python

   # Parameters file template for 3D CSEM modelling.
   # By definition, any 3D CSEM survey should include: physical parameters, a mesh file, source and receivers files. These data
   # are included in the modelParams.py file. Additionally, options for PETSc solvers are defined in a petsc.opts file.
   # In order to avoid a specific parser, modelParams.py file is imported by PETGEM as a Python dictionary. As consequence,
   # the dictionary name and his key names MUST NOT BE changed. All file paths should consider as absolute.

   modelling = {
    # ----- Pyshical parameters -----
    # Source
    # Source frequency. Type: float
    # Optional: NO
    'FREQ': 2.0,
    # Source position(x, y, z). Type: float
    # Optional: NO
    'SRC_POS': [1750.0, 1750.0, -975.0],
    # Source orientarion. Type: int
    # 1 = X-directed source
    # 2 = Y-directed source
    # 3 = Z-directed source
    # Optional: NO
    'SRC_DIREC': 1,
    # Source current. Type: float
    # Optional: NO
    'SRC_CURRENT': 1.0,
    # Source length. Type: float
    # Optional: NO
    'SRC_LENGTH': 1.0,
    # Conductivity model. Type: str
    # Optional: NO
    'CONDUCTIVITY_MODEL_FILE': 'DIPOLE1D/Input_model/conductivityModel.dat',
    # Background conductivity. Type: float
    # Optional: NO
    'CONDUCTIVITY_BACKGROUND': 3.33,

    # ------- Mesh and conductivity model files ------
    # Mesh files
    # Nodes spatial coordinates. Type: str
    # Optional: NO
    'NODES_FILE': 'DIPOLE1D/Input_model/nodes.dat',
    # Elements-nodes connectivity. Type: str
    # Optional: NO
    'MESH_CONNECTIVITY_FILE': 'DIPOLE1D/Input_model/meshConnectivity.dat',
    # Elements-edges connectivity. Type: str
    # Optional: NO
    'DOFS_CONNECTIVITY_FILE': 'DIPOLE1D/Input_model/dofs.dat',
    # Edges-nodes connectivity. Type: str
    # Optional: NO
    'DOFS_NODES_FILE': 'DIPOLE1D/Input_model/dofsNodes.dat',
    # Sparsity pattern for matrix allocation (PETSc)
    'NNZ_FILE': 'DIPOLE1D/Input_model/nnz.dat',
    # Boundaries. Type: str
    # Optional: NO
    'BOUNDARIES_FILE': 'DIPOLE1D/Input_model/boundaries.dat',

    # ------------ Solver -----------
    # Solver options must be set in
    # petsc_solver.opts

    # ------------ Receivers file -----------
    # Name of the file that contains the receivers position. Type: str
    # Optional: NO
    'RECEIVERS_FILE': 'DIPOLE1D/Input_model/receivers.dat',
   }

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
visualization by itself, but it generates output files (ASCII,
`PETSc <https://www.mcs.anl.gov/petsc/>`_ and Matlab formats are supported)
with the electric responses at receivers positions. It also gives timing values
in order to evaluate the performance.
