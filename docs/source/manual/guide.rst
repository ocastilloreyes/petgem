==========
User guide
==========

Here you will find a basic user guide for **PETGEM**. 

CSEM kernel
-----------
The Controlled-Source Electromagnetic (CSEM) kernel in **PETGEM** implements a three-dimensional forward solver based on a high-order vector finite element method formulated on unstructured tetrahedral meshes. This numerical approach ensures accurate representation of complex geological structures and enables scalable computations on parallel architectures. The kernel is optimized for high-order polynomial basis functions, allowing enhanced accuracy in electromagnetic field simulations while reducing the number of unknowns compared to low-order formulations.

Workflow
********
The typical workflow for CSEM forward modeling in **PETGEM** consists of the following steps:

1. Generate or import a mesh using `Gmsh <http://gmsh.info/>`_.  
2. Define the subsurface resistivity model (see, e.g., ``tests/csem_model/generate_resistivity_model.py``)
3. Create a parameter file (see, e.g., ``tests/csem_model/generate_params_file.py``)
4. Execute the CSEM kernel to perform the forward simulation
5. Post-process the results (see, e.g., ``tests/csem_model/compare_responses.py``)  

Parameter files
***************
Parameter files specify the essential configuration for a CSEM forward modeling simulation in **PETGEM**, including the mesh, physical model, solver settings, and output specifications. These files provide the link between the input data and the kernel execution.  

The required parameter file must include the following entries:

- ``mesh_filename``: Path to the tetrahedral mesh in HDF5 format  
- ``source_filename``: Path to the source description file in ASCII format
- ``receivers_filename``: Path to the receiver file in HDF5 format
- ``nord``: Polynomial basis order of the finite elements  
- ``pc_type``: PETSc preconditioner type  
- ``pc_factor_mat_solver_type``: PETSc linear solver type
- ``output_dir``: Directory where the simulation results will be stored
- ``output_filename``: Base name for the output files

Source file
^^^^^^^^^^^

The ``source_filename`` file describes the transmitter configuration. It starts with the operating frequency, and then one row per source with its dipole parameters. The number of sources is determined automatically by the number of non-empty lines after the frequency line. The format is:

.. code-block::

   freq 
   x_pos y_pos z_pos current length dip_angle azimuth_angle
   x_pos y_pos z_pos current length dip_angle azimuth_angle
   x_pos y_pos z_pos current length dip_angle azimuth_angle
   ...    

where:

- ``freq``: Operating frequency (Hz)
- ``x_pos y_pos z_pos``: Cartesian coordinates of the dipole position (m)
- ``current``: Source current amplitude (A)
- ``length``: Dipole length (m)
- ``dip_angle``: Dipole inclination angle (degrees)
- ``azimuth_angle``: Dipole azimuth angle (degrees)


Receiver file
^^^^^^^^^^^^^
The ``receivers_filename`` file specifies the receiver locations as a list of Cartesian coordinates ``(x, y, z)``. Each entry corresponds to one measurement point.  

Pre- and post-processing
************************
**PETGEM** provides a suite of Python scripts for pre- and post-processing tasks, including:

- Mesh conversion to the required HDF5 format  
- Resistivity model generation
- Visualization of results in VTK format  

Example scripts can be found at ``tests/csem_model``. These scripts illustrate the construction of parameter files and input datasets required for CSEM simulations.  

Running PETGEM
**************
A typical command for a parallel execution is:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file path_to_params_file.txt