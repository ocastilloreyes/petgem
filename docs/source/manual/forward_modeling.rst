================
Forward modeling
================

The Controlled-Source Electromagnetic (CSEM) forward kernel in **PETGEM**
implements a three-dimensional forward solver based on a high-order vector
finite element method formulated on unstructured tetrahedral meshes. This
numerical approach ensures accurate representation of complex geological
structures and enables scalable computations on parallel architectures. The
kernel is optimized for high-order polynomial basis functions, allowing
enhanced accuracy in electromagnetic field simulations while reducing the
number of unknowns compared to low-order formulations.

Workflow
--------
The typical workflow for CSEM forward modeling in **PETGEM** consists of the
following steps (the shared stages are described in :doc:`overview`):

1. Generate or import a mesh using `Gmsh <http://gmsh.info/>`_.
2. Define the subsurface conductivity model in ``sigmas.csv``.
3. Run ``utils/preprocess.py -mode forward`` to produce the input bundle and
   the parameter file.
4. Execute the CSEM kernel (``fm.csem``) to perform the forward simulation.
5. Post-process the results with the case ``postprocess.py``.

Parameter files
***************
The forward parameter file (emitted by ``utils/preprocess.py``) provides the
link between the input bundle and the kernel execution. It carries the bundle
path and the solver configuration; the polynomial order is read from the
bundle's ``/nord`` dataset, not from the parameter file. A representative
forward parameter file is:

.. code-block::

   -input_filename <case_dir>/input.h5
   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu
   -output_dir <case_dir>/
   -output_filename responses

where:

- ``-input_filename``: Path to the unified input bundle (HDF5).
- ``-ksp_type`` / ``-pc_type``: PETSc Krylov solver and preconditioner. The
  default is FGMRES with the BDDC preconditioner (deluxe scaling, LU coarse
  solve), which converges in a handful of iterations at low to moderate order.
- ``-output_dir`` / ``-output_filename``: Directory and base name for the
  computed responses.

Source file
^^^^^^^^^^^

The ``-source_filename`` file describes the transmitter configuration for
forward modeling. It starts with the operating frequency, then one row per
source with its dipole parameters. The number of sources is determined
automatically by the number of non-empty lines after the frequency line. The
format is:

.. code-block::

   freq
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

The forward source file is embedded into the bundle by the preprocessing step;
inverse modeling uses a different, multi-frequency source file (see
:doc:`inverse_modeling`).

Receiver file
^^^^^^^^^^^^^
The ``-receiver_filename`` file specifies the receiver locations as a list of
Cartesian coordinates ``(x, y, z)``, one measurement point per row.

Running PETGEM
**************
A typical command for a parallel forward execution is:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file path_to_params_file.txt

or, equivalently, through the unified dispatcher:

.. code-block:: bash

   mpirun -n 4 build/petgem modeling -options_file path_to_params_file.txt

A complete, runnable walkthrough is given in :doc:`examples`.
