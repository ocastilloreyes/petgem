================
Forward modeling
================

``fm.csem`` computes the 3D CSEM response of a conductivity model at a single
frequency, discretizing the electric field with Nédélec elements of order 1 to
6 on an unstructured tetrahedral mesh. The formulation is described in
:doc:`method`.

Workflow
--------
1. Generate or import a mesh (see :doc:`meshing`).
2. Define the conductivity model in ``sigmas.txt``.
3. Run ``utils/preprocess.py -mode fm`` to produce the input bundle and
   the parameter file.
4. Run ``fm.csem``.
5. Post-process the responses.

Kernel options
--------------
``fm.csem`` takes its options from a PETSc options file
(``-options_file params.txt``) or directly on the command line.

**Required**

- ``-input_filename`` - the input bundle (HDF5).
- ``-output_dir`` - output directory (created if absent).
- ``-output_filename`` - output stem; writes ``<output_dir>/<stem>.h5``.

**Optional**

- ``-order <1..6>`` - override the bundle's ``/order``.
- ``-mms`` - run the method-of-manufactured-solutions verification instead of a
  CSEM simulation (see :doc:`testing`).

Everything else in the options file is passed to PETSc (solver, preconditioner,
matrix type); see :doc:`solver`.

``-help intro`` prints a usage summary and exits; ``-help`` prints the full
PETSc option database; ``--version`` prints the version.

Parameter file
**************
The parameter file emitted by ``utils/preprocess.py -mode fm`` is:

.. code-block::

   -input_filename <case_dir>/input.h5
   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu
   -output_dir <case_dir>/
   -output_filename responses

The polynomial order is **not** written here - the kernel reads it from the
bundle's ``/order`` dataset.

Input files
-----------

Source file
***********
``-source_filename`` describes the transmitters. Two layouts are accepted and
auto-detected (see :doc:`formats`). The shipped examples use the second:

.. code-block::

   # 8 fields per row (canonical)
   freq x_pos y_pos z_pos current length dip_angle azimuth_angle

.. code-block::

   # a lone frequency line, then 7-field rows
   freq
   x_pos y_pos z_pos current length dip_angle azimuth_angle

Fields are: frequency (Hz), dipole position ``x y z`` (m), ``current`` (A),
``length`` (m), ``dip_angle`` and ``azimuth_angle`` (degrees).

A forward run is **single-frequency**: the kernel's transmitter set carries one
operating frequency shared by all transmitters. Inverse modeling uses a
multi-frequency source file (see :doc:`inverse_modeling`).

Receiver file
*************
``-receiver_filename`` lists receiver positions, one Cartesian point ``x y z``
per row. Whitespace- or comma-separated; ``#`` comments and blank lines are
allowed.

Running
-------
.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file path/to/params.txt

or through the dispatcher:

.. code-block:: bash

   mpirun -n 4 build/petgem fm -options_file path/to/params.txt

Output
------
The kernel writes a single HDF5 file, ``<output_dir>/<output_filename>.h5``,
holding all six field components (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``,
``Hz``) at the receivers, for every transmitter. The layout is documented in
:doc:`formats`.

A runnable walkthrough is given in :doc:`examples`.
