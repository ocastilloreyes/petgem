================
Inverse modeling
================

``im.csem`` recovers a conductivity model from observed :math:`E_x` data at
several frequencies. The model has one value per material (one row of
``sigmas.txt``), not per cell. Each iteration solves the forward and adjoint
problems for every frequency, computes the gradient of the regularized misfit,
and updates the model with L-BFGS (see :doc:`method`).

Inputs
------
``preprocess.py -mode im.csem`` needs, besides the mesh, receivers and
starting model:

``-im_source_filename``
   One row per (frequency, dipole) pair, 8 fields each:
   ``freq x y z current length dip azimuth``. Row ``k`` matches row ``k`` of
   the observed data.

``-observed_filename``
   Observed ``Ex`` as an ``[N_freq, N_recv]`` array, in HDF5 or as a
   MATLAB-style ``invEx.dat`` text file (see :doc:`formats`). Preprocessing
   checks that its shape matches the sources and the receivers.

``-error_level`` (optional)
   Relative noise level used to weight the misfit. It overrides the
   ``error_level`` attribute of an HDF5 observed-data file.

Materials with a non-zero ``fixed`` column in ``sigmas.txt`` are held fixed:
they are neither updated nor smoothed. Their ids are stored in the bundle.

Kernel options
--------------
``im.csem`` takes the common options of :doc:`overview` and the following.
Defaults apply when an option is set neither in the parameter file nor on the
command line.

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Option
     - Default
     - Meaning
   * - ``-im_max_iter``
     - 80
     - Maximum number of L-BFGS iterations.
   * - ``-im_lbfgs_memory``
     - 5
     - Number of correction pairs kept by L-BFGS.
   * - ``-im_lambda``
     - 0.1
     - Tikhonov regularization weight.
   * - ``-im_error_level``
     - 0.01
     - Relative data error. Overrides the value in the bundle.
   * - ``-im_gtol``
     - 1e-5
     - Stop when the gradient norm falls below this value.
   * - ``-im_rms_tol``
     - 0 (off)
     - Stop when the RMS misfit falls below this value.
   * - ``-im_rms_rtol``
     - 1e-3
     - Stop when the relative RMS improvement stays below this value for
       ``-im_rms_stall_window`` iterations. 0 disables it.
   * - ``-im_rms_stall_window``
     - 3
     - Iterations used by ``-im_rms_rtol``.
   * - ``-im_diag_weight``
     - 0 (off)
     - Self-weight of the neighbor smoother applied to the gradient.
   * - ``-im_fixed_materials``
     - from bundle
     - Comma-separated material ids to hold fixed. Overrides the bundle.
   * - ``-im_snapshot_interval``
     - 0 (off)
     - Write a VTU snapshot of the model every N accepted steps, and of the
       last one.
   * - ``-im_observed_mode``
     - ``external``
     - ``external`` reads ``/observed/Ex`` from the bundle (or from
       ``-im_observed_file``). ``fm_native`` reads an ``fm.csem`` output
       file, using transmitter ``k`` as frequency row ``k``.
   * - ``-im_observed_file``
     - the bundle
     - File with the observed data. Required with ``fm_native``; noise is
       not added by the kernel.

Parameter file
--------------
``preprocess.py -mode im.csem`` writes:

.. code-block::

   -input_filename <case_dir>/<input_filename>
   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu
   -pc_bddc_monolithic
   -ksp_rtol 1.0e-10
   -im_max_iter                150
   -im_lbfgs_memory            2
   -im_lambda                  0.1
   -im_diag_weight             0.0
   -im_gtol                    1.0e-5
   -im_rms_tol                 1.05
   -im_snapshot_interval       1
   -output_dir <case_dir>/
   -output_filename responses_p<order>

``-im_error_level`` and ``-im_fixed_materials`` are left out because their
values come from the bundle. To use MUMPS instead of BDDC, see
:doc:`solver`.

Running
-------
.. code-block:: bash

   mpirun -n 4 build/im.csem -options_file params.txt -im_max_iter 30

Output
------
``<output_dir>/<output_filename>.h5`` holds the recovered conductivity, the
model vector and the RMS history; snapshots go to
``<output_dir>/snapshots/``. See :doc:`formats`.
