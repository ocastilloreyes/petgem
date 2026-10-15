================
Inverse modeling
================

``im.csem`` recovers a conductivity model from observed data at several
frequencies: the field components :math:`E_x, E_y, E_z, H_x, H_y, H_z` and
the ratios :math:`Z_{ij} = E_i/H_j`, alone or jointly. The unknown is the
log-resistivity of every cell, except the cells of materials marked as
fixed. Each iteration solves the forward and
adjoint problems for every frequency, computes the gradient of the regularized
misfit, and updates the model with L-BFGS (see :doc:`method`).

Inputs
------
``preprocess.py -mode im.csem`` needs, besides the mesh, receivers and
starting model:

``-im_source_filename``
   One row per (frequency, dipole) pair, 8 fields each:
   ``freq x y z current length dip azimuth``. Row ``k`` matches row ``k`` of
   the observed data.

``-observed_filename``
   One or more observed-data files, each ``file`` or ``component:file``, with
   one ``[N_freq, N_recv]`` array per component, in HDF5 or as a MATLAB-style
   ``invEx.dat`` text file (see :doc:`formats`). An HDF5 file without a
   component prefix gives all the components it holds; a text file without
   one is ``Ex``. Preprocessing checks that every shape matches the sources and
   the receivers. NaN marks a missing datum.

``-error_level`` (optional)
   Relative noise level used to weight the misfit. It overrides the
   ``error_level`` attribute of an HDF5 observed-data file.

Materials with a non-zero ``fixed`` column in ``sigmas.txt`` (for example,
air) are held fixed: their cells are neither updated nor smoothed. Their ids
are stored in the bundle. The ``sigmas.txt`` values of all materials form the
starting model.

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
     - Stop when the ratio :math:`|g|/|x|` of the gradient norm to the model
       norm falls below this value.
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
   * - ``-im_data_components``
     - ``Ex``
     - Comma-separated data components fitted jointly: ``Ex``, ``Ey``,
       ``Ez``, ``Hx``, ``Hy``, ``Hz``, ``Zxx``, ``Zxy``, ``Zyx``, ``Zyy``.
   * - ``-im_observed_mode``
     - ``external``
     - ``external`` reads ``/observed/<component>`` from the bundle (or from
       ``-im_observed_file``). ``fm_native`` reads an ``fm.csem`` output
       file, using transmitter ``k`` as frequency row ``k``.
   * - ``-im_observed_file``
     - the bundle
     - File with the observed data. Required with ``fm_native``; noise is
       not added by the kernel.
   * - ``-im_gradient_check``
     - off
     - Compare the adjoint derivative of the data misfit with central
       differences at the starting model, write the result and stop. See
       `Gradient check`_.
   * - ``-im_gradient_check_eps``
     - ``1e-1,1e-2,1e-3,1e-4``
     - Step sizes of the gradient check, in :math:`\log\rho`.

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
   mpirun -n 4 build/im.csem -options_file params.txt -im_data_components Ex,Zxy

The bundle may hold more components than the run fits; only those named by
``-im_data_components`` are read.

Output
------
``<output_dir>/<output_filename>.h5`` holds the recovered conductivity, the
model vector, the RMS history and the predicted data of each fitted
component; snapshots go to ``<output_dir>/snapshots/``. See :doc:`formats`.

Gradient check
--------------
``-im_gradient_check`` perturbs :math:`\log\rho` of the cells that are not
fixed along a smooth direction :math:`\delta m` and compares, for each step
:math:`\varepsilon`,

.. math::

   \frac{\Phi_d(m + \varepsilon\,\delta m) - \Phi_d(m - \varepsilon\,\delta m)}{2\varepsilon}
   \quad\text{with}\quad \nabla\Phi_d \cdot \delta m,

where :math:`\Phi_d` is the data misfit and the right-hand side comes from
the adjoint solve. The model smoothing is not applied. The relative
difference should fall as :math:`\varepsilon^2` until round-off. The table
is printed in the run log and written to
``<output_dir>/<output_filename>.h5`` (see :doc:`formats`); no inversion is
run.

.. code-block:: bash

   mpirun -n 4 build/im.csem -options_file params.txt -im_data_components Zxy \
       -im_gradient_check -im_gradient_check_eps 1e-2,1e-3
