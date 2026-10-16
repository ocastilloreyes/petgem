=========================
Time-domain CSEM modeling
=========================

``fm.tdcsem`` computes the 3D transient CSEM response of a conductivity model:
the electric field :math:`E(t)` and :math:`\partial B/\partial t` at the
receivers, for one or more electric dipoles driven by a common source waveform.
It uses the same discretization, boundary condition and dipole source as
``fm.csem``. The field is advanced in time with the second-order backward
differentiation scheme (BDF2) or backward Euler, and the time step is
increased by step doubling (see :doc:`method`).

Scope
-----
- The initial field is zero, so the waveform must start at zero current
  (impulse, Gaussian pulse, step-on or any waveform that starts from rest).
  Step-off waveforms are not supported yet.
- The boundary condition is :math:`n \times E = 0`, as in ``fm.csem``. The
  domain must be large enough for the field to vanish at the boundary over the
  simulated time window.

Inputs
------
Besides the mesh, ``sigmas.txt`` and the receiver table,
``preprocess.py -mode fm.tdcsem`` takes:

- ``-source_filename`` - one transmitter per row, with 7 fields and no
  frequency:

  .. code-block::

     # x y z current length dip azimuth
     0.0 0.0 -1.0 1.0 1.0 0.0 0.0

- ``-tdcsem_waveform_filename`` - the source waveform, two columns ``t s(t)``:
  the time (s) and the current divided by the transmitter current. Times must
  increase strictly and the first value of ``s`` must be 0. Between samples
  ``s`` is interpolated linearly; after the last sample it keeps the last
  value. For example, a step-on reached by a 1 µs ramp:

  .. code-block::

     # t (s)  s(t)
     0.0     0.0
     1.0e-6  1.0

- ``-tdcsem_t_end`` - final time (s), written into the parameter file.

.. code-block:: bash

   # Build the input bundle and the parameter file
   python3 utils/preprocess.py -mode fm.tdcsem -order 2 -case_dir <dir> \
       -mesh_filename mesh.msh -receiver_filename receivers.txt \
       -source_filename sources.txt -tdcsem_waveform_filename waveform.txt \
       -tdcsem_t_end 1.0e-2 -sigma_file sigmas.txt

Kernel options
--------------
``fm.tdcsem`` takes the common options of :doc:`overview` and:

- ``-tdcsem_t_end <s>`` - final time. Required.
- ``-tdcsem_dt0 <s>`` - initial time step. Default: the smallest spacing of
  the waveform samples.
- ``-tdcsem_atsd_steps <n>`` - number of steps at a fixed time step before
  each doubling trial. Default: 100.
- ``-tdcsem_atsd_rtol <r>`` - tolerance of the doubling trial; ``0`` keeps the
  time step fixed. Default: ``1e-2``.
- ``-tdcsem_scheme <bdf2|be>`` - time integration scheme. Default: ``bdf2``.

The time step is not doubled before the last waveform sample. The operator of
each time step size is set up once (one factorization with a direct solver)
and reused for every step of that size.

Parameter file
--------------
``preprocess.py -mode fm.tdcsem`` writes a direct-solver file with the final
time:

.. code-block::

   -input_filename <case_dir>/<input_filename>
   -output_dir <case_dir>/
   -output_filename responses_p<order>

   -dm_mat_type aij
   -ksp_type preonly
   -pc_type lu
   -pc_factor_mat_solver_type mumps
   -tdcsem_t_end <t_end>


Running
-------
.. code-block:: bash

   # Run the simulation
   mpirun -n 4 build/fm.tdcsem -options_file params.txt

The kernel prints a ``Time steps`` table with one row per block of
``-tdcsem_atsd_steps`` steps. Each row gives, at the end of the block:

- the step count, the time and the time step used;
- the doubling test :math:`\|E_{2\Delta t} - E_{\Delta t}\| / \|E - E_0\|`
  and whether the doubling was accepted;
- the KSP and charge-correction iterations and the wall time of the block.

.. code-block::

    Time steps:
       Steps       t (s)      dt (s)   Doubling   Result  KSP its Charge its Wall (s)
          27  2.7000e-05  1.0000e-06  5.524e-03 accepted       28        391      1.3
          54  8.1000e-05  2.0000e-06  2.688e-03 accepted       28        392      1.3
         ...

The final report gives the number of time steps, operator setups and accepted
and rejected doublings.

Output
------
``<output_dir>/<output_filename>.h5`` holds the fields at every computed time
step; the layout is in :doc:`formats`. Read it with ``petgem.readTdResponses``
(see :doc:`python_api`):

.. code-block:: python

   import petgem

   r = petgem.readTdResponses("responses_p2.h5", source=1)
   t, ex = r["times"], r["Ex"]        # ex[j, n]: receiver j at time t[n]

The half-space test in ``tests/tdcsem`` (see :doc:`testing`) is a complete
case with an `empymod`_ reference, and :doc:`td_examples` reproduces the
half-space model of `Cai et al. (2017)`_.

.. _Cai et al. (2017): https://doi.org/10.1016/j.jappgeo.2017.08.005
.. _empymod: https://doi.org/10.1190/geo2016-0626.1
