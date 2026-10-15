=========================
Inverse modeling examples
=========================

``examples/im_csem_block`` and ``examples/im_csem_checkerboard`` run the full
inversion workflow on synthetic data:

.. code-block::

   true model -> fm.csem -> synthetic observations + noise -> im.csem -> recovered model

No observed data is shipped. The forward stage produces it, so each run
exercises the whole chain. Both cases share the scripts in
``examples/shared/`` and the same layout; their ``README.md`` files describe
every file.

Common setup
------------
- A 100 Ω·m half-space under air (:math:`10^{-8}` S/m), with ``z = 0`` at the
  surface and ``z`` negative downward.
- One x-directed electric dipole at ``(0, -4000, 0)`` m, unit current and
  length.
- 441 receivers on a 21 × 21 surface grid.
- Frequencies 1, 10, 50, 100, 300, 800 and 1500 Hz; basis order 2.
- 1 % Gaussian noise on the real and imaginary parts of :math:`E_x`, with a
  fixed seed stored in ``observed.h5``.
- Inversion from a uniform 100 Ω·m model with :math:`\lambda = 0.1`. Only the
  ``INVERT`` region is updated; air and background are fixed in
  ``survey/sigmas_im.txt``.

Each case has two meshes built from one ``geometry/mesh.geo``:
``mesh.msh`` (``AIR``, ``BG``, ``INVERT``) for the inversion and
``mesh_true.msh`` (adding the bodies) for the true model. The bodies are
assigned by re-tagging cells after meshing, so both meshes have the same
topology. ``scripts/build_meshes.py -verify`` checks the shipped tags and
``-force`` re-meshes (needs Gmsh).

Running
-------
From the repository root (shown for ``im_csem_block``; ``im_csem_checkerboard`` is the same):

.. code-block:: bash

   make

   # Forward model of the true model: 7 bundles, 7 solves
   bash examples/im_csem_block/scripts/build_bundles.sh fm
   (cd examples/im_csem_block && sbatch scripts/run_fm.slurm)

   # Stack the responses and add noise -> outputs/observed.h5
   bash examples/im_csem_block/scripts/make_observations.sh

   # Inversion
   bash examples/im_csem_block/scripts/build_bundles.sh im
   (cd examples/im_csem_block && sbatch scripts/run_im.slurm)

   # Check the recovered model against reference/
   python3 examples/im_csem_block/scripts/analyze_inversion.py \
       -run_dir examples/im_csem_block/outputs \
       -reference examples/im_csem_block/reference/reference_metrics.json

Each script stops if the products of the previous stage are missing. Without
SLURM, run the ``mpirun`` line from the ``.slurm`` script directly. The jobs
use BDDC; ``SOLVER=mumps`` selects ``configs/solver_mumps.txt``.

``analyze_inversion.py`` prints the metrics and reports **PASS** when they are
within the tolerances of ``reference_metrics.json``. The reference runs used
112 MPI tasks. The L-BFGS path follows the iterative solves, which depend on
the domain decomposition, so compare runs at that rank count.

``reference/`` also holds the L-BFGS log (``lbfgs_log.txt``, one row per
accepted step) and the RMS history (``rms_history.txt``, one row per
objective-gradient evaluation). The second is longer because it includes the
rejected line-search trials.

Conductive block (``examples/im_csem_block``)
---------------------------------------------
A 200 m cube of 10 Ω·m centered at ``(0, 0, -200)`` m, top 100 m below the
surface. Receivers cover ``x, y`` in ``[-200, 200]`` m at 20 m spacing.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Quantity
     - Reference run
   * - RMS of the true model
     - 0.994
   * - Initial / final RMS
     - 11.058 / 1.0498
   * - Termination
     - ``CONVERGED_RMSTOL`` after 91 steps (99 evaluations)
   * - Minimum recovered resistivity
     - 11.00 Ω·m (true 10)
   * - Lateral / vertical offset of the conductor
     - 3.1 m / +36.6 m (shallower)
   * - Conductor volume (ρ < 30 Ω·m)
     - 3.71 × 10⁶ m³ (true 8.0 × 10⁶)

The conductor is found at the right place and resistivity, but shallow and
smaller than the truth. Source and receivers lie on the surface, so the
airwave dominates the data and the depth resolution is limited.

Checkerboard (``examples/im_csem_checkerboard``)
------------------------------------------------
Four 200 m cubes at 100 to 300 m depth with 100 m gaps: two conductors
(10 Ω·m) at ``(+150, -150)`` and ``(-150, +150)`` m and two resistors
(1000 Ω·m) at ``(+150, +150)`` and ``(-150, -150)`` m. Receivers cover
``x, y`` in ``[-500, 500]`` m at 50 m spacing.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Quantity
     - Reference run
   * - RMS of the true model
     - 0.9864
   * - Initial / final RMS
     - 6.1086 / 1.0494
   * - Termination
     - ``CONVERGED_RMSTOL`` after 90 steps (96 evaluations)
   * - Conductors C1 / C2, mean resistivity
     - 32.1 / 29.5 Ω·m (true 10)
   * - Resistors R1 / R2, mean resistivity
     - 167.7 / 164.4 Ω·m (true 1000)
   * - Wrong-sign cells inside the true bodies
     - 0 of 6330

All four bodies come back with the right sign and position. The conductors
recover more of their contrast than the resistors. The step count is not
reproducible to the unit: two runs took 91 and 90 steps.
