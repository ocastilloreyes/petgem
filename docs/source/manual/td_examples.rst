=========================
Time-domain CSEM examples
=========================

Half-space (``examples/fm_tdcsem_halfspace``)
---------------------------------------------
The half-space model of Cai et al. (2017), Section 5.1: a 500 Ω·m earth under
air (``1e8`` Ω·m), excited by an x-directed electric dipole at
``(-1000, 0, 0)`` m with the unit-area Gaussian pulse of their Fig. 3
(center ``0.5`` µs, standard deviation ``0.1`` µs, samples every ``5e-8`` s).
The receivers are inline on the surface at offsets 1000 and 2000 m. The
response is computed from 0 to 1 s with BDF2 (second-order backward
differentiation formula), an initial time step of ``5e-8`` s and a doubling
trial every 25 steps.
The domain is the 50 × 50 × 50 km cube of the paper with
:math:`n \times E = 0` on its boundary.

The reference is the analytical impulse response of the paper, Eq. (A.8),
for :math:`t > 0`:

.. math::

   E_x(t) = \frac{I\,ds}{8}\left(\frac{\mu^3\sigma}{\pi^3 t^5}\right)^{1/2}
            e^{-\mu\sigma\rho^2/(4t)} .

Running
*******
From the repository root:

.. code-block:: bash

   make

   # 1. Input bundle: order 2, mesh from geometry/mesh.geo.
   bash examples/fm_tdcsem_halfspace/scripts/build_bundles.sh 2

   # 2. Forward solve (MUMPS, configs/params.txt).
   (cd examples/fm_tdcsem_halfspace && mpirun -n 4 ../../build/fm.tdcsem -options_file configs/params.txt \
       -input_filename outputs/input_p2.h5 -output_dir outputs/ \
       -output_filename responses_p2)
   #    or on a cluster:
   (cd examples/fm_tdcsem_halfspace && sbatch scripts/run_fm.slurm)

   # 3. Compare with Eq. (A.8) and plot.
   python3 examples/fm_tdcsem_halfspace/scripts/postprocess.py -order 2

``postprocess.py`` compares :math:`E_x` with Eq. (A.8) from the peak of the
analytical response, :math:`t_{\mathrm{peak}} = \mu\sigma\rho^2/10`. It
prints the median relative misfit per receiver and per decade, writes
``outputs/figure_p<order>.png``, and passes when the median misfit over
``[t_peak, t_max]`` is within the tolerance for both receivers. ``-t_max``
(default ``0.3`` s) and ``-tolerance`` (default ``3`` %) change these limits.

Results
*******
On MareNostrum 5 (56 MPI tasks, MUMPS) the order 2 run (552 080 DOFs) takes
552 time steps and 6 min. Relative misfit of :math:`E_x` against Eq. (A.8),
median / max (%):

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Offset (m)
     - t_peak (s)
     - [t_peak, 0.03] s
     - [t_peak, 0.3] s
   * - 1000
     - 2.5e-4
     - 0.19 / 0.48
     - 0.28 / 2.31
   * - 2000
     - 1.0e-3
     - 0.13 / 0.46
     - 0.31 / 2.34

From 0.05 s the response departs from Eq. (A.8): -1.3 % at 0.1 s, +3 % at
0.5 s and -51 % at 1 s. Cai et al. (2017), Fig. 6, show the same late-time
departure for this domain with the Dirichlet boundary condition.
