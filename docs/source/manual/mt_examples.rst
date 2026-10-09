===========
MT examples
===========

Trapezoidal hill (``examples/mt1``)
-----------------------------------
The 3D trapezoidal hill of Nam et al. (2007), used in Castillo-Reyes et al.
(2022), Section 3.1: a 100 Ω·m half-space with a 450 m high hill (top
450 × 450 m, base 2000 × 2000 m) under air (``1e8`` Ω·m), at 2 Hz. The 41
stations lie along ``y = 0``, ``x = -2000 … 2000`` m, 2.1 m below the surface.
The domain is the cube ``[-L, L]^3`` with ``L = 2000 + nskin·3500`` m, i.e. the
boundaries are ``nskin`` skin depths away from the survey.

The case ships two references at 2 Hz:

- ``reference/emmi3d/`` - an independent 3D solution (EMMI3D).
- ``reference/petgem_2022.h5`` - the responses published in the paper, for
  ``p = 1, 2`` and ``nskin = 1, 2, 4, 6, 8, 10``.

Running
*******
From the repository root:

.. code-block:: bash

   make

   # 1. Input bundle: order 2, 4 skin depths, mesh from geometry/mesh.geo.
   #    A third argument sets hmin (m) or gives an existing Gmsh mesh instead.
   bash examples/mt1/scripts/build_bundles.sh 2 4

   # 2. Forward solve.
   (cd examples/mt1 && mpirun -n 4 ../../build/fm.mt -options_file configs/params.txt \
       -input_filename outputs/input_p2_n4.h5 -output_dir outputs/ \
       -output_filename responses_p2_n4)

   # 3. Compare with the references and plot.
   python3 examples/mt1/scripts/postprocess.py -order 2

``postprocess.py`` reads every ``outputs/responses_p<order>_n<nskin>.h5``
present, prints the median misfit in apparent resistivity (%) and phase (deg)
against both references, writes ``outputs/figure_p<order>.png``, and passes
when every run with ``nskin >= 4`` is within 2 % and 1° of EMMI3D. Phases are
compared in the first-quadrant convention of the references,
:math:`\phi = \mathrm{mod}(-\phi_{\mathrm{fm.mt}}, 180^\circ)`.

Results
*******
Runs on MareNostrum 5 (112 MPI tasks, MUMPS) with the meshes of the paper and
the default 1D equation (``paper``). Median misfit against EMMI3D:

.. list-table::
   :header-rows: 1
   :widths: 10 22 22 22 22

   * - nskin
     - p = 1, ρ_xy / ρ_yx (%)
     - p = 1, φ_xy / φ_yx (deg)
     - p = 2, ρ_xy / ρ_yx (%)
     - p = 2, φ_xy / φ_yx (deg)
   * - 1
     - 2.45 / 4.73
     - 6.31 / 5.99
     - 2.73 / 5.36
     - 6.48 / 6.20
   * - 2
     - 2.34 / 2.00
     - 1.78 / 1.57
     - 1.60 / 0.97
     - 1.91 / 1.72
   * - 4
     - 1.21 / 1.06
     - 0.67 / 0.67
     - 0.49 / 0.20
     - 0.75 / 0.74
   * - 6
     - 0.95 / 0.79
     - 0.32 / 0.30
     - 0.51 / 0.21
     - 0.38 / 0.38
   * - 8
     - 0.95 / 0.69
     - 0.16 / 0.17
     - 0.58 / 0.16
     - 0.23 / 0.22
   * - 10
     - 0.77 / 0.63
     - 0.08 / 0.10
     - 0.62 / 0.29
     - 0.15 / 0.15

With ``nskin = 10``, ``fm.mt`` and the published responses differ by at most
0.53 % in apparent resistivity and 0.25° in phase over the 41 stations, for
both orders. With small domains the boundary error appears mostly in the phase
(about 6° low at ``nskin = 1``) and vanishes as the boundaries move away.

Half-space (``tests/mt``)
-------------------------
A 100 Ω·m half-space under air at 10 Hz on a compact box (±1 km laterally,
2 km of air, skin depth ≈ 1.6 km), used by the test suite (see
:doc:`testing`). With ``-mt_1d_equation h`` the boundary field is the exact 1D
magnetic field of the model, and ``fm.mt`` returns
:math:`\rho_{xy} = \rho_{yx} = 100` Ω·m and :math:`|\phi| = 45^\circ` to within
0.01 % and 0.05° at order 2. On this compact box the default ``paper`` equation
gives 173-188 Ω·m; moving the lateral boundaries to about five skin depths
brings it to 0.3 % and 0.6°.

References
----------
- Nam, M.J., Kim, H.J., Song, Y., Lee, T.J., Son, J.S., Suh, J.H., 2007. 3D
  magnetotelluric modelling including surface topography. Geophysical
  Prospecting 55, 277–287.
- Castillo-Reyes, O., Modesto, D., Queralt, P., Marcuello, A., Ledo, J.,
  Amor-Martin, A., de la Puente, J., García-Castillo, L.E., 2022. 3D
  magnetotelluric modeling using high-order tetrahedral Nédélec elements on
  massively parallel computing platforms. Computers & Geosciences 160, 105030.
