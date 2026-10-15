=====================
Numerical formulation
=====================

This page states the discretization implemented by the kernels. For the
derivation and for validation studies, see :doc:`publications`.

Forward problem
---------------
**PETGEM** solves the frequency-domain CSEM problem for the **total** electric
field :math:`E`, with a constant magnetic permeability :math:`\mu = \mu_0`
(``MU`` in ``include/constants.h``) and a diagonal conductivity tensor
:math:`\sigma = \mathrm{diag}(\sigma_x, \sigma_y, \sigma_z)`:

.. math::

   \nabla \times \nabla \times E \;-\; i\,\omega\,\mu\,\sigma\, E \;=\; F,

where :math:`\omega` is the angular frequency and :math:`F` the source term.
For CSEM, homogeneous Dirichlet boundary conditions :math:`n \times E = 0` are imposed on the domain
boundary (``PETGEM_BC_PEC`` in ``setupNedelecGrid``).

The field is discretized with **Nédélec (edge) vector finite elements** of
polynomial order 1 to 6 on an unstructured tetrahedral mesh. These
:math:`H(\mathrm{curl})`-conforming elements enforce tangential continuity
across faces. Discretization yields the complex-symmetric linear system

.. math::

   A\, e = b, \qquad A = K \;-\; i\,\omega\,\mu\, M_\sigma,

with

.. math::

   K_{ij} = \int_\Omega (\nabla \times N_i)\cdot(\nabla \times N_j)\,d\Omega,
   \qquad
   (M_\sigma)_{ij} = \int_\Omega N_i\cdot \sigma\, N_j\,d\Omega,

where :math:`N_i` are the vector basis functions. This is the operator
assembled by ``assembleMaxwellOperator`` (``src/assembly.c``). Because :math:`A` is
complex, **PETGEM** must be built against a PETSc configured with complex
scalars (see :doc:`install`).

Source term
***********
Each transmitter is a point electric dipole with moment
:math:`p = I\,L\,\hat{d}`, where :math:`I` is the current, :math:`L` the dipole
length, and :math:`\hat{d}` the unit direction obtained by rotating the axis by
the dip and azimuth angles. The dipole is located in its host cell, and the
right-hand side is formed by evaluating the basis functions at the dipole
position:

.. math::

   b_j = N_j(x_s)\cdot p .

Receiver responses are obtained by interpolating the solution :math:`e` at the
receiver positions; the magnetic components are recovered from
:math:`\nabla \times E`. The system is solved with PETSc (see :doc:`solver`).

Discrete gradient
*****************
Alongside :math:`K` and :math:`M_\sigma`, the assembly builds the **discrete
gradient matrix** :math:`G`, mapping the order-:math:`p` :math:`H^1` (nodal)
space into the order-:math:`p` Nédélec space, so that
:math:`\nabla \phi_k = \sum_i G_{ik} N_i` exactly and :math:`K G = 0`. This
matrix spans the curl-kernel of the Nédélec space; it is handed to the BDDC
preconditioner (see :doc:`solver`) and is checked by the test suite (the de
Rham identity :math:`K_e G_e = 0` per cell).

MT forward problem
------------------
``fm.mt`` follows Castillo-Reyes et al. (2022). There is no source in the
domain; the excitation enters through a natural boundary condition. With the
bilinear form of the operator above,

.. math::

   a(E, v) = \int_\Omega (\nabla\times v)\cdot(\nabla\times E)
             - i\,\omega\,\mu\,\sigma\, v\cdot E \, d\Omega,
   \qquad
   l(v) = -i\,\omega\,\mu \int_\Gamma v\cdot\left(n\times\hat{H}\right) d\Gamma,

the total field solves :math:`a(E, v) = l(v)` for all test functions, with no
constrained boundary DOFs (``PETGEM_BC_NATURAL``). The matrix is the same
:math:`A` assembled by ``assembleMaxwellOperator``; the right-hand side is the
boundary integral, assembled by ``assembleMtBoundaryRhs`` (``src/mt.c``) with a
triangle quadrature on every boundary face.

The domain is an axis-aligned box with faces :math:`\Gamma_1` (top) to
:math:`\Gamma_6` (bottom). The imposed field is
:math:`\hat{H} = (0, H(z), 0)` for the x-polarization and
:math:`\hat{H} = (H(z), 0, 0)` for the y-polarization; both are solved together
as two right-hand sides. :math:`H(z)` is the solution of a 1D problem on the
conductivity profile :math:`\sigma(z)` of the lateral faces, with
:math:`H(z_{\max}) = 1` and :math:`H(z_{\min}) = 0`:

.. math::

   H'' + i\,\omega\,\mu\,\sigma(z)\, H = 0
   \quad\text{(default, Eq. 11 of the paper)}, \qquad
   (\rho H')' + i\,\omega\,\mu\, H = 0, \;\; \rho = 1/\sigma
   \quad (\texttt{-mt\_1d\_equation h}).

Both are solved with linear finite elements, with nodes at every interface of
the profile. Within a layer the two equations coincide; they differ in the
interface condition (:math:`H'` continuous, or :math:`\rho H'` continuous) and
therefore in the air, where the second gives the nearly constant magnetic field
of a plane wave. For a 1D model the second equation makes the boundary data
exact on the lateral and top faces, so the response does not depend on the
lateral extent of the box (the bottom must still be deep enough for
:math:`H(z_{\min}) = 0`); the first needs boundaries several skin depths away
from the survey, as studied in the paper.

At each receiver, :math:`E` and :math:`H = \nabla\times E/(i\omega\mu)` of
both polarizations give the impedance tensor and the tipper,

.. math::

   Z = \begin{pmatrix} E_x^{(1)} & E_x^{(2)} \\ E_y^{(1)} & E_y^{(2)} \end{pmatrix}
       \begin{pmatrix} H_x^{(1)} & H_x^{(2)} \\ H_y^{(1)} & H_y^{(2)} \end{pmatrix}^{-1},
   \qquad
   T = \begin{pmatrix} H_z^{(1)} & H_z^{(2)} \end{pmatrix}
       \begin{pmatrix} H_x^{(1)} & H_x^{(2)} \\ H_y^{(1)} & H_y^{(2)} \end{pmatrix}^{-1},

and the apparent resistivity and phase
:math:`\rho_{ij} = |Z_{ij}|^2/(\omega\mu)`,
:math:`\phi_{ij} = \mathrm{atan2}(\mathrm{Im}\,Z_{ij}, \mathrm{Re}\,Z_{ij})`
in degrees. With the :math:`e^{-i\omega t}` convention and :math:`z` up, a
half-space gives :math:`\phi_{xy} = 135^\circ` and :math:`\phi_{yx} = -45^\circ`;
references in the first-quadrant convention use
:math:`\mathrm{mod}(-\phi, 180^\circ)`.

Time-domain CSEM forward problem
--------------------------------
``fm.tdcsem`` follows Cai et al. (2017, *J. Appl. Geophys.* 145, 133-143),
without their hybrid boundary condition. In the quasi-static limit the electric field satisfies

.. math::

   \nabla \times \nabla \times E(t) + \mu\,\sigma\,\frac{\partial E}{\partial t}
   = -\mu\,\frac{\partial J_s}{\partial t},

with :math:`n \times E = 0` on the boundary and :math:`E(0) = 0`. The same
Nédélec discretization gives :math:`K e + \mu M_\sigma\,\dot e = -\mu\,\dot b`,
with the dipole term :math:`b_j = N_j(x_s)\cdot p\,s(t)`, where :math:`s(t)`
is the normalized source waveform. With time step :math:`\Delta t`, each
step solves

.. math::

   \left(K + \frac{a_0\,\mu}{\Delta t} M_\sigma\right) e^{n}
   = -\frac{\mu}{\Delta t} M_\sigma \left(a_1 e^{n-1} + a_2 e^{n-2}\right)
     - \frac{\mu}{\Delta t}\left(a_0 s(t_n) + a_1 s(t_{n-1}) + a_2 s(t_{n-2})\right) b_p,
   \qquad b_{p,j} = N_j(x_s)\cdot p ,

with :math:`(a_0, a_1, a_2) = (3/2, -2, 1/2)` for the second-order backward
differentiation formula (BDF2, default) and :math:`(1, -1, 0)` for backward
Euler (``-tdcsem_scheme be``). Before the first step :math:`e = 0` and
:math:`s = 0`.

:math:`K` and :math:`M_\sigma` are assembled once (``assembleMaxwellOperator``
and ``refillMassMatrix``). The operator is real; on a complex PETSc build the solution
has a zero imaginary part.

The time step is adapted by step doubling. After ``-tdcsem_atsd_steps`` steps
at :math:`\Delta t`, one step of :math:`2\Delta t` is compared with the two
steps of :math:`\Delta t` that cover the same interval; the doubling is
accepted when

.. math::

   \|e_{2\Delta t} - e_{\Delta t}\| \le r\,\|e(t + 2\Delta t) - e(t)\|,

with :math:`r` = ``-tdcsem_atsd_rtol`` and the norm taken over all degrees of
freedom and sources. The solution carried forward is always the
:math:`\Delta t` one. The BDF2 step of :math:`2\Delta t` uses :math:`e(t)`
and :math:`e(t - 2\Delta t)`. A rejected :math:`2\Delta t` operator is kept
for the next trial. At the receivers, :math:`E` is interpolated and
:math:`\partial B/\partial t = -\nabla\times E`.

Charge correction
*****************
With the discrete gradient :math:`G` (:math:`K G = 0`), the semi-discrete
equation and :math:`e(0) = 0` give the discrete charge conservation

.. math::

   G^T \left(M_\sigma\, e(t) + s(t)\, b_p\right) = 0,

the discrete form of :math:`\nabla\cdot(\sigma E) = -\nabla\cdot J_s`.
After every solve the kernel adds a gradient,
:math:`e \leftarrow e + G_i\,\phi`, with

.. math::

   \left(G_i^T M_\sigma G_i\right)\phi = -G_i^T\left(M_\sigma\, e + s(t_n)\, b_p\right),

where :math:`G_i` is :math:`G` restricted to the interior nodes
(:math:`\phi = 0` on the boundary). :math:`G_i^T M_\sigma G_i` is set up once
and solved with CG and GAMG (``rtol 1e-10``).

Inverse problem
---------------
``im.csem`` inverts for the log-resistivity of every cell. The unknown is
the perturbation :math:`x = \log\rho - \log\rho_0` with respect to the
starting model :math:`\rho_0`, and the objective is

.. math::

   \Phi(x) = \frac{1}{N_d}\sum_{c,f,r} \left|w_{cfr}\left(d^{\mathrm{obs}}_{cfr} - d_{cfr}(x)\right)\right|^2
            + \frac{\lambda}{N}\,\|x\|^2,
   \qquad w_{cfr} = \frac{1}{\epsilon\,|d^{\mathrm{obs}}_{cfr}|},

where :math:`c` runs over the fitted components (``-im_data_components``),
:math:`f` over the frequencies, :math:`r` over the receivers,
:math:`N_d` counts the real and imaginary parts of the data that are not
missing,
:math:`N` is the number of cells, :math:`\epsilon` the relative data error
(``-im_error_level``) and :math:`\lambda` the Tikhonov weight
(``-im_lambda``). The regularization term penalizes departure from the
starting model.

The predicted data are field components at the receivers,
:math:`E_i = Q_{E_i} e` and :math:`H_j = Q_{H_j} e / (i\omega\mu)`, or their
ratio :math:`Z_{ij} = E_i/H_j`, with :math:`Q` the receiver interpolation
operators and :math:`e` the solution. The adjoint source of a ratio is

.. math::

   J^T a = Q_{E_i}^T \frac{a}{H_j} - Q_{H_j}^T \frac{a\,E_i}{i\omega\mu\,H_j^2},
   \qquad a = w^2\,\overline{\left(d^{\mathrm{obs}} - Z_{ij}\right)},

and the sources of all fitted components add up.

The gradient is computed with the **adjoint-state method**: one forward and
one adjoint solve per frequency, whatever the number of components, without
forming the Jacobian. The model is
updated with **limited-memory BFGS (L-BFGS)** (``-im_lbfgs_memory``,
``-im_max_iter``). An optional neighbor smoother acts on the gradient
(``-im_diag_weight``), and the cells of fixed materials are excluded from the
update. Options are listed in :doc:`inverse_modeling`.

Verification
------------
The discretization is verified by a **method of manufactured solutions** (MMS)
program, ``build/mms`` (``make mms``, source in ``tests/mms/``), which runs the
fm.csem discretization on a manufactured problem. On the cube :math:`\Omega = [0,L]^3` with
:math:`k = m\pi/L` the exact field is a Hodge split of a solenoidal and a
gradient part,

.. math::

   E^*(x) = a\,S(x) + b\,G(x), \qquad
   S = \begin{pmatrix}\sin ky \sin kz\\ \sin kz \sin kx\\ \sin kx \sin ky\end{pmatrix},
   \qquad G = \frac{\nabla\!\left(\sin kx \sin ky \sin kz\right)}{k},

which satisfies :math:`\nabla\times\nabla\times S = 2k^2 S`,
:math:`\nabla\times G = 0` and :math:`n \times E^* = 0` on
:math:`\partial\Omega` for every integer :math:`m`.
:math:`\nabla\times\nabla\times` annihilates :math:`G`, so the gradient part
is controlled by the mass term alone. The manufactured forcing consistent with
the assembled operator is, per component,

.. math::

   f^*_d = a\left(2k^2 - i\,\omega\,\mu_0\,\sigma_d\right) S_d
           - i\,\omega\,\mu_0\,\sigma_d\,b\,G_d .

The program assembles a volumetric right-hand side from
:math:`f^*` instead of a dipole, and reports relative :math:`L^2` and **energy**
errors against :math:`E^*`, the energy norm being the one the operator induces,

.. math::

   |||v|||^2 = \|\nabla\times v\|_{L^2}^2
               + \omega\mu_0\!\int_\Omega (\sigma v)\cdot\bar v .

Both converge as :math:`O(h^p)` for the first-kind Nédélec family of degree
:math:`p`, and as :math:`O(N_{\mathrm{dof}}^{-p/3})` against the unknown count.
``-mms_diagnostics`` adds error norms recomputed under an over-integrated
quadrature rule, and ``-mms_drop_mass`` runs a negative control in which the
forcing is incomplete and the error does not converge.

The definitions live in ``tests/mms/mms.h`` and are mirrored in
``data/test1/mms_reference.py``; ``data/test1/verify_c_reference.sh`` checks the
two against each other. See ``data/test1/README.md`` for the design and the
verification protocol, and :doc:`testing`.
