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

   \nabla \times \nabla \times E \;-\; i\,\omega\,\mu\,\sigma\, E \;=\; f,

where :math:`\omega = 2\pi f` is the angular frequency. For CSEM, homogeneous
Dirichlet boundary conditions :math:`n \times E = 0` are imposed on the domain
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
boundary integral, assembled by ``assembleMtBoundaryRHS`` (``src/mt.c``) with a
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

Inverse problem
---------------
``im.csem`` recovers a conductivity model :math:`m` - one value per invertable
material - by minimizing a regularized data-misfit functional combining the
difference between observed and predicted responses (weighted by the relative
data-error level, ``-im_error_level``) with a Tikhonov term of weight
:math:`\lambda` (``-im_lambda``) that penalizes departure from the starting
model.

The gradient is assembled by the **adjoint-state method**: each iteration
performs forward and adjoint solves per frequency, avoiding formation of the
full Jacobian. The model is updated with a **limited-memory BFGS (L-BFGS)**
scheme (``-im_lbfgs_memory``, ``-im_max_iter``). An optional neighbor
smoother acts on the gradient (``-im_diag_weight``); materials flagged as
fixed are excluded from the update. Options are listed in
:doc:`inverse_modeling`.

Verification
------------
The discretization is verified by a **method of manufactured solutions** (MMS)
mode, enabled with ``fm.csem -mms``. On the cube :math:`\Omega = [0,L]^3` with
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

In this mode the kernel assembles a volumetric right-hand side from
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

The definitions live in ``include/mms.h`` and are mirrored in
``data/test1/mms_reference.py``; ``data/test1/verify_c_reference.sh`` checks the
two against each other. See ``data/test1/README.md`` for the design and the
verification protocol, and :doc:`testing`.
