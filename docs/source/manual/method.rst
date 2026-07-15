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

where :math:`\omega = 2\pi f` is the angular frequency. Homogeneous Dirichlet
boundary conditions :math:`n \times E = 0` are imposed on the domain boundary.

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
assembled by ``assembleCsemKandM`` (``src/assembly.c``). Because :math:`A` is
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
mode, enabled with ``fm.csem -mms``. On the unit cube :math:`[0,1]^3` the exact
field

.. math::

   E^*(x,y,z) = \big(\sin \pi y \sin \pi z,\;
                     \sin \pi z \sin \pi x,\;
                     \sin \pi x \sin \pi y\big)

satisfies :math:`\nabla\times\nabla\times E^* = 2\pi^2 E^*` and
:math:`n \times E^* = 0` on the boundary, so the manufactured forcing consistent
with the assembled operator is, per component,

.. math::

   f^*_d = \left(2\pi^2 - i\,\omega\,\mu\,\sigma_d\right) E^*_d .

In this mode the kernel assembles a volumetric right-hand side from
:math:`f^*` instead of a dipole, and reports relative :math:`L^2` and
:math:`H(\mathrm{curl})` errors against :math:`E^*`. The definitions live in
``include/mms.h`` and are mirrored in ``tests/mms/mms_reference.py``. See
:doc:`testing`.
