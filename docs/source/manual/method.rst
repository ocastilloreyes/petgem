=====================
Numerical formulation
=====================

This page summarizes the numerical method underlying **PETGEM**. For the full
derivation and validation, see the publications listed in :doc:`publications`.

Forward problem
---------------
**PETGEM** solves the 3D Controlled-Source Electromagnetic (CSEM) problem in the
frequency domain. For a time-harmonic source, the electric field :math:`E`
satisfies

.. math::

   \nabla \times \left( \mu^{-1}\, \nabla \times E \right)
       + i\omega\sigma\, E = -\,i\omega\, J_s,

where :math:`\mu` is the magnetic permeability, :math:`\sigma` the conductivity,
:math:`\omega` the angular frequency, and :math:`J_s` the source current density.

The field is discretized with **high-order Nédélec (edge) vector finite
elements** of order ``nord`` (1 to 6) on an unstructured tetrahedral mesh. These
:math:`H(\mathrm{curl})`-conforming elements enforce tangential continuity
across faces and represent the curl-curl operator accurately while keeping the
number of unknowns low relative to low-order formulations. Discretization yields
the complex linear system

.. math::

   (K + i\omega M)\, e = b,

with

.. math::

   K_{ij} = \int_\Omega (\nabla \times N_i)\cdot \mu^{-1}\,(\nabla \times N_j)\,d\Omega,
   \qquad
   M_{ij} = \int_\Omega N_i\cdot \sigma\, N_j\,d\Omega,
   \qquad
   b_i = -\,i\omega \int_\Omega N_i\cdot J_s\,d\Omega,

where :math:`N_i` are the vector basis functions. The receiver responses are
obtained by interpolating :math:`e` at the receiver locations. The system is
solved per frequency with PETSc (see :doc:`solver`).

Inverse problem
---------------
The inverse kernel recovers a conductivity model :math:`m` (one value per
invertable material) by minimizing the regularized data-misfit functional

.. math::

   \Phi(m) = (d_\mathrm{obs} - d_\mathrm{pre})^H\, C_d^{-1}\,
             (d_\mathrm{obs} - d_\mathrm{pre})
           + \lambda\,(m - m_0)^T\, C_m^{-1}\,(m - m_0),

where :math:`d_\mathrm{obs}` and :math:`d_\mathrm{pre}` are the observed and
predicted responses, :math:`C_d` the data covariance (set from the noise level),
:math:`m_0` the reference model, and :math:`\lambda` the Tikhonov factor
(``-inv_lambda``).

The gradient of :math:`\Phi` is assembled by the **adjoint-state method**: at
each iteration, one forward and one adjoint solve per frequency yield the
sensitivity contribution without forming the full Jacobian. The model is updated
with a **limited-memory BFGS (L-BFGS)** scheme (``-inv_lbfgs_memory``,
``-inv_max_iter``). An optional neighbor-smoothing operator regularizes the
recovered model spatially (``-inv_diag_weight``); materials flagged as fixed are
excluded from the update.

Conventions
-----------
Internally the kernels use the :math:`e^{-i\omega t}` time convention, so the
assembled operator carries the corresponding sign; this is equivalent to the
:math:`e^{+i\omega t}` formulation above under complex conjugation and does not
affect the computed field magnitudes. **PETGEM** must therefore be built against
a PETSc configured with complex scalars (see :doc:`install`).
