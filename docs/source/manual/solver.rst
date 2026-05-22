==============================
Solver options and performance
==============================

**PETGEM** assembles a complex-symmetric linear system per frequency and solves
it with PETSc. This page describes the default solver configurations, how to
choose a solver as the polynomial order increases, and the mesh/parallelism
considerations that drive performance.

Default solver configurations
------------------------------
The parameter file emitted by ``utils/preprocess.py`` carries a working solver
configuration for each mode.

Forward modeling
****************
The forward kernel defaults to a preconditioned iterative solver - FGMRES with
the **BDDC** preconditioner (deluxe scaling, LU coarse solve):

.. code-block::

   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu

This converges in a handful of iterations at low to moderate order and scales
well across MPI ranks.

Inverse modeling
****************
The inverse kernel re-solves the system at every L-BFGS iteration (forward and
adjoint, for every frequency), so it defaults to a **direct** factorization
with MUMPS:

.. code-block::

   -ksp_type preonly
   -pc_type                    lu
   -pc_factor_mat_solver_type  mumps
   -mat_mumps_icntl_14         80
   -mat_mumps_icntl_28         1

Choosing a solver by polynomial order
-------------------------------------
The BDDC preconditioner is built from a topological (order-1) discrete gradient
hint. As the polynomial order rises, the high-order curl-kernel is increasingly
invisible to that order-1 coarse space, and BDDC convergence degrades:

==========  ===========================  ==============================================
``nord``    Recommended forward solver   Notes
==========  ===========================  ==============================================
1-5         BDDC (FGMRES)                Converges in a handful of iterations
6           LU + MUMPS                   BDDC degrades sharply; use a direct solver
==========  ===========================  ==============================================

At ``nord=6``, switch the forward run to a direct solver by overriding the
solver keys on the command line (or editing the parameter file):

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file params_p6.txt \
      -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps

Adaptive BDDC coarse-space enrichment is **not** applicable here: the CSEM
operator is complex-symmetric and indefinite, which violates the
positive-definiteness assumption of the adaptive eigenvalue selection.

Mesh resolution and the h-p tradeoff
------------------------------------
Accuracy scales with both mesh refinement (``h``) and polynomial order (``p``).
Higher order delivers more accuracy per element, so the mesh can be **coarsened
as the order increases** - the canonical case ships a separate ``mesh_p${NORD}.geo``
for each order, growing coarser with ``nord``.

The binding constraint for accuracy is resolving the conductivity contrast
(e.g. a resistive block), controlled by the local element size in that region.
Refining the source line or the far field has little effect on the misfit once
the contrast is resolved. See :doc:`meshing` for the meshing parameters.

Parallelism
-----------
The kernels are MPI-parallel and the numerical result is invariant to the rank
count (within solver tolerance). For best wall-clock performance, match the
number of MPI ranks to the available physical cores - oversubscribing (more
ranks than cores) makes MPI ranks contend and inflates run time. A typical
parallel invocation is:

.. code-block:: bash

   mpirun -n <ranks> build/fm.csem -options_file params.txt
