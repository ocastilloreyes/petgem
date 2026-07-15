==============
Solver options
==============

**PETGEM** assembles a complex-symmetric linear system (see :doc:`method`) and
solves it with PETSc. Any PETSc ``KSP``/``PC`` option can be set in the
parameter file or on the command line; this page describes the defaults the
preprocess emits and the preconditioner **PETGEM** configures itself.

Forward default: BDDC
---------------------
The parameter file emitted by ``utils/preprocess.py -mode fm`` selects
FGMRES with the **BDDC** preconditioner (deluxe scaling, LU coarse solve), on a
``MATIS`` operator:

.. code-block::

   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu

BDDC requires ``-dm_mat_type is``: **PETGEM** configures ``PCBDDC`` only when the
operator is of type ``MATIS``. In that case it registers the **discrete gradient
matrix** :math:`G` assembled alongside the operator, via
``PCBDDCSetDiscreteGradient(pc, G, order, 0, PETSC_TRUE, PETSC_TRUE)``
(``setupBDDCFromPetgemGradient`` in ``src/solver.c``). :math:`G` is built at the
**same polynomial order as the run**, so the coarse space captures the
curl-kernel of the Nédélec space at that order. With a non-``MATIS`` matrix
type, the ``PCBDDC`` setup is skipped and PETSc's option handling applies as
usual.

Inverse default: direct solve
-----------------------------
The inverse kernel re-solves the system at every L-BFGS iteration (forward and
adjoint, for each frequency), so ``utils/preprocess.py -mode im`` emits a
**direct** factorization with MUMPS:

.. code-block::

   -ksp_type preonly
   -pc_type                    lu
   -pc_factor_mat_solver_type  mumps
   -mat_mumps_icntl_14         80
   -mat_mumps_icntl_28         1

Using a direct solver
---------------------
The solver keys can be overridden on the command line without touching the
parameter file. To run the forward kernel with a direct solve:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file params.txt \
      -dm_mat_type aij -ksp_type preonly -pc_type lu \
      -pc_factor_mat_solver_type mumps

Note the ``-dm_mat_type aij``: it overrides the ``is`` in the parameter file,
which exists only to enable BDDC. A direct solve is deterministic and
independent of iterative convergence, which is why the test suite
(:doc:`testing`) uses it for its reference comparisons.

Parallelism
-----------
The kernels are MPI-parallel:

.. code-block:: bash

   mpirun -n <ranks> build/fm.csem -options_file params.txt

The receiver interpolation is written to be MPI-invariant, and the test suite
runs the same comparisons under multiple rank counts. Note that iterative
convergence (and hence the accuracy of an unconverged iterative solve) can
depend on the rank count and the polynomial order; a direct solve does not.
