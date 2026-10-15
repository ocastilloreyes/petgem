==============
Solver options
==============

The kernels solve a complex-symmetric system (see :doc:`method`) with PETSc,
so any ``KSP``/``PC`` option can go in the parameter file or on the command
line. ``fm.csem`` and ``fm.mt`` solve all their right-hand sides in one call
(transmitters, or the two MT polarizations). ``im.csem`` reuses one solver for
the forward and adjoint solves of each frequency. ``fm.tdcsem`` sets up one
solver per time step size and reuses it for every step of that size.

BDDC (iterative)
----------------
The parameter files written for ``fm.csem`` and ``im.csem`` use FGMRES with
BDDC:

.. code-block::

   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu
   -pc_bddc_monolithic

``im.csem`` adds ``-ksp_rtol 1.0e-10``, because the adjoint solution enters the
gradient. ``fm.tdcsem`` with BDDC also needs ``-ksp_rtol 1.0e-10``: with
``1e-8`` the solver error accumulates over the time steps.

When the operator is ``MATIS`` (``-dm_mat_type is``), **PETGEM** sets the
preconditioner to ``PCBDDC`` and passes it the discrete gradient :math:`G` of
the run's order (``setupBddcFromPetgemGradient`` in ``src/solver.c``). With
any other matrix type this step is skipped, so ``-pc_type bddc`` needs
``-dm_mat_type is``.

Direct solve (MUMPS)
--------------------
The ``fm.mt`` and ``fm.tdcsem`` parameter files use a direct solve, and any
kernel accepts the same block:

.. code-block::

   -dm_mat_type aij
   -ksp_type preonly
   -pc_type lu
   -pc_factor_mat_solver_type mumps

Options given later override earlier ones, so a parameter file can be
switched on the command line or with a second options file:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file params.txt \
       -dm_mat_type aij -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps

   mpirun -n 4 build/im.csem -options_file configs/params_im.txt \
       -options_file configs/solver_mumps.txt

The inversion examples ship both blocks as ``configs/solver_bddc.txt`` and
``configs/solver_mumps.txt``. ``im.csem`` uses ``MATIS`` when no
``-dm_mat_type`` is given.

A direct solve does not depend on the number of ranks, which is why the test
suite uses it. With BDDC the iteration count, and so the result of an
unconverged solve, can change with the rank count and the order.

PETGEM-specific options
-----------------------
.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Meaning
   * - ``-petgem_bddc_primal_sigma <S/m>``
     - ``fm.csem`` only. Add the DOFs of cells with conductivity above this
       value to the BDDC primal space. Off by default.
   * - ``-petgem_bddc_primal_check``
     - With the option above, check the selected DOFs against the diagonal of
       the operator.
   * - ``-petgem_validate_gradient``
     - Check the de Rham identity and the structure PCBDDC expects on the
       assembled discrete gradient.
   * - ``-petgem_grad_view``
     - View the discrete gradient matrix (PETSc viewer syntax).
