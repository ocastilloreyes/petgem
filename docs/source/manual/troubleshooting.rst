===============
Troubleshooting
===============

Common build- and run-time issues, with their cause and resolution.

Build
-----
``PETSC_DIR is unset or invalid``
*********************************
The Makefile checks ``PETSC_DIR`` before including PETSc's configuration and
stops with this message when it is empty or points at a directory without
``lib/petsc/conf/variables``. Set it (and ``PETSC_ARCH``) to your PETSc install:

.. code-block:: bash

   make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt

The provided Docker image sets these for you (see :doc:`install`).

Real-vs-complex scalar mismatch
*******************************
**PETGEM** requires a PETSc built with **complex** scalars
(``--with-scalar-type=complex``). Symptoms include a load-time error that a
dataset is "marked as real but PETSc is configured for complex" when reading
HDF5 inputs. Rebuild PETSc with complex scalars, or use the Docker image, which
already does.

Run
---
``Blocksize of layout ... must match that of mapping`` at high order
********************************************************************
A block-size mismatch reported during assembly at ``nord >= 4`` indicates a
stale build picking up an older object file. After any header change, rebuild
from clean:

.. code-block:: bash

   make clean && make

BDDC is very slow or stalls at ``nord=6``
*****************************************
The BDDC preconditioner is built from an order-1 coarse-space hint, which
becomes a poor preconditioner for the order-6 operator (convergence degrades to
hundreds of iterations, or stalls on a coarse mesh). Use a direct solver for
``nord=6``:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file params_p6.txt \
      -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps

See :doc:`solver` for the per-order solver guidance. Adaptive BDDC enrichment is
not a remedy here - the CSEM operator is complex-symmetric and indefinite,
which the adaptive eigenvalue selection does not support.

Run is slow / oversubscribed MPI
********************************
Using more MPI ranks than physical cores makes ranks contend and inflates
wall-clock time. Match ``-n`` to the available cores. The result is invariant
to the rank count (within solver tolerance), so reducing ranks does not change
the answer.

Documentation
-------------
"No API modules generated yet" on the API reference
****************************************************
This indicated the API generator was scanning the wrong directory for headers;
it now scans ``include/`` and fails loudly if no headers are found. Regenerate
with:

.. code-block:: bash

   make docs

See :doc:`contributing` for the documentation build.
