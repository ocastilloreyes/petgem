===============
Troubleshooting
===============

Build
-----
``PETSC_DIR is unset or invalid``
*********************************
The Makefile checks ``PETSC_DIR`` before including PETSc's configuration and
stops with this message when it is empty or points at a directory without
``lib/petsc/conf/variables``. Set it (and ``PETSC_ARCH``):

.. code-block:: bash

   make PETSC_DIR=/path/to/petsc PETSC_ARCH=arch-linux-c-opt

The Docker image sets both (see :doc:`install`).

Real-vs-complex scalar mismatch
*******************************
**PETGEM** assembles a complex operator and requires a PETSc built with
``--with-scalar-type=complex``. A PETSc configured for real scalars fails when
reading the complex HDF5 datasets. Rebuild PETSc with complex scalars, or use
the Docker image.

Stale objects after a header change
***********************************
Object files are not tracked against header changes. After editing anything in
``include/``, rebuild from clean:

.. code-block:: bash

   make clean && make

``doxygen`` / ``Sphinx`` not found
**********************************
``make docs`` checks for both and stops with an explicit message. Install
Doxygen from your package manager, and the Python packages with
``pip install -r docs/requirements.txt``.

Run
---
Missing required option
***********************
The kernels require ``-input_filename``, ``-output_dir``, and
``-output_filename``, and abort naming the one that is missing. Run
``./fm.csem -help intro`` for a usage summary, or ``-help`` for the full option
database.

``-order`` out of range
***********************
The polynomial order must be in ``1..6``; the kernel aborts otherwise. When
``-order`` is not given, the value stored in the bundle's ``/order`` dataset is
used.

BDDC is not being configured
****************************
**PETGEM** sets up ``PCBDDC`` (registering its discrete-gradient matrix) only
when the operator is of type ``MATIS``. If ``-pc_type bddc`` is set without
``-dm_mat_type is``, that setup is skipped. Keep both keys together - the
forward parameter file emits them as a pair. See :doc:`solver`.

``Option -inv_… was renamed to -im_…``
**************************************
Every inversion option now carries the ``-im_`` prefix, matching the ``im.csem``
kernel and the ``petgem im`` subcommand. A parameter file still using the old
``-inv_`` spelling is rejected with an error naming the replacement. Rename the
options (``-inv_max_iter`` → ``-im_max_iter``, and so on); the values and
meanings are unchanged.

``-im_observed_mode fm_native`` aborts
***************************************
In ``fm_native`` mode the observed data lives in an ``fm.csem`` responses file
rather than the bundle, so ``-im_observed_file`` is mandatory and the kernel
aborts without it. See :doc:`inverse_modeling`.

Tests are skipped
*****************
Levels 4-6 need the ``fm.csem`` binary; their fixtures skip when it is not
found. Point ``PETGEM_FM_CSEM`` at the built binary:

.. code-block:: bash

   PETGEM_FM_CSEM=build/fm.csem pytest tests/e2e

See :doc:`testing`.
