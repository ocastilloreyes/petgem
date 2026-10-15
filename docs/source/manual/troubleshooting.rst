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
**PETGEM** needs a PETSc built with ``--with-scalar-type=complex``. Rebuild
PETSc with complex scalars, or use the Docker image.

``doxygen`` / ``Sphinx`` not found
**********************************
``make docs`` checks for both and stops with an explicit message. Install
Doxygen from your package manager, and the Python packages with
``pip install -r docs/requirements.txt``.

Run
---
Missing required option
***********************
The kernels need ``-input_filename``, ``-output_dir`` and ``-output_filename``
and stop naming the one that is missing. ``-help intro`` prints a usage
summary.

``-order`` out of range
***********************
The order must be 1 to 6, whether it comes from ``-order`` or from the
bundle.

BDDC is not being configured
****************************
**PETGEM** passes its discrete gradient to ``PCBDDC`` only when the operator
is ``MATIS``. Set ``-dm_mat_type is`` together with ``-pc_type bddc``. See
:doc:`solver`.

``Cannot find /observed/<component>``
*************************************
A component named in ``-im_data_components`` is not in the bundle. Add its
file to ``-observed_filename`` (``component:file``) and rebuild the bundle.
See :doc:`inverse_modeling`.

``-im_observed_mode fm_native`` aborts
***************************************
``fm_native`` reads the observed data from an ``fm.csem`` output file, which
must be given with ``-im_observed_file``. See :doc:`inverse_modeling`.

Tests are skipped
*****************
Tests skip when their binary or input bundle is missing. Build the binaries
with ``make`` (``make mms`` for level 6), build the bundle with the
``build_bundle.sh`` or ``build_bundles.sh`` script named in the skip message,
or point the ``PETGEM_*`` variables at the binaries. See :doc:`testing`.
