==========
Quickstart
==========

This page runs the ``examples/fm_csem_marine`` marine CSEM case and checks the result
against its reference. You need PETSc with complex scalars, Gmsh and the
Python packages (see :doc:`install`); the Docker image has all of them.

.. code-block:: bash

   make

   # Mesh + input bundle (outputs/mesh.msh, outputs/input.h5)
   bash examples/fm_csem_marine/scripts/build_bundles.sh 1

   # Forward solve
   (cd examples/fm_csem_marine && mpirun -n 4 ../../build/fm.csem -options_file configs/params.txt)

   # Compare |Ex| with the reference
   python3 examples/fm_csem_marine/scripts/postprocess.py -tolerance 0.03

``build_bundles.sh`` runs Gmsh and then ``utils/preprocess.py``; open it to see
the preprocessing arguments. The solve writes ``outputs/responses_p1.h5``.
``postprocess.py`` prints the NRMSD, relative L2 error and MAPE of
:math:`|E_x|`, saves a figure in ``outputs/`` and exits with a non-zero status
when the NRMSD is above ``-tolerance``.

The kernel's ``-order`` would override the bundle, but ``postprocess.py``
takes the order from the bundle to name its input and figure. To run order 2,
rebuild the bundle and change the output name, which ``configs/params.txt``
fixes to ``responses_p1``:

.. code-block:: bash

   bash examples/fm_csem_marine/scripts/build_bundles.sh 2
   (cd examples/fm_csem_marine && mpirun -n 4 ../../build/fm.csem -options_file configs/params.txt \
       -output_filename responses_p2)
   python3 examples/fm_csem_marine/scripts/postprocess.py

Next steps
----------
- :doc:`overview` - workflow, bundle and kernel options
- :doc:`examples` - the forward cases in detail
- :doc:`solver` - choosing the linear solver
