==========
Quickstart
==========

This page runs a first **PETGEM** forward simulation on the ``fm_model``
example: a marine CSEM benchmark with a thin resistive layer. For the concepts
behind each step see :doc:`overview`; for the full walkthrough see
:doc:`examples`.

Prerequisites
-------------
A build environment with PETSc (complex scalars), Gmsh, and the Python helpers -
see :doc:`install`. The provided Docker image ships all of them.

Build
-----
.. code-block:: bash

   make

This builds ``build/fm.csem``, ``build/im.csem``, and ``build/petgem``.

Run the forward example
-----------------------
.. code-block:: bash

   export MODEL_DIR=examples/fm_model
   export ORDER=1

   # 1. Mesh
   gmsh -3 ${MODEL_DIR}/geometry/mesh.geo -o ${MODEL_DIR}/outputs/mesh.msh

   # 2. Assemble the input bundle
   #    (or wrap steps 1-2 with: bash ${MODEL_DIR}/scripts/build_bundles.sh ${ORDER})
   python3 utils/preprocess.py \
      -mode fm \
      -order ${ORDER} \
      -case_dir ${MODEL_DIR} \
      -mesh_filename outputs/mesh.msh \
      -source_filename survey/sources.txt \
      -receiver_filename survey/receivers.txt \
      -sigma_file survey/sigmas.txt \
      -input_filename outputs/input.h5 \
      -params_filename outputs/_pp_p${ORDER}.txt

   # 3. Forward modeling (committed solver options in configs/)
   mpirun -n 4 build/fm.csem \
      -options_file ${MODEL_DIR}/configs/params.txt -order ${ORDER}

   # 4. Compare against the shipped reference
   python3 ${MODEL_DIR}/scripts/postprocess.py -tolerance 0.03

Step 4 reports the NRMSD, relative L2, and MAPE of :math:`|E_x|` against
``reference/reference.h5``, and exits non-zero if the NRMSD exceeds ``-tolerance``.

Selecting the polynomial order
------------------------------
The order is stored in the bundle (``/order``) by the preprocess step, and can
also be overridden at run time with ``-order``:

.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file ${MODEL_DIR}/configs/params.txt -order 2

Because the override bypasses the bundle's value, one bundle can be reused for
any order in ``1..6``.

Using the dispatcher
--------------------
``build/petgem`` runs the same kernel code. The forward run above is equivalent
to:

.. code-block:: bash

   mpirun -n 4 build/petgem fm \
      -options_file ${MODEL_DIR}/configs/params.txt -order ${ORDER}

Next steps
----------
- :doc:`overview` - the shared workflow, the input bundle, the conductivity model
- :doc:`examples` - the shipped example cases
- :doc:`formats` - input and output data formats
- :doc:`solver` - solver options
