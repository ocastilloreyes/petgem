==========
Quickstart
==========

This page gets a first **PETGEM** forward simulation running in a few commands,
using the canonical ``csem_model`` case. For the concepts behind each step see
:doc:`overview`; for the full walkthrough see :doc:`examples`.

Prerequisites
-------------
A working build environment with PETSc (complex scalars) and the Python helpers
- see :doc:`install`. The quickest route is the provided Docker image, which
ships every dependency.

Build
-----
.. code-block:: bash

   make

This builds ``build/fm.csem`` (forward), ``build/im.csem`` (inverse), and the
unified ``build/petgem`` dispatcher.

Run the canonical forward example
----------------------------------
.. code-block:: bash

   # Case directory and polynomial order
   export CSEM_TEST_DIR=examples/csem_model
   export NORD=1

   # 1. Mesh
   gmsh ${CSEM_TEST_DIR}/mesh_p${NORD}.geo -3 -o ${CSEM_TEST_DIR}/mesh_p${NORD}.msh

   # 2. Assemble the input bundle + parameter file (forward mode)
   python3 utils/preprocess.py \
      -mode forward \
      -nord ${NORD} \
      -case_dir ${CSEM_TEST_DIR} \
      -mesh_filename mesh_p${NORD}.msh \
      -receiver_filename receivers.txt \
      -source_filename sources.txt \
      -sigma_file sigmas.txt \
      -input_filename input_p${NORD}.h5 \
      -params_filename params_p${NORD}.txt

   # 3. Forward modeling
   mpirun -n 4 build/fm.csem \
      -options_file ${CSEM_TEST_DIR}/params_p${NORD}.txt

   # 4. Validate against the reference (prints NRMSD)
   python3 ${CSEM_TEST_DIR}/postprocess.py \
      -responses_filename responses_p${NORD}.h5

Step 4 prints the normalized root-mean-square deviation (NRMSD) of ``|Ex|``
against the shipped reference; values below the default tolerance (``0.03``)
indicate a successful run.

Using the unified dispatcher
----------------------------
The single-purpose binaries and the ``petgem`` dispatcher run identical code
paths. The forward run above is equivalent to:

.. code-block:: bash

   mpirun -n 4 build/petgem modeling \
      -options_file ${CSEM_TEST_DIR}/params_p${NORD}.txt

For an inversion run, use ``-mode inverse`` in preprocessing and
``build/im.csem`` (or ``build/petgem inverse``) - see :doc:`examples_inverse`.

Next steps
----------
- :doc:`overview` - the shared workflow, bundle, and conductivity model
- :doc:`forward_modeling` / :doc:`inverse_modeling` - per-mode details
- :doc:`formats` - input/output data formats
- :doc:`solver` - solver choice and performance per polynomial order
