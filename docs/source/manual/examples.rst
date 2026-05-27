=========================
Forward modeling examples
=========================
This section provides example simulations to help users get started with 3D
CSEM forward modeling using **PETGEM**. These examples are designed to be
practical, allowing users to validate the code and learn the typical workflow
for setting up and running simulations. For an inversion walkthrough, see
:doc:`examples_inverse`.

Canonical CSEM example
----------------------
This test case validates **PETGEM** using a 3D CSEM model with a simple
conductivity structure.

Model description
*****************

- Frequency: 2 Hz
- Source position at xyz = [0.0, 0.0, 0.0] m
- Conductivity structures

   - Resistive block: 0.01 S/m
   - Half-space: 1.0 S/m

The domain is discretized with an unstructured tetrahedral mesh generated using
`Gmsh <http://gmsh.info/>`_. The case ships a per-order mesh ``mesh_p${NORD}.geo``
for each polynomial order.

Execution steps
***************

Run the following commands to execute the test case. Set the desired polynomial
order (``1`` through ``6``) by modifying the ``NORD`` variable:

.. code-block:: bash

   # Compile PETGEM
   make

   # Setup environment
   export CSEM_TEST_DIR=tests/cases/csem_model
   export NORD=1        # any order in 1..6

   # Mesh generation
   gmsh ${CSEM_TEST_DIR}/mesh_p${NORD}.geo -3 -o ${CSEM_TEST_DIR}/mesh_p${NORD}.msh

   # Generate the input bundle and parameter file (forward mode)
   python3 utils/preprocess.py \
      -mode forward \
      -nord ${NORD} \
      -case_dir ${CSEM_TEST_DIR} \
      -mesh_filename mesh_p${NORD}.msh \
      -receiver_filename receivers.txt \
      -source_filename sources.txt \
      -sigma_file sigmas.txt \
      -input_filename input_p${NORD}.h5 \
      -params_filename params_p${NORD}.txt \
      -output_vtk model.vtu

   # Forward modeling (parallel example)
   mpirun -n 4 build/fm.csem \
      -options_file ${CSEM_TEST_DIR}/params_p${NORD}.txt

   # Compare results against the reference
   python3 ${CSEM_TEST_DIR}/postprocess.py \
      -responses_filename responses_p${NORD}_src1.h5

Step-by-step
************

1. **Compilation**

   ``make`` builds the PETGEM executables (including ``build/fm.csem``).

2. **Environment setup**

   The environment variables ``CSEM_TEST_DIR`` and ``NORD`` define the test
   case directory and the polynomial order (any value in ``1..6``).

3. **Mesh generation**

   ``mesh_p${NORD}.geo`` defines the geometry and meshing strategy (coarser at
   higher order). ``gmsh`` generates the 3D unstructured tetrahedral mesh
   ``mesh_p${NORD}.msh``.

4. **Input data generation**

   ``utils/preprocess.py -mode forward`` assembles the unified input bundle
   (``input_p${NORD}.h5``) and the matching parameter file
   (``params_p${NORD}.txt``) from the mesh, conductivity table, receivers, and
   sources. The optional ``-output_vtk`` writes the conductivity model for
   visualization.

5. **Forward modeling**

   ``fm.csem`` runs in parallel (4 MPI tasks in this example) using the
   generated parameter file to compute the CSEM responses
   (``responses_p${NORD}_src1.h5``).

6. **Results comparison**

   ``postprocess.py`` validates PETGEM results against the precomputed ModEM
   reference (``reference.h5``) shipped with the case, reporting the normalized
   root-mean-square deviation (NRMSD).

Expected outcome
****************

- Forward responses computed by **PETGEM** are compared to the reference
  solution.
- Agreement is quantified via normalized root-mean-square deviation (NRMSD),
  which must fall below the configured tolerance (default ``0.03``).
- Successful execution confirms that **PETGEM** produces accurate results
  across the supported polynomial orders.

.. figure:: /_static/images/csem_test_p2.png
   :alt: Comparison of Ex component between PETGEM and the reference for nord=2
   :align: center
   :width: 95%

   Comparison of the electric field component **Ex** between **PETGEM** and the
   reference solution for the case ``nord=2``.


Extrae profiling example
------------------------
This example demonstrates how to profile **PETGEM** using `Extrae <https://tools.bsc.es/extrae>`_ to analyze the performance of the CSEM forward modeling kernel. It guides users through the steps to generate execution traces and visualize them with `Paraver <https://tools.bsc.es/paraver>`_, allowing identification of parallel performance bottlenecks and evaluation of load balancing.


Execution steps
***************

Run the following commands to compile **PETGEM** with Extrae support, generate
the mesh and input bundle, execute the forward modeling, and generate the
performance trace:

The Extrae configuration files (``extrae.xml``, ``petgem_labels.txt``, and the
Paraver configuration ``petgem_functions.cfg``) ship with the canonical case
under ``tests/cases/csem_model/extrae``; the mesh, sources, receivers, and
conductivity table are reused from ``tests/cases/csem_model``. This is the same
execution exercised in CI.

.. code-block:: bash

   # Setup environment
   export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:$LD_LIBRARY_PATH
   export CSEM_TEST_DIR=tests/cases/csem_model
   export EXTRAE_DIR=${CSEM_TEST_DIR}/extrae
   export EXTRAE_CONFIG_FILE=${EXTRAE_DIR}/extrae.xml
   export EXTRAE_LABELS=${EXTRAE_DIR}/petgem_labels.txt
   export TRACE_NAME=petgem.prv
   export NORD=1

   # Compile PETGEM with Extrae instrumentation
   make USE_EXTRAE=1

   # Mesh generation
   gmsh ${CSEM_TEST_DIR}/mesh_p${NORD}.geo -3 -o ${CSEM_TEST_DIR}/mesh_p${NORD}.msh

   # Generate the input bundle and parameter file (forward mode)
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

   # Forward modeling (parallel example)
   mpirun -n 4 build/fm.csem.extrae \
       -options_file ${CSEM_TEST_DIR}/params_p${NORD}.txt

   # Merge intermediate files and create the trace
   ${EXTRAE_HOME}/bin/mpi2prv -f ${CSEM_TEST_DIR}/TRACE.mpits -o ${CSEM_TEST_DIR}/${TRACE_NAME}


Step-by-step
************

1. **Environment setup**

   Set environment variables to enable `Extrae <https://tools.bsc.es/extrae>`_ support, define the case and Extrae
   configuration directories, trace name, labels file, and the polynomial order (``NORD=1``):

   - ``LD_LIBRARY_PATH`` to include Extrae libraries
   - ``CSEM_TEST_DIR`` for the case data (mesh, sources, receivers, conductivity)
   - ``EXTRAE_DIR`` for the Extrae configuration files (``tests/cases/csem_model/extrae``)
   - ``EXTRAE_CONFIG_FILE`` for the Extrae configuration (``extrae.xml``)
   - ``EXTRAE_LABELS`` for the labels used in instrumentation (``petgem_labels.txt``)
   - ``TRACE_NAME`` for the output trace
   - ``NORD`` for the finite element basis functions

2. **Compilation**

   ``make USE_EXTRAE=1`` builds PETGEM with Extrae instrumentation enabled
   (``build/fm.csem.extrae``).

3. **Mesh generation**

   ``mesh_p${NORD}.geo`` defines the geometry and meshing strategy.
   ``gmsh`` generates the 3D unstructured tetrahedral mesh
   ``mesh_p${NORD}.msh``.

4. **Input data generation**

   ``utils/preprocess.py -mode forward`` assembles the input bundle and
   parameter file (``params_p${NORD}.txt``) for the selected polynomial order.

5. **Forward modeling**

   ``fm.csem.extrae`` runs in parallel (4 MPI tasks in this example) using the
   generated parameter file to compute the CSEM responses while recording
   performance events.

6. **Trace generation**

   ``${EXTRAE_HOME}/bin/mpi2prv`` merges intermediate Extrae files
   and generates the execution trace
   (``${CSEM_TEST_DIR}/${TRACE_NAME}``) for performance analysis.


Expected outcome
****************

- Execution trace ``petgem.prv`` is created in ``${CSEM_TEST_DIR}``
- The trace can be opened with `Paraver <https://tools.bsc.es/paraver>`_ using the configuration file ``${EXTRAE_DIR}/petgem_functions.cfg``
- Users can analyze **PETGEM** parallel performance, identify bottlenecks, and assess load balancing for further optimizations


.. figure:: /_static/images/extrae_trace.png
   :alt: Extrae trace for PETGEM execution
   :align: center
   :width: 95%

   Extrae trace of a **PETGEM** simulation executed with 4 MPI tasks. Each color represents a distinct instrumented phase of the **PETGEM** execution, highlighting the temporal distribution and concurrency of computational stages across MPI processes.
