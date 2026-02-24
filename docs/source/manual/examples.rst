========
Examples
========
This section provides example simulations to help users get started with 3D CSEM modeling
using **PETGEM**. These examples are designed to be practical, allowing
users to validate the code and learn the typical workflow for setting up and running simulations.

Canonical CSEM example
----------------------
These test cases validate **PETGEM** using 3D CSEM models
with simple stratified resistivity structures. The setup reproduces the canonical model described in:

Castillo-Reyes, O., de la Puente, J. M., Cela, J. M. (2018). *PETGEM: A parallel code for 3D CSEM forward modeling using edge finite elements.* Computers & Geosciences, 119, 123–136. `DOI 10.1016/j.cageo.2018.07.005 <https://doi.org/10.1016/j.cageo.2018.07.005>`_.


Model description
*****************

- Frequency: 2 Hz
- Source position at xyz=[0.0, 0.0,-975.0] m
- Conductivity structures

   - Seawater: 3.33 S/m
   - Sediments: 1.0 S/m
   - Oil reservoir: 0.01 S/m
   - Sediments: 1.0 S/m

The domain is discretized with an unstructured tetrahedral mesh generated using `Gmsh <http://gmsh.info/>`_.

Execution steps
***************

Run the following commands to execute the test cases for ``nord=1`` or ``nord=2``.
Set the desired polynomial order by modifying the ``NORD`` variable:

.. code-block:: bash

   # Compile PETGEM
   make

   # Setup environment
   export CSEM_TEST_DIR=tests/csem_model
   export NORD=1        # Change to 2 if needed

   # Mesh generation
   gmsh ${CSEM_TEST_DIR}/mesh_p${NORD}.geo -3

   # Generate input data
   python3 ${CSEM_TEST_DIR}/generate_input.py \
       -nord ${NORD} \
       -case_dir ${CSEM_TEST_DIR} \
       -mesh_filename mesh_p${NORD}.msh \
       -source_filename sources.txt \
       -receiver_filename receivers.txt

   # Forward modeling (parallel example)
   mpirun -n 4 build/fm.csem \
       -options_file ${CSEM_TEST_DIR}/params_nord${NORD}.txt

   # Compare results with reference
   python3 ${CSEM_TEST_DIR}/compare_responses.py ${NORD}

Step-by-step
************

1. **Compilation**

   ``make`` builds the PETGEM executable (``build/fm.csem``).

2. **Environment setup**

   The environment variables ``CSEM_TEST_DIR`` and ``NORD`` define
   the test case directory and the polynomial order (``nord=1`` or ``nord=2``).

3. **Mesh generation**

   ``mesh_p${NORD}.geo`` defines the geometry and meshing strategy.
   ``gmsh`` generates the 3D unstructured tetrahedral mesh
   (``mesh_p${NORD}.msh``).

4. **Input data generation**

   ``generate_input.py`` creates the required simulation inputs,
   including the resistivity model and parameter file
   (``params_nord${NORD}.txt``), based on the selected polynomial order.

5. **Forward modeling**

   ``fm.csem`` runs in parallel (4 MPI tasks in this test case)
   using the generated parameter file to compute the CSEM responses.

6. **Results comparison**

   ``compare_responses.py ${NORD}`` validates PETGEM results
   against the semi-analytical 1D reference responses
   from Dipole1D.

Expected outcome
****************

- Forward responses computed by **PETGEM** are compared to 1D semi-analytical solutions
- Agreement is quantified via normalized root-mean-square deviation (NRMSD)
- Successful execution confirms that **PETGEM** produces accurate results for both the ``nord=1`` and ``nord=2`` scenarios.

.. figure:: /_static/images/csem_test_p2.png
   :alt: Comparison of Ex component between PETGEM* and Dipole1D for nord=2
   :align: center
   :width: 95%

   Comparison of the electric field component **Ex** between **PETGEM** and the
   semi-analytical reference code **Dipole1D** for the case ``nord=2``. The resulting
   NRMSD is **0.0134**.


Extrae profiling example
------------------------
This example demonstrates how to profile **PETGEM** using `Extrae <https://tools.bsc.es/extrae>`_ to analyze the performance of the CSEM forward modeling kernel. It guides users through the steps to generate execution traces and visualize them with `Paraver <https://tools.bsc.es/paraver>`_, allowing identification of parallel performance bottlenecks and evaluation of load balancing.


Execution steps
***************

Run the following commands to compile **PETGEM** with Extrae support, generate the mesh,
resistivity model, parameter file, execute the forward modeling, and generate the performance trace:

.. code-block:: bash

   # Setup environment
   export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:$LD_LIBRARY_PATH
   export EXTRAE_TEST_DIR=tests/extrae_profiling
   export EXTRAE_CONFIG_FILE=${EXTRAE_TEST_DIR}/extrae.xml
   export TRACE_NAME=petgem.prv
   export EXTRAE_LABELS=${EXTRAE_TEST_DIR}/petgem_labels.txt
   export NORD=1  

   # Compile PETGEM with Extrae instrumentation
   make USE_EXTRAE=1

   # Mesh generation
   gmsh ${EXTRAE_TEST_DIR}/mesh_p${NORD}.geo -3

   # Generate input data
   python3 ${EXTRAE_TEST_DIR}/generate_input.py \
       -nord ${NORD} \
       -case_dir ${EXTRAE_TEST_DIR} \
       -mesh_filename mesh_p${NORD}.msh \
       -source_filename sources.txt \
       -receiver_filename receivers.txt

   # Forward modeling (parallel example)
   mpirun -n 4 build/fm.csem.extrae \
       -options_file ${EXTRAE_TEST_DIR}/params_nord${NORD}.txt

   # Merge intermediate files and create the trace
   ${EXTRAE_HOME}/bin/mpi2prv -f ${EXTRAE_TEST_DIR}/TRACE.mpits -o ${EXTRAE_TEST_DIR}/${TRACE_NAME}   


Step-by-step
************

1. **Environment setup**

   Set environment variables to enable `Extrae <https://tools.bsc.es/extrae>`_ support, define the test directory,
   trace name, labels file, and the polynomial order (``NORD=1``):

   - ``LD_LIBRARY_PATH`` to include Extrae libraries
   - ``EXTRAE_TEST_DIR`` for test case files
   - ``EXTRAE_CONFIG_FILE`` for Extrae configuration
   - ``TRACE_NAME`` for the output trace
   - ``EXTRAE_LABELS`` for the labels used in instrumentation
   - ``NORD`` for the finite element basis functions

2. **Compilation**

   ``make USE_EXTRAE=1`` builds PETGEM with Extrae instrumentation enabled
   (``build/fm.csem.extrae``).

3. **Mesh generation**

   ``mesh_p${NORD}.geo`` defines the geometry and meshing strategy.
   ``gmsh`` generates the 3D unstructured tetrahedral mesh
   (``mesh_p${NORD}.msh``).

4. **Input data generation**

   ``generate_input.py`` creates the required simulation inputs,
   including the resistivity model and parameter file
   (``params_nord${NORD}.txt``), based on the selected polynomial order.

5. **Forward modeling**

   ``fm.csem.extrae`` runs in parallel (4 MPI tasks in this test case)
   using the generated parameter file to compute the CSEM responses
   while recording performance events.

6. **Trace generation**

   ``${EXTRAE_HOME}/bin/mpi2prv`` merges intermediate Extrae files
   and generates the execution trace
   (``${EXTRAE_TEST_DIR}/${TRACE_NAME}``) for performance analysis.


Expected outcome
****************

- Execution trace ``petgem.prv`` is created in ``${EXTRAE_TEST_DIR}``
- The trace can be opened with `Paraver <https://tools.bsc.es/paraver>`_ using the configuration file ``${EXTRAE_TEST_DIR}/petgem_functions.cfg``
- Users can analyze **PETGEM** parallel performance, identify bottlenecks, and assess load balancing for further optimizations


.. figure:: /_static/images/extrae_trace.png
   :alt: Extrae trace for PETGEM execution
   :align: center
   :width: 95%

   Extrae trace of a **PETGEM** simulation executed with 4 MPI tasks. Each color represents a distinct instrumented phase of the **PETGEM** execution, highlighting the temporal distribution and concurrency of computational stages across MPI processes.