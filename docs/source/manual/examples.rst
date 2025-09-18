Examples
========
This section provides example simulations to help users get started with 3D CSEM modeling
using **PETGEM**. These examples are designed to be practical, allowing
users to validate the code and learn the typical workflow for setting up and running simulations.

Canonical CSEM example
----------------------
These test cases validate **PETGEM** using 3D CSEM models
with simple stratified resistivity structures. The setup reproduces the reference experiment described in:

Castillo-Reyes, O., de la Puente, J. M., Cela, J. M. (2018). *PETGEM: A parallel code for 3D CSEM forward modeling using edge finite elements.* Computers & Geosciences, 119, 123–136. https://doi.org/10.1016/j.cageo.2018.07.005 <https://doi.org/10.1016/j.cageo.2018.07.005>_

### Model description

- Frequency: 2 Hz
- Source position: (0.0, 0.0, -975.0) m
- Conductivity structures:
   - Seawater: 3.33 S/m
   - Sediments: 1.0 S/m
   - Oil reservoir: 0.01 S/m
   - Sediments: 1.0 S/m

The domain is discretized with an unstructured tetrahedral mesh generated using **Gmsh**.

### Execution steps

Run the following commands to execute the test cases for `nord=1` or `nord=2`. Replace `[nord]` with `1` or `2` as needed:

.. code-block:: bash

   # Setup environment
   export PETGEM_CSEM_TEST_DIR=tests/csem_model

   # Mesh generation (specific to nord)
   gmsh ${PETGEM_CSEM_TEST_DIR}/mesh_p[nord].geo -3

   # Generate resistivity model
   python3 ${PETGEM_CSEM_TEST_DIR}/generate_resistivity_model.py [nord]

   # Generate parameter file
   python3 ${PETGEM_CSEM_TEST_DIR}/generate_params_file.py [nord]

   # Forward modeling (parallel)
   mpirun -n 4 build/csem_kernel -options_file ${PETGEM_CSEM_TEST_DIR}/params_nord[nord].txt

   # Compare results with reference
   python3 ${PETGEM_CSEM_TEST_DIR}/compare_responses.py [nord]

### Step-by-step

1. **Mesh generation**  
   `mesh_p[nord].geo` defines the geometry and meshing strategy. Gmsh outputs the tetrahedral mesh for simulation

2. **Resistivity model generation**  
   `generate_resistivity_model.py [nord]` builds the layered conductivity distribution for `nord=1` or `nord=2`

3. **Parameter file generation**  
   `generate_params_file.py [nord]` creates `params_nord[nord].txt` containing **PETGEM** runtime options

4. **Forward modeling**  
   `csem_kernel` runs in parallel (4 MPI tasks) to compute the CSEM responses

5. **Results comparison**  
   `compare_responses.py [nord]` validates **PETGEM** output against semi-analytical 1D reference responses (Dipole1D)

### Expected outcome

- Forward responses computed by **PETGEM** are compared to 1D semi-analytical solutions
- Agreement is quantified via normalized root-mean-square deviation (NRMSD)
- Successful execution confirms that **PETGEM** produces accurate results for both the `nord=1` and `nord=2` scenarios


Extrae profiling example
------------------------
This example demonstrates how to profile **PETGEM** using **Extrae** to analyze the
performance of the CSEM forward modeling kernel. It guides users through the steps
to generate execution traces and visualize them with **Paraver**, allowing identification
of parallel performance bottlenecks and evaluation of load balancing.


### Execution steps

Run the following commands to compile **PETGEM** with Extrae support, generate the mesh,
resistivity model, parameter file, execute the forward modeling, and create the trace:

.. code-block:: bash

   # Setup environment
   export EXTRAE_TEST_DIR=tests/extrae_profiling
   export EXTRAE_CONFIG_FILE=${EXTRAE_TEST_DIR}/extrae.xml
   export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:$LD_LIBRARY_PATH
   export TRACE_NAME=petgem.prv
   export EXTRAE_LABELS=${EXTRAE_TEST_DIR}/petgem_labels.txt

   # Compile PETGEM with Extrae instrumentation
   make USE_EXTRAE=1

   # Mesh generation
   gmsh ${EXTRAE_TEST_DIR}/mesh_p1.geo -3

   # Generate resistivity model
   python3 ${EXTRAE_TEST_DIR}/generate_resistivity_model.py 1

   # Generate parameter file
   python3 ${EXTRAE_TEST_DIR}/generate_params_file.py 1

   # Forward modeling (parallel)
   mpirun -n 4 build/csem_kernel -options_file ${EXTRAE_TEST_DIR}/params_nord1.txt

   # Merge intermediate files and create the trace
   $EXTRAE_HOME/bin/mpi2prv -f ${EXTRAE_TEST_DIR}/TRACE.mpits -o ${EXTRAE_TEST_DIR}/${TRACE_NAME}

### Step-by-step

1. **Compile PETGEM with Extrae**  
   Activates instrumentation to record performance events during execution

2. **Mesh generation**  
   `mesh_p1.geo` defines the geometry and meshing strategy. Gmsh outputs the tetrahedral mesh for simulation

3. **Resistivity model generation**  
   `generate_resistivity_model.py 1` builds the layered conductivity distribution for `nord=1`

4. **Parameter file generation**  
   `generate_params_file.py 1` creates `params_nord1.txt` containing **PETGEM** runtime options

5. **Forward modeling**  
   `csem_kernel` runs in parallel (4 MPI tasks) to compute the CSEM responses while collecting performance events

6. **Trace creation**  
   Merges intermediate **Extrae** files and generates the execution trace `${EXTRAE_TEST_DIR}/petgem.prv`

### Expected outcome

- Execution trace `petgem.prv` is created in `${EXTRAE_TEST_DIR}`
- The trace can be opened with **Paraver** using the configuration file at `${EXTRAE_TEST_DIR}/petgem_functions.cfg`
- Users can analyze PETGEM’s parallel performance, identify bottlenecks, and assess load balancing for further optimizations
