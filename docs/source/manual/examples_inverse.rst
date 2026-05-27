=========================
Inverse modeling examples
=========================
This section provides an example inversion to help users get started with 3D
CSEM inverse modeling using **PETGEM**. The example exercises the full
inversion workflow - preprocessing, kernel execution, and model recovery - on
the case shipped under ``tests/cases/inverse``. For the forward workflow, see
:doc:`examples`.

Canonical inversion example
---------------------------
This test case recovers a subsurface conductivity model from multi-frequency
CSEM data using the L-BFGS inverse kernel.

Model description
*****************

- Frequencies: 50, 300, 800, 1500 Hz
- Source position at xyz = [0.0, -4000.0, 0.0] m
- Four materials in ``sigmas.txt``:

   - Air: fixed (excluded from inversion)
   - Ocean: fixed (excluded from inversion)
   - Sediments: invertable
   - Basement: invertable

The starting model is a half-space guess used to seed the inversion (not the
truth model that synthesized the observed data). The observed data
(``observed_data.h5``) contains the complex **Ex** response, sized
``[N_freq, N_recv]``. The case ships a pre-generated mesh ``mesh_p1.msh``.

Execution steps
***************

Run the following commands to execute the inversion. The case is provided at
``nord=1``:

.. code-block:: bash

   # Compile PETGEM
   make

   # Setup environment
   export INVERSE_TEST_DIR=tests/cases/inverse
   export NORD=1

   # Generate the input bundle and parameter file (inverse mode).
   # The bundle embeds the multi-frequency sources, observed data,
   # noise level, and the fixed-material list from sigmas.txt.
   python3 utils/preprocess.py \
      -mode inverse \
      -nord ${NORD} \
      -case_dir ${INVERSE_TEST_DIR} \
      -mesh_filename mesh_p${NORD}.msh \
      -receiver_filename receivers.txt \
      -inv_source_filename sources.txt \
      -observed_filename observed_data.h5 \
      -sigma_file sigmas.txt \
      -input_filename input_p${NORD}.h5 \
      -params_filename params_p${NORD}.txt

   # Inverse modeling (parallel example)
   mpirun -n 4 build/im.csem \
      -options_file ${INVERSE_TEST_DIR}/params_p${NORD}.txt \
      -inv_max_iter 30

Step-by-step
************

1. **Compilation**

   ``make`` builds the PETGEM executables (including ``build/im.csem``).

2. **Environment setup**

   The environment variables ``INVERSE_TEST_DIR`` and ``NORD`` define the test
   case directory and the polynomial order.

3. **Input data generation**

   ``utils/preprocess.py -mode inverse`` assembles the unified input bundle
   (``input_p${NORD}.h5``) and the matching inverse parameter file
   (``params_p${NORD}.txt``). In addition to the mesh, conductivity table, and
   receivers, the bundle embeds the multi-frequency sources
   (``-inv_source_filename``), the observed data (``-observed_filename``), the
   noise level, and the fixed-material list derived from the ``fixed`` column
   of ``sigmas.txt``. The mesh ships pre-generated as ``mesh_p1.msh``.

   Here ``-observed_filename`` is the shipped HDF5 (``observed_data.h5``), but a
   raw MATLAB-style ``invEx.dat`` can be passed directly instead (parsed inline,
   with ``-error_level`` for the noise level) - no separate conversion step
   (see :doc:`formats`).

4. **Inverse modeling**

   ``im.csem`` runs in parallel (4 MPI tasks in this example) using the
   generated parameter file. At each L-BFGS iteration it solves the forward and
   adjoint problems for every frequency, assembles the misfit gradient, and
   updates the conductivity of the invertable materials. ``-inv_max_iter``
   overrides the iteration cap from the parameter file; any ``-inv_*`` knob
   (see :doc:`inverse_modeling`) can be set on the command line.

Expected outcome
****************

- The inverted conductivity of the invertable materials evolves from the
  starting half-space guess toward the structure implied by the observed data.
- The RMS data misfit decreases over the L-BFGS iterations until it reaches the
  ``-inv_rms_tol`` early-stop threshold or the iteration cap.
- With ``-inv_snapshot_interval`` enabled, **PETGEM** writes a conductivity VTU
  snapshot to ``-output_dir`` every N accepted steps, allowing the recovered
  model to be inspected as the inversion progresses.
