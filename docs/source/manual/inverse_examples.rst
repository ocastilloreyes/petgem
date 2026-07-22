=========================
Inverse modeling examples
=========================

One inverse example ships under ``examples/``. It is the reference benchmark
for ``im.csem`` and demonstrates the complete controlled-source EM inversion
workflow end to end:

.. code-block::

   true model -> forward modeling -> synthetic observations -> noise
              -> inversion -> recovered model

For the kernel options, inputs and stopping criteria used below, see
:doc:`inverse_modeling`.

CSEM inversion benchmark
------------------------
``examples/im_model`` recovers a buried conductive block from noisy
multi-frequency surface data. A known model is forward-modeled with
``fm.csem``, contaminated with 1 % Gaussian noise, and inverted with ``im.csem``
from a homogeneous starting model. The inversion recovers the conductor and
fits the data down to the noise level. Every input is provided, and the
synthetic observations are shipped, so the inversion can be reproduced without
repeating the forward stage.

Model
*****
A resistive halfspace containing a single conductive block, beneath an air
layer. Coordinates use **z positive downward**; ``z = 0`` is the air/earth
interface.

.. list-table::
   :header-rows: 1

   * - Region
     - Resistivity
     - Conductivity
   * - Air
     - --
     - ``1e-8`` S/m
   * - Background (earth)
     - ``100`` Ω·m
     - ``0.01`` S/m
   * - Anomaly (target)
     - ``10`` Ω·m
     - ``0.1`` S/m

The anomaly is a 200 × 200 × 200 m cube (volume ``8e6`` m³) centered at
``(0, 0, 200)`` m, i.e. its top is 100 m below the surface.

Acquisition:

- Frequencies: 1, 10, 50, 100, 300, 800, 1500 Hz.
- Source: one x-directed horizontal electric dipole at ``(0, -4000, 0)`` m,
  unit current and length.
- Receivers: a 21 × 21 surface grid (441 points), ``x, y`` in ``[-200, 200]`` m
  at 20 m spacing, ``z = 0``.
- Basis order: 2.

Two meshes define the geometry, both built with the Gmsh built-in kernel:

.. list-table::
   :header-rows: 1

   * - Mesh
     - Physical volumes
     - Role
   * - ``geometry/im_model.msh``
     - AIR, BG, INVERT
     - inversion (homogeneous starting model)
   * - ``geometry/im_true.msh``
     - AIR, BG, INVERT, ANOMALY
     - true model (forward modeling)

The ``INVERT`` region is the invertable subset of cells; ``AIR`` and ``BG`` are
held fixed during the inversion (the ``fixed`` column of
``survey/sigmas_im.txt``).

Both meshes come from the single source ``geometry/im_model.geo``, which meshes
only ``AIR`` and ``BG``; ``INVERT`` and ``ANOMALY`` are then assigned by
*re-tagging* cells whose centroid falls in the corresponding box. Re-tagging
changes material labels only, leaving the mesh topology - and therefore the
discrete gradient that PCBDDC is built from - untouched, whereas embedding the
bodies as geometric volumes would introduce new edges and faces. The step is
scripted, with the mask read from the ``.geo`` itself so it cannot drift from
the refinement fields::

    python3 examples/im_model/scripts/build_meshes.py --verify   # check, no writes
    python3 examples/im_model/scripts/build_meshes.py --force    # re-mesh (needs gmsh)

Under the ``petgem-env`` image this reproduces both shipped meshes
byte-identically.

Directory layout
****************
.. code-block::

   examples/im_model/
     README.md
     geometry/    im_model.geo, im_model.msh, im_true.msh
     survey/      receivers, sources (im + per-frequency fm), frequencies,
                  sigmas_im.txt (starting model), sigmas_true.txt (true model)
     configs/     params_im.txt, params_fm_f<F>.txt
     scripts/     build_meshes.py, gen_survey.py, build_bundles.sh,
                  run_forward.slurm, run_inversion.slurm
     reference/   observed.h5, reference_metrics.json, reference_metrics.md
     outputs/     generated files (bundles, responses, logs, VTU snapshots)

``outputs/`` is git-ignored and disposable: a fresh clone starts empty there and
each workflow recreates what it needs, so nothing in it is required to run the
benchmark. Note that the solver stages write to fixed filenames, so re-running
them overwrites earlier results; since the directory is untracked, an
overwritten run can only be recovered by rerunning the simulation. Copy any run
worth keeping outside ``outputs/`` first.

General, reusable tools live in the ``utils/`` package, not in the example:
``utils/preprocess.py`` (build a bundle), ``utils/make_observed.py`` (assemble
forward responses and add noise), and ``utils/analyze_inversion.py`` (evaluate a
recovered model against a reference).

Running
*******
The commands below run from the repository root. The forward stage uses the
iterative PCBDDC solver and the inversion uses the MUMPS direct solver (see
:doc:`solver`); both come from the shipped parameter files under ``configs/``.

**Quick start (inversion only).** Uses the shipped ``reference/observed.h5``, so
no forward run is required:

.. code-block:: bash

   make

   # 1. Build the inverse input bundle
   python3 utils/preprocess.py -mode im -order 2 \
      -case_dir examples/im_model \
      -mesh_filename geometry/im_model.msh \
      -receiver_filename survey/receivers.txt \
      -im_source_filename survey/sources_im.txt \
      -observed_filename reference/observed.h5 \
      -error_level 0.01 \
      -sigma_file survey/sigmas_im.txt \
      -input_filename outputs/input_im.h5

   # 2. Run the inversion
   mpirun -n 4 build/im.csem \
      -options_file examples/im_model/configs/params_im.txt

   # 3. Evaluate the recovered model
   python3 utils/analyze_inversion.py \
      -run_dir examples/im_model/outputs \
      -reference examples/im_model/reference/reference_metrics.json

**Full workflow (from scratch).** Regenerate the observations by forward
modeling the true model, then invert:

.. code-block:: bash

   # (optional) regenerate the survey definition
   python3 examples/im_model/scripts/gen_survey.py

   # 1. Forward: one bundle + one run per frequency (shown for 1 Hz)
   python3 utils/preprocess.py -mode fm -order 2 \
      -case_dir examples/im_model \
      -mesh_filename geometry/im_true.msh \
      -receiver_filename survey/receivers.txt \
      -source_filename survey/sources_f1.txt \
      -sigma_file survey/sigmas_true.txt \
      -input_filename outputs/input_fm_f1.h5
   mpirun -n 4 build/fm.csem \
      -options_file examples/im_model/configs/params_fm_f1.txt

   # 2. Assemble the 7 responses and add 1% noise
   python3 utils/make_observed.py \
      -case_dir examples/im_model \
      -pattern "outputs/responses_fm_f{freq}_p2.h5" \
      -freqs 1,10,50,100,300,800,1500 \
      -seed 20260720 -error_level 0.01 \
      -out outputs/observed.h5

   # 3. Invert with the just-made observations
   python3 utils/preprocess.py -mode im -order 2 \
      -case_dir examples/im_model \
      -mesh_filename geometry/im_model.msh \
      -receiver_filename survey/receivers.txt \
      -im_source_filename survey/sources_im.txt \
      -observed_filename outputs/observed.h5 \
      -error_level 0.01 -sigma_file survey/sigmas_im.txt \
      -input_filename outputs/input_im.h5
   mpirun -n 4 build/im.csem \
      -options_file examples/im_model/configs/params_im.txt

On a cluster, the ``scripts/build_bundles.sh`` wrapper drives the preprocessing
for all frequencies, and ``scripts/run_forward.slurm`` /
``scripts/run_inversion.slurm`` submit the solver stages via SLURM.

Steps in detail:

1. **Forward modeling.** ``fm.csem`` is monochromatic, so each frequency is a
   separate bundle and run. Each writes ``responses_fm_f<F>_p2.h5`` under
   ``outputs/``.
2. **Observations and noise.** ``make_observed.py`` stacks the 7 responses into
   an ``[N_freq, N_recv]`` array and adds Gaussian noise with standard deviation
   ``error_level·|E_x|`` on the real and imaginary parts independently. The seed
   is fixed and recorded in ``observed.h5``, so the dataset is reproducible.
3. **Inversion.** ``im.csem`` inverts from the uniform 100 Ω·m starting model,
   updating only the ``INVERT`` region, and writes the recovered conductivity,
   log-perturbation and RMS history to ``responses_im_p2.h5`` plus VTU model
   snapshots.
4. **Evaluation.** ``analyze_inversion.py`` reads the RMS history and the final
   VTU snapshot, reports the misfit and recovered-anomaly statistics, and checks
   them against the tolerances in ``reference/reference_metrics.json``.

Expected results
****************
The inversion converges to the noise floor and recovers the conductor. Full
values and acceptance tolerances are in ``reference/reference_metrics.md`` and
``reference/reference_metrics.json``.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Quantity
     - Value
   * - RMS of the true model
     - ≈ 1.0
   * - Initial RMS
     - 11.35
   * - Final RMS
     - 1.05 (``CONVERGED_RMSTOL``, ~90 iterations)
   * - Background resistivity
     - 100 Ω·m
   * - Peak recovered resistivity
     - ≈ 11 Ω·m (true 10)
   * - Conductor lateral offset
     - 6.9 m
   * - Conductor volume (ρ < 30 Ω·m)
     - 3.5e6 m³

``analyze_inversion.py`` prints these and reports **PASS** when they fall within
tolerance. The RMS decreases monotonically at every iteration. The recovered
conductor is correctly located and correctly scaled; its shallow bias and
reduced volume are the expected resolution limits of a single surface receiver
plane with one source, not an error.

Reproducibility
***************
The noise seed is fixed (``20260720``) and recorded in ``observed.h5``, and the
inversion uses a direct solver, so the recovered model is independent of the MPI
rank count. The meshes are provided with their material tags; ``im_model.geo``
documents how the base mesh is built (see :doc:`meshing`).
