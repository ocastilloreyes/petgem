============
Data formats
============

This page documents the files **PETGEM** reads and writes: the text and HDF5
inputs consumed by ``utils/preprocess.py``, the input bundle it produces, and
the files written by ``fm.csem``, ``im.csem`` and ``fm.mt``. The HDF5 files use PETSc's viewer
layout; the Python readers in :doc:`python_api` return NumPy arrays.

Conductivity table (``sigmas.txt``)
-----------------------------------
A whitespace-delimited text table of per-material conductivity, passed via
``-sigma_file`` (relative to ``-case_dir``). The row index is the **0-based
material id**. For a Gmsh mesh this is ``material_id = gmsh:physical - 1``; for
a VTK mesh the distinct cell-data region codes are mapped to rows in ascending
order. See :doc:`meshing`. ``#`` comments and blank lines are allowed.

.. code-block::

   # sigma_x sigma_y sigma_z [fixed]
   0.1 0.1 0.1 1     # material 0 - held fixed during inversion (e.g. air)
   1.0 1.0 1.0 0     # material 1 - invertible
   2.0 2.0 2.0       # material 2 - 'fixed' column omitted -> defaults to 0

- ``sigma_x sigma_y sigma_z`` - the diagonal conductivity (S/m). Use equal
  values for an isotropic material.
- ``fixed`` (optional 4th column) - a non-zero value marks the material as held
  fixed during inversion. Forward modeling ignores it. Every row must have the
  same number of columns.

Sources
-------
Two layouts are accepted and auto-detected. The canonical one carries the
frequency on **every row** (8 fields):

.. code-block::

   freq x_pos y_pos z_pos current length dip_angle azimuth_angle
   ...

The legacy forward layout - a lone frequency line followed by 7-field rows - is
also accepted and converted to the same representation on read:

.. code-block::

   2
   1750.0 1750.0 -975.0 1. 1. 0. 0.

Fields are: frequency (Hz), position ``x y z`` (m), ``current`` (A), ``length``
(m), ``dip_angle`` and ``azimuth_angle`` (degrees).

Forward modeling (``-source_filename``) uses a single frequency shared by all
transmitters; inverse modeling (``-im_source_filename``) carries one row per
``(frequency, dipole)`` pair.

MT frequency (``-mt_frequency_filename``, MT only)
--------------------------------------------------
One frequency (Hz); ``#`` comments are allowed.

.. code-block::

   # Frequency (Hz)
   2.0

Receivers (``-receiver_filename``)
----------------------------------
Receiver positions, one Cartesian point per row. Columns may be separated by
whitespace **or** commas; ``#`` comments and blank lines are allowed.

.. code-block::

   x y z
   ...

Observed data (``-observed_filename``, inverse only)
----------------------------------------------------
The inversion target, in **either** of two forms (``preprocess.py`` selects by
file extension):

**HDF5** (``.h5`` / ``.hdf5``)

- ``/Ex`` - complex ``[N_freq, N_recv]`` dataset (``complex128``, stored as an
  ``{r, i}`` compound). Row order matches the inverse source frequencies;
  column order matches the receiver list.
- ``/frequencies`` - the frequency of each row (Hz).
- optional root attribute ``error_level`` - the relative noise level.

**Raw text** (MATLAB-style ``invEx.dat``), parsed inline - one row per
frequency:

.. code-block::

   freq_label  Re(Ex_1) Im(Ex_1)  Re(Ex_2) Im(Ex_2)  ...  Re(Ex_N) Im(Ex_N)

The leading label is dropped (frequencies come from the inverse sources); the
noise level is supplied with ``-error_level``, or left to the kernel default.

Input bundle (``-input_filename``, default ``input.h5``)
--------------------------------------------------------
``utils/preprocess.py`` writes a single HDF5 bundle that the kernel reads in
full. All modes share the same skeleton - mesh, model, order, and receivers;
CSEM runs add ``/sources``, inverse runs also ``/observed`` and ``/im_meta``,
and MT runs ``/mt``.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Group / dataset
     - Mode
     - Contents
   * - DMPlex blocks
     - all
     - The mesh, written by PETSc's DMPlex viewer into its own top-level groups
       (``topology``, ``topologies``, ``geometry``, ``labels``). The per-cell
       model - ``sigma_x, sigma_y, sigma_z`` plus the material id - is the
       ``model_data`` Vec stored with them. Read back by the kernel; the exact
       nesting is PETSc's, not **PETGEM**'s.
   * - ``/order``
     - all
     - Polynomial order (length-1 Vec)
   * - ``/receivers``
     - all
     - Receiver positions, flattened ``[N_recv * 3]``
   * - ``/sources/*``
     - fm, im
     - Transmitters, as separate Vecs: ``freq``, ``position`` (flattened),
       ``current``, ``length``, ``dipAngle``, ``azimuthAngle`` - one entry each
       per transmitter
   * - ``/observed/Ex``
     - inverse
     - Observed field; the ``@error_level`` attribute holds the relative noise level
   * - ``/im_meta/fixed_materials``
     - inverse
     - 0-based material ids held fixed (int32)
   * - ``/mt/freq``
     - mt
     - Frequency (length-1 Vec)

``petgem.readBundle(path)`` reads a CSEM bundle and returns ``receivers``
(``[N_recv, 3]``), ``order``, ``frequency`` (of the first transmitter) and
``sources`` (``[N_src, 8]``).

Output files
------------
Every kernel writes ``{output_dir}/{output_filename}.h5``, with the same
provenance attributes on the root group:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Root attribute
     - Meaning
   * - ``petgem_version``
     - Version of the code that produced the file
   * - ``simulation_type``
     - ``fm.csem``, ``im.csem`` or ``fm.mt`` - which kernel wrote it
   * - ``input_filename``
     - Path of the input bundle (which carries the mesh, model, and geometry)
   * - ``order``
     - Polynomial order actually used
   * - ``ksp_type`` / ``pc_type``
     - Solver and preconditioner from the options database, including
       command-line overrides (``default`` when unset)
   * - ``mpi_tasks``
     - Number of MPI ranks
   * - ``date``
     - Execution timestamp

Each kernel adds its own attributes, listed below.

Forward responses (``fm.csem``)
*******************************
Layout::

    /                              root attrs (shared provenance block above)
    /sources/src1/                 per-source attrs
    /sources/src1/fields/Ex        complex PETSc Vec, length N_recv
    /sources/src1/fields/Ey
    /sources/src1/fields/Ez
    /sources/src1/fields/Hx
    /sources/src1/fields/Hy
    /sources/src1/fields/Hz
    /sources/src2/                 (one such group per transmitter)
    ...

Additional root attributes: ``num_sources``, ``frequency`` (the run's operating
frequency).

Per-source attributes (``/sources/src{k}``, ``k`` 1-based): ``frequency``,
``x_pos``, ``y_pos``, ``z_pos``, ``current``, ``length``, ``dip_angle``,
``azimuth_angle``.

``petgem.readResponses(path, source=k)`` returns the six fields of transmitter
``k`` with its attributes; ``petgem.readAllResponses(path)`` returns all of
them (see :doc:`python_api`).

MT responses (``fm.mt``)
************************
Fields of both polarizations and the MT transfer functions at the receivers.
Every dataset is a complex PETSc Vec of length ``N_recv``; apparent resistivity
and phase are real values stored in the real part.

Layout::

    /                              root attrs (shared provenance block above)
    /polarizations/x/fields/Ex     x-polarization fields: Ex Ey Ez Hx Hy Hz
    ...
    /polarizations/y/fields/Ex     y-polarization fields: Ex Ey Ez Hx Hy Hz
    ...
    /impedance/xx                  Z_xx  (also xy, yx, yy)
    /apparent_resistivity/xx       rho_xx in Ohm.m  (also xy, yx, yy)
    /phase/xx                      phi_xx in degrees, atan2(Im Z, Re Z)  (also xy, yx, yy)
    /tipper/x                      T_x  (also y)

Additional root attributes: ``frequency``, ``num_receivers``.

The phase follows the :math:`e^{-i\omega t}` convention with :math:`z` up (a
half-space gives ``phi_xy = 135``, ``phi_yx = -45``); see :doc:`method`.

Inversion results (``im.csem``)
*******************************
Layout::

    /                              root attrs (shared provenance block above)
    /conductivity                  recovered model, 3 components per cell
    /log_perturbation              model parameter vector
    /rms_history                   RMS misfit, one entry per objgrad evaluation

Additional root attributes: ``num_frequencies``, ``lambda``, ``error_level``,
``num_iterations``, ``num_objgrad_evaluations``, ``convergence_reason``.

``num_iterations`` counts accepted L-BFGS steps. ``num_objgrad_evaluations``
counts objective-gradient evaluations and is the length of ``/rms_history``;
it is larger because it includes rejected line-search trials, so
``/rms_history`` is not monotone.

With ``-im_snapshot_interval N`` the kernel also writes the model every N
accepted steps, and the last accepted step, to ``snapshots/`` under the output
directory. If no step is accepted, ``iter0000`` holds the initial model::

    {output_dir}/snapshots/iter0001.pvtu       (open this in ParaView)
    {output_dir}/snapshots/iter0001_r0000.vtu  (one piece per rank)

Each snapshot has one cell field, ``rho`` (resistivity in Ohm.m). The
directory name is fixed, so runs that share an output directory overwrite each
other's snapshots. A run on fewer ranks than a previous one leaves the extra
pieces of the old run in place; clear the directory between runs.
