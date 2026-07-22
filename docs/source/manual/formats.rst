============
Data formats
============

This page documents the files **PETGEM** reads and writes: the text and HDF5
inputs consumed by ``utils/preprocess.py``, the input bundle it produces, and
the responses file written by ``fm.csem``. The HDF5 files use PETSc's viewer
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
   1.0 1.0 1.0 0     # material 1 - invertable
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
full. Both modes share the same skeleton - mesh, model, order, receivers, and
``/sources``; inverse runs add ``/observed`` and ``/im_meta``.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Group / dataset
     - Mode
     - Contents
   * - DMPlex blocks
     - both
     - The mesh, written by PETSc's DMPlex viewer into its own top-level groups
       (``topology``, ``topologies``, ``geometry``, ``labels``). The per-cell
       model - ``sigma_x, sigma_y, sigma_z`` plus the material id - is the
       ``model_data`` Vec stored with them. Read back by the kernel; the exact
       nesting is PETSc's, not **PETGEM**'s.
   * - ``/order``
     - both
     - Polynomial order (length-1 Vec)
   * - ``/receivers``
     - both
     - Receiver positions, flattened ``[N_recv * 3]``
   * - ``/sources/*``
     - both
     - Transmitters, as separate Vecs: ``freq``, ``position`` (flattened),
       ``current``, ``length``, ``dipAngle``, ``azimuthAngle`` - one entry each
       per transmitter
   * - ``/observed/Ex``
     - inverse
     - Observed field; the ``@error_level`` attribute holds the relative noise level
   * - ``/im_meta/fixed_materials``
     - inverse
     - 0-based material ids held fixed (int32)

``petgem.readBundle(path)`` returns a dict with ``receivers`` (``[N_recv, 3]``),
``order``, ``frequency`` (that of the first transmitter), and ``sources``
(``[N_src, 8]``: ``freq x y z current length dipAngle azimuthAngle``). The
DMPlex mesh and ``model_data`` stay in the file and are consumed by the C
kernel.

Output files
------------
Both kernels write ``{output_dir}/{output_filename}.h5``. The **root
provenance block is identical** in both, written by one shared routine, so any
PETGEM product can be traced back to the run that made it:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Root attribute
     - Meaning
   * - ``petgem_version``
     - Version of the code that produced the file
   * - ``simulation_type``
     - ``fm`` or ``im`` - which kernel wrote it
   * - ``input_filename``
     - Path of the input bundle (which carries the mesh, model, and geometry)
   * - ``order``
     - Polynomial order actually used
   * - ``ksp_type`` / ``pc_type``
     - Solver and preconditioner, as resolved from the options database
       (``default`` when unset), so the file records the configuration the run
       really used - including command-line overrides
   * - ``mpi_tasks``
     - Number of MPI ranks
   * - ``date``
     - Execution timestamp

Each kernel then adds its own product-specific attributes.

Forward responses (``fm.csem``)
*******************************
All six field components at the receivers, for every transmitter. The output
Vecs are parallel on the kernel communicator, so writes are collective when
PETSc is linked against a parallel HDF5 build.

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

Inversion results (``im.csem``)
*******************************
The recovered model and the convergence history.

Layout::

    /                              root attrs (shared provenance block above)
    /conductivity                  recovered model, 3 components per cell
    /log_perturbation              model parameter vector
    /rms_history                   RMS misfit per iteration

Additional root attributes: ``num_frequencies``, ``lambda``, ``error_level``,
``num_iterations``, ``convergence_reason``.

With ``-im_snapshot_interval`` enabled, the kernel also writes a ParaView
snapshot of the model every N accepted L-BFGS steps, named from the same output
stem::

    {output_dir}/{output_filename}_iter00001.pvtu          (master)
    {output_dir}/{output_filename}_iter00001_p0000.vtu     (one piece per rank)

Each snapshot carries a single cell field, ``rho`` (resistivity in Ohm.m).

Python readers:

- ``petgem.readResponses(path, source=1)`` - the per-source dict (``Ex``…``Hz``
  arrays, plus ``source`` and ``provenance`` attribute dicts).
- ``petgem.readAllResponses(path)`` - ``{'provenance': ..., 'num_sources': N,
  'sources': {1: {...}, ...}}``, each entry shaped like ``readResponses``.

``examples/fm_model/scripts/postprocess.py`` uses these readers to compare
against a reference.
