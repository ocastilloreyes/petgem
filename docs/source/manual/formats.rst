============
Data formats
============

This page documents the data formats consumed and produced by **PETGEM**: the
text and HDF5 inputs read by ``utils/preprocess.py``, the unified input bundle
it writes, and the responses file written by the kernel. All HDF5 files use the
PETSc/`h5py <https://www.h5py.org/>`_ layout; the Python helpers
``petgem.readBundle`` and ``petgem.readResponses`` provide convenient readers.

Conductivity table (``sigmas.txt``)
-----------------------------------
A whitespace-delimited text table of per-material conductivity, passed via
``-sigma_file`` (relative to ``-case_dir``). The row index is the **0-based
material id**. For a Gmsh mesh this maps to the physical group as
``material_id = gmsh:physical - 1``; for a VTK mesh the distinct cell-data
region codes are mapped to rows in ascending order (row ``i`` = ``i``-th
smallest code). See :doc:`meshing` for the per-format mapping. ``#`` comments
and blank lines are allowed (commas are also accepted, so legacy
comma-separated tables still load).

.. code-block::

   # sigma_x sigma_y sigma_z [fixed]
   0.1 0.1 0.1 1     # material 0 - held fixed during inversion (e.g. air)
   1.0 1.0 1.0 0     # material 1 - invertable
   2.0 2.0 2.0       # material 2 - 'fixed' column omitted -> defaults to 0

- ``sigma_x sigma_y sigma_z``: per-axis conductivity (S/m). Use equal values
  for an isotropic material.
- ``fixed`` (optional 4th column): ``1`` marks the material as held fixed
  during inversion (its gradient is zeroed). Forward modeling ignores it.

Sources
-------
Both modes use a single transmitter format: **one row per transmitter, 8
fields** (frequency on every row). Forward modeling (``-source_filename``)
repeats one frequency across its transmitters; inverse modeling
(``-inv_source_filename``) carries one row per ``(frequency, dipole)`` pair:

.. code-block::

   freq x_pos y_pos z_pos current length dip_angle azimuth_angle
   ...

The fields are: frequency (Hz), position ``x y z`` (m), ``current`` (A),
``length`` (m), ``dip_angle`` and ``azimuth_angle`` (degrees). The legacy
forward layout (a lone ``freq`` line followed by 7-field rows) is still
accepted and converted to this form on read.

Receivers (``-receiver_filename``)
----------------------------------
A text file of receiver positions, one Cartesian point per row. Columns may be
separated by whitespace **or** commas (``#`` comments and blank lines allowed),
so receiver lists exported from MATLAB or other tools load without conversion:

.. code-block::

   x y z
   ...

Observed data (``-observed_filename``, inverse only)
----------------------------------------------------
The observed electric field used as the inversion target, in **either** of two
forms (``preprocess.py`` detects the format by extension):

**HDF5** (``.h5`` / ``.hdf5``):

- ``/Ex``: complex ``[N_freq, N_recv]`` dataset (``complex128``, stored as an
  ``{r, i}`` compound). Row order matches the inverse source frequencies;
  column order matches the receiver list.
- ``/frequencies``: the frequency of each row (Hz).
- optional root attribute ``error_level`` (the relative noise level).

**Raw text** (``invEx.dat``, MATLAB-style) - parsed inline by ``preprocess.py``,
so no separate conversion step is needed:

.. code-block::

   freq_label  Re(Ex_1) Im(Ex_1)  Re(Ex_2) Im(Ex_2)  ...  Re(Ex_N) Im(Ex_N)

one row per frequency. The leading label is dropped (frequencies come from the
inverse sources); the noise level is supplied with ``-error_level`` (or left to
the kernel default). The standalone ``utils/convert_invex_to_hdf5.py`` can still
produce a reusable HDF5 from such a file, but is no longer required.

Input bundle (``-input_filename``, default ``input.h5``)
--------------------------------------------------------
``utils/preprocess.py`` assembles a **single unified HDF5 bundle** that the
kernel reads in full. Both modes share the same skeleton - mesh, model, order,
receivers, and ``/sources`` - and inverse runs simply **add** the two groups
that have no forward counterpart (``/observed`` and ``/inv_meta``). Each mode
populates only the fields it needs.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Group / dataset
     - Mode
     - Contents
   * - DMPlex topology + ``model_data``
     - both
     - Mesh (PETSc DMPlex layout) and per-cell conductivity, written by the preprocess
   * - ``/nord``
     - both
     - Polynomial order (length-1 vector)
   * - ``/receivers``
     - both
     - Receiver positions, flattened ``[N_recv * 3]``
   * - ``/sources/*``
     - both
     - Transmitters, one entry per row: freq, position, current, length, dipAngle, azimuthAngle (per entry). Forward repeats one frequency; inverse carries one row per (frequency, dipole).
   * - ``/observed/Ex``
     - inverse
     - Observed field; the ``@error_level`` attribute holds the relative noise level
   * - ``/inv_meta/fixed_materials``
     - inverse
     - 0-based material ids held fixed (int32)

``petgem.readBundle(path)`` returns a dict with ``receivers`` (``[N_recv, 3]``),
``nord``, ``frequency`` (first transmitter), and ``sources`` (``[N_src, 8]``:
``freq x y z current length dipAngle azimuthAngle``). The DMPlex mesh and
``model_data`` are kept in the file but not returned.

Responses file
--------------
The kernel writes one responses file per source, named
``<output_filename>_p<nord>_src<N>.h5`` (e.g. ``responses_p1_src1.h5``).

- ``/fields/{Ex, Ey, Ez, Hx, Hy, Hz}``: the computed field components at the
  receivers (complex on a complex PETSc build).
- ``/source``: attributes describing the transmitter for this file.
- Top-level attributes (provenance): ``petgem_version``, ``input_filename``,
  ``date``, ``nord``, ``mpi_tasks``.

``petgem.readResponses(path)`` returns a dict with the six field arrays plus
``source`` and ``provenance`` attribute dicts. The per-case ``postprocess.py``
scripts use this reader to compare against a reference.
