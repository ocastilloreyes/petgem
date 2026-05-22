============
Data formats
============

This page documents the data formats consumed and produced by **PETGEM**: the
text and HDF5 inputs read by ``utils/preprocess.py``, the unified input bundle
it writes, and the responses file written by the kernel. All HDF5 files use the
PETSc/`h5py <https://www.h5py.org/>`_ layout; the Python helpers
``petgem.readBundle`` and ``petgem.readResponses`` provide convenient readers.

Conductivity table (``sigmas.csv``)
-----------------------------------
A CSV table of per-material conductivity, passed via ``-sigma_file`` (relative
to ``-case_dir``). The row index is the **0-based material id**, which maps to
the Gmsh physical group as ``material_id = gmsh:physical - 1``.

.. code-block::

   # sigma_x, sigma_y, sigma_z [, fixed]
   0.1, 0.1, 0.1, 1     # material 0 - held fixed during inversion (e.g. air)
   1.0, 1.0, 1.0, 0     # material 1 - invertable
   2.0, 2.0, 2.0        # material 2 - 'fixed' column omitted -> defaults to 0

- ``sigma_x, sigma_y, sigma_z``: per-axis conductivity (S/m). Use equal values
  for an isotropic material.
- ``fixed`` (optional 4th column): ``1`` marks the material as held fixed
  during inversion (its gradient is zeroed). Forward modeling ignores it.

Source files
------------
Forward sources (``-source_filename``)
**************************************
A text file whose first non-comment line is the operating frequency (Hz),
followed by one row per dipole:

.. code-block::

   freq
   x_pos y_pos z_pos current length dip_angle azimuth_angle
   ...

Inverse sources (``-inv_source_filename``)
******************************************
A text file with one row per ``(frequency, dipole)`` pair - 8 fields, the
frequency prepended to the dipole parameters:

.. code-block::

   freq x_pos y_pos z_pos current length dip_angle azimuth_angle
   ...

In both cases the fields are: position ``x y z`` (m), ``current`` (A),
``length`` (m), ``dip_angle`` and ``azimuth_angle`` (degrees).

Receivers (``-receiver_filename``)
----------------------------------
A text file of receiver positions, one Cartesian point per row:

.. code-block::

   x y z
   ...

Observed data (``-observed_filename``, inverse only)
----------------------------------------------------
An HDF5 file holding the observed electric field used as the inversion target:

- ``/Ex``: complex ``[N_freq, N_recv]`` dataset (``complex128``, stored as an
  ``{r, i}`` compound). Row order matches the inverse source frequencies;
  column order matches the receiver list.
- ``/frequencies``: the frequency of each row (Hz).

Input bundle (``-input_filename``, default ``input.h5``)
--------------------------------------------------------
``utils/preprocess.py`` assembles a **single unified HDF5 bundle** that the
kernel reads in full. It always contains the mesh, model, order, and receivers;
inverse runs add the inversion groups.

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
     - forward
     - Single-frequency transmitter: frequency, position, current, length, dipAngle, azimuthAngle
   * - ``/inv_sources/*``
     - inverse
     - Multi-frequency transmitter: freq, position, current, length, dipAngle, azimuthAngle (per entry)
   * - ``/observed/Ex``
     - inverse
     - Observed field; the ``@error_level`` attribute holds the relative noise level
   * - ``/inv_meta/fixed_materials``
     - inverse
     - 0-based material ids held fixed (int32)

``petgem.readBundle(path)`` returns a dict with ``receivers`` (``[N_recv, 3]``),
``nord``, ``frequency``, and ``sources`` (``[N_src, 7]``:
``x y z current length dipAngle azimuthAngle``). The DMPlex mesh and
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
