==============================
Frequency-domain CSEM modeling
==============================

``fm.csem`` computes the electric and magnetic fields of one or more dipole
transmitters at a set of receivers, for a 3D conductivity model at a single
frequency. The formulation is in :doc:`method`.

Inputs
------
Besides the mesh and ``sigmas.txt``, ``preprocess.py -mode fm.csem`` takes:

``-source_filename``
   The transmitters. All rows must share one frequency; preprocessing stops
   otherwise, so a multi-frequency survey needs one bundle per frequency.
   Each transmitter is ``freq x y z current length dip azimuth``, in Hz, m,
   A, m and degrees. The frequency can also be given once on the first line,
   followed by 7-field rows (see :doc:`formats`).

``-receiver_filename``
   One ``x y z`` point per row.

``-background_layers_file``, ``-background_sigma_file``
   Background model of the secondary-field formulation (optional): a layered
   model, a conductivity per material, or both (see :doc:`formats`). With
   either, the parameter file selects ``-csem_formulation secondary``.

Parameter file
--------------
``preprocess.py -mode fm.csem`` writes:

.. code-block::

   -input_filename <case_dir>/<input_filename>
   -output_dir <case_dir>/
   -output_filename responses_p<order>

   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu

The kernel takes the common options listed in :doc:`overview` and the field
formulation options below; the rest of the file configures PETSc (see
:doc:`solver`).

``-csem_formulation <total|secondary>``
   ``total`` (default) solves for the total field :math:`\mathbf{E}`.
   ``secondary`` solves for :math:`\mathbf{E} - \mathbf{E}_p`, where
   :math:`\mathbf{E}_p` is the field of the transmitters in a background
   model.

``-csem_primary <layered|model>``
   Background model of the secondary formulation. ``layered``: a layered VTI
   medium; :math:`\mathbf{E}_p` is computed analytically. ``model``: one
   conductivity per material on the same mesh; :math:`\mathbf{E}_p` is a
   finite-element solve with the background conductivity. Default:
   ``layered`` if the background has layers, ``model`` otherwise.

``-csem_background_file <file>``
   HDF5 file with the ``/background`` group. Default: the input bundle.

The run report lists the background model, the number of cells with
:math:`\sigma \neq \sigma_b` and the time of :math:`\mathbf{E}_p`. With the
layered background, the cells that cross a layer interface are reported with
a warning; :math:`\sigma_b` of a cell is the layer at its centroid.

Running
-------
.. code-block:: bash

   # Run the simulation
   mpirun -n 4 build/fm.csem -options_file params.txt

Output
------
``<output_dir>/<output_filename>.h5`` holds ``Ex``, ``Ey``, ``Ez``, ``Hx``,
``Hy`` and ``Hz`` at every receiver, for each transmitter. Read it with
``petgem.readResponses`` or ``petgem.readAllResponses``; the layout is in
:doc:`formats`. With ``-csem_formulation secondary`` these are the total
fields, and the secondary fields are read with
``petgem.readResponses(filename, group="secondary_fields")``.
