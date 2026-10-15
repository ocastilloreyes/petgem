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

Parameter file
--------------
``preprocess.py -mode fm.csem`` writes:

.. code-block::

   -input_filename <case_dir>/<input_filename>
   -dm_mat_type is
   -ksp_type fgmres
   -pc_type bddc
   -pc_bddc_use_deluxe_scaling 1
   -pc_bddc_coarse_pc_type lu
   -pc_bddc_monolithic
   -output_dir <case_dir>/
   -output_filename responses_p<order>

The kernel takes only the common options listed in :doc:`overview`; the rest
of the file configures PETSc (see :doc:`solver`).

Running
-------
.. code-block:: bash

   mpirun -n 4 build/fm.csem -options_file params.txt

Output
------
``<output_dir>/<output_filename>.h5`` holds ``Ex``, ``Ey``, ``Ez``, ``Hx``,
``Hy`` and ``Hz`` at every receiver, for each transmitter. Read it with
``petgem.readResponses`` or ``petgem.readAllResponses``; the layout is in
:doc:`formats`.
