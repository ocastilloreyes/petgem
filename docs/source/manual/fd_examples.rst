==============================
Frequency-domain CSEM examples
==============================

Unit cube (``examples/fm_csem_unit_cube``)
------------------------------------------
A homogeneous cube :math:`[0,1]^3` (:math:`\sigma = 1` S/m) with one 2 Hz
dipole at the center and three receivers. It is the dataset of the test suite
(see :doc:`testing`). The mesh is transfinite, so Gmsh produces the same mesh
on every version.

.. code-block:: bash

   # Build the mesh and the bundle
   bash examples/fm_csem_unit_cube/scripts/build_bundles.sh

``configs/params_p1.txt`` runs it with BDDC. ``reference/`` holds the
direct-solve responses for orders 1 to 3 used as test goldens; they are
regenerated with ``scripts/regen_golden.sh``.


Marine CSEM model (``examples/fm_csem_marine``)
-----------------------------------------------
A thin resistive layer in conductive sediments under 1 km of seawater,
compared against a precomputed reference.

Model
*****
Four layers, each a Gmsh physical volume in ``geometry/mesh.geo``. Row ``i``
of ``survey/sigmas.txt`` is the volume tagged ``i + 1``.

.. list-table::
   :header-rows: 1

   * - Layer
     - Volume tag
     - Depth ``z`` (m)
     - Conductivity (S/m)
   * - Water
     - 4
     - ``0 … -1000``
     - 3.3333
   * - Sediments 1
     - 3
     - ``-1000 … -2000``
     - 1.0
   * - Oil (target)
     - 2
     - ``-2000 … -2100``
     - 0.01
   * - Sediments 2
     - 1
     - ``-2100 … -3500``
     - 1.0

Survey:

- Frequency: 2 Hz.
- Source: x-directed horizontal electric dipole at ``(1750, 1750, -975)`` m,
  25 m above the seafloor, unit current and length.
- Receivers: 58 points on the seafloor at ``z = -990`` m, ``y = 1750`` m.

Running
*******
.. code-block:: bash

   # Build the bundles
   bash examples/fm_csem_marine/scripts/build_bundles.sh 1

   # Run the simulation
   (cd examples/fm_csem_marine && mpirun -n 4 ../../build/fm.csem -options_file configs/params.txt)

   # Post-process the results
   python3 examples/fm_csem_marine/scripts/postprocess.py -tolerance 0.03


``build_bundles.sh <order>`` meshes ``geometry/mesh.geo`` into
``outputs/mesh.msh`` and builds ``outputs/input.h5``. The solver options are
in ``configs/params.txt``. ``postprocess.py`` compares :math:`|E_x|` with
``reference/reference.h5``, prints the NRMSD, relative L2 error and MAPE, and
fails when the NRMSD is above ``-tolerance``. It reads
``outputs/responses_p<order>.h5`` for the order stored in the bundle; use
``-responses_filename`` to compare another file. See :doc:`quickstart` for
running other orders.

.. figure:: /_static/images/fd_marine_p2.png
   :alt: Ex of fm.csem and of the reference along the receiver line
   :align: center
   :width: 95%

   :math:`|E_x|` computed by ``fm.csem`` at order 2 and the reference
   solution.

Extrae profiling
----------------
With ``make USE_EXTRAE=1`` the kernels are instrumented with
`Extrae <https://tools.bsc.es/extrae>`_ and produce traces for
`Paraver <https://tools.bsc.es/paraver>`_. The configuration files are in
``tests/extrae/``.

.. code-block:: bash

   # Build the instrumented binaries
   make USE_EXTRAE=1

   # Set up Extrae
   export LD_LIBRARY_PATH=${EXTRAE_HOME}/lib:$LD_LIBRARY_PATH
   export EXTRAE_CONFIG_FILE=$PWD/tests/extrae/extrae.xml

   # Run the simulation
   mpirun -n 4 build/fm.csem.extrae \
       -input_filename examples/fm_csem_unit_cube/outputs/input.h5 \
       -order 1 -output_dir . -output_filename extrae_smoke

   # Merge the traces into a Paraver file
   ${EXTRAE_HOME}/bin/mpi2prv -f TRACE.mpits -o petgem.prv

Extrae writes its ``TRACE.*`` files to the current directory. Open
``petgem.prv`` in Paraver with ``tests/extrae/petgem_functions.cfg``.

.. figure:: /_static/images/extrae_trace.png
   :alt: Extrae trace of a PETGEM execution
   :align: center
   :width: 95%

   Extrae trace of a run on 4 MPI tasks. Each color is an instrumented phase.
