===============
Mesh generation
===============

**PETGEM** uses unstructured tetrahedral meshes, usually generated with
`Gmsh <http://gmsh.info/>`_ (``.msh``). ``preprocess.py`` also reads VTK
meshes (``.vtk``/``.vtu``); the format is detected from the file. The same
mesh serves every polynomial order.

Physical groups and material ids
--------------------------------
Each volumetric region is tagged with a Gmsh **physical volume**. The tag maps
to a **0-based material id** as ``material_id = gmsh:physical - 1``, and that id
is the row index into ``sigmas.txt`` (see :doc:`formats`).

For example, ``examples/fm_csem_marine`` declares four regions with tags 1-4,
so ``sigmas.txt`` row 0 is the volume tagged 1, row 1 the volume tagged 2, and
so on. Number physical volumes consecutively from 1, and provide one
``sigmas.txt`` row per material.

For a **VTK** mesh there are no Gmsh physical tags; the per-cell region is read
from an integer **cell-data array**, auto-detected among ``cell_scalars``,
``materials_id``, ``material_id``, ``MaterialID``, ``material``, and
``CellEntityIds``. Its codes need not be contiguous or 0-based: the preprocess
maps the distinct codes to 0-based ids in **ascending order**, so ``sigmas.txt``
row ``i`` corresponds to the ``i``-th smallest code (codes ``{10, 20, 30, 40}``
map to rows ``{0, 1, 2, 3}``). The preprocess prints the resulting
``code -> row`` table with per-region cell counts, and stops if there are more
codes than ``sigmas.txt`` rows.

Element sizing
--------------
The shipped ``.geo`` files parameterize the characteristic length with named
constants applied to different parts of the geometry - for example, in
``examples/fm_csem_marine/geometry/mesh.geo``, a coarse far-field size, a fine size
along the source/receiver line, and a size inside the target layer. Adjust them
in the ``.geo`` file to control the mesh.

Generating a mesh
-----------------
.. code-block:: bash

   # Build the mesh
   gmsh -3 examples/fm_csem_marine/geometry/mesh.geo -o examples/fm_csem_marine/outputs/mesh.msh

Pass the ``.msh`` to ``utils/preprocess.py`` with ``-mesh_filename``.

Write the mesh in the MSH 2.2 format (``Mesh.MshFileVersion = 2.2;`` in the
``.geo``, as in the shipped examples). With MSH 4 and several physical volumes,
meshio returns one tetrahedral block per volume and the preprocess reads only
the first.

VTK input
---------
A tetrahedral **VTK** mesh (legacy ``.vtk`` or XML ``.vtu``) is passed to the
same ``-mesh_filename`` argument; no separate script or flag is needed:

.. code-block:: bash

   # Build the input bundle from a VTK mesh
   python3 utils/preprocess.py \
       -mode fm.csem -order 1 \
       -case_dir ${CASE_DIR} \
       -mesh_filename model.vtk \
       -source_filename sources.txt \
       -receiver_filename receivers.txt \
       -sigma_file sigmas.txt

Requirements:

- The file must contain a **tetrahedral** cell block; accompanying
  triangle/line blocks are ignored, and mixed-element meshes are not supported.
- The per-cell material id comes from a cell-data array, mapped to 0-based rows
  as described above. Provide one ``sigmas.txt`` row per distinct region code,
  ordered by ascending code.
