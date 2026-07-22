===============
Mesh generation
===============

**PETGEM** operates on unstructured tetrahedral meshes. The usual workflow
generates them with `Gmsh <http://gmsh.info/>`_ (``.msh``), but the preprocess
stage also accepts **VTK** meshes (``.vtk``/``.vtu``); the format is
auto-detected from the file passed to ``-mesh_filename``. This page covers how
mesh regions map to material ids, and how a mesh is generated.

Physical groups and material ids
--------------------------------
Each volumetric region is tagged with a Gmsh **physical volume**. The tag maps
to a **0-based material id** as ``material_id = gmsh:physical - 1``, and that id
is the row index into ``sigmas.txt`` (see :doc:`formats`).

For example, ``examples/fm_model`` declares four regions with tags 1-4,
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
``code -> row`` table with per-region cell counts, and errors if the number of
codes does not match the ``sigmas.txt`` row count.

Element sizing
--------------
The shipped ``.geo`` files parameterize the characteristic length with named
constants applied to different parts of the geometry - for example, in
``examples/fm_model/geometry/mesh.geo``, a coarse far-field size, a fine size
along the source/receiver line, and a size inside the target layer. Adjust them
in the ``.geo`` file to control the mesh.

Generating a mesh
-----------------
.. code-block:: bash

   gmsh -3 examples/fm_model/geometry/mesh.geo -o examples/fm_model/outputs/mesh.msh

The resulting ``.msh`` is passed to ``utils/preprocess.py`` via
``-mesh_filename``, which embeds the mesh and the per-cell conductivity (looked
up from ``sigmas.txt`` by material id) into the input bundle. A pre-generated
``.msh`` can be used directly.

One mesh, any order
-------------------
The polynomial order is a property of the basis, not of the mesh: the same
tetrahedral mesh serves every order in ``1..6``. Each example therefore ships a
single ``mesh.geo``. The order is chosen at preprocessing time with ``-order``,
and can be overridden at run time with the kernel's ``-order`` flag - which
bypasses the bundle's stored value, so one bundle can be reused for all orders.

VTK input
---------
A tetrahedral **VTK** mesh (legacy ``.vtk`` or XML ``.vtu``) is passed to the
same ``-mesh_filename`` argument; no separate script or flag is needed:

.. code-block:: bash

   python3 utils/preprocess.py \
       -mode fm -order 1 \
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
- Reading is done with `meshio <https://github.com/nschloe/meshio>`_, already a
  preprocess dependency.
