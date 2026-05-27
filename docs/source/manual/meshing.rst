===============
Mesh generation
===============

**PETGEM** operates on unstructured tetrahedral meshes. The canonical workflow
generates them with `Gmsh <http://gmsh.info/>`_ (``.msh``), but the preprocess
stage also accepts **VTK** meshes (``.vtk``/``.vtu``); the format is
auto-detected from the file. This page covers the meshing conventions that
connect a geometry to a **PETGEM** case: how regions map to material ids, and
how element sizing is chosen per polynomial order.

Physical groups and material ids
---------------------------------
Each volumetric region of the model is tagged with a Gmsh **physical volume**.
**PETGEM** maps each physical tag to a **0-based material id** as
``material_id = gmsh:physical - 1``, and that id is the row index into
``sigmas.txt`` (see :doc:`formats`).

For example, the canonical case declares two regions:

.. code-block::

   Physical Volume("Resistive_block", 1) = {1};        // -> material id 0
   Physical Volume("Homogeneous_half_space", 2) = {2}; // -> material id 1

so ``sigmas.txt`` row 0 is the resistive block and row 1 the half-space.
Number physical volumes consecutively from 1, and provide one ``sigmas.txt``
row per material.

For a **VTK** mesh there are no Gmsh physical tags; the per-cell region is read
from an integer **cell-data array** (auto-detected, commonly ``cell_scalars`` -
also ``materials_id``, ``material_id``, ``MaterialID``, ``material`` or
``CellEntityIds``). Its distinct codes need not be contiguous or 0-based: the
preprocess stage maps them to 0-based ids in **ascending order**, so
``sigmas.txt`` row ``i`` corresponds to the ``i``-th smallest code. For example
codes ``{10, 20, 30, 40}`` map to rows ``{0, 1, 2, 3}``. The preprocess run
prints the resulting ``code -> row`` table (with per-region cell counts), and
errors if the number of codes does not match the ``sigmas.txt`` row count.

Element sizing
--------------
The shipped ``.geo`` files parameterize the characteristic length with three
constants, applied to different parts of the geometry:

- ``lc_max``: far-field / domain-boundary element size (coarsest).
- ``lc_min``: element size along the source/receiver line (finest).
- ``lc_block``: element size inside the resistive target.

.. code-block::

   lc_max   = 250.;
   lc_min   = 5.;
   lc_block = 50.;

Resolving the conductivity contrast (the target block) is what controls
accuracy; ``lc_block`` is therefore the most important knob. Refining the
source line (``lc_min``) or the far field (``lc_max``) beyond what is needed
adds elements without improving the misfit.

Per-order meshes
----------------
Because accuracy scales with both element size and polynomial order, the mesh
can be coarsened as the order rises (see the h-p tradeoff in :doc:`solver`).
The canonical case ships a separate ``mesh_p${NORD}.geo`` for each order
``1..6``, progressively coarser at higher order, so every order runs at a
comparable accuracy without over-refining.

Generating a mesh
-----------------
Generate the 3D mesh from a ``.geo`` file with Gmsh:

.. code-block:: bash

   gmsh ${CSEM_TEST_DIR}/mesh_p${NORD}.geo -3 -o ${CSEM_TEST_DIR}/mesh_p${NORD}.msh

The resulting ``.msh`` is passed to ``utils/preprocess.py`` via
``-mesh_filename``, which embeds the mesh and the per-cell conductivity (looked
up from ``sigmas.txt`` by material id) into the input bundle. A pre-generated
``.msh`` can be used directly when a ``.geo`` is not shipped (as in the inverse
case, which ships ``mesh_p1.msh``).

VTK input
---------
A tetrahedral **VTK** mesh (legacy ``.vtk`` or XML ``.vtu``) can be passed to
the same ``-mesh_filename`` argument - no separate script or flag is needed, and
the rest of the command is unchanged:

.. code-block:: bash

   python3 utils/preprocess.py \
       -mode forward -nord 1 \
       -case_dir ${CASE_DIR} \
       -mesh_filename model.vtk \
       -receiver_filename receivers.txt \
       -source_filename sources.txt \
       -sigma_file sigmas.txt

Requirements and behaviour:

- The file must contain a **tetrahedral** cell block (any accompanying
  triangle/line blocks are ignored). Mixed-element meshes are not supported.
- The per-cell material id comes from a cell-data array, mapped to 0-based rows
  as described under *Physical groups and material ids* above. Provide one
  ``sigmas.txt`` row per distinct region code, ordered by ascending code.
- Reading is handled by `meshio <https://github.com/nschloe/meshio>`_ (already a
  preprocess dependency), so any tetrahedral format meshio supports works in
  principle; ``.vtk``/``.vtu`` are the tested paths.
