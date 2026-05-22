===============
Mesh generation
===============

**PETGEM** operates on unstructured tetrahedral meshes generated with
`Gmsh <http://gmsh.info/>`_. This page covers the meshing conventions that
connect a ``.geo`` geometry to a **PETGEM** case: how physical groups map to
material ids, and how element sizing is chosen per polynomial order.

Physical groups and material ids
---------------------------------
Each volumetric region of the model is tagged with a Gmsh **physical volume**.
**PETGEM** maps each physical tag to a **0-based material id** as
``material_id = gmsh:physical - 1``, and that id is the row index into
``sigmas.csv`` (see :doc:`formats`).

For example, the canonical case declares two regions:

.. code-block::

   Physical Volume("Resistive_block", 1) = {1};        // -> material id 0
   Physical Volume("Homogeneous_half_space", 2) = {2}; // -> material id 1

so ``sigmas.csv`` row 0 is the resistive block and row 1 the half-space.
Number physical volumes consecutively from 1, and provide one ``sigmas.csv``
row per material.

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
up from ``sigmas.csv`` by material id) into the input bundle. A pre-generated
``.msh`` can be used directly when a ``.geo`` is not shipped (as in the inverse
case, which ships ``mesh_p1.msh``).
