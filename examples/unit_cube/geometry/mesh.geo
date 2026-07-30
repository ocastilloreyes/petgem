/*********************************************************************
*
* unit_cube - Gmsh geometry for the PETGEM FM-CSEM test dataset
*
* Homogeneous unit cube [0,1]^3 with a single material, driven through the
* same forward workflow as examples/fm (gmsh -> preprocess ->
* fm.csem). It is the small, fast, canonical fixture the whole FM-CSEM test
* suite is built around (see README.md and ../../tests/README.md).
*
* Acquisition (provenance for the generated input bundle):
*    Conductivity  --> isotropic sigma = 1.0 S/m           (sigmas.txt)
*    Source        --> one electric dipole at the centre
*                      (0.5, 0.5, 0.5), 2 Hz, unit I/L      (sources.txt)
*    Receivers     --> three points at y = z = 0.25,
*                      x in {0.25, 0.5, 0.75}               (receivers.txt)
*
* Structured transfinite mesh: unlike an unstructured Delaunay fill, a
* transfinite lattice is DETERMINISTIC (identical node layout and cell
* connectivity across gmsh versions), so CI regenerates a bit-stable bundle.
* N divisions per edge -> (N+1)^3 vertices and 6*N^3 tetrahedra; N = 6 keeps
* the mesh tiny (343 vertices, 1296 tets) yet still activates every DOF entity
* class (edge / face / interior) at orders 1-3.
*
* by the PETGEM team
*********************************************************************/
// #################################################################
// #                        Parameters                             #
// #################################################################
// Divisions per cube edge (structured lattice).
N = 6;

// #################################################################
// #                    Define main points                         #
// #################################################################
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {1, 0, 1};
Point(7) = {1, 1, 1};
Point(8) = {0, 1, 1};

// #################################################################
// #                        Define edges                           #
// #################################################################
Line(1)  = {1, 2};   // bottom face (z = 0)
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};
Line(5)  = {5, 6};   // top face (z = 1)
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};
Line(9)  = {1, 5};   // vertical edges
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// #################################################################
// #                   Define bounding surfaces                    #
// #################################################################
Line Loop(1) = {1, 2, 3, 4};        Plane Surface(1) = {1};   // z = 0
Line Loop(2) = {5, 6, 7, 8};        Plane Surface(2) = {2};   // z = 1
Line Loop(3) = {1, 10, -5, -9};     Plane Surface(3) = {3};   // y = 0
Line Loop(4) = {2, 11, -6, -10};    Plane Surface(4) = {4};   // x = 1
Line Loop(5) = {3, 12, -7, -11};    Plane Surface(5) = {5};   // y = 1
Line Loop(6) = {4, 9, -8, -12};     Plane Surface(6) = {6};   // x = 0

// #################################################################
// #                        Define volume                          #
// #################################################################
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// #################################################################
// #                Structured (transfinite) meshing               #
// #################################################################
Transfinite Line "*"    = N + 1;
Transfinite Surface "*";
Transfinite Volume  "*";

// #################################################################
// #             Single material (gmsh:physical tag 1)             #
// #################################################################
// Row 0 of sigmas.txt (= physical tag 1 - 1) -> sigma = 1.0 S/m.
Physical Volume("cube", 1) = {1};

// Mesh format supported by PETGEM (read via meshio in utils/preprocess.py).
Mesh.MshFileVersion = 2.2;
