/*********************************************************************
*
* tests/mms - Gmsh geometry for the PETGEM MMS order-of-accuracy verification
* of the fm.csem high-order Nedelec discretization.
*
* Homogeneous UNIT CUBE [0,1]^3, single material, sigma = 1 S/m. This is the
* SAME domain and workflow as examples/unit_cube (gmsh -> preprocess ->
* fm.csem); only the refinement knob N changes between runs.
*
* Why [0,1]^3 and not the centred cube: the manufactured field used to verify
* the discretization is
*       E*(x,y,z) = ( sin(pi y) sin(pi z),
*                     sin(pi z) sin(pi x),
*                     sin(pi x) sin(pi y) ),
* whose tangential trace n x E* vanishes on the faces of [0,1]^3 (each
* off-normal component carries a sin(pi{x|y|z}) factor that is 0 at 0 and 1).
* That is exactly the homogeneous Dirichlet condition fm.csem already imposes
* via DMPlexMarkBoundaryFaces, so NO nonzero-BC code path is needed. On the
* centred cube [-0.5,0.5]^3 this trace is NOT zero, which would corrupt the
* test -- do not move the domain without changing E* (see mms_reference.py).
*
* CONVERGENCE SEQUENCE: regenerate this mesh for several N to get a family of
* refined meshes with EXACT, KNOWN size h = 1/N, e.g.
*       gmsh -3 -setnumber N 2  -o mesh_N2.msh  mesh.geo
*       gmsh -3 -setnumber N 4  -o mesh_N4.msh  mesh.geo
*       gmsh -3 -setnumber N 8  -o mesh_N8.msh  mesh.geo
*       gmsh -3 -setnumber N 16 -o mesh_N16.msh mesh.geo
* (add N = 32 for p = 1,2 to reach the asymptotic regime; N = 2,3,4 already
*  suffice for p = 5,6, which hit the round-off floor early.)
*
* Structured transfinite meshing keeps the node layout DETERMINISTIC and the
* tetrahedra uniform, so the measured slopes are clean and reproducible.
* N divisions per edge -> (N+1)^3 vertices and 6*N^3 tetrahedra.
*
* by the PETGEM team (http://petgem.bsc.es/)
*********************************************************************/
// #################################################################
// #                        Parameters                             #
// #################################################################
// Divisions per cube edge (structured lattice). Override on the command
// line with:  gmsh -3 -setnumber N <value> -o mesh_N<value>.msh mesh.geo
If(!Exists(N))
  N = 8;
EndIf

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
