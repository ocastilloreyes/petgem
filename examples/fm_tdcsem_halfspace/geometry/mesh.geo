/*********************************************************************
*
* fm_tdcsem_halfspace - Gmsh geometry for the PETGEM time-domain CSEM
* half-space example
*
* Half-space model of Cai et al. (2017), Section 5.1: 500 ohm-m earth
* under air, x-directed electric dipole at (-1000, 0, 0) m, inline
* receivers on the surface at offsets 1000 and 2000 m.
*
* Domain: cube [-L, L]^3 (50 x 50 x 50 km), surface at z = 0, z up.
*
* Physical volumes: 1 = earth, 2 = air.
*
* Parameters:
*   L     half-width of the domain (m)
*   hmin  element size around the source and the receivers (m)
*   hmax  element size far from the survey (m)
*
*********************************************************************/
L = 25000.0;
hmin = 50.0;
hmax = 2500.0;

// #################################################################
// #                    Main points                                #
// #################################################################
Point(1)  = {-L, -L, -L};
Point(2)  = { L, -L, -L};
Point(3)  = { L,  L, -L};
Point(4)  = {-L,  L, -L};
Point(5)  = {-L, -L, 0.0};
Point(6)  = { L, -L, 0.0};
Point(7)  = { L,  L, 0.0};
Point(8)  = {-L,  L, 0.0};
Point(9)  = {-L, -L, L};
Point(10) = { L, -L, L};
Point(11) = { L,  L, L};
Point(12) = {-L,  L, L};

// #################################################################
// #                    Earth (z < 0)                              #
// #################################################################
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 5};
Line(9) = {1, 5}; Line(10) = {2, 6}; Line(11) = {3, 7}; Line(12) = {4, 8};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Curve Loop(2) = {1, 10, -5, -9};
Plane Surface(2) = {2};
Curve Loop(3) = {2, 11, -6, -10};
Plane Surface(3) = {3};
Curve Loop(4) = {3, 12, -7, -11};
Plane Surface(4) = {4};
Curve Loop(5) = {4, 9, -8, -12};
Plane Surface(5) = {5};
Curve Loop(6) = {5, 6, 7, 8};
Plane Surface(6) = {6};
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};
Physical Volume("Earth", 1) = {1};

// #################################################################
// #                    Air (z > 0)                                #
// #################################################################
Line(13) = {9, 10}; Line(14) = {10, 11}; Line(15) = {11, 12}; Line(16) = {12, 9};
Line(17) = {5, 9}; Line(18) = {6, 10}; Line(19) = {7, 11}; Line(20) = {8, 12};
Curve Loop(7) = {5, 18, -13, -17};
Plane Surface(7) = {7};
Curve Loop(8) = {6, 19, -14, -18};
Plane Surface(8) = {8};
Curve Loop(9) = {7, 20, -15, -19};
Plane Surface(9) = {9};
Curve Loop(10) = {8, 17, -16, -20};
Plane Surface(10) = {10};
Curve Loop(11) = {13, 14, 15, 16};
Plane Surface(11) = {11};
Surface Loop(2) = {6, 7, 8, 9, 10, 11};
Volume(2) = {2};
Physical Volume("Air", 2) = {2};

// Refinement: slab around the source and the receivers
Field[1] = Box;
Field[1].VIn  = hmin;
Field[1].VOut = hmax;
Field[1].XMin = -1200.0; Field[1].XMax = 1200.0;
Field[1].YMin = -200.0;  Field[1].YMax = 200.0;
Field[1].ZMin = -200.0;  Field[1].ZMax = 100.0;
Field[1].Thickness = 10000.0;
Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;
