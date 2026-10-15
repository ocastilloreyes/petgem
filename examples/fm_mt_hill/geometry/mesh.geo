/*********************************************************************
*
* fm_mt_hill - Gmsh geometry for the PETGEM 3D trapezoidal hill MT example
*
* Trapezoidal hill of Nam et al. (2007), Castillo-Reyes et al. (2022),
* Section 3.1: 100 ohm-m half-space with a 450 m high hill (top square
* 450 x 450 m, base square 2000 x 2000 m) under air.
*
* Domain: cube [-L, L]^3 with L = 2000 + nskin * delta, delta = 3500 m
* (skin depth at 2 Hz in 100 ohm-m), surface at z = 0, z up.
*
* Physical volumes: 1 = air, 2 = earth.
*
* Parameters (override with gmsh -setnumber):
*   nskin  number of skin depths between the survey and the boundaries
*   hmin   element size on the hill and along the survey line (m)
*   hmax   element size far from the hill (m)
*
*********************************************************************/
DefineConstant[ nskin = {1, Name "nskin"} ];
DefineConstant[ hmin = {80.0, Name "hmin"} ];
DefineConstant[ hmax = {1750.0, Name "hmax"} ];

delta = 3500.0;
L     = 2000.0 + nskin * delta;

// #################################################################
// #                    Hill (frustum)                             #
// #################################################################
// Base square at z = 0, top square at z = 450
Point(1) = {-1000, -1000, 0};
Point(2) = { 1000, -1000, 0};
Point(3) = { 1000,  1000, 0};
Point(4) = {-1000,  1000, 0};
Point(5) = { -225,  -225, 450};
Point(6) = {  225,  -225, 450};
Point(7) = {  225,   225, 450};
Point(8) = { -225,   225, 450};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Line(5) = {5, 6}; Line(6) = {6, 7}; Line(7) = {7, 8}; Line(8) = {8, 5};
Line(9) = {1, 5}; Line(10) = {2, 6}; Line(11) = {3, 7}; Line(12) = {4, 8};
// Base (hole in the ground surface)
Curve Loop(1) = {1, 2, 3, 4};
// Top
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
// Flanks
Curve Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
Curve Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
Curve Loop(5) = {3, 12, -7, -11};
Plane Surface(5) = {5};
Curve Loop(6) = {4, 9, -8, -12};
Plane Surface(6) = {6};

// #################################################################
// #                    Domain box                                 #
// #################################################################
Point(11) = {-L, -L, -L};
Point(12) = { L, -L, -L};
Point(13) = { L,  L, -L};
Point(14) = {-L,  L, -L};
Point(21) = {-L, -L, 0};
Point(22) = { L, -L, 0};
Point(23) = { L,  L, 0};
Point(24) = {-L,  L, 0};
Point(31) = {-L, -L, L};
Point(32) = { L, -L, L};
Point(33) = { L,  L, L};
Point(34) = {-L,  L, L};
Line(21) = {11, 12}; Line(22) = {12, 13}; Line(23) = {13, 14}; Line(24) = {14, 11};
Line(25) = {21, 22}; Line(26) = {22, 23}; Line(27) = {23, 24}; Line(28) = {24, 21};
Line(29) = {31, 32}; Line(30) = {32, 33}; Line(31) = {33, 34}; Line(32) = {34, 31};
Line(33) = {11, 21}; Line(34) = {12, 22}; Line(35) = {13, 23}; Line(36) = {14, 24};
Line(37) = {21, 31}; Line(38) = {22, 32}; Line(39) = {23, 33}; Line(40) = {24, 34};
// Bottom
Curve Loop(11) = {21, 22, 23, 24};
Plane Surface(11) = {11};
// Earth sides
Curve Loop(12) = {21, 34, -25, -33};
Plane Surface(12) = {12};
Curve Loop(13) = {22, 35, -26, -34};
Plane Surface(13) = {13};
Curve Loop(14) = {23, 36, -27, -35};
Plane Surface(14) = {14};
Curve Loop(15) = {24, 33, -28, -36};
Plane Surface(15) = {15};
// Ground surface (z = 0) with the hill base removed
Curve Loop(20) = {25, 26, 27, 28};
Plane Surface(20) = {20, 1};
// Air sides
Curve Loop(21) = {25, 38, -29, -37};
Plane Surface(21) = {21};
Curve Loop(22) = {26, 39, -30, -38};
Plane Surface(22) = {22};
Curve Loop(23) = {27, 40, -31, -39};
Plane Surface(23) = {23};
Curve Loop(24) = {28, 37, -32, -40};
Plane Surface(24) = {24};
// Top
Curve Loop(25) = {29, 30, 31, 32};
Plane Surface(25) = {25};

// #################################################################
// #                    Volumes                                    #
// #################################################################
Surface Loop(1) = {25, 21, 22, 23, 24, 20, 2, 3, 4, 5, 6};
Volume(1) = {1};
Physical Volume("Air", 1) = {1};
Surface Loop(2) = {11, 12, 13, 14, 15, 20, 2, 3, 4, 5, 6};
Volume(2) = {2};
Physical Volume("Earth", 2) = {2};

// Refinement: hill surfaces and a slab along the survey line (y = 0, |x| <= 2 km)
Field[1] = Distance;
Field[1].SurfacesList = {2, 3, 4, 5, 6};
Field[2] = Threshold;
Field[2].InField  = 1;
Field[2].SizeMin  = hmin;
Field[2].SizeMax  = hmax;
Field[2].DistMin  = 0;
Field[2].DistMax  = delta;
Field[3] = Box;
Field[3].VIn  = hmin;
Field[3].VOut = hmax;
Field[3].XMin = -2200; Field[3].XMax = 2200;
Field[3].YMin = -300;  Field[3].YMax = 300;
Field[3].ZMin = -300;  Field[3].ZMax = 300;
Field[3].Thickness = delta;
Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;
