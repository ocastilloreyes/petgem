/*********************************************************************
*
* Time-domain CSEM half-space test for fm.tdcsem
*
* Box [-4000,4000]^2 x [-4000,4000] m: earth below z = 0 (physical 1)
* and air above (physical 2). Refined around the source at the origin
* and the inline receivers at x = 200..400 m.
*
*********************************************************************/
L = 4000.0;  // half-width (m)

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

Field[1] = Box;
Field[1].VIn  = 20.0;
Field[1].VOut = 800.0;
Field[1].XMin = -60.0; Field[1].XMax = 460.0;
Field[1].YMin = -60.0; Field[1].YMax = 60.0;
Field[1].ZMin = -60.0; Field[1].ZMax = 20.0;
Field[1].Thickness = 1500.0;
Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;
