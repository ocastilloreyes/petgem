/*********************************************************************
*
* Mesh for 3D CSEM modelling using PETGEM.
*
* Frequency: 2 Hz
* Target FEM order: nord=2 (use mesh_p1.geo as the source-of-truth geometry)
* Source position: (0.0, 0.0, 0.0)
* Conductivity:
*   - Resistive block: 0.1 S/m
*   - Half-space:      1.0 S/m
*
* Mesh strategy:
*   - Skin depth based refinement
*   - Local refinement at source
*   - Refinement around resistive block
*
* Author:
*    Octavio Castillo-Reyes (UPC/BSC)
*    octavio.castillo@upc.edu
*    octavio.castillo@bsc.es
*
* Latest update:
*    March 04th, 2026
*********************************************************************/

// ============ Characteristic mesh size ===================
lc_max = 350.;
lc_min = 7.;
lc_block = 75.;

// ================= Resistive block =======================
Point (1) = {400.,  100., -200., lc_block};
Point (2) = {600.,  100., -200., lc_block};
Point (3) = {400., -100., -200., lc_block};
Point (4) = {600., -100., -200., lc_block};
Point (5) = {400.,  100., -400., lc_block};
Point (6) = {600.,  100., -400., lc_block};
Point (7) = {400., -100., -400., lc_block};
Point (8) = {600., -100., -400., lc_block};

Line(1) = {1, 2};
Line(2) = {2, 4};
Line(3) = {4, 3};
Line(4) = {3, 1};
Line(5) = {3, 7};
Line(6) = {1, 5};
Line(7) = {2, 6};
Line(8) = {4, 8};
Line(9) = {8, 6};
Line(10) = {6, 5};
Line(11) = {5, 7};
Line(12) = {7, 8};

Curve Loop(1) = {4, 1, 2, 3};
Plane Surface(1) = {1};
Curve Loop(2) = {8, 9, -7, 2};
Plane Surface(2) = {2};
Curve Loop(3) = {10, -6, 1, 7};
Plane Surface(3) = {3};
Curve Loop(4) = {4, 6, 11, -5};
Plane Surface(4) = {4};
Curve Loop(5) = {12, 9, 10, 11};
Plane Surface(5) = {5};
Curve Loop(6) = {8, -12, -5, -3};
Plane Surface(6) = {6};

Surface Loop(1) = {6, 2, 5, 3, 4, 1};
Volume(1) = {1};
Physical Volume("Resistive_block", 1) = {1};


// ================= Half-space domain =====================
Point (9)  = {-2000.,  2000.,  2000., lc_max};
Point (10) = { 2000.,  2000.,  2000., lc_max};
Point (11) = {-2000., -2000.,  2000., lc_max};
Point (12) = { 2000., -2000.,  2000., lc_max};

Point (13) = {-2000.,  2000., -2000., lc_max};
Point (14) = { 2000.,  2000., -2000., lc_max};
Point (15) = {-2000., -2000., -2000., lc_max};
Point (16) = { 2000., -2000., -2000., lc_max};

Line(13) = {11, 9};
Line(14) = {10, 9};
Line(15) = {12, 10};
Line(16) = {12, 11};
Line(17) = {11, 15};
Line(18) = {9, 13};
Line(19) = {10, 14};
Line(20) = {12, 16};
Line(21) = {16, 15};
Line(22) = {15, 13};
Line(23) = {13, 14};
Line(24) = {14, 16};

Curve Loop(7) = {13, -14, -15, 16};
Plane Surface(7) = {7};
Curve Loop(8) = {15, 19, 24, -20};
Plane Surface(8) = {8};
Curve Loop(9) = {19, -23, -18, -14};
Plane Surface(9) = {9};
Curve Loop(10) = {13, 18, -22, -17};
Plane Surface(10) = {10};
Curve Loop(11) = {17, -21, -20, 16};
Plane Surface(11) = {11};
Curve Loop(12) = {21, 22, 23, 24};
Plane Surface(12) = {12};

Surface Loop(2) = {1,2,3,4,5,6,11,10,7,9,8,12};
Volume(2) = {2};
Physical Volume("Homogeneous_half_space", 2) = {2};

// ================= Mesh refinement =====================
Point(100) = {0.,   0.,  0.,   lc_min};
Point(101) = {200.0, 0.0, 0.0, lc_min};
Point(102) = {225.0, 0.0, 0.0, lc_min};
Point(103) = {250.0, 0.0, 0.0, lc_min};
Point(104) = {275.0, 0.0, 0.0, lc_min};
Point(105) = {300.0, 0.0, 0.0, lc_min};
Point(106) = {325.0, 0.0, 0.0, lc_min};
Point(107) = {350.0, 0.0, 0.0, lc_min};
Point(108) = {375.0, 0.0, 0.0, lc_min};
Point(109) = {400.0, 0.0, 0.0, lc_min};
Point(110) = {425.0, 0.0, 0.0, lc_min};
Point(111) = {450.0, 0.0, 0.0, lc_min};
Point(112) = {475.0, 0.0, 0.0, lc_min};
Point(113) = {500.0, 0.0, 0.0, lc_min};
Point(114) = {525.0, 0.0, 0.0, lc_min};
Point(115) = {550.0, 0.0, 0.0, lc_min};
Point(116) = {575.0, 0.0, 0.0, lc_min};
Point(117) = {600.0, 0.0, 0.0, lc_min};
Point(118) = {625.0, 0.0, 0.0, lc_min};
Point(119) = {650.0, 0.0, 0.0, lc_min};
Point(120) = {675.0, 0.0, 0.0, lc_min};
Point(121) = {700.0, 0.0, 0.0, lc_min};
Point(122) = {725.0, 0.0, 0.0, lc_min};
Point(123) = {750.0, 0.0, 0.0, lc_min};
Point(124) = {775.0, 0.0, 0.0, lc_min};
Point(125) = {800.0, 0.0, 0.0, lc_min};
Point(126) = {825.0, 0.0, 0.0, lc_min};
Point(127) = {850.0, 0.0, 0.0, lc_min};
Point(128) = {875.0, 0.0, 0.0, lc_min};
Point(129) = {900.0, 0.0, 0.0, lc_min};
Point(130) = {925.0, 0.0, 0.0, lc_min};
Point(131) = {950.0, 0.0, 0.0, lc_min};
Point(132) = {975.0, 0.0, 0.0, lc_min};
Point(133) = {1000.0, 0.0, 0.0, lc_min};

// Distance field from all points along the source line
Field[1] = Distance;
Field[1].NodesList = {
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 
    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133};

// Threshold refinement based on distance
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_min;     // finest near source
Field[2].LcMax = lc_max;     // coarse far away
Field[2].DistMin = 10;       // radius of finest mesh
Field[2].DistMax = 500;      // smooth transition to coarse

// ----------------- Resistive block refinement ------------
Field[3] = Box;
Field[3].VIn = lc_block;   // mesh size inside block
Field[3].VOut = lc_max;    // mesh size outside block
Field[3].XMin = 200; Field[3].XMax = 500;
Field[3].YMin = -200; Field[3].YMax = 200;
Field[3].ZMin = -240; Field[3].ZMax = -10;

// ----------------- Combine all refinements ----------------
Field[4] = Min;
Field[4].FieldsList = {2, 3};
Background Field = 4;

// ----------------- Optional high-order smoothing --------
Mesh.MeshSizeExtendFromBoundary = 2;
Mesh.MeshSizeFromCurvature = 1;
Mesh.MeshSizeFromPoints = 1;
Mesh.MshFileVersion = 2.2;
