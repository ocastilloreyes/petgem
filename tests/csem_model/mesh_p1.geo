/*********************************************************************
*
* Mesh for 3D CSEM modelling using PETGEM.
*
* Parameters:
*    Frequency --> 2 Hz
*    Source position [x,y,z] --> [0.0, 0.0, -975.0]
*    Conductivity [Sediments, Oil, Sediments, Water] --> [1.0, 1.0/100.0, 1.0, 1.0/0.3] S/m
*
* For more details about this model, we refer to:
*
* Castillo-Reyes, O., de la Puente, Cela, J. M. (2018). PETGEM: A parallel code for 3D 
* CSEM forward modeling using edge finite elements. Computers & Geosciences, 119, 23–136. 
* https://doi.org/10.1016/j.cageo.2018.07.005
*
* Visit http://gmsh.info/ for details about mesh scripting with Gmsh
*
# Author: Octavio Castillo-Reyes (UPC/BSC) (octavio.castillo@upc.edu; octavio.castillo@bsc.es)
# Latest update: September 10th, 2025
*********************************************************************/

// #################################################################
// #                        Parameters                             #
// #################################################################
// Mesh dimensions
MIN_X = -4000.0;
MAX_X = 4000.0;
MIN_Y = -4000.0;
MAX_Y = 4000.0;
MIN_Z = -3000.0;
MAX_Z = 0.0;
DEPTH_MATERIAL1 = -1000.0;
DEPTH_MATERIAL2 = -2000.0;
DEPTH_MATERIAL3 = -2100.0;

// Mesh cell-size
dg = 400.;

// Transmitter position
SRC_X = 0.0;
SRC_Y = 0.0;
SRC_Z = -975.0;

// Depth of receivers
DEPTH_RECEIVERS = -990.0;

// #################################################################
// #                    Define main points                         #
// #################################################################
Point (1) = {MIN_X, MIN_Y, MIN_Z, dg};
Point (2) = {MIN_X, MAX_Y, MIN_Z, dg};
Point (3) = {MAX_X, MAX_Y, MIN_Z, dg};
Point (4) = {MAX_X, MIN_Y, MIN_Z, dg};
Point (5) = {MIN_X, MIN_Y, DEPTH_MATERIAL3, dg};
Point (6) = {MIN_X, MAX_Y, DEPTH_MATERIAL3, dg};
Point (7) = {MAX_X, MAX_Y, DEPTH_MATERIAL3, dg};
Point (8) = {MAX_X, MIN_Y, DEPTH_MATERIAL3, dg};
Point (9) = {MIN_X, MIN_Y, DEPTH_MATERIAL2, dg};
Point (10) = {MIN_X, MAX_Y, DEPTH_MATERIAL2, dg};
Point (11) = {MAX_X, MAX_Y, DEPTH_MATERIAL2, dg};
Point (12) = {MAX_X, MIN_Y, DEPTH_MATERIAL2, dg};
Point (13) = {MIN_X, MIN_Y, DEPTH_MATERIAL1, 300.};
Point (14) = {MIN_X, MAX_Y, DEPTH_MATERIAL1, 300.};
Point (15) = {MAX_X, MAX_Y, DEPTH_MATERIAL1, 300.};
Point (16) = {MAX_X, MIN_Y, DEPTH_MATERIAL1, 300.};
Point (17) = {MIN_X, MIN_Y, MAX_Z, dg};
Point (18) = {MIN_X, MAX_Y, MAX_Z, dg};
Point (19) = {MAX_X, MAX_Y, MAX_Z, dg};
Point (20) = {MAX_X, MIN_Y, MAX_Z, dg};
Point (21) = {-250., -200., DEPTH_MATERIAL1, 100.};
Point (22) = {1700., -200., DEPTH_MATERIAL1, 100.};
Point (23) = {-250., 200., DEPTH_MATERIAL1, 100.};
Point (24) = {1700., 200., DEPTH_MATERIAL1, 100.};

// #################################################################
// #                 Layer 1: Water (1.0/0.3 S/m)                  #
// #################################################################
Line(1) = {18, 19};
Line(2) = {19, 20};
Line(3) = {20, 17};
Line(4) = {17, 18};
Line(5) = {18, 14};
Line(6) = {17, 13};
Line(7) = {20, 16};
Line(8) = {16, 15};
Line(9) = {15, 19};
Line(11) = {14, 13};
Line(12) = {13, 16};
Line(13) = {15, 14};
Line(14) = {22, 24};
Line(15) = {23, 24};
Line(16) = {21, 23};
Line(17) = {21, 22};
Curve Loop(1) = {7, -12, -6, -3};
Plane Surface(1) = {1};
Curve Loop(2) = {8, 9, 2, 7};
Plane Surface(2) = {2};
Curve Loop(3) = {9, -1, 5, -13};
Plane Surface(3) = {3};
Curve Loop(4) = {5, 11, -6, 4};
Plane Surface(4) = {4};
Curve Loop(5) = {3, 4, 1, 2};
Plane Surface(5) = {5};
Curve Loop(6) = {12, 8, 13, 11};
Curve Loop(7) = {15, -14, -17, 16};
Plane Surface(6) = {6, 7};
Plane Surface(7) = {7};
Surface Loop(1) = {1, 2, 6, 3, 5, 4, 7};
Volume(1) = {1};
Physical Volume ("Water", 1) = {1};

// #################################################################
// #                Layer 2: Sediments (1.0 S/m)                   #
// #################################################################
Line(18) = {16, 12};
Line(19) = {15, 11};
Line(20) = {14, 10};
Line(21) = {13, 9};
Line(22) = {12, 9};
Line(23) = {9, 10};
Line(24) = {10, 11};
Line(25) = {11, 12};
Curve Loop(8) = {18, -25, -19, -8};
Plane Surface(8) = {8};
Curve Loop(9) = {19, -24, -20, -13};
Plane Surface(9) = {9};
Curve Loop(10) = {23, -20, 11, 21};
Plane Surface(10) = {10};
Curve Loop(11) = {21, -22, -18, -12};
Plane Surface(11) = {11};
Curve Loop(12) = {25, 22, 23, 24};
Plane Surface(12) = {12};
Surface Loop(2) = {8, 11, 10, 12, 9, 6, 7};
Volume(2) = {2};
Physical Volume ("Sediments1", 2) = {2};

// #################################################################
// #                 Layer 3: Oil (1.0/100.0 S/m)                  #
// #################################################################
Line(26) = {12, 8};
Line(27) = {11, 7};
Line(28) = {10, 6};
Line(29) = {9, 5};
Line(30) = {7, 6};
Line(31) = {6, 5};
Line(32) = {5, 8};
Line(33) = {8, 7};
Curve Loop(13) = {26, 33, -27, 25};
Plane Surface(13) = {13};
Curve Loop(14) = {27, 30, -28, 24};
Plane Surface(14) = {14};
Curve Loop(15) = {28, 31, -29, 23};
Plane Surface(15) = {15};
Curve Loop(16) = {29, 32, -26, 22};
Plane Surface(16) = {16};
Curve Loop(17) = {33, 30, 31, 32};
Plane Surface(17) = {17};
Surface Loop(3) = {13, 16, 15, 14, 17, 12};
Volume(3) = {3};
Physical Volume ("Oil", 3) = {3};

// #################################################################
// #                 Layer 4: Sediments (1.0 S/m)                  #
// #################################################################
Line(34) = {8, 4};
Line(35) = {7, 3};
Line(36) = {6, 2};
Line(37) = {3, 2};
Line(38) = {2, 1};
Line(39) = {1, 5};
Line(40) = {4, 1};
Line(41) = {4, 3};
Curve Loop(18) = {34, 41, -35, -33};
Plane Surface(18) = {18};
Curve Loop(19) = {35, 37, -36, -30};
Plane Surface(19) = {19};
Curve Loop(20) = {36, 38, 39, -31};
Plane Surface(20) = {20};
Curve Loop(21) = {39, 32, 34, 40};
Plane Surface(21) = {21};
Curve Loop(22) = {41, 37, 38, -40};
Plane Surface(22) = {22};
Surface Loop(4) = {18, 21, 20, 19, 22, 17};
Volume(4) = {4};
Physical Volume ("Sediments2", 4) = {4};

// #################################################################
// #                Mesh refinement                                #
// #################################################################
Point(100) = {0.0, 0.0, -1000.0, 5.};
Point(101) = {58.33, 0.0, -1000.0, 5.};
Point(102) = {116.66, 0.0, -1000.0, 5.};
Point(103) = {175.0, 0.0, -1000.0, 5.};
Point(104) = {233.33, 0.0, -1000.0, 5.};
Point(105) = {291.66, 0.0, -1000.0, 5.};
Point(106) = {350.0, 0.0, -1000.0, 5.};
Point(107) = {408.33, 0.0, -1000.0, 5.};
Point(108) = {466.66, 0.0, -1000.0, 5.};
Point(109) = {525.0, 0.0, -1000.0, 5.};
Point(110) = {583.33, 0.0, -1000.0, 5.};
Point(111) = {641.66, 0.0, -1000.0, 5.};
Point(112) = {700.0, 0.0, -1000.0, 5.};
Point(113) = {758.33, 0.0, -1000.0, 5.};
Point(114) = {816.66, 0.0, -1000.0, 5.};
Point(115) = {875.0, 0.0, -1000.0, 5.};
Point(116) = {933.33, 0.0, -1000.0, 5.};
Point(117) = {991.66, 0.0, -1000.0, 5.};
Point(118) = {1050.0, 0.0, -1000.0, 5.};
Point(119) = {1108.33, 0.0, -1000.0, 5.};
Point(120) = {1166.66, 0.0, -1000.0, 5.};
Point(121) = {1225.0, 0.0, -1000.0, 5.};
Point(122) = {1283.33, 0.0, -1000.0, 5.};
Point(123) = {1341.66, 0.0, -1000.0, 5.};
Point(124) = {1400.0, 0.0, -1000.0, 5.};
Point(125) = {1458.33, 0.0, -1000.0, 5.};
Point(126) = {1516.66, 0.0, -1000.0, 5.};
Point(127) = {1575.0, 0.0, -1000.0, 5.};
Point(128) = {1633.33, 0.0, -1000.0, 5.};

Point(129) = {0.0, 0.0, -975.0, 2.5};
Point(130) = {0.0, 0.0, -980.0, 2.5};
Point(131) = {0.0, 0.0, -985.0, 2.5};
Point(132) = {0.0, 0.0, -990.0, 2.5};
Point(133) = {0.0, 0.0, -995.0, 2.5};
Point(134) = {-5.0, 0.0, -975.0, 2.5};
Point(135) = {5.0, 0.0, -975.0, 2.5};

Point {100:128} In Surface {7};
Point {129:135} In Volume {1};

// Mesh file format supported by PETGEM
Mesh.MshFileVersion = 2.2;