/*********************************************************************
*
* Mesh for 3D CSEM modelling using PETGEM.
*
* Parameters:
*    Frequency --> 2 Hz
*    Source position [x,y,z] --> [1750.0, 1750.0, -975.0]
*    Conductivity [Sediments, Oil, Sediments, Water] --> [1.0, 1.0/100.0, 1.0, 1.0/0.3] S/m
*    Skin depth --> [355.6747, 3556.7143, 355.6747, 194.8111] m
*    Average skin detph --> 1115.7151 m
*    Points per skin depth (rg) --> 1.0
*    Global element size (dg) --> 355.6747/rg m
*    Source element size (ds) --> dg/17.5 m
*    x-dimensions --> [-1000.0, 4500.0] m
*    y-dimensions --> [0.0, 3500.0] m
*    z-dimensions --> [0, -3500.0] m
*
* The reference for this model is that described by:
*
*         Constable, S., Weiss, C.J., 2006. Mapping thin resistors
*         and hydrocarbons with marine em methods: Insights from 1d
*         modeling. Geophysics 71, G43–G51.
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
MIN_X = -1000.0;
MAX_X = 4500.0;
MIN_Y = 0.0;
MAX_Y = 3500.0;
MIN_Z = -3500.0;
MAX_Z = 0.0;
DEPTH_MATERIAL1 = -1000.0;
DEPTH_MATERIAL2 = -2000.0;
DEPTH_MATERIAL3 = -2100.0;

// Mesh cell-size
rg = 1.0;
rs = 8.5;
dg = 350./rg;
ds = dg/rs;

// Transmitter position
SRC_X = 1750.0;
SRC_Y = 1750.0;
SRC_Z = -975.0;

// Depth of receivers
DEPTH_RECEIVERS = -990.0;

// #################################################################
// #                    Define main points                         #
// #################################################################
Point (1) = {MIN_X, MIN_Y, MIN_Z};
Point (2) = {MIN_X, MAX_Y, MIN_Z};
Point (3) = {MAX_X, MAX_Y, MIN_Z};
Point (4) = {MAX_X, MIN_Y, MIN_Z};
Point (5) = {MIN_X, MIN_Y, DEPTH_MATERIAL3};
Point (6) = {MIN_X, MAX_Y, DEPTH_MATERIAL3};
Point (7) = {MAX_X, MAX_Y, DEPTH_MATERIAL3};
Point (8) = {MAX_X, MIN_Y, DEPTH_MATERIAL3};
Point (9) = {MIN_X, MIN_Y, DEPTH_MATERIAL2};
Point (10) = {MIN_X, MAX_Y, DEPTH_MATERIAL2};
Point (11) = {MAX_X, MAX_Y, DEPTH_MATERIAL2};
Point (12) = {MAX_X, MIN_Y, DEPTH_MATERIAL2};
Point (13) = {MIN_X, MIN_Y, DEPTH_MATERIAL1};
Point (14) = {MIN_X, MAX_Y, DEPTH_MATERIAL1};
Point (15) = {MAX_X, MAX_Y, DEPTH_MATERIAL1};
Point (16) = {MAX_X, MIN_Y, DEPTH_MATERIAL1};
Point (17) = {MIN_X, MIN_Y, MAX_Z};
Point (18) = {MIN_X, MAX_Y, MAX_Z};
Point (19) = {MAX_X, MAX_Y, MAX_Z};
Point (20) = {MAX_X, MIN_Y, MAX_Z};

// #################################################################
// #                Layer 1: Sediments (1.0 S/m)                   #
// #################################################################
Line (1) = {1, 2};
Line (2) = {2, 3};
Line (3) = {3, 4};
Line (4) = {4, 1};
Line (5) = {1, 5};
Line (6) = {2, 6};
Line (7) = {3, 7};
Line (8) = {4, 8};
Line (9) = {5, 6};
Line (10) = {6, 7};
Line (11) = {7, 8};
Line (12) = {8, 5};
Line Loop (1) = {1, 2, 3, 4};
Plane Surface (1) = {1};
Line Loop (2) = {1, 6, -9, -5};
Plane Surface (2) = {2};
Line Loop (3) = {2, 7, -10, -6};
Plane Surface (3) = {3};
Line Loop (4) = {-3, 7, 11, -8};
Plane Surface (4) = {4};
Line Loop (5) = {4, 5, -12, -8};
Plane Surface (5) = {5};
Line Loop (6) = {9, 10, 11, 12};
Plane Surface (6) = {6};
// Define Volume
Surface Loop (1) = {1, 2, 3, 4, 5, 6};
Volume (1) = {1};
Physical Volume ("Sediments2", 1) = {1};

// #################################################################
// #                 Layer 2: Oil (1.0/100.0 S/m)                  #
// #################################################################
Line (13) = {5, 9};
Line (14) = {6, 10};
Line (15) = {7, 11};
Line (16) = {8, 12};
Line (17) = {9, 10};
Line (18) = {10, 11};
Line (19) = {11, 12};
Line (20) = {12, 9};
Line Loop (7) = {9, 14, -17, -13};
Plane Surface (7) = {7};
Line Loop (8) = {10, 15, -18, -14};
Plane Surface (8) = {8};
Line Loop (9) = {-11, 15, 19, -16};
Plane Surface (9) = {9};
Line Loop (10) = {12, 13, -20, -16};
Plane Surface (10) = {10};
Line Loop (11) = {17, 18, 19, 20};
Plane Surface (11) = {11};
// Define Volume
Surface Loop (2) = {6,7,8,9,10,11};
Volume (2) = {2};
Physical Volume ("Oil", 2) = {2};

// #################################################################
// #                   Layer 3: Sediments (1.0 S/m)                #
// #################################################################
Line (21) = {9, 13};
Line (22) = {10, 14};
Line (23) = {11, 15};
Line (24) = {12, 16};
Line (25) = {13, 14};
Line (26) = {14, 15};
Line (27) = {15, 16};
Line (28) = {16, 13};
Line Loop (12) = {17, 22, -25, -21};
Plane Surface (12) = {12};
Line Loop (13) = {18, 23, -26, -22};
Plane Surface (13) = {13};
Line Loop (14) = {-19, 23, 27, -24};
Plane Surface (14) = {14};
Line Loop (15) = {20, 21, -28, -24};
Plane Surface (15) = {15};
Line Loop (16) = {25, 26, 27, 28};
Plane Surface (16) = {16};
// Define Volume
Surface Loop (3) = {11, 12, 13, 14, 15, 16};
Volume (3) = {3};
Physical Volume ("Sediments1", 3) = {3};

// #################################################################
// #                 Layer 4: Water (1.0/0.3 S/m)                  #
// #################################################################
Line (29) = {13, 17};
Line (30) = {14, 18};
Line (31) = {15, 19};
Line (32) = {16, 20};
Line (34) = {17, 18};
Line (35) = {18, 19};
Line (36) = {19, 20};
Line (37) = {20, 17};
Line Loop (17) = {25, 30, -34, -29};
Plane Surface (17) = {17};
Line Loop (18) = {26, 31, -35, -30};
Plane Surface (18) = {18};
Line Loop (19) = {-27, 31, 36, -32};
Plane Surface (19) = {19};
Line Loop (20) = {28, 29, -37, -32};
Plane Surface (20) = {20};
Line Loop (21) = {34, 35, 36, 37};
Plane Surface (21) = {21};
// Define Volume
Surface Loop (4) = {16,17,18,19,20,21};
Volume (4) = {4};
Physical Volume ("Water", 4) = {4};

// #################################################################
// #                Mesh refinement                                #
// #################################################################
// Source position
Point (101) = {SRC_X, SRC_Y, SRC_Z};

// Receivers (Inline model)
Point (102) = {58.3333,    1750.0, DEPTH_RECEIVERS};
Point (103) = {116.6667,   1750.0, DEPTH_RECEIVERS};
Point (104) = {175.0,      1750.0, DEPTH_RECEIVERS};
Point (105) = {233.3333,   1750.0, DEPTH_RECEIVERS};
Point (106) = {291.6667,   1750.0, DEPTH_RECEIVERS};
Point (107) = {350.0000,   1750.0, DEPTH_RECEIVERS};
Point (108) = {408.3333,   1750.0, DEPTH_RECEIVERS};
Point (109) = {466.6667,   1750.0, DEPTH_RECEIVERS};
Point (110) = {525.0000,   1750.0, DEPTH_RECEIVERS};
Point (111) = {583.3333,   1750.0, DEPTH_RECEIVERS};
Point (112) = {641.6667,   1750.0, DEPTH_RECEIVERS};
Point (113) = {700.0,      1750.0, DEPTH_RECEIVERS};
Point (114) = {758.3333,   1750.0, DEPTH_RECEIVERS};
Point (115) = {816.6667,   1750.0, DEPTH_RECEIVERS};
Point (116) = {875.0,      1750.0, DEPTH_RECEIVERS};
Point (117) = {933.3333,   1750.0, DEPTH_RECEIVERS};
Point (118) = {991.6667,   1750.0, DEPTH_RECEIVERS};
Point (119) = {1050.0,     1750.0, DEPTH_RECEIVERS};
Point (120) = {1108.3333,  1750.0, DEPTH_RECEIVERS};
Point (121) = {1166.6667,  1750.0, DEPTH_RECEIVERS};
Point (122) = {1225.0,     1750.0, DEPTH_RECEIVERS};
Point (123) = {1283.3333,  1750.0, DEPTH_RECEIVERS};
Point (124) = {1341.6667,  1750.0, DEPTH_RECEIVERS};
Point (125) = {1400.0,     1750.0, DEPTH_RECEIVERS};
Point (126) = {1458.3333,  1750.0, DEPTH_RECEIVERS};
Point (127) = {1516.6667,  1750.0, DEPTH_RECEIVERS};
Point (128) = {1575.0,     1750.0, DEPTH_RECEIVERS};
Point (129) = {1633.3333,  1750.0, DEPTH_RECEIVERS};
Point (130) = {1691.6667,  1750.0, DEPTH_RECEIVERS};
Point (131) = {1750.0,     1750.0, DEPTH_RECEIVERS};
Point (132) = {1808.3333,  1750.0, DEPTH_RECEIVERS};
Point (133) = {1866.6667,  1750.0, DEPTH_RECEIVERS};
Point (134) = {1925.0,     1750.0, DEPTH_RECEIVERS};
Point (135) = {1983.3333,  1750.0, DEPTH_RECEIVERS};
Point (136) = {2041.6667,  1750.0, DEPTH_RECEIVERS};
Point (137) = {2100.0,     1750.0, DEPTH_RECEIVERS};
Point (138) = {2158.3333,  1750.0, DEPTH_RECEIVERS};
Point (139) = {2216.6667,  1750.0, DEPTH_RECEIVERS};
Point (140) = {2275.0,     1750.0, DEPTH_RECEIVERS};
Point (141) = {2333.3333,  1750.0, DEPTH_RECEIVERS};
Point (142) = {2391.6667,  1750.0, DEPTH_RECEIVERS};
Point (143) = {2450.0,     1750.0, DEPTH_RECEIVERS};
Point (144) = {2508.3333,  1750.0, DEPTH_RECEIVERS};
Point (145) = {2566.6667,  1750.0, DEPTH_RECEIVERS};
Point (146) = {2625.0,     1750.0, DEPTH_RECEIVERS};
Point (147) = {2683.3333,  1750.0, DEPTH_RECEIVERS};
Point (148) = {2741.6667,  1750.0, DEPTH_RECEIVERS};
Point (149) = {2800.0000,  1750.0, DEPTH_RECEIVERS};
Point (150) = {2858.3333,  1750.0, DEPTH_RECEIVERS};
Point (151) = {2916.6667,  1750.0, DEPTH_RECEIVERS};
Point (152) = {2975.0,     1750.0, DEPTH_RECEIVERS};
Point (153) = {3033.3333,  1750.0, DEPTH_RECEIVERS};
Point (154) = {3091.6667,  1750.0, DEPTH_RECEIVERS};
Point (155) = {3150.0,     1750.0, DEPTH_RECEIVERS};
Point (156) = {3208.3333,  1750.0, DEPTH_RECEIVERS};
Point (157) = {3266.6667,  1750.0, DEPTH_RECEIVERS};
Point (158) = {3325.0,     1750.0, DEPTH_RECEIVERS};
Point (159) = {3383.3333,  1750.0, DEPTH_RECEIVERS};

// Refinement
Field[1] = Attractor;
Field[1].NodesList = {101:159};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = ds;
Field[2].LcMax = dg;
Field[2].DistMin = 250.;
Field[2].DistMax = 500.;
Background Field = 2;

// #################################################################
// #                Mesh parameters                                #
// #################################################################
Characteristic Length {1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} = dg;

// Mesh file format supported by PETGEM
Mesh.MshFileVersion = 2.2;
