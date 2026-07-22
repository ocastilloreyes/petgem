/*********************************************************************
*
* fm_model - Gmsh geometry for the PETGEM marine CSEM example
*
* Canonical 3D controlled-source electromagnetic (CSEM) benchmark: a
* thin, resistive hydrocarbon layer buried in conductive marine
* sediments beneath a seawater column - the classic setup showing that
* marine CSEM can detect a thin resistor. The four layered materials are
* meshed as separate physical volumes; an x-directed horizontal electric
* dipole (HED) is towed just above the seafloor and the inline Ex
* response is recorded along a seafloor receiver profile.
*
* Layers (top -> bottom), physical volume tag and conductivity:
*    Water      (tag 4)  z in [    0, -1000] m --> 1.0/0.3   = 3.3333 S/m
*    Sediments1 (tag 3)  z in [-1000, -2000] m --> 1.0       S/m
*    Oil        (tag 2)  z in [-2000, -2100] m --> 1.0/100.0 = 0.01   S/m  (100 m thin resistor)
*    Sediments2 (tag 1)  z in [-2100, -3500] m --> 1.0       S/m
*
* Acquisition / meshing parameters:
*    Frequency                  --> 2 Hz
*    Source position [x,y,z]    --> [1750.0, 1750.0, -975.0] m  (25 m above seafloor)
*    Receiver depth             --> -990.0 m  (inline profile at y = 1750 m)
*    Skin depth [Sed, Oil, Sed, Water] --> [355.67, 3556.71, 355.67, 194.81] m
*    Minimum skin depth         --> 194.811 m
*    Points per skin depth (rg) --> 1.0
*    Global element size (dg)   --> skin_depth / rg m
*    Domain x --> [58.33 - 4*skin_depth, 3383.33 + 4*skin_depth] m
*    Domain y --> [0.0, 3500.0] m
*    Domain z --> [0.0, -3500.0] m
*
* Reference model:
*    Castillo-Reyes, O., de la Puente, J., & Cela, J. M. (2018). PETGEM: A parallel code 
*    for 3D CSEM forward modeling using edge finite elements. Computers & Geosciences, 119, 123-136.
*
* See http://gmsh.info/ for details about mesh scripting with Gmsh.
*
* by Octavio Castillo-Reyes, UPC-BSC (octavio.castillo@upc.edu, octavio.castillo@bsc.es)
*********************************************************************/
// #################################################################
// #                        Parameters                             #
// #################################################################
// Minimum skin depth
skin_depth = 194.811;

// Point per skin depth
rg = 1.;
x_pos_first_receiver = 58.3333 - skin_depth*4.;
x_pos_last_receiver  = 3383.3333 + skin_depth*4.;

// Dimensions
MIN_X = x_pos_first_receiver;
MAX_X = x_pos_last_receiver;
MIN_Y = 0.0;
MAX_Y = 3500.0;
MIN_Z = -3500.0;
MAX_Z = 0.0;
DEPTH_MATERIAL1 = -1000.0;
DEPTH_MATERIAL2 = -2000.0;
DEPTH_MATERIAL3 = -2100.0;

// Mesh size
dg =  skin_depth / rg;

// Source position
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
// Receivers (Inline model)
Point (105) = {0.0, 1750.0, DEPTH_RECEIVERS};
Point (106) = {35.714285714285715, 1750.0, DEPTH_RECEIVERS};
Point (107) = {71.42857142857143, 1750.0, DEPTH_RECEIVERS};
Point (108) = {107.14285714285714, 1750.0, DEPTH_RECEIVERS};
Point (109) = {142.85714285714286, 1750.0, DEPTH_RECEIVERS};
Point (110) = {178.57142857142858, 1750.0, DEPTH_RECEIVERS};
Point (111) = {214.28571428571428, 1750.0, DEPTH_RECEIVERS};
Point (112) = {250.0, 1750.0, DEPTH_RECEIVERS};
Point (113) = {285.7142857142857, 1750.0, DEPTH_RECEIVERS};
Point (114) = {321.42857142857144, 1750.0, DEPTH_RECEIVERS};
Point (115) = {357.14285714285717, 1750.0, DEPTH_RECEIVERS};
Point (116) = {392.85714285714283, 1750.0, DEPTH_RECEIVERS};
Point (117) = {428.57142857142856, 1750.0, DEPTH_RECEIVERS};
Point (118) = {464.2857142857143, 1750.0, DEPTH_RECEIVERS};
Point (119) = {500.0, 1750.0, DEPTH_RECEIVERS};
Point (120) = {535.7142857142857, 1750.0, DEPTH_RECEIVERS};
Point (121) = {571.4285714285714, 1750.0, DEPTH_RECEIVERS};
Point (122) = {607.1428571428571, 1750.0, DEPTH_RECEIVERS};
Point (123) = {642.8571428571429, 1750.0, DEPTH_RECEIVERS};
Point (124) = {678.5714285714286, 1750.0, DEPTH_RECEIVERS};
Point (125) = {714.2857142857143, 1750.0, DEPTH_RECEIVERS};
Point (126) = {750.0, 1750.0, DEPTH_RECEIVERS};
Point (127) = {785.7142857142857, 1750.0, DEPTH_RECEIVERS};
Point (128) = {821.4285714285714, 1750.0, DEPTH_RECEIVERS};
Point (129) = {857.1428571428571, 1750.0, DEPTH_RECEIVERS};
Point (130) = {892.8571428571429, 1750.0, DEPTH_RECEIVERS};
Point (131) = {928.5714285714286, 1750.0, DEPTH_RECEIVERS};
Point (132) = {964.2857142857143, 1750.0, DEPTH_RECEIVERS};
Point (133) = {1000.0, 1750.0, DEPTH_RECEIVERS};
Point (134) = {1035.7142857142858, 1750.0, DEPTH_RECEIVERS};
Point (135) = {1071.4285714285713, 1750.0, DEPTH_RECEIVERS};
Point (136) = {1107.142857142857, 1750.0, DEPTH_RECEIVERS};
Point (137) = {1142.857142857143, 1750.0, DEPTH_RECEIVERS};
Point (138) = {1178.5714285714287, 1750.0, DEPTH_RECEIVERS};
Point (139) = {1214.2857142857142, 1750.0, DEPTH_RECEIVERS};
Point (140) = {1250.0, 1750.0, DEPTH_RECEIVERS};
Point (141) = {1285.7142857142858, 1750.0, DEPTH_RECEIVERS};
Point (142) = {1321.4285714285713, 1750.0, DEPTH_RECEIVERS};
Point (143) = {1357.142857142857, 1750.0, DEPTH_RECEIVERS};
Point (144) = {1392.857142857143, 1750.0, DEPTH_RECEIVERS};
Point (145) = {1428.5714285714287, 1750.0, DEPTH_RECEIVERS};
Point (146) = {1464.2857142857142, 1750.0, DEPTH_RECEIVERS};
Point (147) = {1500.0, 1750.0, DEPTH_RECEIVERS};
Point (148) = {1535.7142857142858, 1750.0, DEPTH_RECEIVERS};
Point (149) = {1571.4285714285713, 1750.0, DEPTH_RECEIVERS};
Point (150) = {1607.142857142857, 1750.0, DEPTH_RECEIVERS};
Point (151) = {1642.857142857143, 1750.0, DEPTH_RECEIVERS};
Point (152) = {1678.5714285714287, 1750.0, DEPTH_RECEIVERS};
Point (153) = {1714.2857142857142, 1750.0, DEPTH_RECEIVERS};
Point (154) = {1750.0, 1750.0, DEPTH_RECEIVERS};
Point (155) = {1784.5238081632654, 1750.0, DEPTH_RECEIVERS};
Point (156) = {1819.0476163265307, 1750.0, DEPTH_RECEIVERS};
Point (157) = {1853.5714244897958, 1750.0, DEPTH_RECEIVERS};
Point (158) = {1888.0952326530612, 1750.0, DEPTH_RECEIVERS};
Point (159) = {1922.6190408163266, 1750.0, DEPTH_RECEIVERS};
Point (160) = {1957.1428489795917, 1750.0, DEPTH_RECEIVERS};
Point (161) = {1991.666657142857, 1750.0, DEPTH_RECEIVERS};
Point (162) = {2026.1904653061224, 1750.0, DEPTH_RECEIVERS};
Point (163) = {2060.7142734693875, 1750.0, DEPTH_RECEIVERS};
Point (164) = {2095.238081632653, 1750.0, DEPTH_RECEIVERS};
Point (165) = {2129.7618897959183, 1750.0, DEPTH_RECEIVERS};
Point (166) = {2164.2856979591834, 1750.0, DEPTH_RECEIVERS};
Point (167) = {2198.809506122449, 1750.0, DEPTH_RECEIVERS};
Point (168) = {2233.333314285714, 1750.0, DEPTH_RECEIVERS};
Point (169) = {2267.8571224489797, 1750.0, DEPTH_RECEIVERS};
Point (170) = {2302.380930612245, 1750.0, DEPTH_RECEIVERS};
Point (171) = {2336.90473877551, 1750.0, DEPTH_RECEIVERS};
Point (172) = {2371.4285469387755, 1750.0, DEPTH_RECEIVERS};
Point (173) = {2405.9523551020407, 1750.0, DEPTH_RECEIVERS};
Point (174) = {2440.4761632653062, 1750.0, DEPTH_RECEIVERS};
Point (175) = {2474.9999714285714, 1750.0, DEPTH_RECEIVERS};
Point (176) = {2509.5237795918365, 1750.0, DEPTH_RECEIVERS};
Point (177) = {2544.0475877551016, 1750.0, DEPTH_RECEIVERS};
Point (178) = {2578.571395918367, 1750.0, DEPTH_RECEIVERS};
Point (179) = {2613.0952040816323, 1750.0, DEPTH_RECEIVERS};
Point (180) = {2647.619012244898, 1750.0, DEPTH_RECEIVERS};
Point (181) = {2682.142820408163, 1750.0, DEPTH_RECEIVERS};
Point (182) = {2716.666628571428, 1750.0, DEPTH_RECEIVERS};
Point (183) = {2751.190436734694, 1750.0, DEPTH_RECEIVERS};
Point (184) = {2785.714244897959, 1750.0, DEPTH_RECEIVERS};
Point (185) = {2820.2380530612245, 1750.0, DEPTH_RECEIVERS};
Point (186) = {2854.7618612244896, 1750.0, DEPTH_RECEIVERS};
Point (187) = {2889.2856693877548, 1750.0, DEPTH_RECEIVERS};
Point (188) = {2923.80947755102, 1750.0, DEPTH_RECEIVERS};
Point (189) = {2958.3332857142855, 1750.0, DEPTH_RECEIVERS};
Point (190) = {2992.857093877551, 1750.0, DEPTH_RECEIVERS};
Point (191) = {3027.380902040816, 1750.0, DEPTH_RECEIVERS};
Point (192) = {3061.9047102040813, 1750.0, DEPTH_RECEIVERS};
Point (193) = {3096.4285183673464, 1750.0, DEPTH_RECEIVERS};
Point (194) = {3130.952326530612, 1750.0, DEPTH_RECEIVERS};
Point (195) = {3165.4761346938776, 1750.0, DEPTH_RECEIVERS};
Point (196) = {3199.9999428571427, 1750.0, DEPTH_RECEIVERS};
Point (197) = {3234.523751020408, 1750.0, DEPTH_RECEIVERS};
Point (198) = {3269.047559183673, 1750.0, DEPTH_RECEIVERS};
Point (199) = {3303.571367346938, 1750.0, DEPTH_RECEIVERS};
Point (200) = {3338.0951755102037, 1750.0, DEPTH_RECEIVERS};
Point (201) = {3372.6189836734693, 1750.0, DEPTH_RECEIVERS};
Point (202) = {3407.1427918367344, 1750.0, DEPTH_RECEIVERS};
Point (203) = {3441.6665999999996, 1750.0, DEPTH_RECEIVERS};

// Refinement
Field[1] = Attractor;
Field[1].NodesList = {105:203};
Field[2] = MathEval;
Field[2].F = Sprintf("F1/5.0 + %g", dg/70.);
Field[3] = Min;
Field[3].FieldsList = {2};
Background Field = 3;

// #################################################################
// #                Mesh parameters                                #
// #################################################################
Characteristic Length {1, 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} = dg;
// Mesh format supported by PETGEM
Mesh.MshFileVersion = 2.2;
