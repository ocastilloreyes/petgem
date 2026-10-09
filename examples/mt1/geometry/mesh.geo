/*********************************************************************
*
* mt1 - Gmsh geometry for the PETGEM 3D trapezoidal hill MT example
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
SetFactory("OpenCASCADE");

DefineConstant[ nskin = {1, Name "nskin"} ];
DefineConstant[ hmin = {80.0, Name "hmin"} ];
DefineConstant[ hmax = {1750.0, Name "hmax"} ];

delta = 3500.0;
L     = 2000.0 + nskin * delta;

// Hill: base square at z = 0, top square at z = 450
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
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Ruled ThruSections(1) = {1, 2};

// Earth (half-space + hill) and air
Box(2) = {-L, -L, -L, 2*L, 2*L, L};
Box(3) = {-L, -L, 0, 2*L, 2*L, L};
earth() = BooleanUnion{ Volume{2}; Delete; }{ Volume{1}; Delete; };
air()   = BooleanDifference{ Volume{3}; Delete; }{ Volume{earth()}; };
BooleanFragments{ Volume{earth(), air()}; Delete; }{}

eps = 1.0;
Physical Volume(1) = Volume In BoundingBox{-L-eps, -L-eps, -eps, L+eps, L+eps, L+eps};
Physical Volume(2) = Volume In BoundingBox{-L-eps, -L-eps, -L-eps, L+eps, L+eps, 450+eps};

// Refinement: hill surfaces and a slab along the survey line (y = 0, |x| <= 2 km)
hill() = Surface In BoundingBox{-1000-eps, -1000-eps, -eps, 1000+eps, 1000+eps, 450+eps};
Field[1] = Distance;
Field[1].SurfacesList = {hill()};
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
