/*********************************************************************
*
* MT half-space test for fm.mt
*
* Box [-1000,1000] x [-1000,1000] x [-9600,2000] m: earth below z = 0
* (physical 1) and air above (physical 2).
*
*********************************************************************/
SetFactory("OpenCASCADE");

L  = 1000.0;   // half-width (m)
zb = -9600.0;  // bottom (m)
zt = 2000.0;   // top (m)

Box(1) = {-L, -L, zb, 2*L, 2*L, -zb};
Box(2) = {-L, -L, 0.0, 2*L, 2*L, zt};
BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }

Physical Volume(1) = {1};
Physical Volume(2) = {2};

Field[1] = Box;
Field[1].VIn  = 150.0;
Field[1].VOut = 600.0;
Field[1].XMin = -L; Field[1].XMax = L;
Field[1].YMin = -L; Field[1].YMax = L;
Field[1].ZMin = -1500.0; Field[1].ZMax = 300.0;
Field[1].Thickness = 1500.0;
Background Field = 1;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MshFileVersion = 2.2;
