/*********************************************************************
* im_model_smooth.geo -- CANDIDATE M3: extra-gentle grading (trend point).
*
* Same physical model and geometry as geometry/im_model.geo (homogeneous
* earth under air; z POSITIVE DOWN; transmitter at (0,-4000,0)). The
* INVERT/ANOMALY regions are still parameter MASKS applied by re-tagging
* cells after meshing, NOT geometric volumes, so no artificial interface is
* introduced.
*
* ONLY the meshing strategy changes relative to the original:
*   - graded box transition widened   Thickness 400/200 -> 1500/800
*   - radial size gradient softened    0.32 -> 0.22 (per metre)
*   - far-field cap lowered            h_far 10000 -> 5000
*   - 3D algorithm                     Delaunay(1) -> HXT(10)
* The CRITICAL element sizes are UNCHANGED (h_roi=40, h_an=25, h_near=40),
* so anomaly/ROI resolution -- and thus inversion accuracy -- is preserved.
* The softer grading removes the abrupt fine->coarse size jump that produced
* the high-valence, topologically complex subdomain interfaces implicated in
* the PCBDDC coarse-edge (Nedelec support) failure at order 2.
*********************************************************************/

X  = 30000;  ZA = -60000;  ZB = 30000;   // domain
RX =   400;  RZ =   600;                  // region-of-interest footprint (for refinement only)
AX =   100;  AZT = 100;   AZB = 300;      // true-anomaly footprint (for refinement only)
SY = -4000;
h_roi = 40;  h_an = 25;  h_near = 40;  h_far = 4000;    // h_far 10000 -> 5000

// ---- POINTS: earth top z=0 (1-4), earth bottom z=ZB (5-8), air top z=ZA (9-12)
Point(1)={-X,-X,0}; Point(2)={X,-X,0}; Point(3)={X,X,0}; Point(4)={-X,X,0};
Point(5)={-X,-X,ZB};Point(6)={X,-X,ZB};Point(7)={X,X,ZB};Point(8)={-X,X,ZB};
Point(9)={-X,-X,ZA};Point(10)={X,-X,ZA};Point(11)={X,X,ZA};Point(12)={-X,X,ZA};

// ---- LINES
Line(1)={1,2};Line(2)={2,3};Line(3)={3,4};Line(4)={4,1};        // z=0 (shared)
Line(5)={5,6};Line(6)={6,7};Line(7)={7,8};Line(8)={8,5};        // z=ZB
Line(9)={1,5};Line(10)={2,6};Line(11)={3,7};Line(12)={4,8};     // earth verticals
Line(13)={9,10};Line(14)={10,11};Line(15)={11,12};Line(16)={12,9}; // z=ZA
Line(17)={1,9};Line(18)={2,10};Line(19)={3,11};Line(20)={4,12};    // air verticals

// ---- SURFACES
Line Loop(1)={1,2,3,4};        Plane Surface(1)={1};   // z=0
Line Loop(2)={5,6,7,8};        Plane Surface(2)={2};   // z=ZB
Line Loop(3)={1,10,-5,-9};     Plane Surface(3)={3};
Line Loop(4)={2,11,-6,-10};    Plane Surface(4)={4};
Line Loop(5)={3,12,-7,-11};    Plane Surface(5)={5};
Line Loop(6)={4,9,-8,-12};     Plane Surface(6)={6};
Line Loop(7)={13,14,15,16};    Plane Surface(7)={7};   // z=ZA
Line Loop(8)={1,18,-13,-17};   Plane Surface(8)={8};
Line Loop(9)={2,19,-14,-18};   Plane Surface(9)={9};
Line Loop(10)={3,20,-15,-19};  Plane Surface(10)={10};
Line Loop(11)={4,17,-16,-20};  Plane Surface(11)={11};

// ---- VOLUMES
Surface Loop(1)={1,2,3,4,5,6};        Volume(1)={1};   // earth (BG)
Surface Loop(2)={1,7,8,9,10,11};      Volume(2)={2};   // air (z=0 face shared)
Physical Volume("AIR", 1)={2};
Physical Volume("BG",  2)={1};

// ---- MESH SIZE (refine where the masks will land) -- SMOOTH GRADING
Field[1]=Box; Field[1].VIn=h_roi; Field[1].VOut=h_far;
Field[1].XMin=-RX;Field[1].XMax=RX;Field[1].YMin=-RX;Field[1].YMax=RX;Field[1].ZMin=0;Field[1].ZMax=RZ;
Field[1].Thickness=2500;                                 // 400 -> 1500
Field[2]=Box; Field[2].VIn=h_an; Field[2].VOut=h_far;
Field[2].XMin=-AX;Field[2].XMax=AX;Field[2].YMin=-AX;Field[2].YMax=AX;Field[2].ZMin=AZT;Field[2].ZMax=AZB;
Field[2].Thickness=1200;                                  // 200 -> 800
Field[3]=MathEval; Field[3].F=Sprintf("%g + 0.15*sqrt(x^2 + y^2 + z^2)", h_near);          // 0.32 -> 0.22
Field[4]=MathEval; Field[4].F=Sprintf("%g + 0.15*sqrt(x^2 + (y-(%g))^2 + z^2)", h_near, SY); // 0.32 -> 0.22
Field[10]=Min; Field[10].FieldsList={1,2,3,4};
Background Field=10;

Mesh.MeshSizeExtendFromBoundary=0;
Mesh.MeshSizeFromPoints=0;
Mesh.MeshSizeFromCurvature=0;
Mesh.MeshSizeMax=h_far;
Mesh.Algorithm3D=10;                                     // Delaunay(1) -> HXT(10)
Mesh.MshFileVersion=2.2;
Mesh.Optimize=1;
