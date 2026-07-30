/*********************************************************************
* mesh.geo  --  INVERSION mesh base (im.csem)
*
* A HOMOGENEOUS earth under air 
* In inverse modelling the invertable region is a parameter MASK
* (a logical subset of cells), not a physical body, so it is assigned by
* re-tagging cells AFTER meshing (see the re-tag step that produces
* mesh.msh and mesh_true.msh) rather than by embedding a volume that
* would create an artificial mesh interface with no conductivity contrast.
*
* This .geo yields only:  tag 1 AIR (z>0), tag 2 BG/earth (z<0).
* Starting model is a uniform 100 ohm-m halfspace. Refinement is placed
* (via size fields) at the true-anomaly footprint, the region of interest,
* the survey centre and the transmitter, so the later cell-masks are well
* resolved. Coordinates: z NEGATIVE DOWN, matching examples/fm - depth is a
* negative z, z=0 is the air/earth interface. Transmitter at (0,-4000,0).
*********************************************************************/

X  = 30000;  ZA =  60000;  ZB = -30000;   // domain (ZA air top, ZB earth bottom)
RX =   400;  RZ =  -600;                  // region-of-interest footprint (for refinement only)
AX =   100;  AZT = -100;  AZB = -300;     // true-anomaly footprint: AZT top, AZB bottom (AZB < AZT)
SY = -4000;
h_roi = 40;  h_an = 25;  h_near = 40;  h_far = 5000;   // smooth grading for PCBDDC robustness (h_far was 10000)

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

// ---- MESH SIZE (refine where the masks will land)
// Grading is deliberately SMOOTH: wide box transitions and a gentle radial
// gradient keep the fine->coarse size jump small so the air-earth interface
// stays topologically simple for PCBDDC coarse-edge construction. Critical
// sizes (h_roi/h_an/h_near) are unchanged, so resolution/accuracy is preserved.
// Keep the fine boxes on the earth side of z=0 (do NOT refine across z=0):
// with z negative down that is the z<0 half-space, so ZMin is the deeper bound.
Field[1]=Box; Field[1].VIn=h_roi; Field[1].VOut=h_far;
Field[1].XMin=-RX;Field[1].XMax=RX;Field[1].YMin=-RX;Field[1].YMax=RX;Field[1].ZMin=RZ;Field[1].ZMax=0;
Field[1].Thickness=1500;                                 // smoother transition (was 400)
Field[2]=Box; Field[2].VIn=h_an; Field[2].VOut=h_far;
Field[2].XMin=-AX;Field[2].XMax=AX;Field[2].YMin=-AX;Field[2].YMax=AX;Field[2].ZMin=AZB;Field[2].ZMax=AZT;
Field[2].Thickness=800;                                  // smoother transition (was 200)
Field[3]=MathEval; Field[3].F=Sprintf("%g + 0.22*sqrt(x^2 + y^2 + z^2)", h_near);           // gradient was 0.32
Field[4]=MathEval; Field[4].F=Sprintf("%g + 0.22*sqrt(x^2 + (y-(%g))^2 + z^2)", h_near, SY); // gradient was 0.32
Field[10]=Min; Field[10].FieldsList={1,2,3,4};
Background Field=10;

Mesh.MeshSizeExtendFromBoundary=0;
Mesh.MeshSizeFromPoints=0;
Mesh.MeshSizeFromCurvature=0;
Mesh.MeshSizeMax=h_far;
Mesh.Algorithm3D=10;                                     // HXT: better worst-element quality (was 1=Delaunay)
Mesh.MshFileVersion=2.2;
Mesh.Optimize=1;
