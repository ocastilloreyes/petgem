/*********************************************************************
* im_model_smooth_iface.geo -- CANDIDATE M2: M1 + symmetric air-earth interface.
*
* Identical to CANDIDATE M1 (im_model_smooth.geo) except the region-of-interest
* refinement box is extended a thin slab into the AIR (ZMin 0 -> -ZAIR) so the
* z=0 air/earth interface -- where the 441 receivers sit -- lies INSIDE a
* uniformly h_roi-sized region instead of ON the boundary of the refinement box.
*
* Rationale: in the original and in M1, the ROI box starts exactly at z=0, so
* the air just above the interface coarsens while the earth just below stays
* fine. That fine/coarse step STRADDLING the interface plane produces the
* high-valence interface vertices and complex z=0 connectivity that PCBDDC's
* Nedelec coarse-edge construction must chain through. Making the near-surface
* size symmetric across z=0 removes that straddling step, so interface vertices
* have regular (interior-like) valence.
*
* Physics/geometry unchanged: refining a thin air slab does not alter the
* homogeneous-earth model or the re-tagging masks (air stays tag 1). Critical
* sizes h_roi=40, h_an=25, h_near=40 unchanged -> resolution preserved.
*********************************************************************/

X  = 30000;  ZA = -60000;  ZB = 30000;   // domain
RX =   400;  RZ =   600;                  // region-of-interest footprint (for refinement only)
AX =   100;  AZT = 100;   AZB = 300;      // true-anomaly footprint (for refinement only)
SY = -4000;
ZAIR = 200;                               // air slab (m) above z=0 kept at h_roi (symmetric interface)
h_roi = 40;  h_an = 25;  h_near = 40;  h_far = 5000;

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

// ---- MESH SIZE (refine where the masks will land) -- SMOOTH GRADING + SYMMETRIC INTERFACE
Field[1]=Box; Field[1].VIn=h_roi; Field[1].VOut=h_far;
Field[1].XMin=-RX;Field[1].XMax=RX;Field[1].YMin=-RX;Field[1].YMax=RX;Field[1].ZMin=-ZAIR;Field[1].ZMax=RZ;  // ZMin 0 -> -ZAIR
Field[1].Thickness=1500;
Field[2]=Box; Field[2].VIn=h_an; Field[2].VOut=h_far;
Field[2].XMin=-AX;Field[2].XMax=AX;Field[2].YMin=-AX;Field[2].YMax=AX;Field[2].ZMin=AZT;Field[2].ZMax=AZB;
Field[2].Thickness=800;
Field[3]=MathEval; Field[3].F=Sprintf("%g + 0.22*sqrt(x^2 + y^2 + z^2)", h_near);
Field[4]=MathEval; Field[4].F=Sprintf("%g + 0.22*sqrt(x^2 + (y-(%g))^2 + z^2)", h_near, SY);
Field[10]=Min; Field[10].FieldsList={1,2,3,4};
Background Field=10;

Mesh.MeshSizeExtendFromBoundary=0;
Mesh.MeshSizeFromPoints=0;
Mesh.MeshSizeFromCurvature=0;
Mesh.MeshSizeMax=h_far;
Mesh.Algorithm3D=10;
Mesh.MshFileVersion=2.2;
Mesh.Optimize=1;
