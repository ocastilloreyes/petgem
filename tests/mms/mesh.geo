/*********************************************************************
*
* test1 - Gmsh geometry for the PETGEM MMS order-of-accuracy study
*
* Homogeneous CUBE [0,L]^3 with L = 1000 m, single (anisotropic) material.
* Same workflow as examples/fm (gmsh -> preprocess -> fm.csem); only the
* refinement knob N changes between runs.
*
* WHY THE CORNER SITS AT THE ORIGIN. The manufactured field verified here is
*
*       E*(x,y,z) = a S(x) + b G(x),        k = m pi / L,
*       S = ( sin(ky) sin(kz), sin(kz) sin(kx), sin(kx) sin(ky) ),
*       G = grad( sin(kx) sin(ky) sin(kz) ) / k,
*
* whose tangential trace n x E* vanishes on the faces of [0,L]^3 for every
* integer mode number m: each off-normal component carries a sin(k{x|y|z})
* factor that is zero at 0 and at L, and G is the gradient of a potential that
* vanishes on the whole boundary. That is exactly the homogeneous Dirichlet
* condition fm.csem imposes by constraining boundary DOFs out of the
* PetscSection (src/grid.c), so NO nonzero-BC code path is needed and the
* verification runs on the production assembly and solve. On a centred cube
* [-L/2,L/2]^3 this trace is NOT zero and the test is invalid: do not move or
* rescale the domain without changing E* (see mms_reference.py).
*
* WHY L = 1000 m AT f = 10 Hz. With sigma_ref = 1 S/m the skin depth is
* delta = sqrt(2/(w mu0 sigma)) = 159.155 m = L/(m pi), so with m = 2 the
* manufactured mode satisfies
*
*       rho = 2 k^2 / (w mu0 sigma_ref) = 1      EXACTLY,
*       nu  = w mu0 sigma_ref L^2       = 8 pi^2 = 79.0,
*
* i.e. the curl-curl and the mass term of A = K - i w mu0 M are exactly
* balanced. The conditioning of A scales as kappa ~ p^4 N^2 / nu, so a domain
* much smaller than the skin depth (nu -> 0) drives kappa up and the measured
* slopes report the arithmetic floor kappa*eps instead of the discretization
* error. On the unit cube at the same frequency nu = 7.9e-5, a factor 1e6
* worse. L is duplicated in include/mms.h (MMS_L) and mms_reference.py (L):
* keep the three in step.
*
* CONVERGENCE SEQUENCE: regenerate this mesh for several N to get a family of
* refined meshes with EXACT, KNOWN size h = L/N, e.g.
*       gmsh -3 -setnumber N 4  -o mesh_N4.msh  mesh.geo
*       gmsh -3 -setnumber N 8  -o mesh_N8.msh  mesh.geo
*       gmsh -3 -setnumber N 16 -o mesh_N16.msh mesh.geo
* The per-order lists used by job_marenostrum.slurm are
*       p=1: 16 24 32 48    p=2: 8 12 16 24    p=3: 6 8 12 16
*       p=4:  4  6  8 12    p=5: 4  6  8 10    p=6: 4 6  8 10
* which keep every run under ~7.6e5 DOFs (see mms_reference.py --design).
*
* Since lambda = 2L/m = L here, N is also the number of elements per
* manufactured wavelength.
*
* Structured transfinite meshing keeps the node layout DETERMINISTIC and the
* tetrahedra uniform, so the measured slopes are clean and reproducible.
* N divisions per edge -> (N+1)^3 vertices and 6*N^3 tetrahedra.
*
* by the PETGEM team (http://petgem.bsc.es/)
*********************************************************************/
// #################################################################
// #                        Parameters                             #
// #################################################################
// Divisions per cube edge (structured lattice). Override on the command
// line with:  gmsh -3 -setnumber N <value> -o mesh_N<value>.msh mesh.geo
If(!Exists(N))
  N = 8;
EndIf

// Edge length of the domain [0,L]^3 in metres.
// MUST match MMS_L in include/mms.h and L in mms_reference.py;
// runMMSVerification checks it against the mesh bounding box and aborts on a mismatch.
L = 1000.0;

// #################################################################
// #                    Define main points                         #
// #################################################################
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};
Point(5) = {0, 0, L};
Point(6) = {L, 0, L};
Point(7) = {L, L, L};
Point(8) = {0, L, L};

// #################################################################
// #                        Define edges                           #
// #################################################################
Line(1)  = {1, 2};   // bottom face (z = 0)
Line(2)  = {2, 3};
Line(3)  = {3, 4};
Line(4)  = {4, 1};
Line(5)  = {5, 6};   // top face (z = L)
Line(6)  = {6, 7};
Line(7)  = {7, 8};
Line(8)  = {8, 5};
Line(9)  = {1, 5};   // vertical edges
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

// #################################################################
// #                   Define bounding surfaces                    #
// #################################################################
Line Loop(1) = {1, 2, 3, 4};        Plane Surface(1) = {1};   // z = 0
Line Loop(2) = {5, 6, 7, 8};        Plane Surface(2) = {2};   // z = L
Line Loop(3) = {1, 10, -5, -9};     Plane Surface(3) = {3};   // y = 0
Line Loop(4) = {2, 11, -6, -10};    Plane Surface(4) = {4};   // x = L
Line Loop(5) = {3, 12, -7, -11};    Plane Surface(5) = {5};   // y = L
Line Loop(6) = {4, 9, -8, -12};     Plane Surface(6) = {6};   // x = 0

// #################################################################
// #                        Define volume                          #
// #################################################################
Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

// #################################################################
// #                Structured (transfinite) meshing               #
// #################################################################
Transfinite Line "*"    = N + 1;
Transfinite Surface "*";
Transfinite Volume  "*";

// #################################################################
// #             Single material (gmsh:physical tag 1)             #
// #################################################################
// Row 0 of survey/sigmas.txt (= physical tag 1 - 1) -> sigma = (0.5, 1.0, 2.0).
Physical Volume("cube", 1) = {1};

// Mesh format supported by PETGEM (read via meshio in utils/preprocess.py).
Mesh.MshFileVersion = 2.2;
