---
title: 'PETGEM v2.0.0: A parallel edge-element toolkit for high-order 3D electromagnetic modeling in geophysics'
tags:
  - geophysics
  - computational electromagnetics
  - CSEM
  - finite element method
  - edge elements
  - high-performance computing
  - PETSc
  - C
authors:
  - name: Octavio Castillo-Reyes
    orcid: 0000-0003-4271-5015
    corresponding: true
    affiliation: "1, 2"
affiliations:
  - name: Department of Computer Architecture, Universitat Politècnica de Catalunya (UPC), Barcelona, Spain
    index: 1
  - name: Barcelona Supercomputing Center (BSC), Barcelona, Spain
    index: 2
date: 16 July 2026
bibliography: references.bib
---

# Summary

`PETGEM` (Parallel Edge-element Toolkit for General Electromagnetic Modeling) is
an open-source scientific software package for three-dimensional (3D)
electromagnetic (EM) modeling of the Earth's subsurface in the frequency domain.
It discretizes the total electric field with high-order Nédélec (edge) vector
finite elements of polynomial orders 1-6 on unstructured tetrahedral
meshes, and is written in C on top of the `PETSc` library [@petsc-web-page] and
the Message Passing Interface (MPI). From a single high-order finite-element
core, the package builds two computational kernels and a dispatcher: `fm.csem`,
which computes the controlled-source electromagnetic (CSEM) response of a given
conductivity model, and `im.csem`, which recovers a conductivity model from
observed data by means of a limited-memory BFGS (L-BFGS) optimizer with
adjoint-state gradients. Pre- and post-processing are handled by a lightweight
Python layer that assembles a self-describing HDF5 input bundle consumed by both
kernels. `PETGEM` is designed for and validated on high-performance computing
(HPC) platforms, and has been applied to marine and land CSEM surveys, geothermal
exploration, EM imaging in the presence of metallic infrastructure, and
passive-source (magnetotelluric) configurations
[@CAG.18.CastilloReyes; @GJI.19.CastilloReyes; @TGRS.21.CastilloReyes; @CAG.22.CastilloReyes; @CG.23.CastilloReyes].
The software is distributed under the BSD 3-Clause license, developed openly on
GitHub with continuous integration and containerized builds, and documented
through a versioned user and developer manual.

# Statement of need

Geophysical EM imaging produces subsurface resistivity maps that are essential
for reducing the risk associated with geo-resource exploration and for the
sustainable management of the subsurface. Its applications span hydrocarbon
detection, mineral characterization, geothermal exploration, groundwater studies,
and the monitoring of $\mathrm{CO}_2$ storage reservoirs
[@Eidesmo2002; @Constable2010; @Avdeev2005]. Among these techniques, the CSEM
method has become a mature tool, and its interpretation relies on repeated,
accurate solutions of the forward problem: the numerical computation of the EM
fields for an assumed subsurface conductivity distribution. Because realistic
targets involve complex geometries, sharp resistivity contrasts, and large
computational domains, forward modeling remains numerically demanding and is
recognized as an active and challenging area of research
[@SG.10.Borner; @Newman2014].

Two requirements drive the need for software such as `PETGEM`. First, from a
numerical standpoint, accurate representation of the EM field in geometrically
complex media calls for methods that conform to the physics of the problem.
Nédélec edge elements are the natural choice because they are
$H(\mathrm{curl})$-conforming, enforcing tangential field continuity across
element faces while avoiding the spurious modes that plague nodal
discretizations of Maxwell's equations [@monk2003finite]. High-order variants
further reduce numerical dispersion and improve geometric flexibility on
unstructured tetrahedral meshes. Second, from a computational standpoint, the
solution of large, ill-conditioned, complex-valued sparse linear systems, and
the prospect of solving thousands of such systems within inversion or
machine-learning workflows, requires scalable and efficient implementations that
exploit distributed-memory parallelism [@Newman2014; @puzyrev2016evaluation].
Meeting both requirements simultaneously, in a single reusable and
well-engineered code base, is non-trivial.

Despite a growing open-source landscape for 3D CSEM modeling
[@GJI.21.Werthmuller; @oldenburg20203d], most freely available codes rely on
low-order discretizations, target moderate problem sizes, or are not primarily
engineered for massively parallel execution. Related finite-element efforts
address parts of this space (e.g., adaptive higher-order formulations
for marine CSEM [@Schwarzbach2011], higher-order finite-element EM solvers for
HPC environments [@garcia2017higher], and parallel edge-element CSEM codes
[@cai2017parallelized]). `PETGEM` occupies a distinct point in this landscape by
combining high polynomial order (Nédélec orders 1-6) with a fully MPI-parallel,
`DMPlex`-based mesh and solver infrastructure, and by providing both forward and
inverse kernels within a single, openly developed, tested, and reproducible code
base. Its intended users are computational geophysicists and EM practitioners who
need to run accurate feasibility studies and survey-design simulations, who wish
to generate high-order solutions on large unstructured meshes, and who want a
maintainable platform on which to build further research, including inversion and
data-driven applications [@FEC.23.CastilloReyes; @SG.24.CastilloReyes].

# Software description

`PETGEM` follows a modular architecture in which a single finite-element core is
shared by every simulation mode. The C sources under `src/` separate concerns
into well-defined units: the arbitrary-order $H(\mathrm{curl})$ Nédélec and $H^1$
nodal bases (`fe_nedelec.c`, `fe_nodal.c`), the element-level finite-element
machinery (`fem.c`), the global assembly of the operators and right-hand sides
(`assembly.c`), the mesh and discretization layer built on `PETSc`'s `DMPlex`
(`grid.c`), the linear-system solver interface (`solver.c`), the input/output
subsystem (`io.c`), the forward and inverse CSEM kernels (`fm_csem.c`,
`im_csem.c`, `inversion.c`), a method-of-manufactured-solutions verification mode
(`mms.c`), and a unified dispatcher (`petgem.c`). The corresponding public
interfaces live under `include/`. This separation promotes code reuse,
readability, and extensibility: the same assembled operators, discrete-gradient
matrix, and solver policy serve both the forward and the inverse kernel.

The numerical framework solves the frequency-domain, diffusive form of Maxwell's
equations for the total electric field $E$,
$$ \nabla \times \nabla \times E \;-\; i\,\omega\,\mu\,\sigma\, E \;=\; f, $$
with angular frequency $\omega = 2\pi f$, magnetic permeability $\mu = \mu_0$,
and a diagonal conductivity tensor $\sigma$, subject to homogeneous Dirichlet
conditions $n \times E = 0$ on the domain boundary. Discretization with Nédélec
elements yields the complex-symmetric linear system $A\,e = b$, with
$A = K - i\,\omega\,\mu\, M_\sigma$, where $K$ is the curl-curl stiffness matrix
and $M_\sigma$ the conductivity-weighted mass matrix. Because $A$ is complex,
`PETGEM` is built against a `PETSc` configured with complex scalars. A point
electric dipole enters the right-hand side through the evaluation of the basis
functions at the source location, and receiver responses are obtained by
interpolating the solution and its curl at the receiver positions. Alongside $K$
and $M_\sigma$, the assembly builds the high-order discrete-gradient matrix $G$
that spans the curl-kernel of the Nédélec space and satisfies $K\,G = 0$; this
operator is central to the preconditioning strategy described below.

The end-to-end workflow is deliberately simple and reproducible. A tetrahedral
mesh is generated with `Gmsh` [@geuzaine2009gmsh] (or imported in VTK format),
the conductivity model is
described in a plain-text per-material table, and a single Python preprocessing
command assembles an HDF5 input bundle together with a matching `PETSc` options
file. The kernels read the bundle, solve the problem in parallel under MPI, and
write self-describing HDF5 output whose provenance attributes record the
`PETGEM` version, simulation type, polynomial order, solver configuration, and
MPI task count. Post-processing and validation are performed by small Python
scripts shipped with each example.

# Features and functionality

**High-order edge-element formulation.** `PETGEM` implements high-order
$H(\mathrm{curl})$ Nédélec elements on the reference tetrahedron (orders 1-6),
with closed-form degree-of-freedom counts and a per-entity (edge/face/interior)
ordering that the assembly's local-to-global mapping relies on. The polynomial
order is a property of the basis rather than of the mesh, so a single tetrahedral
mesh serves every order in the supported range; the order is selected at
preprocessing time and can be overridden at run time. This high-order edge-element design is
the methodological foundation established in the `PETGEM` reference publications
[@CAG.18.CastilloReyes; @GJI.19.CastilloReyes].

**Mesh management and parallel infrastructure.** Mesh handling is built on
`PETSc`'s `DMPlex` [@knepley2009mesh; @lange2016efficient], which represents the
unstructured mesh as a topological graph and provides boundary labeling, section
creation, and the closure operations that drive assembly. Delegating mesh
management to `DMPlex` gives `PETGEM` access to scalable, standardized
infrastructure for parallel partitioning through graph partitioners such as
`ParMETIS` [@karypis1997parmetis] and `PT-Scotch` [@pellegrini1996scotch],
distributed data layout, and parallel mesh input/output [@hapla2021fully], while
keeping the toolkit interoperable with the broader `PETSc` ecosystem.

**Solver infrastructure.** The forward kernel assembles the operator as a
distributed `MATIS` matrix and solves for all CSEM sources at once as a single
multi-right-hand-side system. By default it uses flexible GMRES
[@saad1993flexible] preconditioned by Balancing Domain Decomposition by
Constraints (BDDC) [@dohrmann2007approximate]. `PETGEM` registers its
high-order discrete gradient $G$ with the preconditioner so that the BDDC coarse
space captures the curl-kernel of the $H(\mathrm{curl})$ operator, a
well-established requirement for robust preconditioning of edge-element
discretizations. A direct factorization with `MUMPS` [@amestoy2006hybrid] is
also available and is the default for the inverse kernel, which re-solves the
system at every optimization step. Because any `PETSc` `KSP`/`PC` option can be
set from the parameter file or the command line, the solver configuration is
fully controllable without recompilation.

**Inverse modeling.** The `im.csem` kernel recovers a per-material conductivity
model by minimizing a regularized data-misfit functional that combines the
weighted difference between observed and predicted responses with a Tikhonov
term penalizing departure from the starting model. The gradient is assembled by
the adjoint-state method, performing forward and adjoint solves per frequency
and avoiding the explicit formation of the Jacobian, and the model is updated
with an L-BFGS scheme. This inversion layer is built directly on the validated
forward core, so gradients and forward responses share the same high-order
discretization.

**Verification and testing.** Correctness is guarded by a layered test suite. C
harnesses link directly against the production sources to check basis-function
invariants, degree-of-freedom ordering, and element-matrix properties (symmetry,
definiteness, and the discrete de Rham identity $K_e\,G_e = 0$) for every order
1-6. End-to-end tests drive the compiled kernel through the full
assemble–solve–interpolate pipeline on a reference dataset and compare against
committed exact-solve golden references. An independent order-of-accuracy check
uses a method of manufactured solutions [@roache2002code; @garcia2016verification],
in which a known exact field yields closed-form discretization errors that the
kernel must reproduce. The suite is designed so that partial environments run
whatever coverage they can, and reference comparisons use deterministic direct
solves to remain stable across MPI task counts.

**Performance analysis.** The forward kernel can be compiled with `Extrae`
instrumentation to emit execution traces for the `Paraver` performance analyzer
[@extrae2024], enabling systematic identification of bottlenecks and
communication patterns in parallel runs. The tracing configuration ships with
the repository and is exercised as part of the automated pipeline. The
scalability and HPC suitability of the underlying methodology have been
characterized in prior studies of the code
[@GJI.19.CastilloReyes; @EMDPI.22.CastilloReyes; @gell2022hpc; @xavier2024fib].

**Reproducibility, containers, and continuous integration.** A Docker image
provides the full dependency stack (a complex-scalar `PETSc` build with MPI,
`MUMPS`, HDF5, and graph partitioners, together with `Gmsh`, `Extrae`, and the
Python layer), so that builds and runs are reproducible across machines. The
project is developed on GitHub with GitHub Actions workflows that build the
kernels and run the verification suite inside the pinned container image, build
and drift-check the documentation, and, on release, publish a versioned container
image and a tagged GitHub release. Every output file is stamped with provenance
metadata, and the example datasets are regenerated deterministically from
committed geometry and text inputs, closing the loop between the shipped inputs
and the reported results.

# Example applications

Two ready-to-run cases ship with the repository and illustrate the forward
workflow. The **canonical marine CSEM model** is the textbook benchmark in which
a thin, resistive hydrocarbon layer is buried in conductive marine sediments
beneath a seawater column; it exercises the complete preprocess -> `fm.csem` ->
post-process pipeline on a realistic layered earth and is validated against a
precomputed reference by comparing the electric field along an inline seafloor
receiver profile. The **unit-cube** case is a small homogeneous model with a
single dipole and three receivers that, despite its size, activates every
degree-of-freedom entity class and therefore serves as the dataset around which
the automated test suite is built.

Beyond the shipped examples, the methodology implemented in `PETGEM` has been
applied and validated across a range of geophysical settings reported in the
literature: shallow marine hydrocarbon exploration
[@CAG.18.CastilloReyes; @GJI.19.CastilloReyes; @GJI.21.Werthmuller; @EMDPI.22.CastilloReyes],
geothermal reservoir characterization with metallic casing [@TGRS.21.CastilloReyes],
EM surveys in the presence of metallic infrastructure and the associated meshing
strategies [@JCS.CastilloReyes2022; @CG.23.CastilloReyes], and passive-source
magnetotelluric modeling with large resistivity contrasts [@CAG.22.CastilloReyes].
These studies document the accuracy, robustness, and parallel efficiency of the
toolkit under demanding conditions.

# Availability and documentation

`PETGEM` is openly available on GitHub at
<https://github.com/ocastilloreyes/petgem> under the BSD 3-Clause license, with
semantic versioning and tagged releases managed through the repository's release
workflow. Building requires `make` and an existing complex-scalar `PETSc`
installation; a Docker image with the complete dependency stack is provided for a
fully reproducible environment, and a single `make` invocation produces the
forward kernel, the inverse kernel, and the dispatcher. User and developer
documentation is hosted at <https://petgem.readthedocs.io>, is versioned
alongside the code, and covers installation, a quick-start, the numerical
formulation, mesh and data formats, solver options, the example cases, the
testing framework, and contribution guidelines; a Doxygen-generated C API
reference is integrated into the same site. The documentation build is itself
verified in continuous integration, including a drift guard that keeps the manual
consistent with the code. Together, the open repository, permissive license,
container images, automated testing and continuous integration, provenance-aware
input/output, and comprehensive documentation make `PETGEM` a usable,
maintainable, and reproducible platform for the computational-geophysics and EM
modeling community.

# Acknowledgements
This work was partially supported by the Generalitat de Catalunya (AGAUR)
under grant agreement 2021-SGR-00478. The author acknowledges the `PETSc`
development team, particularly Prof. Matthew Knepley, for their support in
the validation and optimization of the software. We also thank Prof. Pilar
Queralt for providing the `DIPOLECODE` reference solutions and the BSC Tools
Department, especially German Llort, for assistance with `Extrae`/`Paraver`
performance analysis.

# References
