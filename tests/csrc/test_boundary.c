/*
 * Filename: test_boundary.c
 * Author: PETGEM test suite
 * Date: 2026-10-09
 *
 * Description:
 * Boundary tools. For a given order (argv[1]) checks:
 *   - the 2D triangle rule (compute2DQuadraturePoints): weights sum to 1/2,
 *     points inside the unit triangle, exact monomials up to degree 2*order+1;
 *   - setupNedelecGrid on a box mesh: global DOF counts for PETGEM_BC_NATURAL
 *     (all DOFs) and PETGEM_BC_PEC (interior DOFs only);
 *   - getBoundaryFaces / computeBoundaryFaceGeometry: face count, total area,
 *     sum of n*area = 0, axis-aligned outward normals on the box planes.
 *
 * Usage: [mpirun -n N] test_boundary <order>   (order in 1..6).  Exit 0 iff all checks pass.
 */
#include "fem.h"
#include "grid.h"
#include "constants.h"
#include "petgem_test.h"
#include <petsc.h>

static const PetscInt  boxFaces[3] = {2, 3, 2};
static const PetscReal boxLower[3] = {0.0, 0.0, -1.0};
static const PetscReal boxUpper[3] = {2.0, 3.0, 0.5};

static PetscReal factorial(PetscInt n)
{
  PetscReal f = 1.0;
  for (PetscInt i = 2; i <= n; i++) f *= (PetscReal)i;
  return f;
}

static void check_quadrature_2d(PetscInt order, PetscMPIInt rank)
{
  Quadrature2D q;
  PetscCallAbort(PETSC_COMM_SELF, computeNum2DQuadraturePoints(order, &q));
  PT_CHECK(q.numPoints == (order + 1) * (order + 1), "order=%d: numPoints=%d", (int)order, (int)q.numPoints);

  PetscCallAbort(PETSC_COMM_SELF, PetscCalloc1(q.numPoints, &q.points));
  for (PetscInt i = 0; i < q.numPoints; i++) PetscCallAbort(PETSC_COMM_SELF, PetscCalloc1(2, &q.points[i]));
  PetscCallAbort(PETSC_COMM_SELF, PetscCalloc1(q.numPoints, &q.weights));
  PetscCallAbort(PETSC_COMM_SELF, compute2DQuadraturePoints(&q));

  PetscReal wsum = 0.0;
  for (PetscInt i = 0; i < q.numPoints; i++) {
    const PetscReal s = q.points[i][0], t = q.points[i][1];
    wsum += q.weights[i];
    PT_CHECK(q.weights[i] > 0.0, "order=%d: weight %d not positive", (int)order, (int)i);
    PT_CHECK(s > 0.0 && t > 0.0 && s + t < 1.0, "order=%d: point %d (%g,%g) outside the triangle", (int)order, (int)i, s, t);
  }
  PT_CLOSE(wsum, 0.5, 1e-14, "order=%d: weights sum %.16g != 1/2", (int)order, wsum);

  /* INT_T s^a t^b = a! b! / (a+b+2)! */
  const PetscInt degree = 2 * order + 1;
  for (PetscInt a = 0; a <= degree; a++) {
    for (PetscInt b = 0; a + b <= degree; b++) {
      PetscReal num = 0.0;
      for (PetscInt i = 0; i < q.numPoints; i++) num += q.weights[i] * PetscPowRealInt(q.points[i][0], a) * PetscPowRealInt(q.points[i][1], b);
      const PetscReal exact = factorial(a) * factorial(b) / factorial(a + b + 2);
      PT_CLOSE(num, exact, 1e-13 * exact + 1e-16, "order=%d: s^%d t^%d integrates to %.16g, exact %.16g", (int)order, (int)a, (int)b, num, exact);
    }
  }

  for (PetscInt i = 0; i < q.numPoints; i++) PetscCallAbort(PETSC_COMM_SELF, PetscFree(q.points[i]));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(q.points));
  PetscCallAbort(PETSC_COMM_SELF, PetscFree(q.weights));
  (void)rank;
}

/* Box of boxFaces hexahedra, each split into 6 Kuhn tetrahedra, built on rank 0 and distributed. */
static void create_box(DM *dm)
{
  DM          dmDist = NULL;
  PetscMPIInt rank;
  PetscInt    numCells = 0, numVertices = 0, *cells = NULL;
  PetscReal  *coords = NULL;

  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank == 0) {
    const PetscInt nx = boxFaces[0], ny = boxFaces[1], nz = boxFaces[2];
    const PetscInt perm[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
    numVertices = (nx + 1) * (ny + 1) * (nz + 1);
    numCells    = 6 * nx * ny * nz;
    PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(3 * numVertices, &coords));
    PetscCallAbort(PETSC_COMM_SELF, PetscMalloc1(4 * numCells, &cells));
    for (PetscInt k = 0; k <= nz; k++)
      for (PetscInt j = 0; j <= ny; j++)
        for (PetscInt i = 0; i <= nx; i++) {
          const PetscInt v = (k * (ny + 1) + j) * (nx + 1) + i;
          coords[3 * v + 0] = boxLower[0] + (boxUpper[0] - boxLower[0]) * i / nx;
          coords[3 * v + 1] = boxLower[1] + (boxUpper[1] - boxLower[1]) * j / ny;
          coords[3 * v + 2] = boxLower[2] + (boxUpper[2] - boxLower[2]) * k / nz;
        }
    PetscInt c = 0;
    for (PetscInt k = 0; k < nz; k++)
      for (PetscInt j = 0; j < ny; j++)
        for (PetscInt i = 0; i < nx; i++)
          for (PetscInt t = 0; t < 6; t++) {
            PetscInt ijk[3] = {i, j, k};
            for (PetscInt s = 0; s < 4; s++) {
              if (s > 0) ijk[perm[t][s - 1]]++;
              cells[4 * c + s] = (ijk[2] * (ny + 1) + ijk[1]) * (nx + 1) + ijk[0];
            }
            c++;
          }
  }
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexCreateFromCellListPetsc(PETSC_COMM_WORLD, 3, numCells, numVertices, 4, PETSC_TRUE, cells, 3, coords, dm));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(cells));
  PetscCallAbort(PETSC_COMM_WORLD, PetscFree(coords));
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexDistribute(*dm, 0, NULL, &dmDist));
  if (dmDist) {
    PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(dm));
    *dm = dmDist;
  }
}

/* Global counts of edges / faces / cells, total and on the "Boundary" label. */
static void count_entities(DM dm, PetscInt all[3], PetscInt bnd[3])
{
  IS              numbering;
  const PetscInt *gidx;
  DMLabel         label;
  PetscInt        pStart, local[6] = {0, 0, 0, 0, 0, 0}, global[6];

  PetscCallAbort(PETSC_COMM_WORLD, DMPlexCreatePointNumbering(dm, &numbering));
  PetscCallAbort(PETSC_COMM_WORLD, ISGetIndices(numbering, &gidx));
  PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetChart(dm, &pStart, NULL));
  PetscCallAbort(PETSC_COMM_WORLD, DMGetLabel(dm, "Boundary", &label));
  for (PetscInt depth = 1; depth <= 3; depth++) {
    PetscInt start, end;
    PetscCallAbort(PETSC_COMM_WORLD, DMPlexGetDepthStratum(dm, depth, &start, &end));
    for (PetscInt p = start; p < end; p++) {
      PetscInt value;
      if (gidx[p - pStart] < 0) continue;
      local[depth - 1]++;
      PetscCallAbort(PETSC_COMM_WORLD, DMLabelGetValue(label, p, &value));
      if (value == 100) local[3 + depth - 1]++;
    }
  }
  PetscCallAbort(PETSC_COMM_WORLD, ISRestoreIndices(numbering, &gidx));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&numbering));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(local, global, 6, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  for (int i = 0; i < 3; i++) { all[i] = global[i]; bnd[i] = global[3 + i]; }
}

static PetscInt global_dofs(DM dm)
{
  Vec      v;
  PetscInt n;
  PetscCallAbort(PETSC_COMM_WORLD, DMCreateGlobalVector(dm, &v));
  PetscCallAbort(PETSC_COMM_WORLD, VecGetSize(v, &n));
  PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&v));
  return n;
}

static void check_grid(PetscInt order, PetgemBoundaryCondition bc, const char *tag)
{
  DM           dm;
  Grid         grid;
  petgemParams params;
  PetscInt     all[3], bnd[3];

  PetscCallAbort(PETSC_COMM_WORLD, PetscMemzero(&params, sizeof(params)));
  params.order = order;
  create_box(&dm);
  PetscCallAbort(PETSC_COMM_WORLD, setupNedelecGrid(params, bc, &dm, &grid));
  PT_CHECK(grid.bc == bc, "[%s] order=%d: grid.bc not stored", tag, (int)order);

  count_entities(dm, all, bnd);
  const PetscInt pe = order, pf = order * (order - 1), pc = order * (order - 1) * (order - 2) / 2;
  const PetscInt total    = all[0] * pe + all[1] * pf + all[2] * pc;
  const PetscInt boundary = bnd[0] * pe + bnd[1] * pf;
  const PetscInt expected = (bc == PETGEM_BC_PEC) ? total - boundary : total;
  const PetscInt n        = global_dofs(dm);
  PT_CHECK(n == expected, "[%s] order=%d: %d global DOFs, expected %d", tag, (int)order, (int)n, (int)expected);

  /* Boundary faces */
  IS              faces;
  const PetscInt *fidx;
  PetscInt        nf, nfGlobal;
  PetscReal       sums[4] = {0, 0, 0, 0}, gsums[4];

  PetscCallAbort(PETSC_COMM_WORLD, getBoundaryFaces(dm, &faces));
  PetscCallAbort(PETSC_COMM_WORLD, ISGetLocalSize(faces, &nf));
  PetscCallAbort(PETSC_COMM_WORLD, ISGetIndices(faces, &fidx));
  for (PetscInt i = 0; i < nf; i++) {
    PetscInt  cell;
    PetscReal vtx[NUM_VERTICES_PER_FACE][NUM_DIMENSIONS], normal[NUM_DIMENSIONS], area;
    PetscCallAbort(PETSC_COMM_WORLD, computeBoundaryFaceGeometry(dm, fidx[i], &cell, vtx, normal, &area));

    PetscInt axis = 0;
    for (PetscInt d = 1; d < NUM_DIMENSIONS; d++) if (PetscAbsReal(normal[d]) > PetscAbsReal(normal[axis])) axis = d;
    PT_CLOSE(PetscAbsReal(normal[axis]), 1.0, 1e-12, "[%s] face %d: normal (%g,%g,%g) not axis-aligned", tag, (int)fidx[i], normal[0], normal[1], normal[2]);
    const PetscReal plane = normal[axis] > 0 ? boxUpper[axis] : boxLower[axis];
    for (PetscInt v = 0; v < NUM_VERTICES_PER_FACE; v++) {
      PT_CLOSE(vtx[v][axis], plane, 1e-12, "[%s] face %d: vertex off the outward box plane", tag, (int)fidx[i]);
    }
    PT_CHECK(area > 0.0, "[%s] face %d: area %g", tag, (int)fidx[i], area);
    for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) sums[d] += area * normal[d];
    sums[3] += area;
  }
  PetscCallAbort(PETSC_COMM_WORLD, ISRestoreIndices(faces, &fidx));
  PetscCallAbort(PETSC_COMM_WORLD, ISDestroy(&faces));

  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(&nf, &nfGlobal, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPIAbort(PETSC_COMM_WORLD, MPI_Allreduce(sums, gsums, 4, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));

  const PetscReal lx = boxUpper[0] - boxLower[0], ly = boxUpper[1] - boxLower[1], lz = boxUpper[2] - boxLower[2];
  const PetscInt  quads = 2 * (boxFaces[0] * boxFaces[1] + boxFaces[1] * boxFaces[2] + boxFaces[0] * boxFaces[2]);
  PT_CHECK(nfGlobal == bnd[1], "[%s] %d boundary faces returned, label has %d", tag, (int)nfGlobal, (int)bnd[1]);
  PT_CHECK(nfGlobal == 2 * quads, "[%s] %d boundary faces, expected %d", tag, (int)nfGlobal, (int)(2 * quads));
  PT_CLOSE(gsums[3], 2.0 * (lx * ly + ly * lz + lx * lz), 1e-12, "[%s] total boundary area %.16g", tag, gsums[3]);
  for (PetscInt d = 0; d < NUM_DIMENSIONS; d++) {
    PT_CLOSE(gsums[d], 0.0, 1e-12, "[%s] sum of n*area component %d = %.3e", tag, (int)d, gsums[d]);
  }

  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&grid.H1dm));
  PetscCallAbort(PETSC_COMM_WORLD, DMDestroy(&dm));
}

int main(int argc, char **argv)
{
  PetscMPIInt rank;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  const PetscInt order = (argc > 1) ? atoi(argv[1]) : 1;
  check_quadrature_2d(order, rank);
  check_grid(order, PETGEM_BC_NATURAL, "natural");
  check_grid(order, PETGEM_BC_PEC, "pec");

  int localFail = pt_failures, anyFail = 0;
  PetscCallMPI(MPI_Allreduce(&localFail, &anyFail, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));
  int rc = 0;
  if (rank == 0) rc = pt_report("test_boundary");
  else if (pt_failures) fprintf(stderr, "rank %d: %d failures\n", rank, pt_failures);
  PetscCall(PetscFinalize());
  return (rc || anyFail) ? 1 : 0;
}
