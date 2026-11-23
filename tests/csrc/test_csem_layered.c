/*
 * Filename: test_csem_layered.c
 * Author: PETGEM test suite
 * Date: 2026-11-20
 *
 * Description:
 * Fields of an electric point dipole in a layered VTI medium (csem_layered.c):
 *   - E and H against empymod (csem_layered_reference.h) for marine, land and
 *     whole-space models;
 *   - one layer against the closed-form whole-space dipole;
 *   - four layers of equal conductivity against one layer;
 *   - reciprocity E_a(x; e_b at x0) = E_b(x0; e_a at x) across layers;
 *   - points on an interface evaluated in the layers above and below:
 *     equal tangential E and H, continuous sigma_v E_z;
 *   - findLayer and findLayerOnSide at and between interfaces.
 *
 * Usage: test_csem_layered   Exit 0 iff all checks pass.
 */
#include "constants.h"
#include "csem_layered.h"
#include "petgem_test.h"
#include <petsc.h>

#include "csem_layered_reference.h"

/* max_d |a_d - b_d| / max_d |b_d| over the three components. */
static PetscReal relativeError(const PetscScalar a[3], const PetscScalar b[3]) {
  PetscReal err = 0.0, scale = 0.0;
  for (int d = 0; d < 3; d++) {
    err   = PetscMax(err, PetscAbsScalar(a[d] - b[d]));
    scale = PetscMax(scale, PetscAbsScalar(b[d]));
  }
  return err / scale;
}

/* E and H at points of the dipole (position, moment) in a model. */
static void dipoleFields(const LayeredModel *model, PetscReal freq, const PetscReal position[3], const PetscReal moment[3], PetscInt n,
                         const PetscReal *points, PetscScalar *E, PetscScalar *H) {
  LayeredDipole dipole;
  PetscCallAbort(PETSC_COMM_SELF, createLayeredDipole(model, freq, position, moment, &dipole));
  PetscCallAbort(PETSC_COMM_SELF, evaluateLayeredDipole(&dipole, n, points, NULL, E, H));
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredDipole(&dipole));
}

/* E and H of the dipole p at xs in a whole space of conductivity sigma, e^{-iwt}. */
static void wholeSpace(PetscReal sigma, PetscReal freq, const PetscReal xs[3], const PetscReal p[3], const PetscReal x[3], PetscScalar E[3],
                       PetscScalar H[3]) {
  const PetscReal   w = angularFrequency(freq);
  const PetscScalar k = PetscSqrtScalar(PETSC_i * w * MU * sigma);
  PetscReal         R[3], u[3], rr = 0.0, up = 0.0;

  for (int d = 0; d < 3; d++) {
    R[d] = x[d] - xs[d];
    rr += R[d] * R[d];
  }
  rr = PetscSqrtReal(rr);
  for (int d = 0; d < 3; d++) {
    u[d] = R[d] / rr;
    up += u[d] * p[d];
  }
  const PetscScalar g = PetscExpScalar(PETSC_i * k * rr) / (4.0 * PETSC_PI * rr), kr = k * rr;
  const PetscScalar a = 1.0 + PETSC_i / kr - 1.0 / (kr * kr), b = -1.0 - 3.0 * PETSC_i / kr + 3.0 / (kr * kr);
  const PetscReal   uxp[3] = {u[1] * p[2] - u[2] * p[1], u[2] * p[0] - u[0] * p[2], u[0] * p[1] - u[1] * p[0]};
  for (int d = 0; d < 3; d++) {
    E[d] = PETSC_i * w * MU * g * (a * p[d] + b * up * u[d]);
    H[d] = (PETSC_i * k - 1.0 / rr) * g * uxp[d];
  }
}

static void check_reference(void) {
  for (size_t c = 0; c < sizeof(layeredReference) / sizeof(layeredReference[0]); c++) {
    const LayeredReference *ref = &layeredReference[c];
    const PetscInt          nl = ref->numLayers, np = ref->numPoints;
    PetscScalar             E[LR_MAX_POINTS * 3], H[LR_MAX_POINTS * 3];
    LayeredModel            model;

    PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(nl, ref->interfaces, ref->sigmaH, ref->sigmaV, &model));
    dipoleFields(&model, ref->freq, ref->position, ref->moment, np, &ref->points[0][0], E, H);
    for (PetscInt q = 0; q < np; q++) {
      PetscScalar Eref[3], Href[3];
      for (int d = 0; d < 3; d++) {
        Eref[d] = ref->E[q][d][0] + PETSC_i * ref->E[q][d][1];
        Href[d] = ref->H[q][d][0] + PETSC_i * ref->H[q][d][1];
      }
      const PetscReal eE = relativeError(&E[q * 3], Eref), eH = relativeError(&H[q * 3], Href);
      PT_CHECK(eE < 1e-7, "[%s] point %d: E relative error %.3e against empymod", ref->name, (int)q, eE);
      PT_CHECK(eH < 1e-7, "[%s] point %d: H relative error %.3e against empymod", ref->name, (int)q, eH);
    }
    PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&model));
  }
}

static void check_whole_space(void) {
  const PetscReal sigma = 0.3, freq = 3.0;
  const PetscReal s1[1] = {sigma}, s4[4] = {sigma, sigma, sigma, sigma}, interfaces4[3] = {-100.0, -300.0, -700.0};
  const PetscReal xs[3] = {20.0, -10.0, -250.0}, p[3] = {0.3, -0.8, 0.5};
  const PetscReal points[7][3] = {{520.0, 290.0, -250.0}, {20.3, -9.8, -400.0},     {20.0, -10.0, -150.0}, {-300.0, 50.0, 40.0},
                                  {140.0, -60.0, -320.0}, {700.0, -900.0, -1200.0}, {21.0, -10.0, -251.0}};
  PetscScalar     E1[21], H1[21], E4[21], H4[21];
  LayeredModel    one, four;

  PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(1, NULL, s1, s1, &one));
  PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(4, interfaces4, s4, s4, &four));
  dipoleFields(&one, freq, xs, p, 7, &points[0][0], E1, H1);
  dipoleFields(&four, freq, xs, p, 7, &points[0][0], E4, H4);
  for (int q = 0; q < 7; q++) {
    PetscScalar E[3], H[3];
    wholeSpace(sigma, freq, xs, p, points[q], E, H);
    /* Key (2009) filter: relative error about 1e-5 at the source depth */
    const PetscReal tol = (points[q][2] == xs[2]) ? 2e-5 : 1e-6;
    const PetscReal eE = relativeError(&E1[q * 3], E), eH = relativeError(&H1[q * 3], H);
    PT_CHECK(eE < tol, "[whole space] point %d: E relative error %.3e against the closed form", q, eE);
    PT_CHECK(eH < tol, "[whole space] point %d: H relative error %.3e against the closed form", q, eH);
    const PetscReal fE = relativeError(&E4[q * 3], &E1[q * 3]), fH = relativeError(&H4[q * 3], &H1[q * 3]);
    PT_CHECK(fE < 1e-12, "[equal layers] point %d: E relative difference %.3e to one layer", q, fE);
    PT_CHECK(fH < 1e-12, "[equal layers] point %d: H relative difference %.3e to one layer", q, fH);
  }
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&one));
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&four));
}

static void check_reciprocity(void) {
  const PetscReal interfaces[4] = {0.0, -1000.0, -2000.0, -2100.0};
  const PetscReal sh[5] = {1e-8, 3.3333, 1.0, 0.01, 1.0}, sv[5] = {1e-8, 3.3333, 0.5, 0.01, 0.5};
  const PetscReal x0[4][3] = {{0.0, 0.0, -975.0}, {0.0, 0.0, -975.0}, {100.0, -50.0, -1500.0}, {0.0, 0.0, -10.0}};
  const PetscReal x1[4][3] = {{1200.0, 500.0, -2050.0}, {800.0, -300.0, 30.0}, {-600.0, 900.0, -2600.0}, {300.0, 200.0, -1990.0}};
  LayeredModel    model;

  PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(5, interfaces, sh, sv, &model));
  for (int c = 0; c < 4; c++) {
    PetscScalar Gx[3][3], Gy[3][3]; /* Gx[a][b] = E_a(x1; e_b at x0), Gy[a][b] = E_a(x0; e_b at x1) */
    PetscReal   err = 0.0, scale = 0.0;
    for (int b = 0; b < 3; b++) {
      const PetscReal p[3] = {b == 0, b == 1, b == 2};
      PetscScalar     E[3];
      dipoleFields(&model, 1.0, x0[c], p, 1, x1[c], E, NULL);
      for (int a = 0; a < 3; a++) Gx[a][b] = E[a];
      dipoleFields(&model, 1.0, x1[c], p, 1, x0[c], E, NULL);
      for (int a = 0; a < 3; a++) Gy[a][b] = E[a];
    }
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        err   = PetscMax(err, PetscAbsScalar(Gx[a][b] - Gy[b][a]));
        scale = PetscMax(scale, PetscAbsScalar(Gx[a][b]));
      }
    PT_CHECK(err < 1e-8 * scale, "[reciprocity] pair %d: max |G_ab(x1, x0) - G_ba(x0, x1)| / max |G| = %.3e", c, err / scale);
  }
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&model));
}

static void check_find_layer(void) {
  const PetscReal interfaces[2] = {0.0, -100.0}, s[3] = {1e-8, 0.1, 0.01};
  const PetscReal z[6]     = {10.0, 0.0, -50.0, -100.0, -100.5, -1e6};
  const PetscInt  layer[6] = {0, 0, 1, 1, 2, 2};
  LayeredModel    model;

  PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(3, interfaces, s, s, &model));
  for (int i = 0; i < 6; i++)
    PT_CHECK(findLayer(&model, z[i]) == layer[i], "findLayer(z = %g) = %d, expected %d", z[i], (int)findLayer(&model, z[i]), (int)layer[i]);
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&model));
}

static void check_interface_side(void) {
  const PetscReal interfaces[2] = {0.0, -100.0}, sh[3] = {1e-8, 0.1, 0.02}, sv[3] = {1e-8, 0.05, 0.02};
  const PetscReal xs[3] = {0.0, 0.0, -40.0}, p[3] = {0.6, 0.3, 0.74}, freq = 1.0;
  const PetscReal points[2][3] = {{350.0, -120.0, -100.0}, {-200.0, 260.0, 0.0}};
  const PetscInt  above[2] = {1, 0}, below[2] = {2, 1};
  PetscScalar     Ea[6], Ha[6], Eb[6], Hb[6], E0[6], H0[6];
  LayeredModel    model;
  LayeredDipole   dipole;
  PetscErrorCode  ierr;

  PetscCallAbort(PETSC_COMM_SELF, createLayeredModel(3, interfaces, sh, sv, &model));
  PetscCallAbort(PETSC_COMM_SELF, createLayeredDipole(&model, freq, xs, p, &dipole));
  PetscCallAbort(PETSC_COMM_SELF, evaluateLayeredDipole(&dipole, 2, &points[0][0], above, Ea, Ha));
  PetscCallAbort(PETSC_COMM_SELF, evaluateLayeredDipole(&dipole, 2, &points[0][0], below, Eb, Hb));
  PetscCallAbort(PETSC_COMM_SELF, evaluateLayeredDipole(&dipole, 2, &points[0][0], NULL, E0, H0));
  for (int q = 0; q < 2; q++) {
    const PetscScalar *ea = &Ea[3 * q], *eb = &Eb[3 * q];
    const PetscReal    nt = PetscMax(PetscAbsScalar(ea[0]), PetscAbsScalar(ea[1]));
    const PetscReal    eT = PetscMax(PetscAbsScalar(ea[0] - eb[0]), PetscAbsScalar(ea[1] - eb[1])) / nt;
    const PetscReal    eH = relativeError(&Ha[3 * q], &Hb[3 * q]);
    const PetscReal    eJ = PetscAbsScalar(sv[above[q]] * ea[2] - sv[below[q]] * eb[2]) / PetscAbsScalar(sv[below[q]] * eb[2]);
    PetscPrintf(PETSC_COMM_SELF, "  interface point %d: tangential E %.3e, H %.3e, sigma_v E_z %.3e\n", q, (double)eT, (double)eH,
                (double)eJ);
    PT_CHECK(eT < 1e-8, "[interface] point %d: tangential E differs by %.3e between the layers above and below", q, eT);
    PT_CHECK(eH < 1e-8, "[interface] point %d: H differs by %.3e between the layers above and below", q, eH);
    PT_CHECK(eJ < 1e-8, "[interface] point %d: sigma_v E_z differs by %.3e between the layers above and below", q, eJ);
    PT_CHECK(relativeError(&E0[3 * q], ea) == 0.0 && relativeError(&H0[3 * q], &Ha[3 * q]) == 0.0,
             "[interface] point %d: layers = NULL differs from the layer above", q);
  }

  {
    const PetscReal x[3]     = {300.0, 0.0, -50.0};
    const PetscInt  wrong[1] = {2};
    PetscScalar     E[3];
    PetscCallAbort(PETSC_COMM_SELF, PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
    ierr = evaluateLayeredDipole(&dipole, 1, x, wrong, E, NULL);
    PetscCallAbort(PETSC_COMM_SELF, PetscPopErrorHandler());
    PT_CHECK(ierr != PETSC_SUCCESS, "[interface] point outside the given layer not rejected");
  }
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredDipole(&dipole));

  {
    const PetscReal z[7]     = {-100.0, -100.0, -100.0 + 1e-7, -50.0, 0.0, 0.0, -1e6};
    const PetscReal zSide[7] = {-150.0, -50.0, -150.0, -150.0, 10.0, -10.0, 10.0};
    const PetscInt  layer[7] = {2, 1, 2, 1, 0, 1, 2};
    for (int i = 0; i < 7; i++)
      PT_CHECK(findLayerOnSide(&model, z[i], zSide[i]) == layer[i], "findLayerOnSide(z = %g, zSide = %g) = %d, expected %d", z[i], zSide[i],
               (int)findLayerOnSide(&model, z[i], zSide[i]), (int)layer[i]);
  }
  PetscCallAbort(PETSC_COMM_SELF, destroyLayeredModel(&model));
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  check_reference();
  check_whole_space();
  check_reciprocity();
  check_find_layer();
  check_interface_side();

  int rc = pt_report("test_csem_layered");
  PetscCall(PetscFinalize());
  return rc;
}
