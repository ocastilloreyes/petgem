#!/usr/bin/env python3
# --------------------------------------------------------------------------
# mms_reference.py
#
# Reference definition of the manufactured solution used to verify the
# order of accuracy of the PETGEM fm.csem high-order Nedelec discretization.
#
# It is the single source of truth for:
#   * the exact field           E*(x)                       -> evaluate_E()
#   * the manufactured forcing   f*(x) = curlcurl E* - i w mu0 sigma E*
#                                                            -> evaluate_f()
#   * the exact norms of E* on [0,1]^3   (denominators of the relative errors)
#   * the C-ready component expressions for E* and f* (printed by --emit-c)
#
# The C kernel (fm.csem) must reproduce these two functions:
#   - evaluate_f() is what the volumetric-source assembly integrates against
#     the Nedelec basis to build the RHS,
#   - evaluate_E() (and its curl) is what the error-norm routine compares the
#     finite-element solution against, cell by cell, at quadrature points.
#
# Derivation (see the module self-test, run:  python3 mms_reference.py):
#   E*(x,y,z) = ( sin(pi y) sin(pi z),
#                 sin(pi z) sin(pi x),
#                 sin(pi x) sin(pi y) )
#   div E* = 0   and   curlcurl E* = 2 pi^2 E*
#   => f*(x) = (2 pi^2 - i w mu0 sigma) E*(x)          [a scalar multiple of E*]
#   E* has homogeneous tangential trace  n x E* = 0  on d[0,1]^3, so it is
#   consistent with the homogeneous Dirichlet BC fm.csem already applies
#   (DMPlexMarkBoundaryFaces): NO nonzero-BC code path is needed.
#
# Author: Octavio Castillo Reyes (UPC/BSC)
# --------------------------------------------------------------------------
import numpy as np

MU0 = 4.0e-7 * np.pi          # free-space magnetic permeability [H/m]
PI  = np.pi
TWO_PI2 = 2.0 * PI * PI       # curlcurl E* = 2 pi^2 E*   (the "stiffness" scalar)

# closed-form norms of E* on the unit cube [0,1]^3 (see self-test)
E_L2_NORM     = np.sqrt(0.75)                    # ||E*||_L2      = sqrt(3)/2
E_CURL_L2_SQ  = 1.5 * PI * PI                    # ||curl E*||^2  = 3 pi^2 / 2
E_HCURL_NORM  = np.sqrt(0.75 + E_CURL_L2_SQ)     # ||E*||_H(curl)


def omega_of(freq_hz):
    """Angular frequency omega = 2 pi f."""
    return 2.0 * PI * freq_hz


def evaluate_E(pts):
    """Exact field E*(x) at points `pts` (shape (N,3)).  Returns (N,3) complex."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    E = np.empty((pts.shape[0], 3), dtype=complex)
    E[:, 0] = np.sin(PI * y) * np.sin(PI * z)
    E[:, 1] = np.sin(PI * z) * np.sin(PI * x)
    E[:, 2] = np.sin(PI * x) * np.sin(PI * y)
    return E


def evaluate_curlE(pts):
    """Exact curl E*(x) at points `pts`.  Returns (N,3) complex.
       (curl E*)_x = pi sin(pi x) [cos(pi y) - cos(pi z)], cyclically."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    sx, sy, sz = np.sin(PI * x), np.sin(PI * y), np.sin(PI * z)
    cx, cy, cz = np.cos(PI * x), np.cos(PI * y), np.cos(PI * z)
    C = np.empty((pts.shape[0], 3), dtype=complex)
    C[:, 0] = PI * sx * (cy - cz)
    C[:, 1] = PI * sy * (cz - cx)
    C[:, 2] = PI * sz * (cx - cy)
    return C


def evaluate_f(pts, freq_hz, sigma=1.0):
    """Manufactured forcing f*(x) = (2 pi^2 - i w mu0 sigma) E*(x).

    This is the volumetric current density the RHS assembly integrates:
        b_j = sum_cells sum_q  w_q |J_cell| * ( N_j(x_q) . f*(x_q) ).
    """
    scal = TWO_PI2 - 1j * omega_of(freq_hz) * MU0 * sigma
    return scal * evaluate_E(pts)


def exact_norms():
    """Closed-form norms of E* on [0,1]^3, used as relative-error denominators."""
    return {
        "L2":     E_L2_NORM,          # sqrt(3)/2            ~ 0.8660254
        "curlL2": np.sqrt(E_CURL_L2_SQ),
        "Hcurl":  E_HCURL_NORM,       # sqrt(3/4 + 3 pi^2/2) ~ 3.9439075
    }


def relative_errors(pts, weights, Eh, curlEh):
    """Relative L2 and H(curl) errors from FE samples at quadrature points.

    Mirror of the C error routine, handy for validating it in pure Python:
      pts     (N,3)  physical quadrature points (all cells concatenated)
      weights (N,)   quadrature weights already multiplied by |J_cell|
      Eh      (N,3)  finite-element field E_h at those points
      curlEh  (N,3)  finite-element curl E_h at those points
    """
    weights = np.asarray(weights, float)
    Eex, Cex = evaluate_E(pts), evaluate_curlE(pts)
    dE, dC = Eh - Eex, curlEh - Cex
    l2_num   = np.sqrt(np.sum(weights * np.sum(np.abs(dE) ** 2, axis=1)))
    curl_num = np.sqrt(np.sum(weights * np.sum(np.abs(dC) ** 2, axis=1)))
    hcurl_num = np.sqrt(l2_num ** 2 + curl_num ** 2)
    n = exact_norms()
    return {"L2": l2_num / n["L2"], "Hcurl": hcurl_num / n["Hcurl"]}


C_TEMPLATE = r"""
/* --- manufactured solution E* and forcing f* (see mms_reference.py) --- */
/* curlcurl E* = 2*pi^2 E*, div E* = 0, n x E* = 0 on the unit-cube boundary */
static inline void mmsExactE(const PetscReal X[3], PetscScalar E[3]) {
    const PetscReal pi = PETSC_PI;
    E[0] = PetscSinReal(pi*X[1]) * PetscSinReal(pi*X[2]);
    E[1] = PetscSinReal(pi*X[2]) * PetscSinReal(pi*X[0]);
    E[2] = PetscSinReal(pi*X[0]) * PetscSinReal(pi*X[1]);
}
static inline void mmsForcingF(const PetscReal X[3], PetscReal omega,
                               PetscReal sigma, PetscScalar F[3]) {
    const PetscReal pi = PETSC_PI;
    const PetscScalar s = (PetscScalar)(2.0*pi*pi) - PETSC_i*(omega*MU*sigma);
    PetscScalar E[3]; mmsExactE(X, E);
    F[0] = s*E[0]; F[1] = s*E[1]; F[2] = s*E[2];
}
"""


def _selftest():
    """Numerically confirm the closed-form forcing and norms."""
    import sympy as sp
    x, y, z = sp.symbols('x y z', real=True)
    a = sp.pi
    E = sp.Matrix([sp.sin(a*y)*sp.sin(a*z), sp.sin(a*z)*sp.sin(a*x), sp.sin(a*x)*sp.sin(a*y)])
    curl = lambda F: sp.Matrix([sp.diff(F[2], y)-sp.diff(F[1], z),
                                sp.diff(F[0], z)-sp.diff(F[2], x),
                                sp.diff(F[1], x)-sp.diff(F[0], y)])
    assert sp.simplify(sp.diff(E[0], x)+sp.diff(E[1], y)+sp.diff(E[2], z)) == 0, "div E* != 0"
    assert sp.simplify(curl(curl(E)) - 2*a**2*E) == sp.zeros(3, 1), "curlcurl E* != 2 pi^2 E*"
    L2sq = sum(sp.integrate(E[i]**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)) for i in range(3))
    cE = curl(E)
    curlsq = sum(sp.integrate(cE[i]**2, (x, 0, 1), (y, 0, 1), (z, 0, 1)) for i in range(3))
    assert abs(float(L2sq) - E_L2_NORM**2) < 1e-12
    assert abs(float(curlsq) - E_CURL_L2_SQ) < 1e-9
    # cross-check the numpy path against a random point
    p = np.array([[0.31, 0.62, 0.17]])
    fnp = evaluate_f(p, freq_hz=1.0, sigma=1.0)[0]
    scal = complex(2*float(a)**2 - 1j*omega_of(1.0)*MU0*1.0)
    Esym = np.array([complex(E[i].subs({x: 0.31, y: 0.62, z: 0.17})) for i in range(3)])
    assert np.allclose(fnp, scal*Esym), "evaluate_f mismatch vs sympy"
    print("[selftest] div E* = 0                                 OK")
    print("[selftest] curlcurl E* = 2*pi^2 E*  =>  f* = (2pi^2 - i w mu0 sigma) E*  OK")
    print(f"[selftest] ||E*||_L2     = {E_L2_NORM:.10f}   (= sqrt(3)/2)")
    print(f"[selftest] ||curl E*||   = {np.sqrt(E_CURL_L2_SQ):.10f}   (= sqrt(3/2) pi)")
    print(f"[selftest] ||E*||_Hcurl  = {E_HCURL_NORM:.10f}")
    print("[selftest] evaluate_f numpy path matches sympy        OK")


if __name__ == "__main__":
    import sys
    if "--emit-c" in sys.argv:
        print(C_TEMPLATE)
    else:
        _selftest()
