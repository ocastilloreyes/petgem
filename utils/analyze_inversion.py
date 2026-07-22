#!/usr/bin/env python3
"""
PETGEM inversion-result analyzer - general CLI utility.

Evaluates a recovered resistivity model produced by im.csem against a
reference description of the true model, and reports the standard inversion
diagnostics:

    - initial / final data misfit (RMS) and termination reason
    - background resistivity
    - peak (minimum) resistivity of the recovered conductor
    - conductor centroid, and its offset from the true centroid
    - conductor volume
    - PASS / FAIL against documented tolerances

The tool is problem-independent: the true model, the anomaly search box and
the acceptance tolerances are read from a reference JSON, so it works for any
CSEM inversion example (see examples/im_model/reference/reference_metrics.json).

Inputs it reads from the run directory:
    responses_im*.h5                     -> rms_history, convergence attributes
    *_iter<NNNNN>_p<RRRR>.vtu            -> recovered model with geometry
                                            (rho cell field); the highest
                                            available iteration is used.

Usage:
    python3 utils/analyze_inversion.py \\
        -run_dir   examples/im_model/outputs \\
        -reference examples/im_model/reference/reference_metrics.json
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None


def _txt(da):
    return np.fromstring(da.text.replace("\n", " "), sep=" ")


def parse_vtu_model(run_dir):
    """Combine all VTU pieces of the highest-iteration snapshot into
    (centroid_xyz, rho, volume) arrays over every cell."""
    vtus = glob.glob(os.path.join(run_dir, "*_iter*_p*.vtu"))
    if not vtus:
        sys.exit(f"ERROR: no VTU snapshots (*_iter*_p*.vtu) found in {run_dir}")
    iters = {int(re.search(r"_iter0*(\d+)_p", f).group(1)) for f in vtus}
    last = max(iters)
    pieces = sorted(f for f in vtus if re.search(rf"_iter0*{last}_p", f))
    cx, cy, cz, rho, vol = [], [], [], [], []
    for fn in pieces:
        piece = ET.parse(fn).getroot().find(".//Piece")
        pts = _txt(piece.find("./Points/DataArray")).reshape(-1, 3)
        conn = offs = rr = None
        for da in piece.find("./Cells"):
            if da.get("Name") == "connectivity":
                conn = _txt(da).astype(int)
            elif da.get("Name") == "offsets":
                offs = _txt(da).astype(int)
        for da in piece.find("./CellData"):
            if da.get("Name") == "rho":
                rr = _txt(da)
        start = 0
        for k, off in enumerate(offs):
            v = pts[conn[start:off]]
            start = off
            c = v.mean(axis=0)
            cx.append(c[0]); cy.append(c[1]); cz.append(c[2])
            rho.append(rr[k])
            vol.append(abs(np.dot(v[1] - v[0],
                                  np.cross(v[2] - v[0], v[3] - v[0]))) / 6.0)
    return (np.array(cx), np.array(cy), np.array(cz),
            np.array(rho), np.array(vol), last)


def read_convergence(run_dir):
    """Return (rms0, rms_final, iterations, reason) from responses_im*.h5."""
    if h5py is None:
        return (None, None, None, None)
    h5s = sorted(glob.glob(os.path.join(run_dir, "responses_im*.h5")))
    if not h5s:
        return (None, None, None, None)
    with h5py.File(h5s[0], "r") as h:
        rms = np.asarray(h["rms_history"])[:, 0] if "rms_history" in h else None
        it = h.attrs.get("num_iterations")
        reason = h.attrs.get("convergence_reason")
        if isinstance(reason, bytes):
            reason = reason.decode()
    if rms is None:
        return (None, None, it, reason)
    return (float(rms[0]), float(rms[-1]), it, reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-run_dir", required=True,
                    help="Directory with responses_im*.h5 and VTU snapshots.")
    ap.add_argument("-reference", required=True,
                    help="Reference metrics JSON.")
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    tm = ref["true_model"]
    box = tm["anomaly_box_m"]
    thr = ref.get("conductor_threshold_ohm_m", 30.0)
    tol = ref.get("tolerances", {})
    true_c = np.array(tm["anomaly_centroid_m"], dtype=float)

    cx, cy, cz, rho, vol, it_used = parse_vtu_model(args.run_dir)
    rms0, rmsf, iters, reason = read_convergence(args.run_dir)

    earth = cz > 1.0
    inbox = ((cx >= box["x"][0]) & (cx <= box["x"][1]) &
             (cy >= box["y"][0]) & (cy <= box["y"][1]) &
             (cz >= box["z"][0]) & (cz <= box["z"][1]))
    cond = earth & (rho < thr)

    bg_med = float(np.median(rho[earth]))
    rho_min = float(rho[earth].min())
    if cond.sum():
        w = vol[cond] / rho[cond]
        rc = np.array([np.average(cx[cond], weights=w),
                       np.average(cy[cond], weights=w),
                       np.average(cz[cond], weights=w)])
        cond_vol = float(vol[cond].sum())
        frac_box = float(vol[cond & inbox].sum() / cond_vol)
    else:
        rc, cond_vol, frac_box = np.array([np.nan] * 3), 0.0, 0.0
    lat_off = float(np.hypot(rc[0] - true_c[0], rc[1] - true_c[1]))
    ver_off = float(rc[2] - true_c[2])

    checks = []

    def chk(name, ok, detail):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:26s} {detail}")

    print("=" * 66)
    print(f"  Inversion analysis - {ref.get('name', args.reference)}")
    print(f"  recovered model: VTU snapshot iter {it_used}, {len(rho)} cells")
    print("=" * 66)

    print("\nConvergence")
    if rmsf is not None:
        print(f"    initial RMS      = {rms0:.3f}")
        print(f"    final   RMS      = {rmsf:.4f}   (target {tol.get('final_rms_max','-')})")
        print(f"    iterations       = {iters}")
        print(f"    termination      = {reason}")
    else:
        print("    (no responses_im*.h5 with rms_history found - skipped)")

    print("\nRecovered model")
    print(f"    true model       : background {tm['background_ohm_m']} ohm.m,"
          f" anomaly {tm['anomaly_ohm_m']} ohm.m at {list(true_c)}")
    print(f"    background median = {bg_med:.1f} ohm.m")
    print(f"    peak (min) rho    = {rho_min:.2f} ohm.m")
    print(f"    conductor centroid= ({rc[0]:.0f}, {rc[1]:.0f}, {rc[2]:.0f})")
    print(f"    lateral offset    = {lat_off:.1f} m")
    print(f"    vertical offset   = {ver_off:.1f} m")
    print(f"    conductor volume  = {cond_vol:.3e} m^3  (rho < {thr:g} ohm.m)")
    print(f"    fraction in box   = {frac_box*100:.0f}%")

    print("\nAcceptance checks")
    if rmsf is not None and "final_rms_max" in tol:
        chk("final RMS <= target", rmsf <= tol["final_rms_max"] + 1e-6,
            f"{rmsf:.4f} <= {tol['final_rms_max']}")
    if "background_ohm_m_pct" in tol:
        p = abs(bg_med - tm["background_ohm_m"]) / tm["background_ohm_m"] * 100
        chk("background resistivity", p <= tol["background_ohm_m_pct"],
            f"{bg_med:.1f} ohm.m ({p:.1f}% off {tm['background_ohm_m']})")
    if "min_resistivity_ohm_m_range" in tol:
        lo, hi = tol["min_resistivity_ohm_m_range"]
        chk("peak resistivity", lo <= rho_min <= hi,
            f"{rho_min:.2f} in [{lo}, {hi}] ohm.m")
    if "lateral_offset_m_max" in tol:
        chk("lateral localization", lat_off <= tol["lateral_offset_m_max"],
            f"{lat_off:.1f} m <= {tol['lateral_offset_m_max']} m")
    if "volume_m3_range" in tol:
        lo, hi = tol["volume_m3_range"]
        chk("conductor volume", lo <= cond_vol <= hi,
            f"{cond_vol:.2e} in [{lo:.1e}, {hi:.1e}] m^3")

    ok = all(checks)
    print("\n" + "=" * 66)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  ({sum(checks)}/{len(checks)} checks)")
    print("=" * 66)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
