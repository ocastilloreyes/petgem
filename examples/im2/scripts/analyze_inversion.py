#!/usr/bin/env python3
"""
PETGEM inversion-result analyzer, MULTI-BODY - general CLI utility.

The companion of ``examples/im1/scripts/analyze_inversion.py``, for true
models made of SEVERAL targets that may be conductive *or* resistive relative
to the background - a checkerboard, a block pair, a layered set of bodies. The
im1 analyzer assumes one conductor and reduces the whole earth to a single
``rho < thr`` population, which merges neighbouring targets and cannot see a
resistor at all.

Reported per body, inside that body's own search region:

    - peak resistivity (minimum for a conductor, maximum for a resistor)
    - anomaly centroid, and its lateral / vertical offset from the true centroid
    - anomaly volume and the fraction of it inside the true body box
    - polarity: whether the region recovered the anomaly with the right sign

Reported once for the model:

    - initial / final data misfit (RMS), step counts and termination reason
    - background resistivity (median over the earth)
    - PASS / FAIL against the documented tolerances

The tool is problem-independent: the bodies, their search regions, the
detection thresholds and the acceptance tolerances all come from a reference
JSON (see examples/im2/reference/reference_metrics.json), so it works for any
multi-target CSEM inversion example.

Reference JSON schema (only the keys this tool reads)::

    {
      "true_model": {
        "background_ohm_m": 100.0,
        "bodies": [
          {"name": "C1",
           "polarity": "conductor",          # or "resistor"
           "resistivity_ohm_m": 10.0,
           "box_m":        {"x": [...], "y": [...], "z": [...]},
           "search_box_m": {"x": [...], "y": [...], "z": [...]}}
        ]
      },
      "conductor_threshold_ohm_m": 30.0,     # rho below this = recovered conductor
      "resistor_threshold_ohm_m": 300.0,     # rho above this = recovered resistor
      "tolerances": {
        "final_rms_max": 1.05,
        "background_ohm_m_pct": 2.0,
        "lateral_offset_m_max": 100.0,             # global default
        "bodies": {"C1": {"mean_ohm_m_range": [...],   # carries the amplitude check
                          "peak_ohm_m_range": [...],
                          "volume_m3_range": [...],
                          "lateral_offset_m_max": 40.0}}   # overrides the global
      }
    }

``search_box_m`` is what separates the bodies: give each one a region that
contains it and no other, so a smeared reconstruction is still attributed to
the right target. For a checkerboard the natural choice is the quadrant.

Inputs it reads from the run directory are the same as the im1 analyzer -
the .h5 carrying /rms_history, and snapshots/iter*_r*.vtu.

Usage:
    python3 examples/im2/scripts/analyze_inversion.py \\
        -run_dir   examples/im2/outputs \\
        -reference examples/im2/reference/reference_metrics.json
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
    # Current layout is run_dir/snapshots/iterNNNN_rRRRR.vtu. Two earlier ones
    # exist in the wild - <stem>_snapshots/model_iterNNNN_rRRRR.vtu, and
    # <stem>_iterNNNNN_pRRRR.vtu flat in run_dir - so accept all three and let
    # an existing results directory still analyse.
    vtus = (glob.glob(os.path.join(run_dir, "snapshots", "iter*_r*.vtu"))
            + glob.glob(os.path.join(run_dir, "*_snapshots", "*iter*_r*.vtu"))
            + glob.glob(os.path.join(run_dir, "*_iter*_p*.vtu")))
    if not vtus:
        sys.exit(f"ERROR: no VTU snapshots found under {run_dir} "
                 f"(looked for snapshots/iter*_r*.vtu and the two older layouts)")
    # No leading underscore in the pattern: it must match "iter0103_r0000.vtu"
    # as well as "..._iter00096_p0110.vtu".
    rank_re = re.compile(r"iter0*(\d+)_[pr]\d+\.vtu$")
    iters = {int(rank_re.search(f).group(1)) for f in vtus}
    last = max(iters)
    pieces = sorted(f for f in vtus
                    if int(rank_re.search(f).group(1)) == last)
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
    """Return (rms0, rms_final, iterations, evaluations, reason).

    Two distinct counters, and conflating them is the classic mistake here:
    ``iterations`` is the accepted-L-BFGS-step count, ``evaluations`` is the
    number of objective-gradient evaluations. The latter is larger by the
    rejected line-search trials, and it is what ``rms_history`` is indexed by -
    which is also why that series is not monotone.

    Files written before ``num_objgrad_evaluations`` existed stored the
    evaluation count in ``num_iterations``; there the step count is
    unrecoverable, so it is reported as None rather than guessed.
    """
    if h5py is None:
        return (None, None, None, None, None)
    # Identify the result by content, not by name: an inversion result is the
    # HDF5 in run_dir that carries /rms_history. Matching on "responses_im*.h5"
    # instead would silently tie -output_filename to a fixed prefix.
    hit = None
    for cand in sorted(glob.glob(os.path.join(run_dir, "*.h5"))):
        try:
            with h5py.File(cand, "r") as h:
                if "rms_history" in h:
                    hit = cand
                    break
        except OSError:
            continue          # not readable HDF5, or busy: not our result file
    if hit is None:
        return (None, None, None, None, None)
    with h5py.File(hit, "r") as h:
        rms = np.asarray(h["rms_history"])[:, 0] if "rms_history" in h else None
        it = h.attrs.get("num_iterations")
        ev = h.attrs.get("num_objgrad_evaluations")
        reason = h.attrs.get("convergence_reason")
        if isinstance(reason, bytes):
            reason = reason.decode()
    if ev is None:          # legacy file: num_iterations actually held evaluations
        it, ev = None, it
    if rms is None:
        return (None, None, it, ev, reason)
    return (float(rms[0]), float(rms[-1]), it, ev, reason)


def _inside(cx, cy, cz, box):
    """Boolean mask of the cells whose centroid falls in an axis-aligned box."""
    return ((cx >= box["x"][0]) & (cx <= box["x"][1]) &
            (cy >= box["y"][0]) & (cy <= box["y"][1]) &
            (cz >= box["z"][0]) & (cz <= box["z"][1]))


def _body_metrics(body, cx, cy, cz, rho, vol, earth, thr_cond, thr_res):
    """Diagnostics for one body: peak, centroid, offsets, volume, polarity.

    The anomaly cells are those in the body's search region that cross the
    detection threshold on the body's own side of the background. Their
    centroid is weighted by volume x contrast, which is 1/rho for a conductor
    and rho for a resistor - in both cases "more anomalous" pulls harder.
    """
    resistor = body["polarity"] == "resistor"
    search = earth & _inside(cx, cy, cz, body["search_box_m"])
    truth = _inside(cx, cy, cz, body["box_m"])
    out = {"name": body["name"], "polarity": body["polarity"],
           "true_rho": float(body["resistivity_ohm_m"]),
           "true_centroid": np.array([np.mean(body["box_m"][a])
                                      for a in ("x", "y", "z")])}
    if not search.any():
        out.update(peak=np.nan, mean=np.nan, centroid=np.full(3, np.nan),
                   volume=0.0, frac_box=0.0, lat_off=np.nan, ver_off=np.nan,
                   detected=False)
        return out

    r = rho[search]
    out["peak"] = float(r.max() if resistor else r.min())
    # Volume-weighted geometric mean over the TRUE body box. Unlike the peak -
    # a single-cell extremum that can overshoot past the true value - this is
    # what actually describes the body, so it carries the amplitude tolerance.
    out["mean"] = float(10.0 ** np.average(np.log10(rho[truth]),
                                           weights=vol[truth])) if truth.any() else np.nan
    hit = search & ((rho > thr_res) if resistor else (rho < thr_cond))
    out["detected"] = bool(hit.any())
    if hit.any():
        w = vol[hit] * (rho[hit] if resistor else 1.0 / rho[hit])
        c = np.array([np.average(cx[hit], weights=w),
                      np.average(cy[hit], weights=w),
                      np.average(cz[hit], weights=w)])
        out["centroid"] = c
        out["volume"] = float(vol[hit].sum())
        out["frac_box"] = float(vol[hit & truth].sum() / out["volume"])
        out["lat_off"] = float(np.hypot(c[0] - out["true_centroid"][0],
                                        c[1] - out["true_centroid"][1]))
        out["ver_off"] = float(c[2] - out["true_centroid"][2])
    else:
        out.update(centroid=np.full(3, np.nan), volume=0.0, frac_box=0.0,
                   lat_off=np.nan, ver_off=np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-run_dir", required=True,
                    help="Run directory: the inversion .h5 plus its snapshots/ pieces.")
    ap.add_argument("-reference", required=True,
                    help="Reference metrics JSON (multi-body schema).")
    args = ap.parse_args()

    ref = json.load(open(args.reference))
    tm = ref["true_model"]
    if "bodies" not in tm:
        sys.exit("ERROR: this reference has no true_model.bodies - it describes a "
                 "single anomaly, so use examples/im1/scripts/analyze_inversion.py instead.")
    bodies = tm["bodies"]
    thr_cond = ref.get("conductor_threshold_ohm_m", 30.0)
    thr_res = ref.get("resistor_threshold_ohm_m", 300.0)
    tol = ref.get("tolerances", {})
    btol = tol.get("bodies", {})

    cx, cy, cz, rho, vol, it_used = parse_vtu_model(args.run_dir)
    rms0, rmsf, iters, evals, reason = read_convergence(args.run_dir)

    # Earth cells only, excluding a thin band around the interface. Depth is a
    # NEGATIVE z (the convention both examples use), so the earth is z < 0.
    earth = cz < -1.0
    bg_med = float(np.median(rho[earth]))
    res = [_body_metrics(b, cx, cy, cz, rho, vol, earth, thr_cond, thr_res)
           for b in bodies]

    checks = []

    def chk(name, ok, detail):
        checks.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:30s} {detail}")

    print("=" * 74)
    print(f"  Inversion analysis - {ref.get('name', args.reference)}")
    print(f"  recovered model: VTU snapshot iter {it_used}, {len(rho)} cells")
    print("=" * 74)

    print("\nConvergence")
    if rmsf is not None:
        print(f"    initial RMS      = {rms0:.3f}")
        print(f"    final   RMS      = {rmsf:.4f}   (target {tol.get('final_rms_max','-')})")
        if iters is not None:
            print(f"    L-BFGS steps     = {iters}   (accepted)")
        else:
            print("    L-BFGS steps     = n/a  (legacy file: only the evaluation count was stored)")
        print(f"    objgrad evals    = {evals}   (indexes rms_history)")
        print(f"    termination      = {reason}")
    else:
        print("    (no .h5 with rms_history found in run_dir - skipped)")

    print("\nBackground")
    print(f"    true             = {tm['background_ohm_m']:g} ohm.m")
    print(f"    median over earth= {bg_med:.1f} ohm.m")

    print(f"\nBodies  (conductor: rho < {thr_cond:g} ohm.m,"
          f"  resistor: rho > {thr_res:g} ohm.m)")
    hdr = ("    name  polarity   true rho    peak rho    mean rho      centroid (x,y,z)"
           "        lat off   vert off      volume   in box")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for m in res:
        c = m["centroid"]
        cstr = ("        (nan)          " if np.isnan(c[0])
                else f"({c[0]:6.0f},{c[1]:6.0f},{c[2]:6.0f})")
        print(f"    {m['name']:<5s} {m['polarity']:<10s} "
              f"{m['true_rho']:7.1f}  {m['peak']:10.2f}  {m['mean']:10.1f}   {cstr} "
              f"{m['lat_off']:8.1f}  {m['ver_off']:+9.1f}  "
              f"{m['volume']:10.3e}  {m['frac_box']*100:5.0f}%")
    # z is negative down, so a positive vertical offset means the recovered
    # body sits shallower than the truth. Spelled out because the sign alone
    # is ambiguous.
    print("    (vertical offset > 0 = recovered shallower than true; z is negative down)")
    print("    (peak = single-cell extremum in the quadrant; mean = volume-weighted "
          "over the true box)")

    print("\nAcceptance checks")
    if rmsf is not None and "final_rms_max" in tol:
        chk("final RMS <= target", rmsf <= tol["final_rms_max"] + 1e-6,
            f"{rmsf:.4f} <= {tol['final_rms_max']}")
    if "background_ohm_m_pct" in tol:
        p = abs(bg_med - tm["background_ohm_m"]) / tm["background_ohm_m"] * 100
        chk("background resistivity", p <= tol["background_ohm_m_pct"],
            f"{bg_med:.1f} ohm.m ({p:.1f}% off {tm['background_ohm_m']})")
    for m in res:
        # Polarity first: a checkerboard is only resolved if every target comes
        # back with the right sign and none of them leaks into its neighbours.
        chk(f"{m['name']}: {m['polarity']} detected", m["detected"],
            f"peak {m['peak']:.2f} ohm.m in its search region")
        t = btol.get(m["name"], {})
        if "peak_ohm_m_range" in t:
            lo, hi = t["peak_ohm_m_range"]
            chk(f"{m['name']}: peak resistivity",
                bool(lo <= m["peak"] <= hi),
                f"{m['peak']:.2f} in [{lo:g}, {hi:g}] ohm.m")
        if "mean_ohm_m_range" in t:
            lo, hi = t["mean_ohm_m_range"]
            chk(f"{m['name']}: mean resistivity",
                bool(lo <= m["mean"] <= hi),
                f"{m['mean']:.1f} in [{lo:g}, {hi:g}] ohm.m")
        if "volume_m3_range" in t and m["detected"]:
            lo, hi = t["volume_m3_range"]
            chk(f"{m['name']}: anomaly volume", bool(lo <= m["volume"] <= hi),
                f"{m['volume']:.2e} in [{lo:.1e}, {hi:.1e}] m^3")
        # A per-body bound overrides the global one: a body recovered as a few
        # tens of cells has a correspondingly noisy centroid, and holding it to
        # the same bound as a well-resolved one measures noise, not position.
        lmax = t.get("lateral_offset_m_max", tol.get("lateral_offset_m_max"))
        if lmax is not None and m["detected"]:
            chk(f"{m['name']}: lateral localization",
                bool(m["lat_off"] <= lmax),
                f"{m['lat_off']:.1f} m <= {lmax} m")

    ok = all(checks)
    print("\n" + "=" * 74)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  ({sum(checks)}/{len(checks)} checks)")
    print("=" * 74)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
