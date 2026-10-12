"""Readers of im.csem results shared by the example analyzers (im_csem_block, im_csem_checkerboard)."""
import glob
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
    # an existing results directory still analyze.
    vtus = (glob.glob(os.path.join(run_dir, "snapshots", "iter*_r*.vtu"))
            + glob.glob(os.path.join(run_dir, "*_snapshots", "*iter*_r*.vtu"))
            + glob.glob(os.path.join(run_dir, "*_iter*_p*.vtu")))
    if not vtus:
        sys.exit(f"ERROR: no VTU snapshots found under {run_dir} "
                 f"(looked for snapshots/iter*_r*.vtu and the two older layouts)")
    # No leading underscore in the pattern: it must match "iter0103_r0000.vtu"
    # as well as "..._iter00096_p0110.vtu".
    rank_re = re.compile(r"iter0*(\d+)_[pr]\d+\.vtu$")
    iters = {int(rank_re.search(name).group(1)) for name in vtus}
    last = max(iters)
    pieces = sorted(name for name in vtus
                    if int(rank_re.search(name).group(1)) == last)
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
        for j, off in enumerate(offs):
            v = pts[conn[start:off]]
            start = off
            c = v.mean(axis=0)
            cx.append(c[0]); cy.append(c[1]); cz.append(c[2])
            rho.append(rr[j])
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
