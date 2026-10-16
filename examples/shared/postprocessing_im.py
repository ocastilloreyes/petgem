#!/usr/bin/env python3
# --------------------------------------------------------------------------
# postprocessing_im.py - im.csem inversion post-processing.
#
# Consumes the outputs/ directory of an inversion example (im_csem_block,
# im_csem_checkerboard): the result file responses_im_p<order>.h5 and the VTU
# snapshots in outputs/snapshots/, plus reference/reference_metrics.json for
# the true bodies. Produces the convergence panel and the model sections, plus
# a console summary of the model at the plotted iterations.
#
# One EPS (and PNG) per panel:
#
#   im_rms.eps  im_rms_legend.eps                         RMS per evaluation
#   im_section_iterNNNN.eps  im_section_colorbar.eps      model section per iteration
#
# Encoding: colour = log10 resistivity (CMAP_RHO), dashed outline = true body,
# dotted line = target RMS (black) and RMS of the true model (gray).
#
# Usage (optionally write the panels somewhere else, e.g. docs/source/_static):
#   python3 examples/shared/postprocessing_im.py examples/im_csem_block
#   python3 examples/shared/postprocessing_im.py examples/im_csem_checkerboard ../figures
# --------------------------------------------------------------------------
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

# --- figure format -------------------
SIZE_MARKER = 5
SIZE_FONT = 11
LINE_WIDTH = 2
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'Serif'
matplotlib.rcParams['font.size'] = SIZE_FONT
COLORS_FIELDS = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628']
FIG_SIZE_PANEL = (4.5, 3.5)             # one panel, as in postprocessing_oc.py
FIG_SIZE_LEGEND = (9.0, 0.6)            # legend strip, width of a two-panel row
DPI_EPS = 1200
DPI_PNG = 150
CMAP_RHO = 'RdYlBu'                     # red = conductive, blue = resistive
LOG_RHO_RANGE = (1.0, 3.0)              # 10 to 1000 ohm.m
CONDUCTOR = 30.0                        # ohm.m, as in analyze_inversion.py
RESISTOR = 250.0                        # ohm.m, as in analyze_inversion.py
ITERS = [10, 30, 60]                    # plotted iterations, plus the last one

# Section of each case: axis normal to the plane, its position (m) and the extent (m).
CASES = {
    "im_csem_block": {"normal": 1, "value": 0.0,
                      "extent": {0: (-400.0, 400.0), 2: (-600.0, 0.0)}},
    "im_csem_checkerboard": {"normal": 2, "value": -200.0,
                             "extent": {0: (-550.0, 550.0), 1: (-550.0, 550.0)}},
}
AXIS_NAME = "xyz"
EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

FIG_RMS = "im_rms.eps"
FIG_SECTION = "im_section_iter{:04d}.eps"
FIG_LEGEND = {"rms": "im_rms_legend.eps", "section": "im_section_colorbar.eps"}


def load_result(run_dir):
    """RMS history and run attributes of the inversion result file."""
    for cand in sorted(glob.glob(os.path.join(run_dir, "*.h5"))):
        with h5py.File(cand, "r") as h:
            if "rms_history" in h:
                attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                         for k, v in h.attrs.items()}
                return np.asarray(h["rms_history"])[:, 0], attrs
    sys.exit(f"error: no inversion result (.h5 with /rms_history) in '{run_dir}'")


def load_reference(case_dir):
    """True bodies, target RMS and RMS of the true model from reference_metrics.json."""
    path = os.path.join(case_dir, "reference", "reference_metrics.json")
    if not os.path.exists(path):
        sys.exit(f"error: reference file '{path}' not found")
    ref = json.load(open(path))
    tm = ref["true_model"]
    if "bodies" in tm:
        bodies = [(b["box_m"], b["resistivity_ohm_m"]) for b in tm["bodies"]]
    else:
        bodies = [(tm["anomaly_box_m"], tm["anomaly_ohm_m"])]
    return (bodies, ref["tolerances"]["final_rms_max"],
            ref.get("observations", {}).get("rms_true_model"))


def snapshot_iterations(run_dir):
    """Sorted iteration numbers of the snapshots in run_dir/snapshots/."""
    names = glob.glob(os.path.join(run_dir, "snapshots", "iter*.pvtu"))
    if not names:
        sys.exit(f"error: no snapshots in '{os.path.join(run_dir, 'snapshots')}'")
    return sorted(int(os.path.basename(n)[4:8]) for n in names)


def _array(da):
    return np.fromstring(da.text.replace("\n", " "), sep=" ")


def load_snapshot(run_dir, it):
    """Cell vertices (n, 4, 3) and resistivity (n) of one snapshot, over all pieces."""
    pieces = sorted(glob.glob(os.path.join(run_dir, "snapshots", f"iter{it:04d}_r*.vtu")))
    verts, rho = [], []
    for fn in pieces:
        piece = ET.parse(fn).getroot().find(".//Piece")
        pts = _array(piece.find("./Points/DataArray")).reshape(-1, 3)
        conn = [_array(da).astype(int) for da in piece.find("./Cells")
                if da.get("Name") == "connectivity"][0]
        verts.append(pts[conn].reshape(-1, 4, 3))
        rho.append([_array(da) for da in piece.find("./CellData")
                    if da.get("Name") == "rho"][0])
    return np.concatenate(verts), np.concatenate(rho)


def section(verts, normal, value):
    """Polygons of the cells cut by the plane x[normal] = value, and their cell ids."""
    d = verts[:, :, normal] - value
    cut = np.where((d.min(axis=1) < 0) & (d.max(axis=1) > 0))[0]
    keep = [a for a in range(3) if a != normal]
    polys = []
    for c in cut:
        pts = []
        for i, j in EDGES:
            di, dj = d[c, i], d[c, j]
            if di * dj < 0:
                p = verts[c, i] + di / (di - dj) * (verts[c, j] - verts[c, i])
                pts.append(p[keep])
        pts = np.array(pts)
        ctr = pts.mean(axis=0)
        polys.append(pts[np.argsort(np.arctan2(pts[:, 1] - ctr[1], pts[:, 0] - ctr[0]))])
    return polys, cut, keep


def volume(verts):
    a = verts[:, 1] - verts[:, 0]
    b = verts[:, 2] - verts[:, 0]
    c = verts[:, 3] - verts[:, 0]
    return np.abs(np.einsum('ij,ij->i', a, np.cross(b, c))) / 6.0


def panel():
    """A single-axes figure of the size of one postprocessing_oc.py panel."""
    return plt.subplots(figsize=FIG_SIZE_PANEL, constrained_layout=True)


def save(fig, out, outdir):
    eps = os.path.join(outdir, out)
    png = eps.replace(".eps", ".png")
    fig.savefig(eps, format="eps", dpi=DPI_EPS, bbox_inches="tight")
    fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {eps} and {png}")


def legend_strip(handles, out, outdir, ncol):
    """Legend-only EPS of the width of a two-panel row; ncol shrinks until it fits."""
    width = FIG_SIZE_LEGEND[0]
    fig = plt.figure(figsize=(width, 2.0))
    while True:
        leg = fig.legend(handles=handles, loc='center', ncol=ncol,
                         fontsize=SIZE_FONT, frameon=True)
        fig.canvas.draw()
        bbox = leg.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if bbox.width <= width - 0.1 or ncol <= 1:
            break
        leg.remove()
        ncol -= 1
    fig.set_size_inches(width, bbox.height + 0.15)
    save(fig, out, outdir)


def colorbar_strip(out, outdir):
    """Colorbar-only EPS of the width of a two-panel row."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_LEGEND, constrained_layout=True)
    sm = plt.cm.ScalarMappable(cmap=CMAP_RHO, norm=plt.Normalize(*LOG_RHO_RANGE))
    cb = fig.colorbar(sm, cax=ax, orientation='horizontal')
    cb.set_label(r'$\log_{10}\rho$ ($\Omega\cdot$m)')
    save(fig, out, outdir)


def figure_rms(rms, target, rms_true, outdir):
    """RMS of every objective-gradient evaluation."""
    fig, ax = panel()
    ev = np.arange(len(rms))
    ax.semilogy(ev, rms, color=COLORS_FIELDS[1], marker='o', linestyle='-',
                markersize=SIZE_MARKER, markerfacecolor=COLORS_FIELDS[1],
                markeredgecolor='k', linewidth=LINE_WIDTH, zorder=3)
    ax.axhline(target, color='k', linestyle=':', linewidth=LINE_WIDTH, zorder=1)
    if rms_true is not None:
        ax.axhline(rms_true, color='gray', linestyle=':', linewidth=LINE_WIDTH, zorder=1)
    ax.set_xlabel(r'Evaluation')
    ax.set_ylabel(r'RMS')
    ax.set_xlim(0, len(rms))
    ax.grid(True, which='major', linestyle=':', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    save(fig, FIG_RMS, outdir)

    handles = [Line2D([], [], color=COLORS_FIELDS[1], marker='o', linestyle='-',
                      markersize=SIZE_MARKER, markeredgecolor='k',
                      linewidth=LINE_WIDTH, label=r'\texttt{im.csem}'),
               Line2D([], [], color='k', linestyle=':', linewidth=LINE_WIDTH,
                      label=rf'Target RMS $= {target:g}$')]
    if rms_true is not None:
        handles.append(Line2D([], [], color='gray', linestyle=':', linewidth=LINE_WIDTH,
                              label=rf'RMS of the true model $= {rms_true:g}$'))
    legend_strip(handles, FIG_LEGEND["rms"], outdir, ncol=3)


def figure_section(verts, rho, it, case, bodies, outdir):
    """Model section at one iteration, with the outline of the true bodies."""
    polys, cut, keep = section(verts, case["normal"], case["value"])
    fig, ax = panel()
    pc = PolyCollection(polys, cmap=CMAP_RHO, edgecolors='face', linewidths=0.2)
    pc.set_array(np.log10(rho[cut]))
    pc.set_clim(*LOG_RHO_RANGE)
    ax.add_collection(pc)
    for box, _ in bodies:
        lo = case["value"] >= box[AXIS_NAME[case["normal"]]][0]
        hi = case["value"] <= box[AXIS_NAME[case["normal"]]][1]
        if not (lo and hi):
            continue
        (x0, x1), (y0, y1) = box[AXIS_NAME[keep[0]]], box[AXIS_NAME[keep[1]]]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color='k',
                linestyle='--', linewidth=1.2, zorder=3)
    ax.set_xlim(*case["extent"][keep[0]])
    ax.set_ylim(*case["extent"][keep[1]])
    ax.set_aspect('equal')
    ax.set_xlabel(rf'${AXIS_NAME[keep[0]]}$ (m)')
    ax.set_ylabel(rf'${AXIS_NAME[keep[1]]}$ (m)')
    save(fig, FIG_SECTION.format(it), outdir)


def summary(run_dir, iters):
    """Resistivity range and conductor/resistor volume in the earth per iteration."""
    print(f"\nModel per iteration ({run_dir})")
    print(f"{'iter':>5} {'rho_min':>9} {'rho_max':>9} {'V(rho<30)':>11} {'V(rho>250)':>11}")
    print("-" * 49)
    for it, (verts, rho) in iters.items():
        earth = verts.mean(axis=1)[:, 2] < -1.0
        vol = volume(verts)
        print(f"{it:>5} {rho[earth].min():>9.2f} {rho[earth].max():>9.1f} "
              f"{vol[earth & (rho < CONDUCTOR)].sum():>11.3e} "
              f"{vol[earth & (rho > RESISTOR)].sum():>11.3e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: postprocessing_im.py <case_dir> [outdir]")
    case_dir = sys.argv[1].rstrip("/")
    name = os.path.basename(os.path.abspath(case_dir))
    if name not in CASES:
        sys.exit(f"error: unknown case '{name}' (expected one of {list(CASES)})")
    run_dir = os.path.join(case_dir, "outputs")
    outdir = sys.argv[2] if len(sys.argv) > 2 else run_dir
    os.makedirs(outdir, exist_ok=True)

    rms, attrs = load_result(run_dir)
    bodies, target, rms_true = load_reference(case_dir)
    available = snapshot_iterations(run_dir)
    plotted = [it for it in ITERS if it in available] + [available[-1]]
    iters = {it: load_snapshot(run_dir, it) for it in plotted}

    summary(run_dir, iters)
    figure_rms(rms, target, rms_true, outdir)
    for it, (verts, rho) in iters.items():
        figure_section(verts, rho, it, CASES[name], bodies, outdir)
    colorbar_strip(FIG_LEGEND["section"], outdir)
