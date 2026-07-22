#!/usr/bin/env python3
"""Build the two tagged meshes of the im_model CSEM inversion benchmark.

``geometry/im_model.geo`` meshes only two physical volumes: AIR (tag 1, z<0)
and BG (tag 2, z>0). The INVERT parameter region and the ANOMALY body are NOT
meshed as geometric volumes - they are assigned by RE-TAGGING cells after
meshing, which changes only material labels and leaves the mesh topology
untouched. That matters: PCBDDC's Nedelec support keys off the discrete
gradient, whose structure depends on topology (edges -> nodes) alone, so
embedding the bodies as volumes would add edges and faces and can trip the
preconditioner. A conductivity contrast is a material value, not a mesh
feature.

Produces, into ../geometry/:

    im_model.msh   AIR, BG, INVERT            inversion starting model -> im.csem
    im_true.msh    AIR, BG, INVERT, ANOMALY   true model              -> fm.csem

The tagging rule, applied to each tetrahedron's centroid (z POSITIVE DOWN):

    z < 0                                              -> 1 AIR
    |x|,|y| <= RX  and  0 <= z <= RZ                    -> 3 INVERT
      and, in the true model only,
      |x|,|y| <= AX  and  AZT <= z <= AZB               -> 4 ANOMALY
    everything else in the earth                        -> 2 BG

RX/RZ/AX/AZT/AZB are read from im_model.geo, which already declares them for
its refinement fields, so the mask and the refinement can never drift apart.

Run from anywhere; needs gmsh on PATH only for --force:

    python3 scripts/build_meshes.py --verify    # check shipped meshes, no writes
    python3 scripts/build_meshes.py --force     # re-mesh with gmsh, overwrite

With the gmsh shipped in the petgem-env image, --force reproduces the two
meshes BYTE-IDENTICALLY, so regenerating is safe there. Under a different gmsh
version the mesh may differ, and the shipped reference/observed.h5 - which was
forward-modelled on the current im_true.msh - would no longer correspond to it:
rerun the forward stage (scripts/build_bundles.sh fm + scripts/run_forward.slurm,
then utils/make_observed.py) before inverting. Check with --verify first.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "..", "geometry", "im_model.geo")
GEOM = os.path.join(HERE, "..", "geometry")

TAG_AIR, TAG_BG, TAG_INVERT, TAG_ANOMALY = 1, 2, 3, 4
NAMES = {TAG_AIR: "AIR", TAG_BG: "BG", TAG_INVERT: "INVERT", TAG_ANOMALY: "ANOMALY"}


def geo_constants():
    """Read RX, RZ, AX, AZT, AZB from im_model.geo (its own refinement fields)."""
    txt = open(GEO).read()
    out = {}
    for key in ("RX", "RZ", "AX", "AZT", "AZB"):
        m = re.search(rf"\b{key}\s*=\s*(-?[0-9.]+)\s*;", txt)
        if not m:
            sys.exit(f"ERROR: {key} not found in {GEO}")
        out[key] = float(m.group(1))
    return out


def classify(c, k, true_model):
    """Physical tag for a cell centroid c=(x,y,z); z is positive down."""
    x, y, z = c
    if z < 0.0:
        return TAG_AIR
    if abs(x) <= k["RX"] and abs(y) <= k["RX"] and z <= k["RZ"]:
        if (true_model and abs(x) <= k["AX"] and abs(y) <= k["AX"]
                and k["AZT"] <= z <= k["AZB"]):
            return TAG_ANOMALY
        return TAG_INVERT
    return TAG_BG


def read_msh(path):
    """Parse a gmsh 2.2 ASCII mesh into (lines, coords, element index list)."""
    lines = open(path).read().split("\n")
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = {}
    for k in range(nn):
        p = lines[ni + 2 + k].split()
        coords[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    return lines, coords, ni, nn, ei, ne


def centroid(tokens, coords):
    """Centroid of a gmsh 2.2 element line, or None if it is not a tetrahedron."""
    t = [int(x) for x in tokens]
    if t[1] != 4:                       # 4 = 4-node tetrahedron
        return None, t
    nodes = t[3 + t[2]:]
    xs = [coords[n] for n in nodes]
    return (sum(p[0] for p in xs) / 4.0,
            sum(p[1] for p in xs) / 4.0,
            sum(p[2] for p in xs) / 4.0), t


def retag(src, dst, k, true_model):
    """Write dst = src with every tetrahedron's physical tag reassigned."""
    lines, coords, ni, nn, ei, ne = read_msh(src)
    used, elems = set(), []
    for j in range(ne):
        toks = lines[ei + 2 + j].split()
        c, t = centroid(toks, coords)
        if c is None:
            elems.append(" ".join(toks))
            continue
        t[3] = classify(c, k, true_model)       # tags[0] = physical tag
        used.add(t[3])
        elems.append(" ".join(str(v) for v in t))

    names = [f'3 {tag} "{NAMES[tag]}"' for tag in sorted(used)]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", str(len(names)), *names, "$EndPhysicalNames"]
    out += lines[ni:ei + 2] + elems + lines[ei + 2 + ne:]
    with open(dst, "w") as f:
        f.write("\n".join(out))
    return used, len(elems)


def verify(k):
    """Check the shipped meshes already satisfy the tagging rule. No writes."""
    ok = True
    for name, tm in (("im_model.msh", False), ("im_true.msh", True)):
        path = os.path.join(GEOM, name)
        if not os.path.isfile(path):
            print(f"  {name:14s} MISSING")
            ok = False
            continue
        lines, coords, ni, nn, ei, ne = read_msh(path)
        counts, bad = {}, 0
        for j in range(ne):
            c, t = centroid(lines[ei + 2 + j].split(), coords)
            if c is None:
                continue
            want = classify(c, k, tm)
            counts[t[3]] = counts.get(t[3], 0) + 1
            if want != t[3]:
                bad += 1
        tally = ", ".join(f"{NAMES[t]}={n}" for t, n in sorted(counts.items()))
        print(f"  {name:14s} {sum(counts.values()):6d} tets  [{tally}]  "
              f"mismatches={bad}")
        ok = ok and bad == 0
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="Check the shipped meshes against the rule; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Re-mesh with gmsh and OVERWRITE the shipped meshes.")
    args = ap.parse_args()

    k = geo_constants()
    print(f"mask from im_model.geo: RX={k['RX']:g} RZ={k['RZ']:g} "
          f"AX={k['AX']:g} AZT={k['AZT']:g} AZB={k['AZB']:g}")

    if args.verify or not args.force:
        ok = verify(k)
        if not args.force:
            if not args.verify:
                print("\nNothing written. Pass --force to re-mesh and overwrite,\n"
                      "or --verify to silence this hint.")
            sys.exit(0 if ok else 1)

    if not shutil.which("gmsh"):
        sys.exit("ERROR: gmsh not found on PATH. Run inside the petgem-env image:\n"
                 "  docker run --rm -v \"$PWD\":/workspace -w /workspace "
                 "petgem-env:latest \\\n"
                 "      python3 examples/im_model/scripts/build_meshes.py --force")

    print("\nNOTE: with the petgem-env gmsh this reproduces the shipped meshes\n"
          "byte-identically. Under a different gmsh version the mesh may change,\n"
          "and reference/observed.h5 - forward-modelled on the current\n"
          "im_true.msh - would need the forward stage rerun before inverting.\n")

    base = os.path.join(GEOM, "_base.msh")
    subprocess.run(["gmsh", "-3", GEO, "-o", base], check=True)

    for name, tm in (("im_model.msh", False), ("im_true.msh", True)):
        used, n = retag(base, os.path.join(GEOM, name), k, tm)
        tally = ", ".join(NAMES[t] for t in sorted(used))
        print(f"  wrote geometry/{name}: {n} elements, tags [{tally}]")
    os.remove(base)

    print()
    verify(k)


if __name__ == "__main__":
    main()
