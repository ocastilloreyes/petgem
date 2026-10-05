#!/usr/bin/env python3
"""Build the two tagged meshes of the im2 CSEM inversion benchmark.

``geometry/mesh.geo`` meshes only two physical volumes: AIR (tag 1, z>0)
and BG (tag 2, z<0). The INVERT parameter region and the four anomaly bodies
are NOT meshed as geometric volumes - they are assigned by RE-TAGGING cells
after meshing, which changes only material labels and leaves the mesh topology
untouched. A conductivity contrast is a material value, not a mesh feature.

Produces, into ../geometry/:

    mesh.msh        AIR, BG, INVERT              inversion starting model -> im.csem
    mesh_true.msh   AIR, BG, INVERT, COND, RES   true model               -> fm.csem

The true model is a CHECKERBOARD of four 200 m cubes at the same depth, two
conductive and two resistive, so the two anomaly resistivities share two tags
rather than four: COND (tag 4) carries both 10 ohm-m cubes and RES (tag 5) both
1000 ohm-m cubes.

The tagging rule, applied to each tetrahedron's centroid (z NEGATIVE DOWN, the
convention examples/fm uses: depth is a negative z, z=0 is the interface):

    z > 0                                              -> 1 AIR
    |x|,|y| <= RX  and  RZ <= z <= 0                    -> 3 INVERT
      and, in the true model only, inside a cube
      | |x| - AC | <= AH, | |y| - AC | <= AH,
      AZB <= z <= AZT                                   -> 4 COND  if x*y < 0
                                                        -> 5 RES   if x*y > 0
    everything else in the earth                        -> 2 BG

The ``x*y`` test is the checkerboard: the two cubes on the (+,-) and (-,+)
diagonal are conductive, the two on the (+,+) and (-,-) diagonal resistive.

RZ, AZT and AZB are negative (AZT is the shallower, AZB the deeper bound).

RX/RZ/AC/AH/AZT/AZB are read from mesh.geo, which already declares them for
its refinement fields, so the mask and the refinement can never drift apart.

Run from anywhere; needs gmsh on PATH only for --force:

    python3 scripts/build_meshes.py --verify    # check shipped meshes, no writes
    python3 scripts/build_meshes.py --force     # re-mesh with gmsh, overwrite

With the gmsh shipped in the petgem-env image, --force reproduces the two
meshes BYTE-IDENTICALLY, so regenerating is safe there. Under a different gmsh
version the mesh may differ, which changes the discretisation and therefore the
forward responses: rerun the whole pipeline (see the case README) and refresh
reference/reference_metrics.json. Check with --verify first.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "..", "geometry", "mesh.geo")
GEOM = os.path.join(HERE, "..", "geometry")

TAG_AIR, TAG_BG, TAG_INVERT, TAG_COND, TAG_RES = 1, 2, 3, 4, 5
NAMES = {TAG_AIR: "AIR", TAG_BG: "BG", TAG_INVERT: "INVERT",
         TAG_COND: "COND", TAG_RES: "RES"}


def geo_constants():
    """Read RX, RZ, AC, AH, AZT, AZB from mesh.geo (its own refinement fields)."""
    txt = open(GEO).read()
    out = {}
    for key in ("RX", "RZ", "AC", "AH", "AZT", "AZB"):
        m = re.search(rf"\b{key}\s*=\s*(-?[0-9.]+)\s*;", txt)
        if not m:
            sys.exit(f"ERROR: {key} not found in {GEO}")
        out[key] = float(m.group(1))
    return out


def classify(c, k, true_model):
    """Physical tag for a cell centroid c=(x,y,z); z is NEGATIVE down.

    RZ/AZT/AZB carry their sign, so the depth tests read as plain interval
    checks with the deeper (more negative) bound on the left.
    """
    x, y, z = c
    if z > 0.0:
        return TAG_AIR
    if abs(x) <= k["RX"] and abs(y) <= k["RX"] and k["RZ"] <= z <= 0.0:
        if (true_model
                and abs(abs(x) - k["AC"]) <= k["AH"]
                and abs(abs(y) - k["AC"]) <= k["AH"]
                and k["AZB"] <= z <= k["AZT"]):
            # Checkerboard: opposite-sign quadrants conductive, same-sign resistive.
            return TAG_COND if x * y < 0.0 else TAG_RES
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
    for name, tm in (("mesh.msh", False), ("mesh_true.msh", True)):
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
    print(f"mask from mesh.geo: RX={k['RX']:g} RZ={k['RZ']:g} AC={k['AC']:g} "
          f"AH={k['AH']:g} AZT={k['AZT']:g} AZB={k['AZB']:g}")

    if args.verify or not args.force:
        ok = verify(k)
        if not args.force:
            if not args.verify:
                print("\nNothing written. Pass --force to re-mesh and overwrite,\n"
                      "or --verify to silence this hint.")
            sys.exit(0 if ok else 1)

    if not shutil.which("gmsh"):
        sys.exit("ERROR: gmsh not found on PATH. Run inside the petgem-env image:\n"
                 "  docker run --rm -u $(id -u):$(id -g) -v \"$PWD\":/workspace "
                 "-w /workspace petgem-env:latest \\\n"
                 "      python3 examples/im2/scripts/build_meshes.py --force")

    print("\nNOTE: with the petgem-env gmsh this reproduces the shipped meshes\n"
          "byte-identically. Under a different gmsh version the mesh may change,\n"
          "and the reference metrics - measured on the current\n"
          "mesh_true.msh - would need the forward stage rerun before inverting.\n")

    base = os.path.join(GEOM, "_base.msh")
    subprocess.run(["gmsh", "-3", GEO, "-o", base], check=True)

    for name, tm in (("mesh.msh", False), ("mesh_true.msh", True)):
        used, n = retag(base, os.path.join(GEOM, name), k, tm)
        tally = ", ".join(NAMES[t] for t in sorted(used))
        print(f"  wrote geometry/{name}: {n} elements, tags [{tally}]")
    os.remove(base)

    print()
    verify(k)


if __name__ == "__main__":
    main()
