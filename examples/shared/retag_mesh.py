"""Retag the cells of a gmsh 2.2 mesh by a centroid rule (examples im_csem_block, im_csem_checkerboard).

A case script supplies the mesh.geo constants it needs, the tag names and a
``classify(centroid, constants, true_model)`` rule, then calls ``main``. The
case writes two meshes into its geometry/ directory:

    mesh.msh        inversion starting model  (true_model=False)  -> im.csem
    mesh_true.msh   true model                (true_model=True)   -> fm.csem
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

MESHES = (("mesh.msh", False), ("mesh_true.msh", True))


def geo_constants(geo, keys):
    """Read `KEY = value;` assignments from mesh.geo."""
    txt = open(geo).read()
    out = {}
    for key in keys:
        m = re.search(rf"\b{key}\s*=\s*(-?[0-9.]+)\s*;", txt)
        if not m:
            sys.exit(f"ERROR: {key} not found in {geo}")
        out[key] = float(m.group(1))
    return out


def read_msh(path):
    """Parse a gmsh 2.2 ASCII mesh into (lines, coords, ni, nn, ei, ne)."""
    lines = open(path).read().split("\n")
    ni = lines.index("$Nodes")
    nn = int(lines[ni + 1])
    coords = {}
    for i in range(nn):
        p = lines[ni + 2 + i].split()
        coords[int(p[0])] = (float(p[1]), float(p[2]), float(p[3]))
    ei = lines.index("$Elements")
    ne = int(lines[ei + 1])
    return lines, coords, ni, nn, ei, ne


def centroid(tokens, coords):
    """Centroid of a gmsh 2.2 element line, or None if it is not a tetrahedron."""
    t = [int(tok) for tok in tokens]
    if t[1] != 4:                       # 4 = 4-node tetrahedron
        return None, t
    nodes = t[3 + t[2]:]
    xs = [coords[node] for node in nodes]
    return (sum(point[0] for point in xs) / 4.0,
            sum(point[1] for point in xs) / 4.0,
            sum(point[2] for point in xs) / 4.0), t


def retag(src, dst, classify, names, k, true_model):
    """Write dst = src with every tetrahedron's physical tag reassigned."""
    lines, coords, ni, nn, ei, ne = read_msh(src)
    used, elems = set(), []
    for i in range(ne):
        toks = lines[ei + 2 + i].split()
        c, t = centroid(toks, coords)
        if c is None:
            elems.append(" ".join(toks))
            continue
        t[3] = classify(c, k, true_model)       # tags[0] = physical tag
        used.add(t[3])
        elems.append(" ".join(str(tok) for tok in t))

    physical = [f'3 {tag} "{names[tag]}"' for tag in sorted(used)]
    out = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
           "$PhysicalNames", str(len(physical)), *physical, "$EndPhysicalNames"]
    out += lines[ni:ei + 2] + elems + lines[ei + 2 + ne:]
    with open(dst, "w") as f:
        f.write("\n".join(out))
    return used, len(elems)


def verify(geom, classify, names, k):
    """Check the shipped meshes already satisfy the tagging rule. No writes."""
    ok = True
    for name, tm in MESHES:
        path = os.path.join(geom, name)
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
        tally = ", ".join(f"{names[tag]}={count}" for tag, count in sorted(counts.items()))
        print(f"  {name:14s} {sum(counts.values()):6d} tets  [{tally}]  "
              f"mismatches={bad}")
        ok = ok and bad == 0
    return ok


def main(doc, case, keys, names, classify, geom):
    """CLI shared by the case scripts: -verify the shipped meshes or -force a re-mesh."""
    parser = argparse.ArgumentParser(description=doc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-verify", action="store_true",
                        help="Check the shipped meshes against the rule; write nothing.")
    parser.add_argument("-force", action="store_true",
                        help="Re-mesh with gmsh and OVERWRITE the shipped meshes.")
    args = parser.parse_args()

    geo = os.path.join(geom, "mesh.geo")
    k = geo_constants(geo, keys)
    print("mask from mesh.geo: " + " ".join(f"{key}={k[key]:g}" for key in keys))

    if args.verify or not args.force:
        ok = verify(geom, classify, names, k)
        if not args.force:
            if not args.verify:
                print("\nNothing written. Pass -force to re-mesh and overwrite,\n"
                      "or -verify to silence this hint.")
            sys.exit(0 if ok else 1)

    if not shutil.which("gmsh"):
        sys.exit("ERROR: gmsh not found on PATH. Run inside the petgem-env image:\n"
                 "  docker run --rm -u $(id -u):$(id -g) -v \"$PWD\":/workspace "
                 "-w /workspace petgem-env:latest \\\n"
                 f"      python3 examples/{case}/scripts/build_meshes.py -force")

    print("\nNOTE: with the petgem-env gmsh this reproduces the shipped meshes\n"
          "byte-identically. Under a different gmsh version the mesh may change,\n"
          "and the reference metrics - measured on the current\n"
          "mesh_true.msh - would need the forward stage rerun before inverting.\n")

    base = os.path.join(geom, "_base.msh")
    subprocess.run(["gmsh", "-3", geo, "-o", base], check=True)

    for name, tm in MESHES:
        used, n = retag(base, os.path.join(geom, name), classify, names, k, tm)
        tally = ", ".join(names[tag] for tag in sorted(used))
        print(f"  wrote geometry/{name}: {n} elements, tags [{tally}]")
    os.remove(base)

    print()
    verify(geom, classify, names, k)
