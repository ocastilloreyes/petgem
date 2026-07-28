#!/usr/bin/env python3
"""Compare candidate meshes for the im_model inversion benchmark.

Order-independent analysis of a tetrahedral mesh (nodes + connectivity) that
quantifies the four things a good CSEM inversion mesh must balance:

  1. geometric quality   - dihedral angles, aspect, mean-ratio, slivers
  2. mesh topology        - edges/faces, non-manifold faces, triple lines, Euler
  3. grading smoothness   - size jump across interior faces (abrupt transitions)
  4. air-earth interface  - valence of the z=0 nodes, in-plane and in 3D
  5. PCBDDC connectivity  - edge line-graph valence / chain-capable fraction

(3), (4) and (5) are the metrics implicated in the PCBDDC order-2 coarse-edge
(Nedelec support) failure: abrupt fine->coarse transitions and a topologically
complex air-earth interface raise vertex valence and produce the branching,
non-chain interface connectivity that PCBDDC's coarse-edge construction must
walk. A smoother, interface-symmetric mesh lowers all three.

    python3 analyze_candidates.py MESH.msh            # human-readable report
    python3 analyze_candidates.py MESH.msh --tsv NAME # one machine-readable row
"""
import argparse
import collections
import os
import sys

import numpy as np

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
FACES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
ZTOL = 1e-6


def read_msh(path):
    L = open(path).read().split("\n")
    ni = L.index("$Nodes")
    nn = int(L[ni + 1])
    coord = np.zeros((nn + 1, 3))
    for k in range(nn):
        p = L[ni + 2 + k].split()
        coord[int(p[0])] = [float(p[1]), float(p[2]), float(p[3])]
    ei = L.index("$Elements")
    ne = int(L[ei + 1])
    tets = []
    for k in range(ne):
        t = [int(x) for x in L[ei + 2 + k].split()]
        if t[1] != 4:
            continue
        nt = t[2]
        tets.append(t[3 + nt:3 + nt + 4])
    return coord, np.array(tets, dtype=np.int64)


def quality(coord, tets):
    vol, aspect, eta, mindih, size = [], [], [], [], []
    for tet in tets:
        v = coord[tet]
        sv = np.dot(v[1] - v[0], np.cross(v[2] - v[0], v[3] - v[0])) / 6.0
        vol.append(sv)
        el = np.array([np.linalg.norm(v[a] - v[b]) for a, b in EDGES])
        aspect.append(el.max() / max(el.min(), 1e-300))
        eta.append(12.0 * (3.0 * abs(sv)) ** (2.0 / 3.0) / max((el ** 2).sum(), 1e-300))
        size.append(el.mean())
        nrm = []
        for f in [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]:
            a, b, c = v[f[0]], v[f[1]], v[f[2]]
            n = np.cross(b - a, c - a)
            ln = np.linalg.norm(n)
            nrm.append(n / ln if ln > 0 else n)
        md = 180.0
        for i in range(4):
            for j in range(i + 1, 4):
                cosang = max(-1.0, min(1.0, -np.dot(nrm[i], nrm[j])))
                md = min(md, np.degrees(np.arccos(cosang)))
        mindih.append(md)
    return (np.array(vol), np.array(aspect), np.array(eta),
            np.array(mindih), np.array(size))


def topology(coord, tets):
    edge_set = set()
    adj = collections.defaultdict(set)           # full 3D nodal adjacency
    face_tets = collections.defaultdict(list)     # face -> list of tet indices
    for ti, tet in enumerate(tets):
        for a, b in EDGES:
            e = tuple(sorted((tet[a], tet[b])))
            edge_set.add(e)
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
        for f in FACES:
            face_tets[tuple(sorted((tet[f[0]], tet[f[1]], tet[f[2]])))].append(ti)
    n_faces = len(face_tets)
    bnd = [f for f, c in face_tets.items() if len(c) == 1]
    interior = [(f, c) for f, c in face_tets.items() if len(c) == 2]
    nonmanifold = sum(1 for c in face_tets.values() if len(c) > 2)
    bnd_edge = collections.Counter()
    for f in bnd:
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            bnd_edge[tuple(sorted((f[a], f[b])))] += 1
    triple = sum(1 for c in bnd_edge.values() if c != 2)
    used = sorted(adj)
    val = np.array([len(adj[v]) for v in used])
    seen, comps = set(), 0
    for s in used:
        if s in seen:
            continue
        comps += 1
        st = [s]
        seen.add(s)
        while st:
            u = st.pop()
            for w in adj[u]:
                if w not in seen:
                    seen.add(w)
                    st.append(w)
    return dict(n_edges=len(edge_set), n_faces=n_faces, n_bnd=len(bnd),
                interior=interior, nonmanifold=nonmanifold, triple=triple,
                val=val, comps=comps, nv=len(used), adj=adj, edges=edge_set)


def interface_metrics(coord, adj):
    """Valence of the z=0 air-earth interface nodes, in 3D and in-plane."""
    on = {v for v in adj if abs(coord[v][2]) < ZTOL}
    if not on:
        return None
    val3d = np.array([len(adj[v]) for v in on])            # all volume neighbours
    valin = np.array([sum(1 for w in adj[v] if w in on) for v in on])  # in-plane
    return dict(n=len(on), val3d=val3d, valin=valin)


def grading(coord, size, interior):
    """Size ratio across each interior face (local coarse<->fine jump)."""
    r = np.array([max(size[a], size[b]) / max(min(size[a], size[b]), 1e-300)
                  for _, (a, b) in interior])
    return r


def line_graph(adj, edges):
    """Order-1 edge line-graph: edge-dof valence = shared-vertex adjacency."""
    lval = np.array([(len(adj[a]) - 1) + (len(adj[b]) - 1) for a, b in edges])
    return lval


def p(a):
    return (a.min(), np.percentile(a, 50), np.percentile(a, 99), a.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--tsv", metavar="NAME", default=None,
                    help="emit one tab-separated summary row labelled NAME")
    args = ap.parse_args()
    if not os.path.isfile(args.mesh):
        sys.exit(f"ERROR: {args.mesh} not found")

    coord, tets = read_msh(args.mesh)
    vol, aspect, eta, mindih, size = quality(coord, tets)
    t = topology(coord, tets)
    ifc = interface_metrics(coord, t["adj"])
    gr = grading(coord, size, t["interior"])
    lval = line_graph(t["adj"], t["edges"])

    slivers = int((mindih < 5.0).sum())
    poor = int((eta < 0.05).sum())
    euler = t["nv"] - t["n_edges"] + t["n_faces"] - len(tets)
    chain = int((lval == 2).sum())
    g15 = 100.0 * (gr > 1.5).mean()
    g20 = 100.0 * (gr > 2.0).mean()

    if args.tsv:
        cols = [args.tsv, coord.shape[0] - 1, len(tets), t["n_edges"], t["n_faces"],
                f"{mindih.min():.2f}", slivers, f"{aspect.max():.1f}",
                f"{np.median(eta):.3f}", int(t["val"].max()), int(np.median(t["val"])),
                t["nonmanifold"], t["triple"], euler,
                (ifc["n"] if ifc else 0),
                (int(ifc["val3d"].max()) if ifc else 0),
                (int(ifc["valin"].max()) if ifc else 0),
                f"{gr.max():.2f}", f"{g15:.1f}", f"{g20:.1f}",
                f"{100.0*chain/t['n_edges']:.1f}"]
        print("\t".join(str(c) for c in cols))
        return

    print(f"mesh: {args.mesh}")
    print(f"  {coord.shape[0]-1} nodes, {len(tets)} tets, {t['n_edges']} edges, "
          f"{t['n_faces']} faces (bnd {t['n_bnd']})\n")

    print("=== 1. geometric quality ===")
    lo, med, p99, hi = p(mindih)
    print(f"  min dihedral (deg)  min/med/99%/max = {lo:.2f} / {med:.2f} / {p99:.2f} / {hi:.2f}")
    print(f"  slivers (<5 deg) = {slivers}      poor tets (eta<0.05) = {poor}")
    lo, med, p99, hi = p(aspect)
    print(f"  edge-ratio aspect   min/med/99%/max = {lo:.2f} / {med:.2f} / {p99:.2f} / {hi:.2f}")
    lo, med, p99, hi = p(eta)
    print(f"  mean-ratio eta      min/med/99%/max = {lo:.3f} / {med:.3f} / {p99:.3f} / {hi:.3f}  (1=ideal)\n")

    print("=== 2. mesh topology ===")
    print(f"  non-manifold faces = {t['nonmanifold']}   triple lines = {t['triple']}   "
          f"components = {t['comps']}   Euler chi = {euler}")
    print(f"  vertex valence (3D) min/med/max = {t['val'].min()} / "
          f"{int(np.median(t['val']))} / {t['val'].max()}\n")

    print("=== 3. grading smoothness (size ratio across interior faces) ===")
    lo, med, p99, hi = p(gr)
    print(f"  neighbour size ratio min/med/99%/max = {lo:.2f} / {med:.2f} / {p99:.2f} / {hi:.2f}")
    print(f"  faces with jump > 1.5x = {g15:.2f}%     > 2.0x = {g20:.2f}%")
    print("  (lower = smoother; abrupt jumps drive interface valence)\n")

    print("=== 4. air-earth interface (z=0) ===")
    if ifc:
        print(f"  interface nodes = {ifc['n']}")
        print(f"  3D valence  min/med/max = {ifc['val3d'].min()} / "
              f"{int(np.median(ifc['val3d']))} / {ifc['val3d'].max()}")
        print(f"  in-plane valence min/med/max = {ifc['valin'].min()} / "
              f"{int(np.median(ifc['valin']))} / {ifc['valin'].max()}")
        print(f"  high-valence (>2x median 3D) interface nodes = "
              f"{int((ifc['val3d'] > 2*np.median(ifc['val3d'])).sum())}\n")
    else:
        print("  (no nodes exactly on z=0)\n")

    print("=== 5. PCBDDC edge-dof connectivity (order-1 line graph) ===")
    lo, med, p99, hi = p(lval)
    print(f"  edge-dof valence min/med/99%/max = {lo:.0f} / {med:.0f} / {p99:.0f} / {hi:.0f}")
    print(f"  valence==2 (chain-capable) edges = {chain}  ({100.0*chain/t['n_edges']:.1f}%)")
    print("  (order-2: each mesh edge -> a 2-clique; higher valence + branching")
    print("   near a rough interface is what the coarse-edge walk mishandles)")


if __name__ == "__main__":
    main()
