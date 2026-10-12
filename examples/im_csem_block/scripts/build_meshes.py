#!/usr/bin/env python3
"""Build the two tagged meshes of the im_csem_block CSEM inversion benchmark.

``geometry/mesh.geo`` meshes only two physical volumes: AIR (tag 1, z>0)
and BG (tag 2, z<0). The INVERT parameter region and the ANOMALY body are not
meshed as geometric volumes - they are assigned by re-tagging cells after
meshing, which changes only material labels and leaves the mesh topology
untouched. A conductivity contrast is a material value, not a mesh
feature.

Produces, into ../geometry/:

    mesh.msh   AIR, BG, INVERT            inversion starting model -> im.csem
    mesh_true.msh    AIR, BG, INVERT, ANOMALY   true model              -> fm.csem

The tagging rule, applied to each tetrahedron's centroid (z NEGATIVE DOWN, the
convention examples/fm_csem_marine uses: depth is a negative z, z=0 is the interface):

    z > 0                                              -> 1 AIR
    |x|,|y| <= RX  and  RZ <= z <= 0                    -> 3 INVERT
      and, in the true model only,
      |x|,|y| <= AX  and  AZB <= z <= AZT               -> 4 ANOMALY
    everything else in the earth                        -> 2 BG

RZ, AZT and AZB are negative (AZT is the shallower, AZB the deeper bound).

RX/RZ/AX/AZT/AZB are read from mesh.geo, which already declares them for
its refinement fields, so the mask and the refinement can never drift apart.

Run from anywhere; needs gmsh on PATH only for -force:

    python3 scripts/build_meshes.py -verify    # check shipped meshes, no writes
    python3 scripts/build_meshes.py -force     # re-mesh with gmsh, overwrite

With the gmsh shipped in the petgem-env image, -force reproduces the two
meshes BYTE-IDENTICALLY, so regenerating is safe there. Under a different gmsh
version the mesh may differ, which changes the discretization and therefore the
forward responses: rerun the whole pipeline (see the case README) and refresh
reference/reference_metrics.json. Check with -verify first.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
import retag_mesh  # noqa: E402

GEOM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "geometry")
KEYS = ("RX", "RZ", "AX", "AZT", "AZB")

TAG_AIR, TAG_BG, TAG_INVERT, TAG_ANOMALY = 1, 2, 3, 4
NAMES = {TAG_AIR: "AIR", TAG_BG: "BG", TAG_INVERT: "INVERT", TAG_ANOMALY: "ANOMALY"}


def classify(c, k, true_model):
    """Physical tag for a cell centroid c=(x,y,z); z is NEGATIVE down.

    RZ/AZT/AZB carry their sign, so the depth tests read as plain interval
    checks with the deeper (more negative) bound on the left.
    """
    x, y, z = c
    if z > 0.0:
        return TAG_AIR
    if abs(x) <= k["RX"] and abs(y) <= k["RX"] and k["RZ"] <= z <= 0.0:
        if (true_model and abs(x) <= k["AX"] and abs(y) <= k["AX"]
                and k["AZB"] <= z <= k["AZT"]):
            return TAG_ANOMALY
        return TAG_INVERT
    return TAG_BG


if __name__ == "__main__":
    retag_mesh.main(__doc__, "im_csem_block", KEYS, NAMES, classify, GEOM)
