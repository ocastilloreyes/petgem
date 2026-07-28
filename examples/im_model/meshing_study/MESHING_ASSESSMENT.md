# im_model meshing assessment — improving PCBDDC robustness without losing accuracy

> **STATUS:** M1 has been **promoted** to `geometry/im_model.geo` and re-meshed
> (`im_model.msh`/`im_true.msh` are now the smooth 10206-node mesh). Remaining
> step: regenerate `reference/observed.h5` from the forward stage on MN5 (§8),
> which is a solver run. Until then, the shipped `reference/observed.h5` is
> **stale** (it was forward-modelled on the old 7464-node mesh).

## Goal

Redesign the Gmsh input for the `im_model` CSEM inversion benchmark so the
generated meshes keep the same physics and resolution but avoid the interface
topologies that trigger the PCBDDC order-2 coarse-edge (`PCBDDCNedelecSupport`)
failure. This applies the outcome of the PCBDDC topology study, whose findings
are the premises here:

- the discrete gradient is mathematically correct (De Rham ≈ 1e-19, rank N−1,
  edge rows = order+1 at every order);
- the failure is **not** in the FEM formulation;
- the failure depends on mesh / interface topology — a smoother mesh of the
  **same physical model** eliminated it while passing the same FEM validation;
- abrupt mesh-size transitions induce the unfavourable interface connectivity;
- meshing affects not only geometric quality but the connectivity graph PCBDDC
  walks when it builds coarse edges.

## 1. Files reviewed

`examples/im_model` contains a single mesh input: `geometry/im_model.geo`
(meshed by `scripts/build_meshes.py`, which re-tags cells into AIR/BG/INVERT/
ANOMALY — the INVERT/ANOMALY regions are parameter **masks**, not geometric
volumes, so the mesh topology is set entirely by `im_model.geo`).

## 2. Aggressive features in the original `im_model.geo`

| choice | original value | why it is a risk |
|---|---|---|
| ROI box transition `Field[1].Thickness` | `400` | fine (h=40) jumps to far (h=10000) over only 400 m |
| anomaly box transition `Field[2].Thickness` | `200` | fine (h=25) jumps to far over only 200 m |
| radial gradient (MathEval `Field[3,4]`) | `0.32` /m | size grows steeply away from the survey/Tx |
| far-field cap `h_far` | `10000` | large end-member ⇒ large fine→coarse ratio |
| 3D algorithm | `Algorithm3D=1` (Delaunay) | poorer worst-element quality than HXT here |

All of these steepen the size gradient near the refined region and the z=0
air-earth interface, which raises interface-vertex valence and produces the
branching, non-chain interface connectivity PCBDDC's coarse-edge walk mishandles
at order 2.

## 3. Candidates generated and compared

Same geometry, same physical model, same **critical** sizes in every candidate
(`h_roi=40`, `h_an=25`, `h_near=40`) so anomaly/ROI resolution — and inversion
accuracy — is preserved. Only the meshing strategy changes.

| candidate | file | strategy |
|---|---|---|
| **M0** original | `../geometry/im_model.geo` | baseline (fails PCBDDC at order 2) |
| **M1** smooth | `im_model_smooth.geo` | Thickness 400/200→**1500/800**, gradient 0.32→**0.22**, h_far 10000→**5000**, Delaunay→**HXT** |
| **M2** smooth+interface | `im_model_smooth_iface.geo` | M1 **plus** ROI box extended a 200 m slab into the air (z=0 symmetric) |
| **M3** gentle | `im_model_gentle.geo` | even softer: Thickness 2500/1200, gradient 0.15, h_far 4000 (trend point) |

Reproduce: `gmsh -3 <candidate>.geo -o out.msh` then
`python3 analyze_candidates.py out.msh`.

## 4. Quantified comparison (`results/candidate_summary.tsv`)

| metric | M0 original | M1 smooth | M2 +interface | M3 gentle |
|---|---|---|---|---|
| nodes | 7464 | 10206 | 11137 | 19034 |
| tetrahedra | 44447 | 57866 | 63802 | 110438 |
| edges | 52297 | 69462 | 76329 | 131659 |
| **min dihedral (°)** | 9.16 | **11.02** | 10.55 | 12.78 |
| slivers (<5°) | 0 | 0 | 0 | 0 |
| max edge-ratio aspect | 7.1 | 7.3 | 6.8 | 5.7 |
| **max grading ratio** (size jump across a face) | 3.73 | **3.15** | 3.15 | 2.68 |
| faces with >2.0× jump | 0.16% | 0.14% | 0.13% | 0.02% |
| non-manifold faces / triple lines | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Euler χ | 1 | 1 | 1 | 1 |
| **z=0 interface max 3D valence** | 21 | **19** | **24** | 20 |
| z=0 interface max in-plane valence | 9 | 8 | 8 | 8 |
| global max vertex valence | 30 | 43 | 40 | 32 |

## 5. Findings

**(a) Softer grading is the effective lever.** From M0→M1→M3 the two topology
metrics that track the empirical PCBDDC outcome move monotonically in the good
direction: worst dihedral 9.16°→11.02°→12.78° and max grading ratio
3.73→3.15→2.68. M1 already removes the abrupt fine→coarse jump, and M1 is the
configuration **independently validated to eliminate the order-2 failure** on
this physical model.

**(b) Global max valence is a red herring.** M1's *global* max valence rose
(30→43) and its max edge-dof valence rose (53→75), yet M1 is the mesh that
works. Those high-valence vertices are **interior** to the refined region (HXT
packing), not on the interface: the z=0 **interface** max 3D valence *fell*
(21→19) and in-plane valence fell (9→8). The predictor of PCBDDC robustness is
**interface-and-grading complexity, not global max valence.**

**(c) The interface-symmetry idea (M2) backfires — measured, then rejected.**
Extending the fine box a slab into the air to make z=0 sit *inside* a uniform
region was intended to lower interface valence. It does the opposite: z=0 nodes
gain fine neighbours *above* as well as below, so interface max 3D valence rises
to **24** (worst of all candidates) and the mesh grows ~10% for no benefit.
**Do not refine across the air-earth interface.** Keep the fine box starting at
z=0 (earth side only) and let grading smoothness — not interface refinement —
control interface topology.

**(d) M3 shows diminishing returns.** Going gentler than M1 buys a little more
smoothness (3.15→2.68) and quality (11.0°→12.8°) but at **2.5× the node count**
(19034 vs 7464), i.e. a large forward/adjoint cost increase for a marginal
topology gain past the point where the failure is already gone.

**(e) The order-2 FEM structure is unchanged and is *not* the fix.** Every
candidate still has the intrinsic per-mesh-edge 2-clique at order 2
(`analyze_graph.py`); no chain-capable (valence-2) edges exist in any 3D mesh.
The mesh does not remove the 2-clique — it changes whether the partition
interfaces force PCBDDC to split those 2-cliques into odd-sized coarse edges.

## 6. Recommendation — new default meshing strategy

Adopt **M1** as the default for `examples/im_model`. Concretely, change
`geometry/im_model.geo`:

```diff
- h_roi = 40;  h_an = 25;  h_near = 40;  h_far = 10000;
+ h_roi = 40;  h_an = 25;  h_near = 40;  h_far = 5000;

- Field[1].Thickness=400;
+ Field[1].Thickness=1500;

- Field[2].Thickness=200;
+ Field[2].Thickness=800;

- Field[3]=MathEval; Field[3].F=Sprintf("%g + 0.32*sqrt(x^2 + y^2 + z^2)", h_near);
- Field[4]=MathEval; Field[4].F=Sprintf("%g + 0.32*sqrt(x^2 + (y-(%g))^2 + z^2)", h_near, SY);
+ Field[3]=MathEval; Field[3].F=Sprintf("%g + 0.22*sqrt(x^2 + y^2 + z^2)", h_near);
+ Field[4]=MathEval; Field[4].F=Sprintf("%g + 0.22*sqrt(x^2 + (y-(%g))^2 + z^2)", h_near, SY);

- Mesh.Algorithm3D=1;
+ Mesh.Algorithm3D=10;
```

`im_model_smooth.geo` in this directory is exactly that file, ready to promote.
**Do not** adopt M2 (interface refinement) or M3 (over-refined).

**Design rules for future im_model meshes:**
1. Keep the critical sizes (`h_roi`, `h_an`, `h_near`) — they set accuracy.
2. Grade **smoothly**: box `Thickness` ≳ 3–4× the local element size and a
   radial gradient ≤ ~0.22 /m, so no interior face jumps more than ~3×.
3. Cap the far field modestly (`h_far ≈ 5000`), not at 10000+.
4. Keep the fine box on the **earth side of z=0**; never refine across the
   air-earth interface.
5. Prefer `Algorithm3D=10` (HXT) for the better worst-element quality here.

## 7. Accuracy is preserved

Same domain, same boundary surface topology (χ=1, 0 non-manifold faces, 0 triple
lines), same anomaly and ROI element sizes. The mesh is ~30% larger (7464→10206
nodes) — an acceptable cost for solver robustness. Cell counts and forward
numbers **will** differ from the original mesh; that is expected and correct as
long as the anomaly/ROI resolution (unchanged here) keeps the fields physically
significant.

## 8. Adopting the change

Promoting M1 to `geometry/im_model.geo` regenerates `im_model.msh`/`im_true.msh`,
so the shipped `reference/observed.h5` (forward-modelled on the current
`im_true.msh`) must be regenerated to stay consistent:

```bash
# 1. promote (copy M1 over the default, or apply the diff above)
cp examples/im_model/meshing_study/im_model_smooth.geo examples/im_model/geometry/im_model.geo
# 2. re-mesh (petgem-env gmsh for reproducibility)
python3 examples/im_model/scripts/build_meshes.py --force
# 3. rerun the forward stage on MN5 and rebuild observed.h5
#    scripts/build_bundles.sh fm  ->  scripts/run_forward.slurm  ->  utils/make_observed.py
```
