# Handoff: a slabreg branch that registers through joint mesh relaxation

For the session working on JM238 in `e11bio/slabreg`. Goal: a branch that offers **joint relaxation** as an
alternative registration route — crack closing, seam registration and the flatten solved as one spring-mesh
energy over tissue *pieces* — behind a config switch, leaving today's path untouched.

## Where things are

- POC repo: `gelmesh` (REPO_URL). Read `README.md` first; the spec lives at
  https://claude.ai/artifact/JfjoUoi3SDT4HPck8rP3Eq (sections "Joint relaxation" and "Mapping onto slabreg").
- Validated on a two-slab phantom with ground truth (`gelmesh run`); the README table has the numbers and
  the caveats, and an independent review pass is folded in (it caught a metric artefact and three
  conventions bugs; the README says which).
- Test suite: `python -m pytest -q tests` (unit level, no phantom). Benchmark: `gelmesh bench`.

## What the branch should do (first implementation = 2-D, on the face; the 3-D mesh is follow-on)

The POC's solver is a 3-D mesh with column coupling (`K_COL`) that makes the xy warp uniform through z —
which is exactly slabreg's model (per-column flatten + one xy field per face). So the first slabreg
implementation can be **2-D on the cracked face** and hand s05 a composite per-pixel field through the
path that already exists for the model-free engine. Nothing downstream changes.

```
s02a  flat faces (as today)                      ──┐
s02c  NEW: crack mask + side labels per face       │  pieces = connected components of tissue & ~crack
s05   NEW engine `joint`:                          │
        seeds per interface (seam AND crack) ──────┤  landmarks serve gains interface kind "crack"
        extend: grid_extend per piece (seam),      │  masked per piece: the consensus gate must not look across the crack
                strip matcher (crack)              │  port gelmesh/strip.py
        relax: pieces + interface springs          │  2-D: one lattice per piece on the face; or SOFIMA mesh
        sample field per piece, composite by label ┘  -> write_per_slab_elastic_fields(prebuilt=..., warp='mls')
s06a  score whole face + a crack-band score
s07   unchanged (renders flatten + field in one pull)
```

### Contract to meet (verified against `src/slabreg` on 2026-11-15; re-verify, it moves)

| piece | what the code does | what the branch must do |
|---|---|---|
| prebuilt field (`s05_landmarks.py` ~1024, 1066–1112) | backward root→gel map, `disp[0]=dy, disp[1]=dx`, run-scale px, `origin_yx` in root px; `ess` = unitless confidence raster; bypasses `decide` and the bake; written as `warp: mls` internally (no `warp=` kwarg); the loop runs over `per_gel_fits` and stamps `engine: model_free` + `_model_free_provenance` unconditionally | write the composite directly. **No NaN** anywhere (non-finite samples refuse under the default gate). The joint engine must still build a `PerGelFit` per slab (as `_lattice_pairs_from_field` + `fit_from_points` do, ~2456) for band sizing/compose, and the stamp needs a `joint` engine label. The piece-label raster used for compositing must be in the ROOT (post-warp) frame, chosen as `gelmesh.relax.render` does (which piece's backward map lands on its own tissue), not the moving-face `crack_mask`; overlaps need a tie rule, residual gaps take the nearer piece |
| anisotropy gate (`warp_qa.assess_field`, `config.py` ~1179) | p99 of strided-gradient singular-value ratio vs `max_anisotropy` 1.7, action `fail`; `folded_frac > 0` only warns | a jump is a one-sample ridge, so p99 passes unless the crack covers > 1 % of samples; expect the fold warning. Exclude a crack band from the stats rather than loosening the gate |
| extender (`landmark_extend.py` ~441–487, `ng_zncc_match.py` ~39–57, `extend_seams.py` ~124) | no mask argument; faces arrive in the face slab's dtype (uint8 on jm200/JM238); the no-data gate counts NON-FINITE pixels (25 % per window) and `_window` fills NaN with the finite mean; neighbour consensus k=8 within 12 px; round trip 1.5 px | **zeroing the other piece does not engage the no-data gate** (zeros are finite) and makes windows lock onto the tissue/zero edge. Either cast the face to float32 and set the other piece to NaN before `grid_extend`, or add a mask argument upstream in vreg's `grid_extend` (check its nodata test is finite-based too). Never let consensus look across the crack |
| s07 fade (`s07_tps_warp.py` ~264–331, `config.py` ~1316–1336, ~2061–2094) | the field fades toward `tps.far_level` away from the anchor face `seam_side_for_root`; with `tps.fade_extent: None` (default) w falls linearly from the anchor face to the slab's most distant plane | a crack that runs through the slab re-opens **progressively through the slab**, not only at the far face. Acceptable on JM238 only if the cracked face **is** that slab's anchor face and the crack is shallow — check `seam_side_for_root` and look at mid-slab planes before committing |
| s06a NCC (`s06a` ~429–533) | per-64-px-block NCC masked by `valid = va & vb`, area-weighted median; gates `ncc_gain ≥ 0.01`, `measured_frac ≥ 0.5` | AND a crack-band mask into `valid` for a second, band-only score; report both, gate on the whole face |
| mask carrier | none exists (`absent_sides` drops a whole face; `data_window` is s02-only; surface-QC classes are not read by s05–s07) | new config key `interfaces[].crack_mask` (path to a 2-D uint8 zarr in the face frame) is the minimum |
| `tps` bake | a baked TPS lattice cannot hold a jump; `FieldCache` reuses fields by content key | force `warp: mls` for a cracked face; make the crack mask + piece labels part of the stamp inputs so markers don't resume a crack-less solve |

### Things from `CLAUDE.md` that will bite here

- Use `~/miniforge3/envs/slabreg/bin/python`; sibling packages (volara*, e11-models) are MATERIALIZED —
  editing a checkout changes nothing until reinstalled. Never run from `/home`; read/write `s3://e11-hpc/...`
  directly, never `/shared/s3/e11-hpc`.
- `--rerun` does not re-flatten a face slab that validates; a change to what the face carries must bump
  `SLAB_FORMAT_VERSION`. A new stage number must be placed in `pipes.STAGE_STEPS`. Stage order is
  `(number, sub)`; a new sub-stage is e.g. `s02c`.
- `DatasetConfig` is `extra="forbid"`: every new key needs a docs entry (`docs/config/dataset-yaml.md`) and
  `test_docs_configref.py` / `test_run_yamls_load.py` must stay green. Real dataset YAMLs live in the PM repo.
- s05's blockwise solve takes `.seam-solve.lock`; a crashed run leaves it behind on purpose.
- Manual CSV frame: BigWarp six columns `(x, y)` in `landmark.landmark_scale` pixels, must match `--scale`
  for `landmarks serve`; a crack "interface" served between two pieces of the SAME face slab needs the side
  label as a layer so the annotator knows which side a point is on.
- The two seam-face artifacts have different frames (`intermediates/flat_faces` pre-warp vs
  `registration/seam_faces` post-warp). The crack branch works in the pre-warp face frame.

### Suggested order of work

0. **Gauge.** Decide the seam convention once: a slab's first tissue plane → 0 and its LAST plane → ZS−1
   (slab-local), so the next slab starts at ZS and no plane is double-booked. The POC's first metric run
   got this wrong and reported a 5× seam-band gain that was really the double-booked plane.
1. **Pieces.** `s02c crack-pieces`: read the face slab, gap mask from no-tissue + (optional) slice-loss
   heightmap, thin to a curve, side labels by connected components; write `crack_mask` + `piece_label` zarrs
   in the face frame. For JM238 you can hand-draw the mask in neuroglancer first to unblock everything else.
2. **Seeds.** `landmarks serve --interface crack`: piece A1 face vs piece A2 face, both from one face slab;
   CSV name carries the piece ids. ~5 pairs per crack per z-level is enough with extension.
3. **Extension.** Seam: `grid_extend` per piece with the other piece zeroed. Crack: port `gelmesh/strip.py`
   (`strip_centres`, `_shells`, `_best_shift`, `match_strips`) — it is numpy-only, ~200 lines; its inputs are
   per-piece warped faces + masks over a z window (use the face slab's planes, `±tiff_half_width`).
4. **Relax.** 2-D first: one node lattice per piece (stride 16–32 px), springs = `(a, b, w)` tables from
   steps 2–3. The internal term must let a piece rotate rigidly at no cost (the flap rotates ~0.1 rad on
   the phantom; a plain linear-spring lattice penalises that and under-rotates). Two options: reuse
   SOFIMA's `relax_mesh` with `gelmesh.relax.make_force` (what the POC does; ~100 ns/step/node on CPU,
   1000 FIRE steps converge), or fit each piece's affine from its pairs first and solve the linearised
   residual as a sparse least-squares (`lsqr`/CG). Carry pairs onto nodes bilinearly, not nearest-node.
   Two to three rounds (extend → relax).
5. **Composite field** per pixel by piece label → `write_per_slab_elastic_fields(prebuilt=…)`.
6. **QC.** s06a crack-band score; s11 dip as the acceptance number. Compare against today's engine on the
   same seam.

### Validation plan on JM238

- Which seam / which face is cracked: FILL IN (dataset YAML path in the PM repo, `interfaces[]` entry).
- Run the existing engine and the `joint` engine on that seam at the same `--scale`; report s06a whole-face
  NCC, crack-band NCC, and s11 dip for both. Expect the whole-face numbers close and the crack band to move.
- Check the far-face re-opening question (fade) by rendering the slab's other face.
- Visual: `registration/seam_faces` planes above/below the seam with the crack band marked.

### Open questions for Jeff

- Repo destination for `gelmesh` (create `e11bio/gelmesh`, or vendor `strip.py` into slabreg only?).
- JM238 dataset YAML path and the cracked interface.
- Whether the 3-D mesh route (SOFIMA `relax_mesh`, pieces through the slab depth, no far-face re-opening)
  should be scoped now or after the 2-D branch lands.
