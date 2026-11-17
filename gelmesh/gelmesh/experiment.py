"""The phantom experiment: variants, truth-derived anchors, ground-truth frame and error summaries.

Variants (combine with suffixes):
    CONNECTED    slab A is ONE mesh (slabreg today: a smooth warp cannot hold the crack's jump)
    JOINT        slab A cut into flap (A1) + main (A2); crack anchors; seam extension; one relaxation
    NONEIGHBOUR  slab A alone (no slab B): crack anchors + flatten only -- the crack-closure floor
    _EXT         crack-face strip matcher runs in rounds >= 1
    _DENSE       crack anchors at four z-levels, ~40 per level (stand-in for a perfect face matcher)
    _FEW         2 crack anchors per level instead of 6-8 (starved: what the matcher must make up for)

Anchors are drawn from the phantom's ground truth (what an annotator would place).  The error metric is the
displacement between each rendered voxel's source and the truth source, in the ROOT piece's frame (slab B's
own xy, both slabs flattened in z) -- springs fix relative placement only, so the frame has to be the root's.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from .geom import ZT, Y, X, ZS, ZV
from .phantom import side_of, crack_dist, dispA_xy, zcoefA, dispB_xy, zcoefB
from .solve import Problem, solve, render_solution


def parse_variant(variant):
    flags = dict(dense='_DENSE' in variant, ext='_EXT' in variant, few='_FEW' in variant)
    base = variant.replace('_DENSE', '').replace('_EXT', '').replace('_FEW', '')
    assert base in ('CONNECTED', 'JOINT', 'NONEIGHBOUR'), variant
    return base, flags


def build_problem(d, variant, n_anchor=8, seed=5):
    rng = np.random.default_rng(seed)
    base, fl = parse_variant(variant)
    A, tA, lA, sA, B, tB, lB, sB = (d[k] for k in ('A', 'tA', 'lA', 'sA', 'B', 'tB', 'lB', 'sB'))
    if base == 'CONNECTED':
        pieces = ['A', 'B']; labels = {'A': (np.where(tA, 1, 0).astype(np.int8), 1), 'B': (lB, 2)}; pol = {1: 'A', -1: 'A', 2: 'B'}
    elif base == 'JOINT':
        pieces = ['A1', 'A2', 'B']; labels = {'A1': (lA, 1), 'A2': (lA, -1), 'B': (lB, 2)}; pol = {1: 'A1', -1: 'A2', 2: 'B'}
    else:
        pieces = ['A1', 'A2']; labels = {'A1': (lA, 1), 'A2': (lA, -1)}; pol = {1: 'A1', -1: 'A2'}
    vols = {'A': A, 'A1': A, 'A2': A, 'B': B}
    tissue = {'A': tA} if base == 'NONEIGHBOUR' else {'A': tA, 'B': tB}
    pb = Problem(vols, labels, pieces, pol, root='B' if 'B' in pieces else 'A2', tissue=tissue,
                 seam=('A', 'B') if 'B' in pieces else None, side_labvol=lA)
    if base != 'CONNECTED': pb.crack_faces = ('A1', 'A2')
    # --- seam anchors: n_anchor per crack side, away from the crack, truth partner on B's top face
    if pb.seam:
        dcr = crack_dist(np.stack(np.mgrid[0:Y, 0:X][::-1], -1).reshape(-1, 2)).reshape(Y, X)
        for lab in (1, -1):
            cand = np.argwhere(pb.okA & (pb.labA_face == lab) & (dcr > 20))
            for y, x in cand[rng.choice(len(cand), n_anchor, replace=False)]:
                a = np.array([x, y, pb.zA[y, x]]); s = sA[int(round(pb.zA[y, x])), y, x]
                tp = s[:2]; u = dispB_xy(tp[None])[0]; bx, by = tp[0] + u[0], tp[1] + u[1]
                iy, ix = int(round(by)), int(round(bx))
                if not (0 <= iy < Y and 0 <= ix < X): continue
                bz = pb.zB[iy, ix]                                        # B's top face at that xy: what an annotator pairs
                if not np.isfinite(bz): continue
                b = np.array([bx, by, bz])
                pb.seam_anchor.append((pol[lab], a, 'B', b))
    # --- crack anchors: points on the flap face and their truth-adjacent partner on the main face
    if base != 'CONNECTED':
        for zc in ((10, 20, 30, 40) if fl['dense'] else (14, 30)):
            lab = lA[zc]; tis = tA[zc]
            fe = (lab > 0) & ndi.binary_dilation((lab < 0) | ~tis); fy, fx = np.where(fe)
            mp = np.argwhere((lab < 0) & ndi.binary_dilation((lab > 0) | ~tis, iterations=3))
            tr = cKDTree(sA[zc][mp[:, 0], mp[:, 1], :2]); order = np.argsort(fx)
            per_level = 40 if fl['dense'] else (2 if fl['few'] else 6)
            for j in order[::max(1, len(order)//per_level)]:
                pf = sA[zc, fy[j], fx[j], :2]; dd, i = tr.query(pf)
                if dd < 14:
                    my, mx = mp[i]; a = np.array([fx[j], fy[j], zc], float); b = np.array([mx, my, zc], float)
                    T = (b - a) + (sA[zc, fy[j], fx[j]] - sA[zc, my, mx])   # truth offset between the two source points
                    pb.crack_pairs.append(('A1', a, 'A2', a + T))
    return pb, base, fl


def truth_source(frame='B'):
    """Gauge-free truth: the global frame is the root slab's own XY frame with both slabs flattened in z
    (A -> [0,ZS), B -> [ZS,2ZS)).  For each global voxel g the truth-local point p satisfies
    p_xy + u_root(p_xy) = g_xy; its source in its own slab volume is that slab's forward map of p.
    Returns (source xyz (ZT,Y,X,3), truth-local xy of each column (Y,X,2))."""
    yy, xx = np.mgrid[0:Y, 0:X]; G = np.stack([xx.ravel(), yy.ravel()], 1).astype(float); P = G.copy()
    if frame == 'B':
        for _ in range(30): P = G - dispB_xy(P)
    else:
        for _ in range(30): P = G - dispA_xy(P, np.full(len(P), -1))
    sa = side_of(P); uA = dispA_xy(P, sa); aA, bA = zcoefA(P, sa); aB, bB = zcoefB(P)
    out = np.zeros((ZT, Y, X, 3), np.float32)
    for z in range(ZT):
        if z < ZS: src = np.c_[P + uA, aA + (1 + bA) * z]               # truth plane z of slab A -> global z
        else:      src = np.c_[G, aB + (1 + bB) * (z - ZS)]           # truth plane z-ZS of slab B -> global z
        out[z] = src.reshape(Y, X, 3)
    return out, P.reshape(Y, X, 2)


def regions(P, border=24):
    """P: truth-local xy of each global column (from truth_source), so the crack band follows the crack"""
    zz = np.arange(ZT)[:, None, None] * np.ones((1, Y, X), np.int8)
    dcr = crack_dist(P.reshape(-1, 2)).reshape(Y, X)
    b = np.zeros((Y, X), bool); b[:border] = b[-border:] = True; b[:, :border] = b[:, -border:] = True
    inner = ~np.broadcast_to(b, (ZT, Y, X))
    near_crack = (np.broadcast_to(dcr, (ZT, Y, X)) < 80) & (zz < ZS) & inner
    near_seam = (np.abs(zz - ZS) < 6) & inner
    far = ~near_crack & ~near_seam & inner
    return dict(near_crack=near_crack, near_seam=near_seam, far=far), zz


def run_variant(d, variant, log=print, rounds=3, n_anchor=8, seed=5, truth=None, config=None):
    pb, base, fl = build_problem(d, variant, n_anchor, seed)
    sol = solve(pb, rounds=rounds, crack_ext=fl['ext'], connected=(base == 'CONNECTED'), log=log, config=config)
    out, chosen, src = render_solution(pb, sol, (ZT, Y, X))
    TS, P = truth if truth is not None else truth_source('A2' if base == 'NONEIGHBOUR' else 'B')
    err = np.linalg.norm(src - TS, axis=-1); exy = np.linalg.norm(src[..., :2] - TS[..., :2], axis=-1); valid = chosen >= 0
    reg, zz = regions(P)
    if base == 'NONEIGHBOUR': valid &= zz < ZS
    inA = (zz < ZS) if base == 'NONEIGHBOUR' else np.ones_like(valid)
    # per-piece masks near the crack: the matcher's effect lives on the flap and is invisible in a mixed number
    flap = chosen == sol['pieces'].index('A1') if 'A1' in sol['pieces'] else np.zeros_like(valid)
    main = chosen == sol['pieces'].index('A2') if 'A2' in sol['pieces'] else np.zeros_like(valid)
    reg['near_crack_flap'] = reg['near_crack'] & flap; reg['near_crack_main'] = reg['near_crack'] & main
    if base == 'NONEIGHBOUR': reg.pop('near_seam')          # no seam without a neighbour
    stats = {}
    for name, m in reg.items():
        mm = m & valid & inA
        if not mm.any(): continue
        stats[name] = dict(median=float(np.median(err[mm])), p90=float(np.percentile(err[mm], 90)),
                           xy_median=float(np.median(exy[mm])), xy_p90=float(np.percentile(exy[mm], 90)),
                           unfilled=float((~valid & m & inA).sum() / max(1, (m & inA).sum())))
        log(f"  {name}: median {stats[name]['median']:.2f} vx, 90th {stats[name]['p90']:.2f} (xy {stats[name]['xy_median']:.2f} / {stats[name]['xy_p90']:.2f}), unfilled {stats[name]['unfilled']*100:.1f}%")
    return dict(variant=variant, pieces=sol['pieces'], x=sol['x'], out=out, chosen=chosen, err=err.astype(np.float16), valid=valid,
                stats=stats, history=sol['history'], n_seam=len(sol['seam_pairs']), n_crack=len(pb.crack_pairs),
                n_crack_auto=len(sol['crack_auto']), seam_pairs=sol['seam_pairs'], crack_pairs=pb.crack_pairs, crack_auto=sol['crack_auto'])
