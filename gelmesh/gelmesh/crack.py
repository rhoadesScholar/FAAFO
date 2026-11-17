"""Crack-face correspondence extension (the "strip matcher"): see `strip.py` for the matcher itself."""
import numpy as np
from .geom import X, Y, ZS
from .relax import warp_piece_planes
from . import strip


def _interp_src(src_plane, mask, q):
    """bilinear source coordinate at global xy q from the valid samples around it (None if none)"""
    x0, y0 = int(np.floor(q[0])), int(np.floor(q[1]))
    if not (0 <= x0 < X-1 and 0 <= y0 < Y-1): return None
    fx, fy = q[0] - x0, q[1] - y0
    w = np.array([(1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy]); idx = [(y0, x0), (y0, x0+1), (y0+1, x0), (y0+1, x0+1)]
    ok = np.array([mask[i] and np.isfinite(src_plane[i]).all() for i in idx])
    if not ok.any(): return None
    w = np.where(ok, w, 0.0); v = np.array([src_plane[i] if o else np.zeros(3) for i, o in zip(idx, ok)])
    return (w[:, None] * v).sum(0) / w.sum()


def crack_extend(x, pieces, labels, vols, p1='A1', p2='A2', zs=(6, 12, 18, 24, 30, 36), zwin=8, log=None, **kw):
    """strip-match the two crack pieces in the current global frame -> correspondence pairs in source coords
    [(p1, a, p2, b)] whose target is 'a and b coincide' (both points sit on the boundary line between the two
    face rows, see strip.match_strips).  Source coordinates are interpolated from each piece's source map."""
    allz = list(range(max(0, min(zs)-zwin), min(ZS, max(zs)+zwin+1)))
    W1, M1, S1 = warp_piece_planes(x, pieces, p1, vols[p1], labels, allz)
    W2, M2, S2 = warp_piece_planes(x, pieces, p2, vols[p2], labels, allz)
    W = np.stack([W1, W2]); M = np.stack([M1, M2])
    pairs, info = [], []
    for z in zs:
        k = allz.index(z)
        for q1, q2, sc, dn, dt in strip.match_strips(W, M, k, zwin=zwin, **kw):
            a = _interp_src(S1[k], M1[k], q1); b = _interp_src(S2[k], M2[k], q2)
            if a is None or b is None: continue
            pairs.append((p1, a.astype(float), p2, b.astype(float))); info.append((z, q1, q2, sc, dn, dt))
    if log: log(f'  crack extension: {len(pairs)} pairs over {len(zs)} planes')
    return pairs, info

