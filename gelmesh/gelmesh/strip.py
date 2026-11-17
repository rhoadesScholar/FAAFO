"""Crack-face strip matcher: automatic extension of crack correspondences from a current (anchor-level) solution.

Works in the GLOBAL (relaxed) frame, where the two crack pieces already sit roughly in place.  At a set of
z-planes, every 16-px grid point near the midline between piece 1 and piece 2 tissue becomes a strip centre;
the local normal n (piece 2 -> piece 1) and tangent t come from the two distance transforms.  Per strip:

  * the ACROSS-crack offset comes from gap closure: the per-t first tissue row of piece 1 (n1) and last row of
    piece 2 (n2) should be adjacent, so the normal correction is -(median(n1 - n2) - 1);
  * the ALONG-crack offset comes from texture continuity: a thin shell of piece-1 tissue just inside its face,
    mirrored against the shell of piece 2 just inside its face, ZNCC'd over tangential shifts.  A neurite or
    nucleus that crossed the crack appears in both shells at the same t, so the ZNCC peaks at the true slide.

Gates: minimum ZNCC, margin of the best peak over the best peak elsewhere (>3 px away), and neighbour
consensus among strips within `cons_radius` px in the same plane.  (A round trip is a no-op here: matching
2 -> 1 is the same ZNCC with the roles swapped, so it always returns -dt.  Consensus along the crack is the
gate that does that job.)  Output: pairs of global-frame points (p1 on piece 1's face, p2 on piece 2's face) that
should be adjacent after relaxation, plus the score; the caller maps them back to piece-source coordinates.
"""
import numpy as np
from scipy import ndimage as ndi


def strip_centres(m1, m2, spacing=16, reach=12):
    """strip centres ON piece 1's face: boundary pixels of m1 that lie within `reach` px of piece-2 tissue,
    greedily thinned to `spacing`; local normal = inward normal of piece 1 (points from piece 2 into piece 1).
    Anchoring on one face keeps the geometry defined whether the pieces currently overlap or leave a gap."""
    b1 = m1 & ~ndi.binary_erosion(m1, iterations=2, border_value=1)   # the image border is not a face
    d2 = ndi.distance_transform_edt(~m2)
    cand = np.argwhere(b1 & (d2 < reach))                     # (y, x)
    if len(cand) == 0: return []
    sd = ndi.distance_transform_edt(m1) - ndi.distance_transform_edt(~m1)   # signed: + inside piece 1
    sd = ndi.gaussian_filter(sd, 2.0)
    gyy, gxx = np.gradient(sd)
    kept = []
    for y, x in cand[np.argsort(cand[:, 1] * m1.shape[0] + cand[:, 0])]:
        if kept and np.min(np.hypot(*(np.array(kept) - [x, y]).T)) < spacing: continue
        kept.append((x, y))
    out = []
    for x, y in kept:
        n = np.array([gxx[y, x], gyy[y, x]]); nn = np.linalg.norm(n)
        if nn < 1e-3: continue
        n /= nn; t = np.array([-n[1], n[0]])
        out.append((np.array([x, y], float), n, t))
    return out


def _sample(img, pts):
    """img (Y,X), pts (...,2) xy -> bilinear sample, 0 outside"""
    return ndi.map_coordinates(img, [pts[..., 1], pts[..., 0]], order=1, mode='constant', cval=0.0)


def _local(c, n, t, L, D):
    tt, nn = np.mgrid[-L//2:L//2, -D:D+1].astype(float)        # (L, 2D+1)
    return c + tt[..., None]*t + nn[..., None]*n, tt[:, 0], nn[0]


def _zncc(a, b, ok, min_valid):
    """mean over shell rows (columns of a, b) of the per-row ZNCC along t, each row normalised over its own
    valid samples (the row at the face is darkened by blending with the empty side; rows differ in gain)."""
    cnt = ok.sum(0); use = cnt >= min_valid
    if not use.any(): return -1.0
    a = a[:, use]; b = b[:, use]; ok = ok[:, use]; cnt = cnt[use]
    a = np.where(ok, a - (a.sum(0) / cnt), 0.0); b = np.where(ok, b - (b.sum(0) / cnt), 0.0)
    num = (a*b).sum(0); den = np.sqrt((a*a).sum(0) * (b*b).sum(0)); good = den > 0
    return float((num[good] / den[good]).mean()) if good.any() else -1.0


def _shells(W, M, c, n, t, L, D, depth, zsel):
    """W, M: (2, nz, Y, X).  Shells over the face patch (t x z): per plane in `zsel`, the boundary rows n1, n2
    and `depth` rows just inside each face.  Returns n1 (nz, L), n2 (nz, L), S1, S2 (L, nz*depth): row-major
    columns (plane, depth) so that _best_shift's per-column normalisation treats every (plane, depth) row alike."""
    P, tt, nn = _local(c, n, t, L, D)
    nz = len(zsel); n1 = np.full((nz, L), np.nan); n2 = np.full((nz, L), np.nan)
    S1 = np.full((L, nz*depth), np.nan); S2 = np.full((L, nz*depth), np.nan)
    for zi, z in enumerate(zsel):
        m1 = _sample(M[0, z].astype(np.float32), P) > 0.5; m2 = _sample(M[1, z].astype(np.float32), P) > 0.5
        w1 = _sample(W[0, z], P); w2 = _sample(W[1, z], P)
        for i in range(L):
            a = np.where(m1[i])[0]; b = np.where(m2[i])[0]
            if len(a):
                n1[zi, i] = nn[a.min()]; k = a.min() + 1                 # one row inside the face
                if k + depth <= 2*D+1: S1[i, zi*depth:(zi+1)*depth] = w1[i, k:k+depth]
            if len(b):
                n2[zi, i] = nn[b.max()]; k = b.max() - 1
                if k - depth + 1 >= 0: S2[i, zi*depth:(zi+1)*depth] = w2[i, k-depth+1:k+1][::-1]
    return n1, n2, S1, S2


def _best_shift(S1, S2, S, min_valid):
    """ZNCC of S1 shifted by dt (S1[i+dt] vs S2[i]) over dt in [-S, S]. Returns (dt, score, margin)."""
    L = S1.shape[0]; scores = np.full(2*S+1, np.nan)
    for j, dt in enumerate(range(-S, S+1)):
        i0, i1 = max(0, -dt), min(L, L-dt)
        a = S1[i0+dt:i1+dt]; b = S2[i0:i1]; ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum(0).max() < min_valid: continue
        scores[j] = _zncc(np.where(ok, a, 0.0), np.where(ok, b, 0.0), ok, min_valid)
    if not np.isfinite(scores).any(): return None
    j = int(np.nanargmax(scores)); dt = j - S
    far = np.abs(np.arange(2*S+1) - j) > 3
    margin = scores[j] - (np.nanmax(scores[far]) if np.isfinite(scores[far]).any() else -1.0)
    return dt, float(scores[j]), float(margin)


def match_strips(W, M, z0, zwin=8, L=64, D=12, depth=3, S=12, min_score=0.3, min_margin=0.08,
                 spacing=16, cons_radius=48, cons_dev=4.0, max_gap=10):
    """W, M: (2, nz, Y, X) warped images / tissue masks of piece 1 and piece 2 over a stack of global planes.
    Strips are seeded from the masks at plane index z0 and matched over planes z0-zwin..z0+zwin (the face is
    a surface; one plane carries too little texture).  Returns [(p1, p2, score, dn, dt)]: p1 carried by piece 1
    and p2 by piece 2, both on the boundary line between the two faces at plane z0 (global xy), that should
    COINCIDE after relaxation."""
    nz = W.shape[1]; zsel = [z for z in range(z0-zwin, z0+zwin+1) if 0 <= z < nz]; zc = zsel.index(z0)
    cand = []
    for c, n, t in strip_centres(M[0, z0], M[1, z0], spacing):
        n1, n2, S1, S2 = _shells(W, M, c, n, t, L, D, depth, zsel)
        both = np.isfinite(n1[zc]) & np.isfinite(n2[zc])
        if both.sum() < L//2: continue
        gap = np.median(n1[zc][both] - n2[zc][both]) - 1.0        # >0: open gap of that many px; <0: overlap
        if abs(gap) > max_gap: continue
        r = _best_shift(S1, S2, S, L//3)
        if r is None: continue
        dt, sc, mg = r
        if sc < min_score or mg < min_margin or abs(dt) > S - 3: continue     # a peak at the edge of the search is not a peak
        i1 = L//2 + dt; i2 = L//2
        if not (0 <= i1 < L and np.isfinite(n1[zc, i1]) and np.isfinite(n2[zc, i2])): continue
        # both points on the shared boundary line between the two face rows (half a pixel outside each face),
        # so "coincide" is the right spring target; putting them ON the face rows would over-close by 1 px
        p1 = c + dt*t + (n1[zc, i1] - 0.5)*n; p2 = c + (n2[zc, i2] + 0.5)*n
        cand.append(dict(c=c, n=n, t=t, p1=p1, p2=p2, score=sc, margin=mg, dn=-gap, dt=dt))
    # neighbour consensus on the tangential shift, expressed as a global-frame vector
    out = []
    if not cand: return out
    C = np.array([k['c'] for k in cand]); V = np.array([k['dt']*k['t'] + k['dn']*k['n'] for k in cand])
    for i, k in enumerate(cand):
        nb = np.where((np.linalg.norm(C - C[i], axis=1) < cons_radius) & (np.arange(len(cand)) != i))[0]
        if len(nb) < 2: continue
        med = np.median(V[nb], 0)
        if np.linalg.norm(V[i] - med) > cons_dev: continue
        out.append((k['p1'], k['p2'], k['score'], k['dn'], k['dt']))
    return out
