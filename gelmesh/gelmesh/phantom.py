"""Two-slab phantom for the joint relaxation test.
Truth volume T (80, Y, X) is cut into slab A = T[0:40] (top) and slab B = T[40:80] (bottom).
Slab A: a crack from the interior to the slab edge; the flap rotates about the crack tip and slides
(open gap along part of the crescent, closed sheared seam along the rest), tilts in z, and the whole
slab has a gentle z-bend.  Slab B: an independent smooth XY warp (rotation + translation + wobble)
and its own z-bend.  Each slab is rendered into its own 48-plane volume (curved surfaces, no-tissue
elsewhere).  Axes: arrays (z, y, x); points (x, y, z)."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from matplotlib.path import Path as _Path
from .geom import ZT, Y, X, ZS, ZV
rng = np.random.default_rng(23)

# ---------------- crack geometry in slab A (truth XY coords) ----------------
H = np.array([150., 150.]); K = np.array([430., 140.]); E = np.array([558., 0.])
_t = np.linspace(0, 1, 300)
_cres = np.stack([H[0] + (K[0]-H[0])*_t, H[1] + 110*np.sin(np.pi*_t) + 10*np.sin(2*np.pi*_t)], 1)
_seg = np.stack([K[0] + (E[0]-K[0])*_t, K[1] + (E[1]-K[1])*_t], 1)[1:]
CRACK = np.concatenate([_cres, _seg], 0)
_tree = cKDTree(CRACK)
_FLAP = _Path(np.concatenate([CRACK, [[E[0], -1.], [H[0]-200, -1.], [H[0]-200, H[1]]]], 0))
def side_of(xy):
    """+1 flap, -1 main (slab A only)"""
    return np.where(_FLAP.contains_points(xy), 1, -1)
def crack_dist(xy): return _tree.query(xy)[0]

THETA, TX, TILT, LAM = -0.09, 24.0, 10.0, 140.0
def _taper(x):
    t = np.clip((x - H[0]) / 120., 0, 1); return t*t*(3-2*t)
def flap_disp(xy):
    r = xy - H; c, s = np.cos(THETA), np.sin(THETA)
    rot = np.stack([c*r[:, 0] - s*r[:, 1], s*r[:, 0] + c*r[:, 1]], 1)
    return _taper(xy[:, 0])[:, None] * (rot - r + np.array([TX, 0.]))

# ---------------- forward maps (truth-local slab coords -> slab volume coords) ----------------
def dispA_xy(xy, side):
    uf = flap_disp(xy); d, i = _tree.query(xy)
    um = 0.35 * flap_disp(CRACK[i]) * np.exp(-d / LAM)[:, None]
    return np.where((side > 0)[:, None], uf, um)
def zcoefA(xy, side):
    """z' = z + a + b z : flap tilt (grows with distance from hinge) + slab bend + margin"""
    rad = np.linalg.norm(xy - H, axis=1)
    tilt = np.where(side > 0, TILT * np.clip(rad / 400., 0, 1), 0.0)
    a = 4.0 + tilt * 0.75 + 3.0 * np.sin(np.pi * xy[:, 0] / X) + 2.0 * np.cos(np.pi * xy[:, 1] / Y)
    b = tilt * 0.2 / ZS
    return a, b
THB, TB = 0.035, np.array([-14.0, 9.0])
def dispB_xy(xy, side=None):
    c0 = np.array([X/2, Y/2]); r = xy - c0; c, s = np.cos(THB), np.sin(THB)
    rot = np.stack([c*r[:, 0] - s*r[:, 1], s*r[:, 0] + c*r[:, 1]], 1) - r
    wob = np.stack([6*np.sin(2*np.pi*xy[:, 1]/Y), 5*np.cos(2*np.pi*xy[:, 0]/X)], 1)
    return rot + TB + wob
def zcoefB(xy, side=None):
    a = 4.0 - 2.5 * np.sin(np.pi * xy[:, 0] / X + 0.6) + 3.0 * np.sin(np.pi * xy[:, 1] / Y)
    return a, np.full(len(xy), 0.03)   # mild through-thickness stretch

# ---------------- pristine volume ----------------
def pristine():
    v = np.full((ZT, Y, X), 60.0, np.float32)
    for _ in range(620):
        p0 = rng.uniform([0, 0, 0], [X, Y, ZT]); ang = rng.uniform(0, np.pi); el = rng.uniform(-0.2, 0.2)
        tt = np.linspace(-1, 1, 500); amp = rng.uniform(10, 40); fr = rng.uniform(1, 3); ph = rng.uniform(0, 6)
        xs = p0[0] + 350*tt*np.cos(ang) + amp*np.sin(fr*np.pi*tt+ph)*np.sin(ang)
        ys = p0[1] + 350*tt*np.sin(ang) - amp*np.sin(fr*np.pi*tt+ph)*np.cos(ang)
        zs = p0[2] + 350*tt*el + 4*np.sin(2*np.pi*tt+ph)
        ok = (xs >= 0) & (xs < X) & (ys >= 0) & (ys < Y) & (zs >= 0) & (zs < ZT)
        v[zs[ok].astype(int), ys[ok].astype(int), xs[ok].astype(int)] = 255
    v = ndi.gaussian_filter(v, (0.7, 1.2, 1.2)) * 1.6
    zz, yy, xx = np.mgrid[0:ZT, 0:Y, 0:X]
    for _ in range(840):
        c = rng.uniform([3, 6, 6], [ZT-3, Y-6, X-6]); r = rng.uniform(4, 7)
        m = ((xx-c[2])**2 + (yy-c[1])**2 + ((zz-c[0])*1.3)**2) < r*r
        v[m] = np.maximum(v[m], 170)
    v += rng.normal(0, 6, v.shape).astype(np.float32)
    v[:, :, ::50] += 25; v[:, ::50, :] += 25; v[::10] += 12
    return np.clip(v, 0, 255).astype(np.float32)

# ---------------- render one slab ----------------
def render_slab(vol40, disp_xy, zcoef, sides):
    """vol40: truth slab (ZS,Y,X). Returns slab volume (ZV,Y,X), tissue mask, piece label, src (x,y,z truth-local)."""
    yy, xx = np.mgrid[0:Y, 0:X]; Q = np.stack([xx.ravel(), yy.ravel()], 1).astype(np.float64)
    inv = {}
    for s in sides:
        side = np.full(len(Q), s); P = Q.copy()
        for _ in range(25): P = Q - disp_xy(P, side)
        res = np.linalg.norm(P + disp_xy(P, side) - Q, axis=1)
        ok = (res < 0.5) & (P[:, 0] >= 0) & (P[:, 0] <= X-1) & (P[:, 1] >= 0) & (P[:, 1] <= Y-1)
        if len(sides) > 1: ok &= side_of(P) == s
        inv[s] = (P, ok)
    out = np.zeros((ZV, Y, X), np.float32); tissue = np.zeros((ZV, Y, X), bool)
    label = np.zeros((ZV, Y, X), np.int8); src = np.full((ZV, Y, X, 3), np.nan, np.float32)
    for s in sides:
        P, ok = inv[s]; a, b = zcoef(P, np.full(len(P), s))
        for zi in range(ZV):
            z0 = (zi - a) / (1 + b)
            okz = ok & (z0 >= 0) & (z0 <= ZS-1) & ~tissue[zi].ravel()
            vals = ndi.map_coordinates(vol40, [z0, P[:, 1], P[:, 0]], order=1, mode='nearest')
            o = out[zi].ravel(); o[okz] = vals[okz]; out[zi] = o.reshape(Y, X)
            t_ = tissue[zi].ravel(); t_[okz] = True; tissue[zi] = t_.reshape(Y, X)
            l_ = label[zi].ravel(); l_[okz] = s; label[zi] = l_.reshape(Y, X)
            sr = src[zi].reshape(-1, 3); sr[okz] = np.stack([P[:, 0], P[:, 1], z0], 1)[okz]; src[zi] = sr.reshape(Y, X, 3)
    out[~tissue] = rng.normal(0, 3, (~tissue).sum())
    return out, tissue, label, src

def surfaces(tissue):
    """per-column top and bottom tissue plane (float, NaN where no tissue)"""
    any_ = tissue.any(0); top = np.where(any_, np.argmax(tissue, 0), np.nan).astype(float)
    bot = np.where(any_, ZV - 1 - np.argmax(tissue[::-1], 0), np.nan).astype(float)
    return top, bot

def build(log=print):
    """pristine volume -> two imaged slabs.  Returns the dict that `gelmesh.solve` consumes:
    T (truth), A/tA/lA/sA (slab A volume, tissue mask, piece label, truth-local source), B/tB/lB/sB."""
    import time
    t0 = time.time(); T = pristine(); log(f'pristine volume {round(time.time()-t0)}s')
    A, tA, lA, sA = render_slab(T[:ZS], dispA_xy, zcoefA, (1, -1)); log(f'slab A (cracked) {round(time.time()-t0)}s')
    B, tB, lB, sB = render_slab(T[ZS:], dispB_xy, zcoefB, (-1,)); lB[:] = np.where(tB, 2, 0); log(f'slab B {round(time.time()-t0)}s')
    return dict(T=T, A=A, tA=tA, lA=lA, sA=sA, B=B, tB=tB, lB=lB, sB=sB)


def load(path):
    """load a phantom pickle and zero the no-tissue voxels (the solver's inputs are the slab volumes as imaged)"""
    import pickle
    d = pickle.load(open(path, 'rb'))
    for k in ('A', 'B'): d[k] = np.where(d['t' + k], d[k], 0).astype(np.float32)
    return d
