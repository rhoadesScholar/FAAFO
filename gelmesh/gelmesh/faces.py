"""Seam faces and their automatic correspondence extension.

The seam between slab A (moving) and slab B (fixed/root) is handled the way slabreg's s05 does it: a face image
per slab (the volume sampled on its detected surface), a few manual anchors -> affine -> coarse-to-fine targeted
SOFIMA flow on the faces (round 0), then slabreg-extender style warp-then-match in later rounds: the moving face
is resampled through the current mesh and the residual is matched with zero targeting, which removes the local
rotation/scale from the correlation.  Gates: residual to the seed (RES_GATE), 3x3 neighbour consensus (CONS_DEV).
SOFIMA's own `max_deviation` median filter is NOT used: it deletes the genuine step at a crack.
"""
import numpy as np
from scipy import ndimage as ndi
from sofima import flow_field, flow_utils
from .geom import X, Y, ZV, STEP, PATCH, FP, FS

def face(vol, surf, offset):
    yy, xx = np.mgrid[0:Y, 0:X]; z = surf + offset; ok = np.isfinite(z)
    img = np.zeros((Y, X), np.float32); img[ok] = ndi.map_coordinates(vol, [z[ok], yy[ok], xx[ok]], order=1)
    return img, ok, z

def affine_fit(a, b):
    """2-D affine b ~ M a + t from anchor pairs (x,y)"""
    A_ = np.c_[a[:, :2], np.ones(len(a))]; sol, *_ = np.linalg.lstsq(A_, b[:, :2], rcond=None); return sol

def targeting_from_affine(sol, side_mask):
    """pre_targeting_field for flow_field: offset (pre - post) at each face node, [2, ny, nx] (x, y)"""
    ny, nx = (Y - (FP[0]-FS[0]))//FS[0], (X - (FP[1]-FS[1]))//FS[1]
    gy, gx = np.mgrid[0:ny, 0:nx]; px = gx*FS[1] + FP[1]//2; py = gy*FS[0] + FP[0]//2
    P = np.stack([px.ravel(), py.ravel(), np.ones(px.size)], 1); Q = P @ sol
    f = np.stack([(Q[:, 0] - px.ravel()).reshape(ny, nx), (Q[:, 1] - py.ravel()).reshape(ny, nx)]).astype(np.float32)
    return f

def _down2d(a, k):
    y, x = (a.shape[0]//k)*k, (a.shape[1]//k)*k; return a[:y, :x].reshape(y//k, k, x//k, k).mean((1, 3))

CONS_DEV, RES_GATE = 10.0, 14.0     # px: neighbour-consensus deviation, residual-to-seed gate
def _consensus(f, max_dev=None, min_nb=3):
    """reject a flow vector that disagrees with the median of its valid neighbours (3x3 ring)"""
    max_dev = CONS_DEV if max_dev is None else max_dev
    out = f.copy(); ok = np.isfinite(f[0]); ny, nx = ok.shape
    for j in range(ny):
        for i in range(nx):
            if not ok[j, i]: continue
            ys, xs = slice(max(0, j-1), j+2), slice(max(0, i-1), i+2)
            nb = ok[ys, xs].copy(); nb[min(j, 1), min(i, 1)] = False
            if nb.sum() < min_nb: out[:, j, i] = np.nan; continue
            med = np.nanmedian(f[:, ys, xs][:, nb], axis=1)
            if np.hypot(*(f[:, j, i] - med)) > max_dev: out[:, j, i] = np.nan
    return out

def _at_positions(field_nodes, pos0, step, n_out, out_step):
    """resample a field given on node centres (entry i <-> position pos0 + i*step) onto SOFIMA's targeting
    grid (entry j <-> position j*out_step), edge-clamped.  SOFIMA reads a pre_targeting_field at index
    round((patch_start + patch//2) / targeting_step), i.e. entry j stands at position j*targeting_step --
    NOT at node i's own index, so a node field must be shifted by patch//(2*step) entries."""
    gy, gx = np.mgrid[0:n_out[0], 0:n_out[1]]
    c = [(gy*out_step[0] - pos0[0])/step[0], (gx*out_step[1] - pos0[1])/step[1]]
    return np.stack([ndi.map_coordinates(field_nodes[k], c, order=1, mode='nearest') for k in range(2)]).astype(np.float32)


def face_flow(pre, post, post_mask, tg, log=None):
    """Coarse-to-fine SOFIMA masked xcorr on the two faces (post = slab A's face, pre = slab B's face).
    Level 4 (patch 32 = 128 px, step 8 = 32 px) targeted by `tg` (a field on the fine node grid), then
    level 1 (patch 64, step 16) targeted by the level-4 flow; gates: residual to the coarse seed (RES_GATE)
    and 3x3 neighbour consensus (CONS_DEV).  All targeting fields are resampled onto SOFIMA's read positions."""
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    k = 4; cp, cs = (32, 32), (8, 8)
    pre_c, post_c = _down2d(pre, k), _down2d(post, k); m_c = _down2d(post_mask.astype(np.float32), k) > 0.5
    ny, nx = tg.shape[1:]
    node0 = (FP[0]//2, FP[1]//2)
    # coarse level: nodes at (cs*m + cp//2) coarse px; targeting grid entry j <-> j*cs coarse px = j*cs*k full px
    nyc = (pre_c.shape[0] - (cp[0]-cs[0]))//cs[0]; nxc = (pre_c.shape[1] - (cp[1]-cs[1]))//cs[1]
    tg_c = _at_positions(tg / k, (node0[0]/k, node0[1]/k), (FS[0]/k, FS[1]/k), (nyc + 2, nxc + 2), cs)
    fc = mfc.flow_field(pre_c, post_c, cp, cs, post_mask=m_c, max_masked=0.7, batch_size=1024,
                        pre_targeting_field=tg_c, pre_targeting_step=cs)
    fc = flow_utils.clean_flow(fc[:, None], min_peak_ratio=1.1, min_peak_sharpness=1.1, max_magnitude=0, max_deviation=0, dim=2)[:, 0]
    # coarse flow lives on coarse node centres (full px: cs*k*m + cp//2*k); plausibility vs the seed, then fill
    cpos0 = (cp[0]//2*k, cp[1]//2*k); cstep = (cs[0]*k, cs[1]*k)
    gy, gx = np.mgrid[0:fc.shape[1], 0:fc.shape[2]]
    cc = [(gy*cstep[0] + cpos0[0] - node0[0])/FS[0], (gx*cstep[1] + cpos0[1] - node0[1])/FS[1]]
    tg_on_c = np.stack([ndi.map_coordinates(tg[q], cc, order=1, mode='nearest') for q in range(2)])
    fc_full = fc * k
    bad = ~np.isfinite(fc_full[0]) | (np.hypot(*(fc_full - tg_on_c)) > 24)        # coarse residual to the seed must be plausible
    fc_full[:, bad] = tg_on_c[:, bad]
    # fine level: targeting grid entry j <-> j*FS full px; the coarse flow is on coarse node centres
    fcf = _at_positions(fc_full, cpos0, cstep, (ny + 2, nx + 2), FS)
    f = mfc.flow_field(pre, post, FP, FS, post_mask=post_mask, max_masked=0.7, batch_size=1024,
                       pre_targeting_field=fcf, pre_targeting_step=FS)
    f = flow_utils.clean_flow(f[:, None], min_peak_ratio=1.1, min_peak_sharpness=1.1, max_magnitude=0, max_deviation=0, dim=2)[:, 0]
    gy, gx = np.mgrid[0:f.shape[1], 0:f.shape[2]]
    cc = [(gy*FS[0] + node0[0] - cpos0[0])/cstep[0], (gx*FS[1] + node0[1] - cpos0[1])/cstep[1]]
    seed_on_nodes = np.stack([ndi.map_coordinates(fc_full[q], cc, order=1, mode='nearest') for q in range(2)])
    f[:, np.hypot(*(f - seed_on_nodes)) > RES_GATE] = np.nan                       # fine residual to the coarse seed
    f = _consensus(f)
    return f


def _dense_offset(tg):
    """offset field on the face node grid [2, ny, nx] -> dense [2, Y, X] (bilinear, edge-clamped)"""
    yy, xx = np.mgrid[0:Y, 0:X]; c = [(yy - FP[0]//2)/FS[0], (xx - FP[1]//2)/FS[1]]
    return np.stack([ndi.map_coordinates(tg[k], c, order=1, mode='nearest') for k in range(2)])

def warp_and_match(faceB, faceA, maskA_valid, tg):
    """slabreg-extender style: resample the moving face (A) into the fixed frame (B) through the current warp,
    match with zero targeting, and return pairs (a in A-face coords, b in B-face coords).  Removes the local
    rotation/scale from the correlation so the residual is small and the peak sharp."""
    D = _dense_offset(tg)                                             # b - a at face positions (forward offset)
    yy, xx = np.mgrid[0:Y, 0:X].astype(float)
    # inverse: for a B-frame position q, the A-frame source is q - D(q) (first-order; D is smooth)
    ax, ay = xx - D[0], yy - D[1]
    faceA_w = ndi.map_coordinates(faceA, [ay, ax], order=1, mode='constant', cval=0.0)
    valid_w = ndi.map_coordinates(maskA_valid.astype(np.float32), [ay, ax], order=1, mode='constant', cval=0.0) > 0.5
    mfc = flow_field.JAXMaskedXCorrWithStatsCalculator()
    f = mfc.flow_field(faceB, faceA_w, FP, FS, post_mask=~valid_w, max_masked=0.7, batch_size=1024)
    f = flow_utils.clean_flow(f[:, None], min_peak_ratio=1.1, min_peak_sharpness=1.1, max_magnitude=RES_GATE, max_deviation=0, dim=2)[:, 0]
    f = _consensus(f)
    ny, nx = f.shape[1:]; gy, gx = np.mgrid[0:ny, 0:nx]; qx = gx*FS[1] + FP[1]//2; qy = gy*FS[0] + FP[0]//2
    out = []
    for j, i in zip(*np.where(np.isfinite(f[0]))):
        q = np.array([qx[j, i], qy[j, i]], float); a = q - D[:, int(q[1]), int(q[0])]; b = q + f[:, j, i]
        out.append((a, b))
    return out

def flow_pairs(ab, zA, zB, label_face, piece_of_label):
    """(a, b) face pairs -> 3-D correspondence pairs tagged with A's piece"""
    pairs = []
    for a, b in ab:
        if not (0 <= a[0] < X-1 and 0 <= a[1] < Y-1 and 0 <= b[0] < X-1 and 0 <= b[1] < Y-1): continue
        ya, xa = int(round(a[1])), int(round(a[0])); yb, xb = int(round(b[1])), int(round(b[0]))
        za = zA[ya, xa]; zb = zB[yb, xb]
        if not (np.isfinite(za) and np.isfinite(zb)): continue
        lab = label_face[ya, xa]
        if lab not in piece_of_label: continue
        pairs.append((piece_of_label[lab], np.array([a[0], a[1], za]), 'B', np.array([b[0], b[1], zb])))
    return pairs


def targeting_from_mesh(x, pieces, pieceA, nodes, zA, zB):
    """offset (B-frame xy) - (A-frame xy) implied by the relaxed mesh at each face node"""
    ia, ib = pieces.index(pieceA), pieces.index('B')
    ny, nx = (Y - (FP[0]-FS[0]))//FS[0], (X - (FP[1]-FS[1]))//FS[1]
    gy, gx = np.mgrid[0:ny, 0:nx]; px = gx*FS[1] + FP[1]//2; py = gy*FS[0] + FP[0]//2
    za = np.nan_to_num(zA[py, px], nan=ZV-8); zb = np.nan_to_num(zB[py, px], nan=8)
    def samp(xm, z):
        c = [(z - PATCH[0]//2)/STEP[0], (py - PATCH[1]//2)/STEP[1], (px - PATCH[2]//2)/STEP[2]]
        return np.stack([ndi.map_coordinates(xm[k], c, order=1, mode='nearest') for k in range(2)])
    xa = samp(x[:, ia], za); xb = samp(x[:, ib], zb)
    return (xa - xb).astype(np.float32)        # a + xa = b + xb  ->  b - a = xa - xb

