"""Joint spring-mesh relaxation of several pieces on SOFIMA primitives.

Every piece (a connected component of tissue after cutting along cracks; the neighbour slab is a piece too)
is one lattice in a batched SOFIMA mesh x[3, P, z, y, x] (node DISPLACEMENTS, xyz).  Forces:

  * SOFIMA's `elastic_mesh_3d` per piece (internal Hookean springs, stiffness k)
  * zero-length correspondence springs between (piece_a, point_a) and (piece_b, point_b): seam pairs,
    crack-face pairs, anchors and automatic extensions alike -- want  a + OFF_a + x_a == b + OFF_b + x_b
  * surface springs pulling the z of every surface sample to its target plane (the flatten)
  * column coupling pulling each node's xy toward its z-column mean (K_COL -> inf is slabreg's z-uniform
    xy warp; 0 a free 3-D mesh); the data sit on one face, so without it the in-plane warp decays with
    distance from that face
  * SOFIMA's own k0 term on the root piece's xy (prev = 0 there, NaN elsewhere): the global gauge.

Springs attach to the node NEAREST the point that lies on the piece's own tissue (a half-stride approximation;
bilinear carrying would be the next step).  The spring target is computed from the exact points, so the
attachment error is the local displacement gradient times the node offset, not the offset itself.
"""
import numpy as np
import jax.numpy as jnp
from scipy import ndimage as ndi
from connectomics.common import bounding_box
from sofima import mesh, map_utils
from .geom import X, Y, ZV, STEP, PATCH, STRIDE_XYZ, OFF

K_PAIR, K_SURF, K_COL = 0.3, 0.3, 1.0

CONFIG = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.1, k=0.03, stride=STRIDE_XYZ, num_iters=1000,
                                max_iters=80000, stop_v_max=0.003, dt_max=1000, start_cap=10, final_cap=10,
                                prefer_orig_order=True)


def nearest_node(nodes, labvol, piece_label, p):
    """index of the nearest grid node whose rest position lies on tissue of the given label (fallback: nearest)"""
    d = np.linalg.norm(nodes - p, axis=-1)
    ni = np.clip(nodes.astype(int), 0, [X-1, Y-1, ZV-1])
    ok = labvol[ni[..., 2], ni[..., 1], ni[..., 0]] == piece_label
    dd = np.where(ok, d, d + 1e6); return np.unravel_index(np.argmin(dd), d.shape)


def make_force(pieces, nodes, labels, pairs, surf_springs, k_pair=None, k_surf=None, k_col=None):
    """pieces: list of names; labels: {name: (labvol, label value)}; pairs: [(pieceA, a, pieceB, b)] with
    a, b points (x,y,z) in each piece's own volume that should coincide in the global frame;
    surf_springs: [(piece, point xyz, target displacement in z)].  Returns (mesh_force, n_pairs, n_surf)."""
    k_pair = K_PAIR if k_pair is None else k_pair; k_surf = K_SURF if k_surf is None else k_surf
    k_col = K_COL if k_col is None else k_col
    pid = {n: i for i, n in enumerate(pieces)}
    PA, PB, T = [], [], []
    for pa, a, pb, b in pairs:
        if pa not in pid or pb not in pid: continue
        ia = nearest_node(nodes, *labels[pa], a); ib = nearest_node(nodes, *labels[pb], b)
        PA.append((pid[pa],) + tuple(ia)); PB.append((pid[pb],) + tuple(ib))
        T.append((b + OFF[pb]) - (a + OFF[pa]))                   # want x_a - x_b = T
    PA = jnp.asarray(np.array(PA, dtype=int).reshape(-1, 4).T); PB = jnp.asarray(np.array(PB, dtype=int).reshape(-1, 4).T)
    T = jnp.asarray(np.array(T, dtype=np.float32).reshape(-1, 3).T)
    SN, ST = [], []
    for p, pt, tz in surf_springs:
        if p not in pid: continue
        i = nearest_node(nodes, *labels[p], pt); SN.append((pid[p],) + tuple(i)); ST.append(tz)
    SN = jnp.asarray(np.array(SN, dtype=int).reshape(-1, 4).T); ST = jnp.asarray(np.array(ST, dtype=np.float32))

    def force(x, k, stride, prefer_orig_order):
        f = mesh.elastic_mesh_3d(x, k, stride, prefer_orig_order)
        if T.shape[1]:
            xa = x[:, PA[0], PA[1], PA[2], PA[3]]; xb = x[:, PB[0], PB[1], PB[2], PB[3]]
            pull = k_pair * (xb + T - xa)
            f = f.at[:, PA[0], PA[1], PA[2], PA[3]].add(pull)
            f = f.at[:, PB[0], PB[1], PB[2], PB[3]].add(-pull)
        if ST.shape[0]:
            xz = x[2, SN[0], SN[1], SN[2], SN[3]]
            f = f.at[2, SN[0], SN[1], SN[2], SN[3]].add(k_surf * (ST - xz))
        if k_col > 0:
            f = f.at[:2].add(k_col * (jnp.mean(x[:2], axis=2, keepdims=True) - x[:2]))
        return f
    return force, int(T.shape[1]), int(ST.shape[0])


FORCE_TOL, MAX_CHUNKS = 0.05, 8


def relax(prev, force, config=None, log=None):
    """FIRE relaxation from rest.  `prev` carries the k0 anchor (0 on the root piece's xy, NaN elsewhere).
    SOFIMA's stop test (v_max < stop_v_max) also passes on a FIRE velocity reset, so convergence is checked
    here on max |force| and the relaxation is continued in chunks of `config.num_iters` until it is below
    FORCE_TOL or MAX_CHUNKS is reached.  A diverged mesh (non-finite nodes) is retried with a smaller step
    ceiling: the stiff terms can exceed FIRE's stable step at dt_max=1000 on a sparse grid.
    Returns (x, n_steps)."""
    cfg = CONFIG if config is None else config
    for attempt in range(3):
        x = jnp.zeros_like(prev); pv = jnp.asarray(prev); total = 0; fmax = np.inf
        for chunk in range(MAX_CHUNKS):
            x, e, n = mesh.relax_mesh(x, pv, cfg, mesh_force=force); total += int(n)
            if not bool(jnp.isfinite(x).all()): break
            # total force = custom mesh_force + SOFIMA's own k0 term toward `prev` where prev is finite (the gauge)
            ftot = force(x, cfg.k, cfg.stride, cfg.prefer_orig_order) + cfg.k0 * jnp.nan_to_num(pv - x)
            fmax = float(jnp.abs(ftot).max())
            if fmax < FORCE_TOL: break
        x = np.array(x)
        if np.isfinite(x).all():
            if log: log(f'  relax: {total} steps, max |force| {fmax:.2e}' + ('' if fmax < FORCE_TOL else f' (not below {FORCE_TOL} after {MAX_CHUNKS} chunks)'))
            return x, total
        if log: log(f'  relax diverged at dt_max={cfg.dt_max}; retrying with dt_max={cfg.dt_max/10}')
        cfg = mesh.IntegrationConfig(**{**cfg.to_dict(), 'dt_max': cfg.dt_max / 10})
    raise RuntimeError('mesh relaxation diverged')


def invert(xmap):
    """forward node displacement map (3, z, y, x) -> inverse map on the same grid, holes filled by extrapolation"""
    box = bounding_box.BoundingBox(start=(0, 0, 0), size=xmap.shape[1:][::-1])
    return map_utils.fill_missing(map_utils.invert_map(xmap, box, box, STRIDE_XYZ[::-1]), extrapolate=True)


def dense_source(inv_map, zz, yy, xx):
    """global-relative voxel coords -> source position (x,y,z) in the piece's own volume"""
    coords = [(zz - PATCH[0]//2)/STEP[0], (yy - PATCH[1]//2)/STEP[1], (xx - PATCH[2]//2)/STEP[2]]
    off = np.stack([ndi.map_coordinates(inv_map[k], coords, order=1, mode='nearest') for k in range(3)], -1)
    return np.stack([xx, yy, zz], -1) + off


def render(pieces, x, vols, labels, out_shape, zchunk=8):
    """compose the global volume from per-piece inverse maps.  A voxel takes the first piece (in list order)
    whose inverse map lands on that piece's own tissue -- side-consistent, so a closed crack shows no seam
    bleed.  z-chunked to bound memory.  Returns (volume, chosen piece index or -1, source xyz)."""
    ZT_, Y_, X_ = out_shape
    out = np.zeros(out_shape, np.float32); chosen = np.full(out_shape, -1, np.int8); src_sel = np.full(out_shape + (3,), np.nan, np.float32)
    invs = [invert(x[:, pi]) for pi in range(len(pieces))]
    for z0 in range(0, ZT_, zchunk):
        zz, yy, xx = np.mgrid[z0:z0+zchunk, 0:Y_, 0:X_].astype(np.float32)
        for pi, name in enumerate(pieces):
            vol, (labvol, lv) = vols[name], labels[name]
            src = dense_source(invs[pi], zz - OFF[name][2], yy - OFF[name][1], xx - OFF[name][0]).astype(np.float32)
            si = np.round(src).astype(int)
            inb = (si[..., 0] >= 0) & (si[..., 0] < X) & (si[..., 1] >= 0) & (si[..., 1] < Y) & (si[..., 2] >= 0) & (si[..., 2] < ZV)
            sic = np.clip(si, 0, [X-1, Y-1, ZV-1])
            ok = inb & (labvol[sic[..., 2], sic[..., 1], sic[..., 0]] == lv) & (chosen[z0:z0+zchunk] < 0)
            vals = ndi.map_coordinates(vol, [src[..., 2], src[..., 1], src[..., 0]], order=1, mode='nearest')
            o = out[z0:z0+zchunk]; o[ok] = vals[ok]; c = chosen[z0:z0+zchunk]; c[ok] = pi; ss = src_sel[z0:z0+zchunk]; ss[ok] = src[ok]
    return out, chosen, src_sel


def warp_piece_planes(x, pieces, name, vol, labels, zs):
    """one piece warped into the global frame at planes `zs` (no cross-piece exclusivity): image, own-tissue
    mask and source map.  Used by the crack-face matcher, which needs both pieces where they overlap."""
    inv = invert(x[:, pieces.index(name)]); labvol, lv = labels[name]
    W = np.zeros((len(zs), Y, X), np.float32); M = np.zeros((len(zs), Y, X), bool); SRC = np.full((len(zs), Y, X, 3), np.nan, np.float32)
    for k, z in enumerate(zs):
        zz, yy, xx = np.mgrid[z:z+1, 0:Y, 0:X].astype(np.float32)
        src = dense_source(inv, zz - OFF[name][2], yy - OFF[name][1], xx - OFF[name][0])[0].astype(np.float32)
        si = np.round(src).astype(int)
        inb = (si[..., 0] >= 0) & (si[..., 0] < X) & (si[..., 1] >= 0) & (si[..., 1] < Y) & (si[..., 2] >= 0) & (si[..., 2] < ZV)
        sic = np.clip(si, 0, [X-1, Y-1, ZV-1]); ok = inb & (labvol[sic[..., 2], sic[..., 1], sic[..., 0]] == lv)
        vals = ndi.map_coordinates(vol, [src[..., 2], src[..., 1], src[..., 0]], order=1, mode='nearest')
        W[k][ok] = vals[ok]; M[k] = ok; SRC[k][ok] = src[ok]
    return W, M, SRC
