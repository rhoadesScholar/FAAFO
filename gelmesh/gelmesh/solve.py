"""The joint solve: pieces + interfaces (seam, crack) + flatten, relaxed together over a few extension rounds.

    round 0:  anchors only (seam affine -> targeted face flow; crack anchors)          -> relax
    round r:  seam warp-and-match through the current mesh; crack strip matcher        -> relax

`solve()` is variant-agnostic; the variants of the experiment (CONNECTED / JOINT / NONEIGHBOUR, _EXT, _DENSE,
_FEW) only change which pieces exist and which anchors are supplied -- see `experiment.py`.
"""
import time
import numpy as np
from .geom import X, Y, ZS, ZV, OFF, STEP, PATCH, grid_shape, node_xyz
from .phantom import surfaces
from . import faces as F
from .relax import make_force, relax, render
from .crack import crack_extend


class Problem:
    """everything the solver needs, independent of where the anchors come from"""

    def __init__(self, vols, labels, pieces, piece_of_label, root, tissue, seam=('A', 'B'), side_labvol=None):
        self.vols, self.labels, self.pieces, self.piece_of_label, self.root = vols, labels, pieces, piece_of_label, root
        self.tissue = tissue                       # {slab: tissue mask (ZV,Y,X)}
        self.seam = seam                           # (moving slab, fixed slab) or None
        self.shape = grid_shape(); self.nodes = node_xyz(self.shape)
        self.seam_anchor, self.crack_pairs = [], []
        self.crack_faces = None                    # (piece 1, piece 2) across the crack, or None
        # surfaces and faces
        self.top, self.bot = {}, {}
        for s, t in tissue.items(): self.top[s], self.bot[s] = surfaces(t)
        if seam:
            a, b = seam
            self.faceA, self.okA, self.zA = F.face(vols[a], self.bot[a], 0.0)
            self.faceB, self.okB, self.zB = F.face(vols[b], self.top[b], 0.0)
            # side labels on the moving face (for anchors per side and per-piece masking); in the CONNECTED
            # variant the piece label is binary, so the experiment passes the true side labels explicitly
            self.labA_face = np.zeros((Y, X), np.int8)
            labvol = side_labvol if side_labvol is not None else self._slab_labvol(a)
            iy, ix = np.where(self.okA); self.labA_face[iy, ix] = labvol[np.clip(self.zA[iy, ix].astype(int), 0, ZV-1), iy, ix]

    def _slab_labvol(self, slab):
        for p in self.pieces:
            if p.startswith(slab): return self.labels[p][0]
        raise KeyError(slab)

    def surface_springs(self):
        """every node column, both surfaces of every slab, target plane: first tissue plane -> 0, last -> ZS-1
        (slab-local).  A slab of ZS planes then fills exactly ZS global planes and the next slab starts at ZS:
        the seam gauge is single-valued (a bottom target of ZS would double-book plane ZS with the slab below)."""
        surf = []
        gy, gx = np.mgrid[0:self.shape[1], 0:self.shape[2]]; px = gx*STEP[2] + PATCH[2]//2; py = gy*STEP[1] + PATCH[1]//2
        for slab in self.tissue:
            labvol = self._slab_labvol(slab); top, bot = self.top[slab], self.bot[slab]
            for y, x in zip(py.ravel(), px.ravel()):
                if not np.isfinite(top[y, x]): continue
                for zsurf, target in ((top[y, x], 0.0), (bot[y, x], float(ZS - 1))):
                    lab = labvol[int(zsurf), y, x]; p = self.piece_of_label.get(lab)
                    if p: surf.append((p, np.array([x, y, zsurf]), target - zsurf))
        return surf


def solve(pb, rounds=3, crack_ext=False, connected=False, log=print, config=None, strip_kw=None):
    """returns dict(x, pieces, seam_pairs, crack_auto, history).  `connected`: slab A is one piece (today's
    smooth mesh) and the seam extension is unmasked."""
    surf = pb.surface_springs()
    pieces, labels, vols = pb.pieces, pb.labels, pb.vols
    x = None; seam_pairs = []; crack_auto = []; history = []
    has_seam = pb.seam is not None
    n_rounds = rounds if (has_seam or crack_ext) else 1
    for r in range(n_rounds):
        if has_seam:
            seam_pairs = []
            moving_labels = (1, -1) if not connected else (1,)
            for lab in moving_labels:
                piece = pb.piece_of_label[lab] if not connected else pieces[0]
                sel = (pb.labA_face == lab) if not connected else pb.okA
                pm = ~(pb.okA & sel)
                if r == 0:
                    anc = [(a, b) for pa, a, pb_, b in pb.seam_anchor if (connected or pa == piece)]
                    sol = F.affine_fit(np.array([a for a, b in anc]), np.array([b for a, b in anc])); tg = F.targeting_from_affine(sol, None)
                    f = F.face_flow(pb.faceB, pb.faceA, pm, tg)
                    ny_, nx_ = f.shape[1:]; gy_, gx_ = np.mgrid[0:ny_, 0:nx_]
                    from .geom import FP, FS
                    qx_ = gx_*FS[1] + FP[1]//2; qy_ = gy_*FS[0] + FP[0]//2
                    ab = [(np.array([qx_[j, i], qy_[j, i]], float), np.array([qx_[j, i] + f[0, j, i], qy_[j, i] + f[1, j, i]]))
                          for j, i in zip(*np.where(np.isfinite(f[0])))]
                else:
                    tg = F.targeting_from_mesh(x, pieces, piece, pb.nodes, pb.zA, pb.zB)
                    ab = F.warp_and_match(pb.faceB, pb.faceA, ~pm, tg)
                pol = pb.piece_of_label if not connected else {1: pieces[0], -1: pieces[0]}
                seam_pairs += F.flow_pairs(ab, pb.zA, pb.zB, pb.labA_face, pol)
            log(f'  round {r}: {len(seam_pairs)} seam pairs from extension')
        if crack_ext and r > 0 and pb.crack_faces:
            p1, p2 = pb.crack_faces
            crack_auto, _ = crack_extend(x, pieces, labels, vols, p1=p1, p2=p2, zs=tuple(range(4, ZS-2, 4)), log=log, **(strip_kw or {}))
        pairs = seam_pairs + pb.seam_anchor + pb.crack_pairs + crack_auto
        force, npair, nsurf = make_force(pieces, pb.nodes, labels, pairs, surf)
        if r == 0:
            per_piece = {p: sum(1 for q, _, _ in surf if q == p) for p in pieces}
            missing = [p for p, n in per_piece.items() if n == 0]
            if missing: raise ValueError(f'pieces with no surface springs (unconstrained flatten): {missing}')
        prev = np.full((3, len(pieces)) + pb.shape, np.nan, np.float32)
        prev[:2, pieces.index(pb.root)] = 0.0          # root piece: xy pinned to its own frame; z free for the flatten
        t0 = time.time(); x, n = relax(prev, force, config, log); dt = time.time() - t0
        log(f'  relax {n} steps, {npair} pair springs, {nsurf} surface springs, {dt:.0f}s')
        history.append(dict(round=r, steps=n, n_pairs=npair, n_surf=nsurf, relax_s=dt, n_seam=len(seam_pairs), n_crack_auto=len(crack_auto)))
    return dict(x=x, pieces=pieces, seam_pairs=seam_pairs, crack_auto=crack_auto, history=history)


def render_solution(pb, sol, out_shape):
    return render(sol['pieces'], sol['x'], pb.vols, pb.labels, out_shape)
