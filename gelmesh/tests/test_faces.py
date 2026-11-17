"""Contracts of gelmesh.faces: the 3x3 neighbour-consensus gate and the face-pair -> 3-D pair lift."""
import numpy as np
import pytest

from gelmesh import faces as F, geom

X, Y = geom.X, geom.Y


def test_consensus_nans_the_outlier_and_keeps_its_neighbours():
    """a flow vector farther than CONS_DEV from the median of its valid 3x3 neighbours is dropped; the neighbours,
    whose medians the outlier cannot move, are kept (the median -- not the mean -- is what makes this one-sided)."""
    f = np.ones((2, 3, 3), np.float32); f[:, 1, 1] = (20.0, 1.0)
    out = F._consensus(f)
    assert np.isnan(out[:, 1, 1]).all()
    keep = np.ones((3, 3), bool); keep[1, 1] = False
    assert np.all(out[:, keep] == 1.0)
    assert np.all(f[:, 1, 1] == (20.0, 1.0))                         # input untouched


def test_consensus_keeps_a_small_deviation_and_respects_max_dev():
    """the gate is max_dev (default CONS_DEV = 10 px): a 9-px deviation passes by default and fails at max_dev=5."""
    f = np.ones((2, 3, 3), np.float32); f[0, 1, 1] = 10.0           # 9 px from the neighbours' median
    assert np.isfinite(F._consensus(f)).all()
    assert np.isnan(F._consensus(f, max_dev=5.0)[:, 1, 1]).all()


def test_consensus_drops_a_node_with_fewer_than_three_valid_neighbours():
    """a vector needs >= min_nb valid neighbours to be judged at all; with too few it is dropped, not trusted."""
    g = np.full((2, 3, 3), np.nan, np.float32); g[:, 0, 0] = g[:, 0, 1] = g[:, 1, 1] = 1.0   # each has 2 neighbours
    assert np.isnan(F._consensus(g)).all()
    assert np.isfinite(F._consensus(g, min_nb=2)[:, [0, 0, 1], [0, 1, 1]]).all()
    h = np.ones((2, 3, 3), np.float32); h[:, 0, 0] = h[:, 2, 2] = np.nan                     # corners: (0,2) keeps 3
    out = F._consensus(h)
    assert np.isnan(out[:, 0, 0]).all() and np.isnan(out[:, 2, 2]).all()
    assert np.isfinite(out[:, 0, 2]).all() and np.isfinite(out[:, 1, 1]).all()


@pytest.fixture
def face_frames():
    zA = np.full((Y, X), 30.0); zB = np.full((Y, X), 5.0); zA[10, 10] = np.nan; zB[20, 20] = np.nan
    lab = np.zeros((Y, X), np.int8); lab[:, :300] = 1; lab[:, 300:] = -1
    return zA, zB, lab, {1: 'A1', -1: 'A2'}


def test_flow_pairs_lifts_face_pairs_to_3d_and_tags_the_piece_by_label(face_frames):
    """(a, b) face pairs become (piece_of_label[label_face[a]], (ax, ay, zA[a]), 'B', (bx, by, zB[b])): the moving
    piece comes from the label under a, z from each face's own surface, b always on slab B."""
    zA, zB, lab, pol = face_frames
    ab = [(np.array([50., 60.]), np.array([52., 61.])), (np.array([350.4, 60.6]), np.array([352., 61.]))]
    pairs = F.flow_pairs(ab, zA, zB, lab, pol)
    assert len(pairs) == 2
    (pa, a, pb, b), (pa2, a2, pb2, b2) = pairs
    assert (pa, pb, pa2, pb2) == ('A1', 'B', 'A2', 'B')
    assert np.array_equal(a, [50., 60., 30.]) and np.array_equal(b, [52., 61., 5.])
    assert np.allclose(a2, [350.4, 60.6, 30.]) and np.array_equal(b2, [352., 61., 5.])  # xy kept sub-pixel


def test_flow_pairs_skips_out_of_bounds_nan_z_and_unknown_labels(face_frames):
    """a pair is dropped when either point leaves the face, when either face has no surface (NaN z) under it, or
    when the label under a maps to no piece -- never raised, never clipped into the face."""
    zA, zB, lab, pol = face_frames
    good = (np.array([50., 60.]), np.array([52., 61.]))
    bad = [(np.array([X, 60.]), np.array([52., 61.])), (np.array([50., 60.]), np.array([52., Y])),
           (np.array([-1., 60.]), np.array([52., 61.])), (np.array([50., 60.]), np.array([X - 1., 61.])),
           (np.array([10., 10.]), np.array([52., 61.])), (np.array([50., 60.]), np.array([20., 20.]))]
    assert F.flow_pairs(bad, zA, zB, lab, pol) == []
    assert len(F.flow_pairs(bad + [good], zA, zB, lab, pol)) == 1
    assert F.flow_pairs([good], zA, zB, lab, {-1: 'A2'}) == []                 # label 1 under a has no piece
    lab0 = np.zeros((Y, X), np.int8)
    assert F.flow_pairs([good], zA, zB, lab0, pol) == []                      # label 0 (no tissue) likewise


def test_affine_fit_and_targeting_round_trip():
    """targeting_from_affine returns (b - a) at each face node (x, y) for the affine fitted to anchor pairs."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 400, (12, 2)); M = np.array([[1.01, 0.02], [-0.02, 0.99]]); t = np.array([3.0, -2.0])
    b = a @ M + t
    sol = F.affine_fit(a, b)
    assert np.allclose(sol[:2], M) and np.allclose(sol[2], t)
    tg = F.targeting_from_affine(sol, None)
    ny, nx = (Y - (geom.FP[0] - geom.FS[0])) // geom.FS[0], (X - (geom.FP[1] - geom.FS[1])) // geom.FS[1]
    assert tg.shape == (2, ny, nx) and tg.dtype == np.float32
    px, py = geom.FP[1] // 2, geom.FP[0] // 2                                 # node (0, 0)
    assert np.allclose(tg[:, 0, 0], np.array([px, py]) @ M + t - [px, py], atol=1e-4)
