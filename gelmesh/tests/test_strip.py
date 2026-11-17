"""Contracts of gelmesh.strip: the shell ZNCC shift search, strip-centre geometry, and match_strips end to end on a
synthetic textured plane."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from gelmesh import strip, geom

L, NCOL, S = 64, 6, 12


def _shells(seed, dt, noise=0.02):
    """S1, S2 of shape (L, NCOL) with S2[i] == S1[i + dt] (+ noise): the convention _best_shift searches"""
    rng = np.random.default_rng(seed)
    base = ndi.gaussian_filter1d(rng.normal(size=(L + 2 * S + 10, NCOL)), 1.5, axis=0)
    S1 = base[S:S + L]; S2 = base[S + dt:S + dt + L] + noise * rng.normal(size=(L, NCOL))
    return S1, S2


@pytest.mark.parametrize('dt', [-7, 0, 5])
def test_best_shift_recovers_a_known_shift_with_high_score_and_margin(dt):
    """_best_shift returns dt such that S1[i + dt] matches S2[i]; a sign flip or off-by-one here would place
    every automatic crack pair on the wrong side of the true slide."""
    S1, S2 = _shells(0, dt)
    got, score, margin = strip._best_shift(S1, S2, S, L // 3)
    assert got == dt
    assert score > 0.95 and margin > 0.5


def test_best_shift_on_noise_scores_low():
    """two unrelated shells must not produce a confident peak (min_score gates on this value)."""
    rng = np.random.default_rng(3)
    got, score, margin = strip._best_shift(rng.normal(size=(L, NCOL)), rng.normal(size=(L, NCOL)), S, L // 3)
    assert score < 0.3 and margin < 0.3


def test_best_shift_tolerates_nan_columns_and_returns_none_when_nothing_is_valid():
    """invalid (NaN) samples are excluded per column instead of poisoning the ZNCC; an all-NaN shell yields None."""
    S1, S2 = _shells(1, 4)
    S1[:, 2] = np.nan; S2[:5, 0] = np.nan
    got, score, margin = strip._best_shift(S1, S2, S, L // 3)
    assert got == 4 and score > 0.95
    assert strip._best_shift(np.full((L, NCOL), np.nan), S2, S, L // 3) is None


def test_zncc_normalises_each_column_over_its_own_valid_samples():
    """_zncc is a mean of per-column ZNCCs, each column normalised by its own mean/gain: a column with a
    different gain still scores 1, and a column with too few valid samples is left out."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(L, 3)); b = a.copy(); b[:, 1] = 5 * a[:, 1] + 3          # gain and offset on column 1
    ok = np.ones((L, 3), bool)
    assert strip._zncc(a, b, ok, L // 3) == pytest.approx(1.0, abs=1e-6)
    b[:, 2] = -a[:, 2]; ok2 = ok.copy(); ok2[3:, 2] = False                         # anti-correlated but invalid
    assert strip._zncc(np.where(ok2, a, 0), np.where(ok2, b, 0), ok2, L // 3) == pytest.approx(1.0, abs=1e-6)
    assert strip._zncc(a, b, ok, L // 3) == pytest.approx(1 / 3, abs=1e-6)
    assert strip._zncc(a, b, np.zeros_like(ok), L // 3) == -1.0


def _halves(gap, H=200, W=200):
    yy, xx = np.mgrid[0:H, 0:W]
    return xx < 100, xx >= 100 + gap


def test_strip_centres_sit_on_piece_1_face_with_inward_normals_and_spacing():
    """centres are boundary pixels of piece 1 near piece 2, thinned to `spacing`; the normal points from piece 2
    INTO piece 1 (here -x) and the tangent is its 90-degree rotation -- the frame every shell is sampled in."""
    m1, m2 = _halves(0)
    cs = strip.strip_centres(m1, m2, spacing=16, reach=12)
    assert len(cs) >= 10
    C = np.array([c for c, n, t in cs]); N = np.array([n for c, n, t in cs]); T = np.array([t for c, n, t in cs])
    # image-border rows (y < 2 or y > H-3) count as "boundary" under binary_erosion's zero border, so a centre can
    # be seeded there up to `reach` px INSIDE piece 1 (e.g. (89, 0) here); the face contract is checked on the
    # interior rows, and every centre must still be on piece 1
    inner = (C[:, 1] >= 2) & (C[:, 1] <= m1.shape[0] - 3)
    assert inner.sum() >= 10
    assert np.all((C[inner, 0] >= 98) & (C[inner, 0] <= 99)), C[inner]
    assert np.all(m1[C[:, 1].astype(int), C[:, 0].astype(int)])                   # every centre is ON piece 1
    assert np.allclose(N[inner], [-1.0, 0.0], atol=0.05)
    assert np.allclose(T, np.stack([-N[:, 1], N[:, 0]], 1))
    d = np.linalg.norm(C[:, None] - C[None], axis=-1); d[np.eye(len(C), dtype=bool)] = np.inf
    assert d.min() >= 16


def test_strip_centres_follow_reach_across_a_gap():
    """a 5-px open crack still seeds centres (reach=12); a 40-px gap seeds none, so the matcher never pairs tissue
    that is not actually adjacent."""
    assert len(strip.strip_centres(*_halves(5), spacing=16, reach=12)) >= 10
    assert strip.strip_centres(*_halves(40), spacing=16, reach=12) == []
    assert strip.strip_centres(*_halves(5), spacing=16, reach=4) == []


def _textured_plane(seed, dty, gap=0, nz=5):
    """W, M (2, nz, Y, X): one x-coherent texture split at x = 100; piece 1's image is shifted by +dty along y"""
    Y, X = geom.Y, geom.X
    rng = np.random.default_rng(seed)
    tex = ndi.gaussian_filter(rng.normal(size=(Y, X)), (1.5, 6))
    blobs = np.zeros((Y, X)); blobs[rng.integers(0, Y, 400), rng.integers(0, X, 400)] = 1.0
    tex = tex / tex.std() + ndi.gaussian_filter(blobs, 3.0) * 40
    tex = ((tex - tex.mean()) / tex.std()).astype(np.float32)
    tex1 = ndi.shift(tex, (dty, 0), order=1, mode='constant')                     # tex1[y] = tex[y - dty]
    yy, xx = np.mgrid[0:Y, 0:X]; m1 = xx < 100; m2 = xx >= 100 + gap
    W = np.zeros((2, nz, Y, X), np.float32); M = np.zeros((2, nz, Y, X), bool)
    W[0] = np.where(m1, tex1, 0); W[1] = np.where(m2, tex, 0); M[0] = m1; M[1] = m2
    return W, M


@pytest.mark.parametrize('seed, dty', [(1, 4), (2, -3), (3, 0)])
def test_match_strips_recovers_the_tangential_slide(seed, dty):
    """pairs (p1, p2) are truth-adjacent points: with piece 1's image translated by +dty along y (the crack tangent),
    p1 - p2 must be (0, dty) -- i.e. p1 - p2 equals the displacement of piece 1 RELATIVE to piece 2, so the
    solver's 'make them coincide' target undoes the translation.  (The returned dt is in units of the tangent
    t = (-n_y, n_x); with n = (-1, 0) that is t = (0, -1), so dt == -dty here.)"""
    W, M = _textured_plane(seed, dty)
    out = strip.match_strips(W, M, z0=2, zwin=2)
    assert len(out) >= 15, len(out)
    d = np.array([p1 - p2 for p1, p2, *_ in out])
    frac = np.mean(np.abs(d[:, 1] - dty) < 1.5)
    assert frac >= 0.9, (frac, d[:, 1])
    assert np.all(np.abs(d[:, 0]) < 0.75)                                          # both points on the boundary line, not on the face rows (an on-row pair would over-close by 1 px)
    dts = np.array([o[4] for o in out]); assert np.median(dts) == -dty
    assert all(o[2] >= 0.3 for o in out)
    assert np.allclose([o[3] for o in out], 0.0, atol=1.0)                        # closed crack: no normal gap


def test_match_strips_reports_an_open_gap_as_negative_dn():
    """dn is the normal correction that closes the crack: a 3-px open gap between the faces gives dn ~ -3 and the
    pair points sit half a pixel outside each face (p2 - p1 spans the gap)."""
    W, M = _textured_plane(4, 2, gap=3)
    out = strip.match_strips(W, M, z0=2, zwin=2)
    assert len(out) >= 15
    assert np.allclose([o[3] for o in out], -3.0, atol=1.0)
    d = np.array([p2 - p1 for p1, p2, *_ in out])
    assert np.all(np.abs(d[:, 0] - 3) < 1.0)
    assert np.mean(np.abs(d[:, 1] + 2) < 1.5) >= 0.9


def test_match_strips_rejects_an_open_gap_beyond_max_gap():
    """max_gap bounds the normal closure a strip may propose; a 7-px gap under max_gap=5 yields no pairs."""
    W, M = _textured_plane(4, 0, gap=7)
    assert strip.match_strips(W, M, z0=2, zwin=2, max_gap=5) == []
    assert len(strip.match_strips(W, M, z0=2, zwin=2, max_gap=10)) >= 10
