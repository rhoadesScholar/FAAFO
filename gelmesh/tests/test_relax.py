import jax.numpy as jnp
"""Contracts of gelmesh.relax: pair-spring target convention, column coupling, the divergence guard, and the
inverse-map / dense-source axis conventions."""
import numpy as np
import pytest
from sofima import mesh

from gelmesh import geom, relax as R
from conftest import SHAPE, PIECES


def _z_pin(nodes, piece):
    """surface springs holding every node of `piece` at z displacement 0 (pins the z gauge of that piece)"""
    return [(piece, nodes[i, j, k].copy(), 0.0) for i in range(SHAPE[0]) for j in range(SHAPE[1]) for k in range(SHAPE[2])]


def test_pair_spring_makes_the_two_points_coincide_in_the_global_frame(nodes, labels, fast_config):
    """A pair (pa, a, pb, b) must relax to a + OFF_a + x_a == b + OFF_b + x_b, OFF keyed by piece NAME; a sign flip in
    T, or OFF ignored, would leave the two points ZS planes (or 16 px) apart after relaxation."""
    a = nodes[1, 2, 3].copy()                       # (64, 48, 16) on A2, exactly on a node
    b = nodes[1, 2, 2].copy()                       # (48, 48, 16) on B, exactly on a node
    force, npair, nsurf = R.make_force(PIECES, nodes, labels, [('A2', a, 'B', b)], _z_pin(nodes, 'B'), k_col=0.0)
    assert (npair, nsurf) == (1, int(np.prod(SHAPE)))
    prev = np.full((3, len(PIECES)) + SHAPE, np.nan, np.float32); prev[:2, 1] = 0.0   # B is the root: xy pinned by k0
    x, n = R.relax(prev, force, fast_config)
    assert x.shape == (3, 2) + SHAPE and n > 0
    xa, xb = x[:, 0, 1, 2, 3], x[:, 1, 1, 2, 2]
    ga, gb = a + geom.OFF['A2'] + xa, b + geom.OFF['B'] + xb
    assert np.allclose(ga, gb, atol=0.5), (ga, gb)
    # the root barely moves; A2 carries the whole correction, INCLUDING slab B's z offset (OFF['B'][2] = ZS)
    assert np.abs(xb).max() < 0.1
    assert abs(xa[2] - geom.ZS) < 0.5, xa
    assert abs(xa[0] - (b[0] - a[0])) < 0.5, xa


def test_pair_spring_target_sign_is_xa_minus_xb_equals_T(nodes, labels):
    """make_force's pull is k_pair * (x_b + T - x_a) with T = (b + OFF_b) - (a + OFF_a); evaluated at x = 0 the
    force on a must point from a toward b's global position, and the reaction on b must be its negative."""
    import jax.numpy as jnp
    a = nodes[1, 2, 3].copy(); b = nodes[1, 2, 2].copy()
    force, _, _ = R.make_force(PIECES, nodes, labels, [('A2', a, 'B', b)], [], k_pair=0.3, k_col=0.0)
    f = np.array(force(jnp.zeros((3, 2) + SHAPE, jnp.float32), 0.03, geom.STRIDE_XYZ, True))
    T = (b + geom.OFF['B']) - (a + geom.OFF['A2'])
    assert np.allclose(f[:, 0, 1, 2, 3], 0.3 * T, atol=1e-5)
    assert np.allclose(f[:, 1, 1, 2, 2], -0.3 * T, atol=1e-5)
    # at rest the elastic term is zero everywhere else
    f[:, 0, 1, 2, 3] = 0; f[:, 1, 1, 2, 2] = 0
    assert np.abs(f).max() < 1e-5


def test_nearest_node_prefers_a_node_on_the_pieces_own_tissue(nodes, labels):
    """springs attach to the nearest node ON the piece's tissue; attaching to the geometric nearest regardless of
    label would hang a crack-face spring on the other piece's node."""
    labvol = np.zeros((geom.ZV, geom.Y, geom.X), np.int8); labvol[:, :, :60] = 1       # tissue of label 1 only at x < 60
    p = np.array([70., 48., 16.])                        # nearest node geometrically is x = 64 (off tissue)
    i = R.nearest_node(nodes, labvol, 1, p)
    assert tuple(nodes[i]) == (48., 48., 16.)
    assert tuple(nodes[R.nearest_node(nodes, labels['A2'][0], 1, p)]) == (64., 48., 16.)


def test_column_coupling_propagates_the_face_warp_down_the_column(nodes, labels, fast_config):
    """K_COL > 0 pulls each node's xy toward its z-column mean; a single xy pull on the TOP layer must reach the
    bottom layer when K_COL is 1 and decay to ~0 when K_COL is 0 (else the in-plane warp dies off the face)."""
    a = nodes[2, 2, 3].copy()                                     # top layer of A2
    b = a + np.array([8., 0., -float(geom.ZS)])                   # same global xyz, +8 in x  ->  T = (8, 0, 0)
    ratios = {}
    for k_col in (0.0, 1.0):
        force, _, _ = R.make_force(PIECES, nodes, labels, [('A2', a, 'B', b)], [], k_col=k_col)
        prev = np.full((3, 2) + SHAPE, np.nan, np.float32); prev[:2] = 0.0     # both pieces k0-pinned in xy
        x, _ = R.relax(prev, force, fast_config)
        col = x[0, 0, :, 2, 3]                                    # x displacement down A2's column (bottom..top)
        assert col[2] > 1.0, col                                  # the top node did move toward b
        ratios[k_col] = col[0] / col[2]
    assert ratios[1.0] > 0.4, ratios
    assert abs(ratios[0.0]) < 0.1, ratios
    assert ratios[1.0] > 3 * abs(ratios[0.0])


def test_relax_raises_after_three_diverged_attempts(fast_config):
    """a non-finite mesh must surface as RuntimeError('mesh relaxation diverged'), not as NaN node positions
    silently rendered into the volume."""
    def nan_force(x, k, stride, prefer_orig_order):
        return mesh.elastic_mesh_3d(x, k, stride, prefer_orig_order) * np.nan
    prev = np.full((3, 2) + SHAPE, np.nan, np.float32); prev[:2] = 0.0
    cfg = mesh.IntegrationConfig(**{**fast_config.to_dict(), 'num_iters': 5, 'max_iters': 5})
    msgs = []
    with pytest.raises(RuntimeError, match='mesh relaxation diverged'):
        R.relax(prev, nan_force, cfg, log=msgs.append)
    assert len(msgs) == 3 and all('retrying' in m for m in msgs)


def test_relax_retries_with_dt_max_divided_by_ten(monkeypatch, fast_config):
    """the retry path lowers dt_max tenfold per attempt and returns the first finite result; the retried config
    must otherwise be the caller's (not the module CONFIG)."""
    calls = []
    def fake_relax_mesh(x0, prev, cfg, mesh_force=None):
        calls.append(cfg)
        return (np.full(x0.shape, np.nan) if len(calls) < 3 else np.ones(x0.shape)), [], 7 * len(calls)
    monkeypatch.setattr(R.mesh, 'relax_mesh', fake_relax_mesh)
    prev = np.full((3, 2) + SHAPE, np.nan, np.float32)                        # no k0 anchor, zero mesh force: converged after the first finite chunk
    x, n = R.relax(prev, lambda *a: jnp.zeros(prev.shape), fast_config)
    assert np.all(x == 1.0) and n == 21
    assert [c.dt_max for c in calls] == [1000, 100, 10]
    assert all(c.num_iters == fast_config.num_iters and c.k0 == fast_config.k0 for c in calls)


def test_invert_and_dense_source_identity():
    """an all-zero displacement map inverts to zero and dense_source returns the query coords as (x, y, z)."""
    xmap = np.zeros((3,) + SHAPE, np.float32)
    inv = R.invert(xmap)
    assert inv.shape == xmap.shape and np.abs(inv).max() == 0
    zz, yy, xx = np.mgrid[0:24, 0:96, 0:128].astype(np.float32)
    src = R.dense_source(inv, zz, yy, xx)
    assert src.shape == zz.shape + (3,)
    assert np.array_equal(src, np.stack([xx, yy, zz], -1))


@pytest.mark.parametrize('comp, name', [(0, 'x'), (1, 'y'), (2, 'z')])
def test_dense_source_of_a_translation_is_its_inverse(comp, name):
    """maps are (3, z, y, x) with components (x, y, z): a forward +5 along one component makes dense_source return
    query - 5 along THAT component and leave the others untouched (a swapped axis order would move the wrong one)."""
    xmap = np.zeros((3,) + SHAPE, np.float32); xmap[comp] = 5.0
    inv = R.invert(xmap)
    assert np.allclose(inv[comp], -5.0) and np.abs(np.delete(inv, comp, axis=0)).max() == 0
    zz, yy, xx = np.mgrid[0:24, 0:96, 0:128].astype(np.float32)
    src = R.dense_source(inv, zz, yy, xx)
    expect = np.stack([xx, yy, zz], -1); expect[..., comp] -= 5.0
    assert np.allclose(src, expect, atol=1e-4)
