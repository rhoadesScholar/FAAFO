"""Small contracts: geom's node grid, experiment.parse_variant, bench.synthetic_force, cli.table."""
import numpy as np
import pytest
import jax.numpy as jnp

from gelmesh import geom, experiment, bench, cli


def test_node_xyz_follows_the_sofima_node_convention():
    """node (gz, gy, gx) rests at (gx*STEP_x + PATCH_x//2, gy*STEP_y + PATCH_y//2, gz*STEP_z + PATCH_z//2) as (x, y, z);
    every spring attachment, inverse map and dense_source assumes exactly this placement."""
    nodes = geom.node_xyz((3, 6, 8))
    assert nodes.shape == (3, 6, 8, 3) and nodes.dtype == float
    assert tuple(nodes[0, 0, 0]) == (geom.PATCH[2] // 2, geom.PATCH[1] // 2, geom.PATCH[0] // 2)
    assert tuple(nodes[1, 2, 3] - nodes[0, 0, 0]) == (3 * geom.STEP[2], 2 * geom.STEP[1], 1 * geom.STEP[0])
    gs = geom.grid_shape()
    full = geom.node_xyz(gs)
    assert np.all(full[..., 0] < geom.X) and np.all(full[..., 1] < geom.Y) and np.all(full[..., 2] < geom.ZV)
    assert geom.STRIDE_XYZ == (geom.STEP[2], geom.STEP[1], geom.STEP[0])
    assert tuple(geom.OFF['B']) == (0, 0, geom.ZS) and not geom.OFF['A'].any()


@pytest.mark.parametrize('variant, base, flags', [
    ('JOINT_FEW_EXT', 'JOINT', dict(dense=False, ext=True, few=True)),
    ('JOINT', 'JOINT', dict(dense=False, ext=False, few=False)),
    ('NONEIGHBOUR_DENSE', 'NONEIGHBOUR', dict(dense=True, ext=False, few=False)),
    ('CONNECTED', 'CONNECTED', dict(dense=False, ext=False, few=False)),
    ('NONEIGHBOUR_FEW_EXT', 'NONEIGHBOUR', dict(dense=False, ext=True, few=True)),
])
def test_parse_variant_splits_base_and_suffixes(variant, base, flags):
    """variant names are a base in {CONNECTED, JOINT, NONEIGHBOUR} plus any of _DENSE/_EXT/_FEW; every name in
    cli.ALL must parse this way or `gelmesh run` dies on it."""
    assert experiment.parse_variant(variant) == (base, flags)


def test_every_cli_variant_parses():
    for v in cli.ALL:
        experiment.parse_variant(v)


@pytest.mark.parametrize('bad', ['JOIN', 'JOINT_DENSE_X', 'joint', ''])
def test_parse_variant_rejects_an_unknown_base(bad):
    """an unknown base is an AssertionError naming the variant, not a silently-defaulted experiment."""
    with pytest.raises(AssertionError):
        experiment.parse_variant(bad)


def test_synthetic_force_runs_on_a_tiny_mesh():
    """bench's stand-in force (elastic + pair + surface + column) returns a finite field of the mesh's shape, with the
    documented spring counts (one pair per face node, two surface springs per face node per piece)."""
    P, shape = 2, (3, 4, 5)
    force, n_pair, n_surf = bench.synthetic_force(P, shape, rng=np.random.default_rng(0))
    assert (n_pair, n_surf) == (4 * 5, 2 * 4 * 5 * P)
    x = jnp.zeros((3, P) + shape, jnp.float32)
    f = np.array(force(x, 0.03, geom.STRIDE_XYZ, True))
    assert f.shape == x.shape and np.isfinite(f).all() and np.abs(f).max() > 0
    f2 = np.array(force(x + 0.5, 0.03, geom.STRIDE_XYZ, True))
    assert np.isfinite(f2).all()


def test_cli_table_formats_a_summary():
    """cli.table renders one markdown row per variant from the run summary dict (stats + counts); a renamed stats
    key would KeyError here before it did at the end of a multi-hour run."""
    st = {k: dict(median=1.234, p90=2.5, unfilled=0.0) for k in ('near_crack', 'near_seam', 'far')}
    summary = {'JOINT': dict(stats=st, history=[], n_seam=10, n_crack=4, n_crack_auto=0),
               'JOINT_EXT': dict(stats=st, history=[dict(round=0)], n_seam=11, n_crack=4, n_crack_auto=37)}
    rows = cli.table(summary).splitlines()
    assert len(rows) == 4 and rows[0].startswith('| variant |') and rows[1].startswith('|---')
    assert rows[2].startswith('| JOINT |') and '1.23 / 2.50' in rows[2] and rows[2].rstrip().endswith('| 10 | 4 | 0 |')
    assert rows[3].rstrip().endswith('| 11 | 4 | 37 |')
    assert cli.table({}).count('\n') == 1
