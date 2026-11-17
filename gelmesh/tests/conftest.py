"""Shared fixtures: a tiny batched mesh (2 pieces) whose every node is tissue of its piece, and a short FIRE
config.  Nothing here touches the phantom pickle."""
import numpy as np
import pytest
from sofima import mesh

from gelmesh import geom

SHAPE = (3, 6, 8)              # (z, y, x) nodes -> rest positions x in [16, 128], y in [16, 96], z in [8, 24]
PIECES = ['A2', 'B']           # OFF['A2'] = 0, OFF['B'] = (0, 0, ZS): the slab-B z offset is exercised


@pytest.fixture(scope='session')
def nodes():
    return geom.node_xyz(SHAPE)


@pytest.fixture(scope='session')
def labels():
    """{piece: (labvol, label)} with the whole (ZV, Y, X) volume labelled as that piece's tissue"""
    return {'A2': (np.full((geom.ZV, geom.Y, geom.X), 1, np.int8), 1),
            'B': (np.full((geom.ZV, geom.Y, geom.X), 2, np.int8), 2)}


@pytest.fixture(scope='session')
def fast_config():
    """the module CONFIG with a short reporting interval (num_iters) so small meshes stop as soon as they settle"""
    return mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.1, k=0.03, stride=geom.STRIDE_XYZ, num_iters=100,
                                  max_iters=20000, stop_v_max=0.003, dt_max=1000, start_cap=10, final_cap=10,
                                  prefer_orig_order=True)
