"""Shared geometry of the POC: phantom volume and mesh grid.

Axes: arrays are (z, y, x); points are (x, y, z).  All sizes are voxels.
"""
import numpy as np

ZT, Y, X = 80, 400, 600          # truth volume (both slabs stacked)
ZS = 40                           # slab thickness in truth planes
ZV = 64                           # rendered slab volume height (room for the z-bend and the flap tilt)

STEP = (8, 16, 16)                # mesh node stride (z, y, x) on each slab volume
PATCH = (16, 32, 32)              # node offset: node i sits at i*STEP + PATCH//2 (SOFIMA convention)
STRIDE_XYZ = (STEP[2], STEP[1], STEP[0])
FP, FS = (64, 64), (16, 16)       # 2-D face flow patch / step (seam extension)

# nominal piece offsets (x, y, z) in the global frame: slab B sits ZS planes below slab A
OFF = {'A1': np.zeros(3), 'A2': np.zeros(3), 'A': np.zeros(3), 'B': np.array([0., 0., float(ZS)])}


def grid_shape():
    return tuple((v - (p - s)) // s for v, p, s in zip((ZV, Y, X), PATCH, STEP))


def node_xyz(shape):
    gz, gy, gx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    return np.stack([gx*STEP[2] + PATCH[2]//2, gy*STEP[1] + PATCH[1]//2, gz*STEP[0] + PATCH[0]//2], -1).astype(float)
