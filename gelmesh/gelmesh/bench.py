"""Cost of one relaxation vs mesh size, for scaling estimates.

Times `mesh.relax_mesh` with the full custom force (elastic + pair springs + surface springs + column coupling)
on synthetic batched meshes of increasing node count, a fixed number of FIRE steps, and reports seconds per
step per node and peak memory.  Pair/surface spring counts scale with the face area (one per face node).
The phantom's production run is ~1000-2000 steps per round and 3 rounds.
"""
import json, time, resource
import numpy as np
import jax, jax.numpy as jnp
from sofima import mesh
from .geom import STRIDE_XYZ


def synthetic_force(P, shape, k_pair=0.3, k_surf=0.3, k_col=1.0, pairs_per_face_node=1.0, rng=None):
    rng = rng or np.random.default_rng(0)
    nz, ny, nx = shape
    n_face = ny * nx; n_pair = int(n_face * pairs_per_face_node); n_surf = 2 * n_face * P
    PA = np.stack([rng.integers(0, P, n_pair), np.full(n_pair, nz-1), rng.integers(0, ny, n_pair), rng.integers(0, nx, n_pair)])
    PB = np.stack([rng.integers(0, P, n_pair), np.zeros(n_pair, int), rng.integers(0, ny, n_pair), rng.integers(0, nx, n_pair)])
    T = rng.normal(0, 2, (3, n_pair)).astype(np.float32)
    SN = np.stack([rng.integers(0, P, n_surf), rng.choice([0, nz-1], n_surf), rng.integers(0, ny, n_surf), rng.integers(0, nx, n_surf)])
    ST = rng.normal(0, 1, n_surf).astype(np.float32)
    PA, PB, T, SN, ST = map(jnp.asarray, (PA, PB, T, SN, ST))

    def force(x, k, stride, prefer_orig_order):
        f = mesh.elastic_mesh_3d(x, k, stride, prefer_orig_order)
        xa = x[:, PA[0], PA[1], PA[2], PA[3]]; xb = x[:, PB[0], PB[1], PB[2], PB[3]]
        pull = k_pair * (xb + T - xa)
        f = f.at[:, PA[0], PA[1], PA[2], PA[3]].add(pull); f = f.at[:, PB[0], PB[1], PB[2], PB[3]].add(-pull)
        xz = x[2, SN[0], SN[1], SN[2], SN[3]]; f = f.at[2, SN[0], SN[1], SN[2], SN[3]].add(k_surf * (ST - xz))
        f = f.at[:2].add(k_col * (jnp.mean(x[:2], axis=2, keepdims=True) - x[:2]))
        return f
    return force, n_pair, n_surf


def time_relax(P, shape, steps=200, repeats=2):
    cfg = mesh.IntegrationConfig(dt=0.001, gamma=0.0, k0=0.1, k=0.03, stride=STRIDE_XYZ, num_iters=steps, max_iters=steps,
                                 stop_v_max=0.0, dt_max=1000, start_cap=10, final_cap=10, prefer_orig_order=True)
    force, n_pair, n_surf = synthetic_force(P, shape)
    prev = np.full((3, P) + tuple(shape), np.nan, np.float32); prev[:2, 0] = 0.0
    x0 = jnp.zeros_like(prev); pv = jnp.asarray(prev)
    mesh.relax_mesh(x0, pv, mesh.IntegrationConfig(**{**cfg.to_dict(), 'num_iters': 2, 'max_iters': 2}), mesh_force=force)   # compile
    best = None
    for _ in range(repeats):
        t = time.time(); x, e, n = mesh.relax_mesh(x0, pv, cfg, mesh_force=force); np.asarray(x); dt = time.time() - t
        best = dt if best is None else min(best, dt)
    nodes = P * int(np.prod(shape))
    return dict(P=P, shape=list(shape), nodes=nodes, n_pair=n_pair, n_surf=n_surf, steps=int(n), seconds=best,
                s_per_step=best / int(n), ns_per_step_node=1e9 * best / int(n) / nodes,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, device=str(jax.devices()[0]))


def run_bench(out, sizes=None, steps=200):
    sizes = sizes or [24, 48, 96, 192]
    rows = []
    for s in sizes:
        shape = (7, s, int(s * 1.5))        # the phantom grid is (7, 24, 36): z x y x x nodes
        r = time_relax(3, shape, steps); rows.append(r)
        print(f"P=3 shape={shape} nodes={r['nodes']:,} pairs={r['n_pair']:,}: {r['s_per_step']*1e3:.1f} ms/step, "
              f"{r['ns_per_step_node']:.0f} ns/step/node, peak RSS {r['peak_rss_mb']:.0f} MB  [{r['device']}]", flush=True)
    json.dump(rows, open(out, 'w'), indent=1); print('wrote', out)
    return rows
