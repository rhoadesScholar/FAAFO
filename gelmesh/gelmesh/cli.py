"""gelmesh command line:  phantom | run | figure | bench"""
import argparse, json, os, pickle, sys, time
import numpy as np

ALL = ['CONNECTED', 'JOINT', 'JOINT_EXT', 'JOINT_DENSE', 'JOINT_FEW', 'JOINT_FEW_EXT',
       'NONEIGHBOUR', 'NONEIGHBOUR_EXT', 'NONEIGHBOUR_DENSE', 'NONEIGHBOUR_FEW', 'NONEIGHBOUR_FEW_EXT']


def cmd_phantom(a):
    from .phantom import build
    d = build()
    pickle.dump(d, open(a.out, 'wb'), protocol=4); print('wrote', a.out)


def cmd_run(a):
    from .phantom import load
    from .experiment import run_variant, truth_source
    from . import relax as R
    if a.k_col is not None: R.K_COL = a.k_col
    if a.k_pair is not None: R.K_PAIR = a.k_pair
    d = load(a.phantom); res = {}
    TB, TA = truth_source('B'), truth_source('A2')      # (source map, truth xy) each
    for v in (a.variants or ALL):
        print(v, flush=True)
        r = run_variant(d, v, rounds=a.rounds, seed=a.seed, truth=TA if v.startswith('NONEIGHBOUR') else TB)
        res[v] = r
    summary = {v: dict(stats=r['stats'], history=r['history'], n_seam=r['n_seam'], n_crack=r['n_crack'], n_crack_auto=r['n_crack_auto']) for v, r in res.items()}
    json.dump(summary, open(a.out + '.json', 'w'), indent=1)
    for r in res.values():
        for k in ('seam_pairs', 'crack_pairs', 'crack_auto'): r.pop(k, None)
    pickle.dump(res, open(a.out, 'wb'), protocol=4); print('wrote', a.out, 'and', a.out + '.json')
    print(table(summary))


def table(summary):
    rows = ['| variant | near crack med / p90 | flap med / p90 | main med / p90 | seam band med / p90 | far med / p90 | unfilled near crack | seam pairs | crack anchors | crack autos |',
            '|---|---|---|---|---|---|---|---|---|---|']
    def mp(st, k): return f"{st[k]['median']:.2f} / {st[k]['p90']:.2f}" if k in st else '—'
    for v, s in summary.items():
        st = s['stats']
        rows.append(f"| {v} | {mp(st, 'near_crack')} | {mp(st, 'near_crack_flap')} | {mp(st, 'near_crack_main')} | {mp(st, 'near_seam')} | {mp(st, 'far')} | "
                    f"{st['near_crack']['unfilled']*100:.1f}% | {s['n_seam']} | {s['n_crack']} | {s['n_crack_auto']} |")
    return '\n'.join(rows)


def cmd_figure(a):
    from .figure import make_figure
    make_figure(a.phantom, a.results, a.out)


def cmd_bench(a):
    from .bench import run_bench
    run_bench(a.out, sizes=a.sizes, steps=a.steps)


def main(argv=None):
    p = argparse.ArgumentParser(prog='gelmesh'); sp = p.add_subparsers(dest='cmd', required=True)
    q = sp.add_parser('phantom', help='build the two-slab phantom'); q.add_argument('--out', default='phantom2.pkl'); q.set_defaults(fn=cmd_phantom)
    q = sp.add_parser('run', help='run experiment variants'); q.add_argument('variants', nargs='*'); q.add_argument('--phantom', default='phantom2.pkl')
    q.add_argument('--out', default='results.pkl'); q.add_argument('--rounds', type=int, default=3); q.add_argument('--seed', type=int, default=5)
    q.add_argument('--k-col', type=float, default=None); q.add_argument('--k-pair', type=float, default=None); q.set_defaults(fn=cmd_run)
    q = sp.add_parser('figure', help='figure from results'); q.add_argument('--phantom', default='phantom2.pkl'); q.add_argument('--results', default='results.pkl')
    q.add_argument('--out', default='fig_joint.png'); q.set_defaults(fn=cmd_figure)
    q = sp.add_parser('bench', help='relaxation cost vs mesh size'); q.add_argument('--out', default='bench.json')
    q.add_argument('--sizes', nargs='*', type=int, default=None); q.add_argument('--steps', type=int, default=200); q.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv); a.fn(a)


if __name__ == '__main__':
    main()
