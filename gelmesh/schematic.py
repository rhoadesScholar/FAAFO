"""README schematic: what the joint relaxation is (left) and how a solve runs (right)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

C_FLAP, C_MAIN, C_B = '#d9822b', '#2b7bba', '#5a9e5a'
C_CRACK, C_SEAM, C_SURF, C_COL, C_GAUGE = '#c0392b', '#1f4e79', '#2e7d32', '#8e8e8e', '#111111'

fig = plt.figure(figsize=(15, 6.8), dpi=130)
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.06)
ax = fig.add_subplot(gs[0]); ax.set_xlim(-0.6, 10.6); ax.set_ylim(-0.3, 7.6); ax.axis('off')

# ---------- left: cross-section through the stack, pieces as lattices ----------
def lattice(ax, poly, color, nx, nz, label=None, shift=(0, 0), rot=0.0, alpha=1.0):
    """draw a piece: filled polygon + a sparse node lattice clipped to it (visual only)"""
    P = np.array(poly, float)
    c = P.mean(0); R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    P = (P - c) @ R.T + c + np.array(shift)
    patch = Polygon(P, closed=True, facecolor=color, alpha=0.13 * alpha, edgecolor=color, lw=1.6)
    ax.add_patch(patch)
    xs = np.linspace(P[:, 0].min() + 0.25, P[:, 0].max() - 0.25, nx); zs = np.linspace(P[:, 1].min() + 0.25, P[:, 1].max() - 0.25, nz)
    from matplotlib.path import Path
    path = Path(P)
    pts = np.array([(x, z) for z in zs for x in xs]); inside = path.contains_points(pts)
    pts = pts[inside]
    ax.scatter(pts[:, 0], pts[:, 1], s=9, color=color, zorder=3, alpha=alpha)
    # lattice links (nearest neighbours on the grid, only where both inside)
    grid = {(round(x, 3), round(z, 3)) for x, z in pts}
    dx = xs[1] - xs[0]; dz = zs[1] - zs[0]
    for x, z in pts:
        for nx_, nz_ in ((x + dx, z), (x, z + dz)):
            if (round(nx_, 3), round(nz_, 3)) in grid:
                ax.plot([x, nx_], [z, nz_], color=color, lw=0.6, alpha=0.6 * alpha, zorder=2)
    if label: ax.text(P[:, 0].min() + 0.15, P[:, 1].min() + 0.12, label, ha='left', va='bottom', color=color, fontsize=9.5, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85), zorder=7)
    return P

# slab A (top): main body + flap, cut by a crack that is open on the left and closed/sheared on the right
# crack curve in the x-z section: a slanted face
main = [(0, 4.0), (5.6, 4.0), (6.3, 6.6), (0, 6.6)]
flap = [(5.6, 4.0), (10, 4.0), (10, 6.6), (6.3, 6.6)]
PB = lattice(ax, [(0, 0.6), (10, 0.6), (10, 3.3), (0, 3.3)], C_B, 11, 4, 'piece B — neighbour slab (root)')
PM = lattice(ax, main, C_MAIN, 7, 4, 'piece A2 — main body')
PF = lattice(ax, flap, C_FLAP, 5, 4, 'piece A1 — flap', shift=(0.55, 0.2), rot=-0.08)

# springs
def spring(ax, p, q, color, lw=1.8, zig=6, amp=0.07, zorder=5):
    p, q = np.array(p), np.array(q); d = q - p; L = np.linalg.norm(d); u = d / L; n = np.array([-u[1], u[0]])
    t = np.linspace(0, 1, 2 * zig + 2); pts = p + np.outer(t, d)
    pts[1:-1] += np.outer(amp * np.where(np.arange(1, len(t) - 1) % 2 == 0, 1, -1), n)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, zorder=zorder, solid_capstyle='round')

# crack-face springs: between the main face and the (displaced) flap face
for z in (4.4, 5.1, 5.8, 6.4):
    xm = 5.6 + (z - 4.0) * (0.7 / 2.6)                 # main face position at this z
    xf = xm + 0.75 + 0.1 * (z - 5.2)                   # flap face (displaced)
    spring(ax, (xm, z), (xf, z + 0.25), C_CRACK)
# seam springs between A faces and B top
for x in (0.8, 2.0, 3.2, 4.4, 6.6, 7.8, 9.0):
    zb = 3.3; za = 4.0 + (0.35 if x > 5.6 else 0.0)
    spring(ax, (x, zb), (x + (0.25 if x > 5.6 else 0.0), za), C_SEAM, zig=4, amp=0.06)
# surface springs to target planes (flatten): dashed horizontal targets + short springs
for z_t, pieces_x in ((6.6, [(1.0, 6.75), (3.0, 6.85)]), (3.3, [(1.2, 3.48), (3.6, 3.55)]), (0.6, [(2.2, 0.42), (7.5, 0.4)])):
    ax.plot([-0.3, 10.3], [z_t, z_t], ls='--', color=C_SURF, lw=1.0, alpha=0.8, zorder=1)
    for x, z in pieces_x:
        spring(ax, (x, z), (x, z_t), C_SURF, zig=3, amp=0.05, lw=1.5)
# column coupling: a vertical bracket on one column of A2
ax.plot([2.4, 2.4], [4.25, 6.35], color=C_COL, lw=5, alpha=0.35, zorder=1)
ax.annotate('', xy=(2.4, 6.35), xytext=(2.4, 4.25), arrowprops=dict(arrowstyle='<->', color=C_COL, lw=1.3))
# gauge pin on B
ax.plot([5.0], [0.6], marker='v', color=C_GAUGE, ms=11, zorder=6); ax.plot([5.0, 5.0], [0.6, 1.2], color=C_GAUGE, lw=1.6)

# legend text
leg = [
    (C_CRACK, 'crack-face springs: a few manual anchors + strip matcher'),
    (C_SEAM, 'seam springs: manual anchors → affine → warp-then-match extension'),
    (C_SURF, 'surface springs: detected surfaces pulled to target planes (the flatten)'),
    (C_COL, 'column coupling: xy warp uniform through z (slabreg\'s field model)'),
    (C_GAUGE, 'gauge: root piece xy pinned (k₀); everything else is relative'),
]
ax.text(5.0, 7.35, 'One energy over all pieces:\nSOFIMA elastic lattice per piece + interface springs + flatten + column coupling + gauge',
        ha='center', va='center', fontsize=10, fontweight='bold', zorder=8, bbox=dict(facecolor='white', edgecolor='none', pad=2))
for i, (c, t) in enumerate(leg):
    fig.text(0.03, 0.115 - i * 0.025, '■', color=c, fontsize=11, va='center')
    fig.text(0.045, 0.115 - i * 0.025, t, fontsize=9.3, va='center', color='#222222')
ax.set_ylim(-1.0, 8.0)

# ---------- right: the solve as a loop ----------
bx = fig.add_subplot(gs[1]); bx.set_xlim(0, 10); bx.set_ylim(0, 10); bx.axis('off')
def box(x, y, w, h, text, color='#f4f4f4', ec='#555555', fs=9.6, bold=False):
    bx.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.25', facecolor=color, edgecolor=ec, lw=1.3))
    bx.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', color='#111111')
def arrow(p, q, text=None, color='#333333', rad=0.0):
    bx.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=14, color=color, lw=1.4, connectionstyle=f'arc3,rad={rad}'))
    if text: bx.text((p[0] + q[0]) / 2 + 0.15, (p[1] + q[1]) / 2, text, fontsize=8.6, color=color, va='center')

box(0.6, 8.6, 8.8, 1.0, 'inputs: per-slab volumes, detected surfaces, crack mask → pieces,\n~5–20 manual pairs per interface (seam and crack)', color='#fbf3e6', fs=8.8)
box(0.6, 7.0, 8.8, 0.9, 'round 0 — seeds only: seam pairs → affine → targeted SOFIMA flow\non the faces; crack anchors as springs', fs=8.8)
box(1.6, 5.35, 6.8, 0.95, 'RELAX  all pieces together (SOFIMA FIRE)\nmax |total force| < 1e-4', color='#e8eef7', ec=C_SEAM, fs=9.6, bold=True)
box(0.6, 3.25, 4.1, 1.35, 'seam extension\nwarp moving face through\ncurrent mesh → match residual\n(residual + consensus gates)', color='#eaf0f8', ec=C_SEAM, fs=8.6)
box(5.3, 3.25, 4.1, 1.35, 'crack-face extension\nstrip matcher on t × z face\npatches: gap → normal offset,\nshell ZNCC → slide', color='#fbeaea', ec=C_CRACK, fs=8.6)
box(0.6, 1.1, 8.8, 1.1, 'output: per-piece displacement sampled on each face, composited by\npiece label → slabreg prebuilt field (mls) → s07 renders as today', color='#eaf5ea', ec=C_SURF, fs=8.8)
arrow((5.0, 8.6), (5.0, 7.9)); arrow((5.0, 7.0), (5.0, 6.3))
arrow((3.6, 5.35), (2.65, 4.6)); arrow((6.4, 5.35), (7.35, 4.6))
arrow((2.65, 3.25), (3.6, 2.2), color='#999999'); arrow((7.35, 3.25), (6.4, 2.2), color='#999999')
# loop arrows back up to RELAX (rounds 1..2)
bx.add_patch(FancyArrowPatch((0.6, 3.9), (1.6, 5.8), arrowstyle='-|>', mutation_scale=14, color=C_SEAM, lw=1.4, connectionstyle='arc3,rad=0.55'))
bx.add_patch(FancyArrowPatch((9.4, 3.9), (8.4, 5.8), arrowstyle='-|>', mutation_scale=14, color=C_CRACK, lw=1.4, connectionstyle='arc3,rad=-0.55'))
bx.text(5.0, 2.65, 'rounds 1–2: extend → relax', ha='center', fontsize=9, color='#444444', style='italic')
bx.text(5.0, 9.85, 'How a solve runs', ha='center', va='center', fontsize=10.5, fontweight='bold')
bx.text(5.0, 0.45, 'Pieces never share mesh links across a crack, so the solution can hold the jump;\nthe neighbour slab\'s seam pairs pin the slide a crack alone leaves free.', ha='center', fontsize=8.8, color='#333333')

fig.savefig('schematic.png', bbox_inches='tight', facecolor='white')
print('wrote schematic.png')
