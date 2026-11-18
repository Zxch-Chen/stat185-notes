# Visual examples for "Why k-means wants round clusters" and kernel k-means (Section 10.1).
# Prints every number quoted in the captions. Run from the notes folder:
#   uv run --python 3.12 --with numpy --with scipy --with matplotlib python figures/kmeans_geometry_figures.py
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal as mvn
BLUE, ORANGE, GREEN, PURPLE, RED, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#C53030", "#1F2933", "#A0AEC0"
LBLUE, LORANGE = "#DCEBFA", "#FCE6D4"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])
rng = np.random.default_rng(7)

# ================================================================ figure 1: the geometry of the nearest-center rule
f, ax = plt.subplots(1, 4, figsize=(11.4, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.45, 1.45]})

# (a) two centers, the bisector, a test point
c1, c2 = np.array([-1.0, -.4]), np.array([1.4, .8])
gx, gy = np.meshgrid(np.linspace(-3, 3, 400), np.linspace(-2.4, 2.4, 400))
G = np.c_[gx.ravel(), gy.ravel()]
side = (((G - c1)**2).sum(1) < ((G - c2)**2).sum(1)).reshape(gx.shape)
a = ax[0]; a.contourf(gx, gy, side, levels=[-.5, .5, 1.5], colors=[LORANGE, LBLUE])
m = (c1 + c2)/2; n = (c2 - c1)/np.linalg.norm(c2 - c1); t = np.array([-n[1], n[0]])
a.plot(*np.c_[m - 4*t, m + 4*t], c=INK, lw=1.3)
a.annotate("", xy=c2, xytext=c1, arrowprops=dict(arrowstyle="->", color=GRAY, lw=1, ls="--"))
x = np.array([-.3, 1.5])
d1, d2 = np.linalg.norm(x - c1), np.linalg.norm(x - c2)
print(f"(a) test point {x}: |x-c1| = {d1:.2f}, |x-c2| = {d2:.2f}")
for c, col, lab in [(c1, BLUE, r"$c_1$"), (c2, ORANGE, r"$c_2$")]:
    a.plot(*np.c_[x, c], c=col, lw=1); a.scatter(*c, marker="X", s=70, c=col, ec=INK, lw=.6, zorder=4)
    a.text(c[0] + .12, c[1] - .32, lab, color=col, fontsize=10)
a.scatter(*x, s=22, c=INK, zorder=5); a.text(x[0] - .25, x[1] + .2, r"$x$", fontsize=10)
a.scatter(*m, s=18, c="white", ec=INK, zorder=5)
a.text(-2.85, -2.2, "perpendicular bisector\nthrough the midpoint", fontsize=7)
a.set_xlim(-3, 3); a.set_ylim(-2.4, 2.4); a.set_aspect("equal"); noticks(a)
a.set_title(f"(a) $x$ joins $c_2$:\n$|x-c_2|={d2:.2f}<|x-c_1|={d1:.2f}$")

# (b) Voronoi cells of 6 centers are convex; a ring cannot be one cell
C = np.array([(-1.6, 1.1), (0.3, 1.6), (1.8, .6), (1.2, -1.4), (-.8, -1.3), (0, 0)])
cols = np.array([BLUE, ORANGE, GREEN, PURPLE, "#D53F8C", "#D69E2E"])
lab = ((G[:, None] - C[None])**2).sum(-1).argmin(1).reshape(gx.shape)
a = ax[1]; a.contourf(gx, gy, lab, levels=np.arange(-.5, 6), colors=[c + "40" for c in cols])
vor = Voronoi(np.vstack([C, [[-20, -20], [20, -20], [20, 20], [-20, 20]]]))
voronoi_plot_2d(vor, ax=a, show_points=False, show_vertices=False, line_colors=INK, line_width=1)
a.scatter(*C.T, marker="X", s=55, c=cols, ec=INK, lw=.6, zorder=4)
th = np.linspace(0, 2*np.pi, 300); a.plot(1.55*np.cos(th), 1.55*np.sin(th), c=RED, lw=1.6, ls="--")
a.set_xlim(-3, 3); a.set_ylim(-2.4, 2.4); a.set_aspect("equal"); noticks(a)
a.set_title("(b) every cell is a convex polygon;\na ring (red) crosses five cells")

# (c, d) two long parallel clusters: the cut along vs the cut across
n = 200
L1 = np.c_[rng.uniform(-4, 4, n), .6 + rng.normal(0, .15, n)]
L2 = np.c_[rng.uniform(-4, 4, n), -.6 + rng.normal(0, .15, n)]
X = np.vstack([L1, L2]); truth = np.r_[np.zeros(n), np.ones(n)]
def loss(lab): return sum(((X[lab == j] - X[lab == j].mean(0))**2).sum() for j in (0, 1))
along, across = truth.astype(int), (X[:, 0] > 0).astype(int)
la, lc = loss(along), loss(across)
print(f"(c,d) long clusters: loss of the true split (cut along) = {la:.0f}, crosswise split = {lc:.0f}, ratio {la/lc:.2f}")
for a, labs, L, ttl in [(ax[2], along, la, "(c) the true split"), (ax[3], across, lc, "(d) the split $k$-means prefers")]:
    cc = np.array([X[labs == j].mean(0) for j in (0, 1)])
    a.scatter(*X.T, s=4, c=np.where(labs == 0, BLUE, ORANGE), lw=0)
    a.scatter(*cc.T, marker="X", s=60, c=[BLUE, ORANGE], ec=INK, lw=.6, zorder=4)
    mm = cc.mean(0); nn = (cc[1] - cc[0])/np.linalg.norm(cc[1] - cc[0]); tt = np.array([-nn[1], nn[0]])
    a.plot(*np.c_[mm - 6*tt, mm + 6*tt], c=INK, lw=1)
    for j in (0, 1):  # a few error bars from points to their center
        idx = np.where(labs == j)[0][::25]
        for i in idx: a.plot(*np.c_[X[i], cc[j]], c=GRAY, lw=.5, zorder=0)
    a.set_xlim(-4.5, 4.5); a.set_ylim(-1.5, 1.5); a.set_aspect("equal"); noticks(a)
    a.set_title(f"{ttl}\nloss $L={L:.0f}$")
save("kmeans_bisector")

# ================================================================ figure 2: GMM boundaries curve, k-means boundaries don't
f, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
mu1, S1 = np.array([0, 0]), .35**2*np.eye(2)
mu2, S2 = np.array([2.6, 0]), np.diag([1.6**2, 1.1**2])
gx2, gy2 = np.meshgrid(np.linspace(-3, 7, 500), np.linspace(-3.5, 3.5, 400))
G2 = np.c_[gx2.ravel(), gy2.ravel()]
lr = (mvn(mu1, S1).logpdf(G2) - mvn(mu2, S2).logpdf(G2)).reshape(gx2.shape)
for a, ttl in [(ax[0], "(a) $k$-means: straight bisector"), (ax[1], "(b) GMM: a curved boundary")]:
    for mu, S, col in [(mu1, S1, BLUE), (mu2, S2, ORANGE)]:
        w, V = np.linalg.eigh(S)
        for r in (1, 2):
            e = matplotlib.patches.Ellipse(mu, 2*r*np.sqrt(w[1]), 2*r*np.sqrt(w[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                                           fill=False, ec=col, lw=1, ls="-" if r == 1 else ":")
            a.add_patch(e)
        a.scatter(*mu, marker="X", s=55, c=col, ec=INK, lw=.6, zorder=4)
    a.set_xlim(-3, 7); a.set_ylim(-3.5, 3.5); a.set_aspect("equal"); noticks(a); a.set_title(ttl)
mid = (mu1 + mu2)/2
ax[0].contourf(gx2, gy2, (gx2 < mid[0]).astype(float), levels=[-.5, .5, 1.5], colors=[LORANGE, LBLUE], zorder=0)
ax[0].axvline(mid[0], c=INK, lw=1.2)
ax[1].contourf(gx2, gy2, (lr > 0).astype(float), levels=[-.5, .5, 1.5], colors=[LORANGE, LBLUE], zorder=0)
ax[1].contour(gx2, gy2, lr, levels=[0], colors=INK, linewidths=1.2)
# where does the GMM boundary cross the line between the means, and how far does the blue region reach?
xs = np.linspace(-3, 7, 20001); l1 = mvn(mu1, S1).logpdf(np.c_[xs, 0*xs]) - mvn(mu2, S2).logpdf(np.c_[xs, 0*xs])
ins = xs[l1 > 0]
print(f"GMM boundary on the axis: blue region from x = {ins.min():.2f} to {ins.max():.2f}; k-means bisector at x = {mid[0]:.2f}")
ax[1].text(-2.8, -3.2, f"blue only from $x={ins.min():.2f}$ to ${ins.max():.2f}$", fontsize=7.5)
ax[0].text(-2.8, -3.2, f"bisector at $x={mid[0]:.1f}$", fontsize=7.5)
save("kmeans_vs_gmm_boundary")

# ================================================================ figure 3: kernel k-means, term by term, and its curved boundary
def ring(r, n): t = rng.uniform(0, 2*np.pi, n); return np.c_[r*np.cos(t), r*np.sin(t)] + rng.normal(0, .1, (n, 2))
R = np.vstack([ring(1, 150), ring(3, 150)]); rt = np.r_[np.zeros(150), np.ones(150)].astype(int)
s = .8
def kern(A, B): return np.exp(-((A[:, None] - B[None])**2).sum(-1)/(2*s**2))
K = kern(R, R)
def kdist(Kxx, Kx, lab, j):  # ||phi(x) - c_j||^2 from kernel values only
    m = lab == j
    return Kxx - 2*Kx[..., m].mean(-1) + K[np.ix_(m, m)].mean()
def kkmeans(lab, it=100):
    for _ in range(it):
        D = np.stack([kdist(np.diag(K), K, lab, j) for j in (0, 1)], 1)
        new = D.argmin(1)
        if (new == lab).all(): break
        lab = new
    obj = D[np.arange(len(lab)), lab].sum()
    return lab, obj
best = None
for st in range(30):
    lab0 = rng.integers(0, 2, len(R))
    lab, obj = kkmeans(lab0)
    if len(set(lab)) == 2 and (best is None or obj < best[1]): best = (lab, obj)
lab = best[0]
acc = max((lab == rt).mean(), (lab != rt).mean())
if (lab == rt).mean() < .5: lab = 1 - lab
print(f"kernel k-means (RBF, sigma={s}), best of 30 random starts: accuracy {acc:.3f}, objective {best[1]:.1f}")
# term-by-term for one point on the inner ring
i0 = np.argmin(((R - np.array([1, 0]))**2).sum(1))
terms = {}
for j in (0, 1):
    m = lab == j
    t1, t2, t3 = K[i0, i0], -2*K[i0, m].mean(), K[np.ix_(m, m)].mean()
    terms[j] = (t1, t2, t3, t1 + t2 + t3)
    print(f"point {R[i0].round(2)} vs {'inner' if j == 0 else 'outer'} cluster: K_ii {t1:.3f}, -2*avg sim {t2:.3f}, +tightness {t3:.3f} -> distance^2 {t1+t2+t3:.3f}")

f, ax = plt.subplots(1, 3, figsize=(10.6, 3.1), gridspec_kw={"width_ratios": [1, 1.15, 1]})
a = ax[0]; sc = a.scatter(*R.T, s=7, c=K[i0], cmap="viridis", lw=0)
a.scatter(*R[i0], s=90, facecolor="none", ec=RED, lw=1.6); a.set_aspect("equal"); noticks(a)
plt.colorbar(sc, ax=a, fraction=.046, pad=.02).set_label(r"$k(x_i,x_j)$", fontsize=8)
a.set_title(r"(a) similarity of every point to $x_i$ (red)")
a = ax[1]; w = .36
names = [r"$K_{ii}$", r"$-\frac{2}{|C|}\sum_j K_{ij}$", r"$+\frac{1}{|C|^2}\sum K_{jj'}$", r"$=\|\varphi(x_i)-c\|^2$"]
for j, col, off in [(0, BLUE, -w/2), (1, ORANGE, w/2)]:
    a.bar(np.arange(4) + off, terms[j], width=w, color=col, label="inner ring" if j == 0 else "outer ring")
    for q, v in enumerate(terms[j]): a.text(q + off, v + (.03 if v >= 0 else -.09), f"{v:.2f}", ha="center", fontsize=6.8)
a.set_ylim(-1.0, 1.25); a.axhline(0, c=INK, lw=.8); a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=7.5); a.legend(fontsize=7.5, loc="lower left")
a.set_title(r"(b) the distance from $x_i$ to each center, term by term")
a = ax[2]
gx3, gy3 = np.meshgrid(np.linspace(-4.2, 4.2, 220), np.linspace(-4.2, 4.2, 220))
G3 = np.c_[gx3.ravel(), gy3.ravel()]; KG = kern(G3, R)
DG = np.stack([kdist(1.0, KG, lab, j) for j in (0, 1)], 1)
a.contourf(gx3, gy3, DG.argmin(1).reshape(gx3.shape), levels=[-.5, .5, 1.5], colors=[LBLUE, LORANGE])
a.contour(gx3, gy3, (DG[:, 0] - DG[:, 1]).reshape(gx3.shape), levels=[0], colors=INK, linewidths=1)
a.scatter(*R.T, s=5, c=np.where(lab == 0, BLUE, ORANGE), lw=0); a.set_aspect("equal"); noticks(a)
a.set_title(f"(c) kernel $k$-means: {100*acc:.0f}%,\na circular boundary in the plane")
save("kernel_kmeans_terms")

# ================================================================ figure 4: the three kinds of quadratic boundary
def coeffs(mu1, S1, mu2, S2, p1=.5, p2=.5):
    """g(x) = log p1 N(x;mu1,S1) - log p2 N(x;mu2,S2) = x^T Q x + b^T x + f; boundary g = 0."""
    P1, P2 = np.linalg.inv(S1), np.linalg.inv(S2)
    Q = -.5*(P1 - P2); b = P1 @ mu1 - P2 @ mu2
    f0 = -.5*(mu1 @ P1 @ mu1 - mu2 @ P2 @ mu2) - .5*np.log(np.linalg.det(S1)/np.linalg.det(S2)) + np.log(p1/p2)
    return Q, b, f0
cases = [("(a) shared $\\Sigma$: a line", np.array([0, 0]), np.array([[1.2, .5], [.5, .6]]), np.array([3, .5]), np.array([[1.2, .5], [.5, .6]])),
         ("(b) one inside the other: an ellipse", np.array([0, 0]), .35**2*np.eye(2), np.array([2.6, 0]), np.diag([1.6**2, 1.1**2])),
         ("(c) crossed shapes: a hyperbola", np.array([0, 0]), np.diag([1.6**2, .4**2]), np.array([1.5, .8]), np.diag([.4**2, 1.6**2]))]
f, ax = plt.subplots(1, 3, figsize=(10.4, 3.25))
gx4, gy4 = np.meshgrid(np.linspace(-4, 6, 500), np.linspace(-4, 4, 400)); G4 = np.c_[gx4.ravel(), gy4.ravel()]
for a, (ttl, m1, S1, m2, S2) in zip(ax, cases):
    Q, b, f0 = coeffs(m1, S1, m2, S2)
    A_, B_, C_ = Q[0, 0], 2*Q[0, 1], Q[1, 1]
    kind = "line" if np.allclose(Q, 0) else ("ellipse" if B_**2 - 4*A_*C_ < 0 else "hyperbola")
    print(f"{ttl}: a={A_:.3f} b={B_:.3f} c={C_:.3f} d={b[0]:.3f} e={b[1]:.3f} f={f0:.3f}; b^2-4ac={B_**2-4*A_*C_:.3f} -> {kind}")
    g = (np.einsum("ni,ij,nj->n", G4, Q, G4) + G4 @ b + f0).reshape(gx4.shape)
    a.contourf(gx4, gy4, (g > 0).astype(float), levels=[-.5, .5, 1.5], colors=[LORANGE, LBLUE])
    a.contour(gx4, gy4, g, levels=[0], colors=INK, linewidths=1.3)
    for mu, S, col in [(m1, S1, BLUE), (m2, S2, ORANGE)]:
        w, V = np.linalg.eigh(S)
        for r in (1, 2):
            a.add_patch(matplotlib.patches.Ellipse(mu, 2*r*np.sqrt(w[1]), 2*r*np.sqrt(w[0]), angle=np.degrees(np.arctan2(V[1, 1], V[0, 1])),
                                                  fill=False, ec=col, lw=1, ls="-" if r == 1 else ":"))
        a.scatter(*mu, marker="X", s=45, c=col, ec=INK, lw=.6, zorder=4)
    a.set_xlim(-4, 6); a.set_ylim(-4, 4); a.set_aspect("equal"); noticks(a); a.set_title(ttl)
    a.text(-3.85, -3.75, f"$b^2-4ac={B_**2-4*A_*C_:.2f}$" if kind != "line" else "$a=b=c=0$", fontsize=8,
           bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
save("gmm_conics")
