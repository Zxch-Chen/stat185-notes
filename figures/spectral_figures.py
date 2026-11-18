# Generates the spectral-clustering figures (Section 10.6) and prints every number quoted in the text.
# Run from the notes folder:
#   uv run --python 3.12 --with numpy --with scipy --with scikit-learn --with matplotlib python figures/spectral_figures.py
import itertools, warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, PURPLE, RED, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#C53030", "#1F2933", "#A0AEC0"
COLS = np.array([BLUE, ORANGE, GREEN, PURPLE])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])

# ------------------------------------------------------------ the board's six-node graph with a bridge of weight alpha
EDGES = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
def lap(a):
    A = np.zeros((6, 6))
    for i, j in EDGES: A[i, j] = A[j, i] = 1
    A[2, 3] = A[3, 2] = a
    return np.diag(A.sum(1)) - A, A
for a in [1, .5, .1, .01, 0]:
    w, V = np.linalg.eigh(lap(a)[0])
    print(f"alpha={a}: eigenvalues {w.round(4).tolist()}  v2 {V[:, 1].round(3).tolist()}  2a/3={2*a/3:.4f}")
# trace identity and brute-force best ratio cut at alpha = 1
L1, A1 = lap(1.0); best = None
for r in range(1, 6):
    for C in itertools.combinations(range(6), r):
        C = set(C); Cc = set(range(6)) - C
        W = sum(A1[i, j] for i in C for j in Cc)
        Y = np.zeros((6, 2)); Y[list(C), 0] = 1/np.sqrt(len(C)); Y[list(Cc), 1] = 1/np.sqrt(len(Cc))
        tr = np.trace(Y.T @ L1 @ Y); assert abs(tr - (W/len(C) + W/len(Cc))) < 1e-12
        if best is None or tr < best[0]: best = (tr, sorted(C))
w1 = np.linalg.eigvalsh(L1)
print(f"alpha=1: best sum_l W(C_l,C_l^c)/|C_l| = {best[0]:.4f} at {best[1]}; lambda1+lambda2 = {w1[0]+w1[1]:.4f}")

f, ax = plt.subplots(1, 3, figsize=(10.4, 2.9), gridspec_kw={"width_ratios": [1.05, 1.2, 1.2]})
P = np.array([(0, .7), (0, -.7), (1.1, 0), (2.4, 0), (3.5, .7), (3.5, -.7)])
for i, j in EDGES: ax[0].plot(*P[[i, j]].T, c=INK, lw=1.2)
ax[0].plot(*P[[2, 3]].T, c=RED, lw=2, ls="--")
ax[0].text(1.75, .14, r"$\alpha$", color=RED, ha="center", fontsize=11)
for t, p in enumerate(P):
    ax[0].scatter(*p, s=260, c="white", edgecolors=INK, zorder=3); ax[0].text(*p, str(t+1), ha="center", va="center", zorder=4)
ax[0].set_xlim(-.5, 4); ax[0].set_ylim(-1.1, 1.1); ax[0].set_aspect("equal"); ax[0].axis("off")
ax[0].set_title(r"(a) two triangles, bridge weight $\alpha$")
al = np.linspace(0, 1, 201); ev = np.array([np.linalg.eigvalsh(lap(a)[0])[:3] for a in al])
ax[1].plot(al, ev[:, 0], c=GRAY, lw=2, label=r"$\lambda_1$")
ax[1].plot(al, ev[:, 1], c=BLUE, lw=2, label=r"$\lambda_2$")
ax[1].plot(al, ev[:, 2], c=ORANGE, lw=2, label=r"$\lambda_3$")
ax[1].plot(al, 2*al/3, c=INK, lw=1, ls=":", label=r"ratio cut of the bridge, $2\alpha/3$")
ax[1].set_xlabel(r"bridge weight $\alpha$"); ax[1].set_ylabel("eigenvalue of $L$"); ax[1].legend(fontsize=7.5, loc="center right")
ax[1].set_title(r"(b) $\lambda_2$ is the price of cutting the bridge")
for a, col, off in [(1, BLUE, -.2), (.1, ORANGE, .2)]:
    v = np.linalg.eigh(lap(a)[0])[1][:, 1]; v = v*np.sign(v[0])
    ax[2].bar(np.arange(1, 7) + off, v, width=.38, color=col, label=rf"$\alpha={a}$")
ax[2].axhline(0, c=INK, lw=.8); ax[2].set_xticks(range(1, 7)); ax[2].set_xlabel("vertex")
ax[2].set_ylabel(r"entry of $\bm v_2$" if False else r"entry of $v_2$"); ax[2].legend(fontsize=7.5)
ax[2].set_title(r"(c) the sign of $v_2$ splits the graph")
save("spec_bridge")

# ------------------------------------------------------------ the pipeline: two rings and a blob
rng = np.random.default_rng(4)
def ring(r, n): t = rng.uniform(0, 2*np.pi, n); return np.c_[r*np.cos(t), r*np.sin(t)] + rng.normal(0, .07, (n, 2))
X = np.vstack([ring(1, 250), ring(2.5, 250), rng.normal((5, 0), .3, (120, 2))])
truth = np.r_[np.zeros(250), np.ones(250), 2*np.ones(120)].astype(int)
def acc(lab):
    M = np.zeros((3, 3))
    for a, b in zip(truth, lab): M[a, b] += 1
    r, c = linear_sum_assignment(-M); return M[r, c].sum()/len(lab)
sigma = .35
D2 = ((X[:, None] - X[None])**2).sum(-1)
A = np.exp(-D2/(2*sigma**2)); np.fill_diagonal(A, 0)
d = A.sum(1); Ls = np.eye(len(X)) - A/np.sqrt(np.outer(d, d))
w, V = np.linalg.eigh(Ls)
print("L_sym eigenvalues 1..8:", w[:8].round(4).tolist(), " gaps:", np.diff(w[:8]).round(4).tolist())
Y = V[:, :3]; U = Y/np.linalg.norm(Y, axis=1, keepdims=True)
lab = KMeans(3, n_init=20, random_state=0).fit_predict(U)
km = KMeans(3, n_init=20, random_state=0).fit_predict(X)
print(f"spectral accuracy {acc(lab):.3f}; k-means on raw points {acc(km):.3f}")
cm = np.array([U[lab == j].mean(0) for j in range(3)]); cm /= np.linalg.norm(cm, axis=1, keepdims=True)
ang = np.degrees(np.arccos(np.clip(cm @ cm.T, -1, 1)))
print("angles between cluster directions (deg):", ang[np.triu_indices(3, 1)].round(1).tolist())
raw_norms = [np.linalg.norm(Y[truth == j], axis=1) for j in range(3)]
print("row lengths of Y before normalizing, per cluster (mean, sd):", [(r.mean().round(3), r.std().round(3)) for r in raw_norms],
      " 1/sqrt(vol share) check:", [np.sqrt(d[truth == j].sum()/d.sum()).round(3) for j in range(3)])
order = np.argsort(truth, kind="stable")
print("between-group affinity: max", A[np.ix_(truth == 0, truth == 1)].max().round(6), "| within ring median", np.median(A[np.ix_(truth == 0, truth == 0)]).round(4))

# compact blobs for comparison: the eigengap is clear there
rng2 = np.random.default_rng(2)
Bl = np.vstack([rng2.normal(m, .55, (100, 2)) for m in [(0, 0), (4, .5), (1.8, 3.4)]])
AB = np.exp(-((Bl[:, None] - Bl[None])**2).sum(-1)/(2*.7**2)); np.fill_diagonal(AB, 0); dB = AB.sum(1)
wB = np.linalg.eigvalsh(np.eye(len(Bl)) - AB/np.sqrt(np.outer(dB, dB)))
print("blobs (sigma=0.7) L_sym eigenvalues 1..8:", wB[:8].round(4).tolist(), " gaps:", np.diff(wB[:8]).round(3).tolist())

f = plt.figure(figsize=(10.6, 2.95))
a0 = f.add_subplot(1, 4, 1); a0.scatter(*X.T, s=4, c=COLS[km], lw=0); a0.set_aspect("equal"); noticks(a0)
a0.set_title(f"(a) $k$-means on the points: {100*acc(km):.0f}%")
a1 = f.add_subplot(1, 4, 2); a1.imshow(A[np.ix_(order, order)]**.25, cmap="Blues", interpolation="nearest"); noticks(a1)
a1.set_title(r"(b) affinity $A$, sorted by group")
a3 = f.add_subplot(1, 4, 3, projection="3d")
uu, vv = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]
a3.plot_wireframe(np.cos(uu)*np.sin(vv), np.sin(uu)*np.sin(vv), np.cos(vv), color=GRAY, lw=.3, alpha=.5)
a3.scatter(*U.T, s=6, c=COLS[lab], depthshade=False)
a3.set_xticks([]); a3.set_yticks([]); a3.set_zticks([]); a3.set_box_aspect((1, 1, 1))
a3.set_title(f"(c) rows of $Y$, normalized: {100*acc(lab):.0f}%", pad=-2)
a2 = f.add_subplot(1, 4, 4)
a2.plot(range(1, 11), wB[:10], "o-", c=GREEN, ms=4, label="three round blobs")
a2.plot(range(1, 11), w[:10], "s-", c=BLUE, ms=3.5, label="rings + blob (panel a)")
a2.annotate("jump after $\\lambda_3$", xy=(3.5, (wB[2]+wB[3])/2), xytext=(4.6, .2), fontsize=8, arrowprops=dict(arrowstyle="->", color=INK, lw=.8))
a2.set_xlabel("index $j$"); a2.set_ylabel(r"$\lambda_j$ of $L_{\rm sym}$"); a2.set_xticks(range(1, 11)); a2.legend(fontsize=7, loc="center right")
a2.set_title("(d) choosing $k$ from the eigenvalues")
save("spec_pipeline")
