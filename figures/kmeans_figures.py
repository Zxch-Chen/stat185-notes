# Generates the k-means figures and prints every number quoted in their captions and examples.
# Run from the notes folder: uv run --with numpy,scipy,scikit-learn,matplotlib figures/kmeans_figures.py
import warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import comb, factorial
from sklearn.cluster import SpectralClustering
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, PURPLE, INK = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#1F2933"
COLS = np.array([BLUE, ORANGE, GREEN, PURPLE])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])

def loss(X, lab, C): return ((X - C[lab])**2).sum()
def lloyd(X, C, iters=100):
    hist = []
    for _ in range(iters):
        lab = ((X[:, None] - C[None])**2).sum(2).argmin(1)
        hist.append(loss(X, lab, C))
        newC = np.array([X[lab == l].mean(0) if (lab == l).any() else C[l] for l in range(len(C))])
        if np.allclose(newC, C): break
        C = newC
    return lab, C, hist
def plusplus(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = ((X[:, None] - np.array(C)[None])**2).sum(2).min(1)
        C.append(X[rng.choice(len(X), p=d2/d2.sum())])
    return np.array(C)
def best_of(X, k, starts, rng, pp=True):
    runs = [lloyd(X, plusplus(X, k, rng) if pp else X[rng.choice(len(X), k, replace=False)]) for _ in range(starts)]
    return min(runs, key=lambda r: r[2][-1])

# ------------------------------------------------- counting partitions (Stirling numbers of the second kind)
S2 = lambda n, k: sum((-1)**(k-j)*comb(k, j)*j**n for j in range(k+1))//factorial(k)
print("S(10,4) =", S2(10, 4), " S(19,4) =", S2(19, 4), " S(4,2) =", S2(4, 2))

# ------------------------------------------------- the stuck example from the lecture sketch
eps = 0.1
X4 = np.array([[0, 2], [0, -2], [3, eps], [3, -eps]], float)
lab, C, h = lloyd(X4, np.array([[3, eps], [3, -eps], [0, 0]], float))
print("stuck: labels", lab, "centers", C.round(3).tolist(), "loss", h[-1], "after", len(h), "assignment step(s)")
lab, C, h = lloyd(X4, np.array([[0, 2], [0, -2], [3, 0]], float))
print("optimum: loss", round(h[-1], 4), "= 2 eps^2 =", 2*eps**2)

# ------------------------------------------------- pairwise form of the within-cluster loss
rng = np.random.default_rng(0)
Z = rng.normal(size=(7, 3))
lhs = ((Z[:, None] - Z[None])**2).sum()/(2*len(Z)); rhs = ((Z - Z.mean(0))**2).sum()
print("pairwise identity:", round(lhs, 6), round(rhs, 6))

# ------------------------------------------------- k-means on three cluster shapes, and a graph-based fix
rng = np.random.default_rng(3)
round_ = np.vstack([rng.normal(m, .45, size=(150, 2)) for m in [(0, 0), (3, 0), (1.5, 2.6)]])
R = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
long_ = np.vstack([(rng.normal(size=(200, 2))*[2.2, .22]) @ R.T + m for m in [(0, 0), (0, 1.6)]])
th = rng.uniform(0, 2*np.pi, 500); r = np.where(np.arange(500) < 200, 1, 3) + rng.normal(0, .12, 500)
rings = np.c_[r*np.cos(th), r*np.sin(th)]; ring_truth = (np.arange(500) >= 200).astype(int)
truth_long = np.repeat([0, 1], 200)
def agree(lab, truth): return max((lab == truth).mean(), (lab != truth).mean())
f, ax = plt.subplots(1, 4, figsize=(10.4, 2.9))
for a, (X, k, t) in zip(ax[:3], [(round_, 3, "(a) round: k-means works"), (long_, 2, "(b) stretched: k-means cuts across"),
                                (rings, 2, "(c) rings: k-means cuts in half")]):
    lab, C, h = best_of(X, k, 10, rng)
    a.scatter(*X.T, c=COLS[lab], s=6, lw=0); a.scatter(*C.T, marker="X", s=90, c="white", edgecolor=INK, lw=1.2)
    a.set_title(t); a.set_aspect("equal"); noticks(a)
    if k == 2: print(t, "agreement with true groups:", round(agree(lab, truth_long if X is long_ else ring_truth), 3))
sc = SpectralClustering(2, affinity="nearest_neighbors", n_neighbors=10, random_state=0).fit_predict(rings)
print("spectral on rings agreement:", round(agree(sc, ring_truth), 3))
ax[3].scatter(*rings.T, c=COLS[sc], s=6, lw=0); ax[3].set_title("(d) rings: eigenmaps, then k-means")
ax[3].set_aspect("equal"); noticks(ax[3])
save("kmeans_shapes")

# ------------------------------------------------- random starts versus k-means++, and the elbow
rng = np.random.default_rng(7)
centers = np.array([(0, 0), (5, 0), (0, 5), (5, 5), (2.5, 2.5), (8, 2.5)])
B = np.vstack([rng.normal(m, .7, size=(80, 2)) for m in centers])
k = 6
rand = [lloyd(B, B[rng.choice(len(B), k, replace=False)]) for _ in range(30)]
pp = [lloyd(B, plusplus(B, k, rng)) for _ in range(30)]
fr, fp = np.array([r[2][-1] for r in rand]), np.array([r[2][-1] for r in pp])
best = min(fr.min(), fp.min())
tol = 1.01*best
print(f"k=6, 30 starts each. best loss {best:.1f}. random starts reaching it: {(fr <= tol).sum()}/30, "
      f"worst {fr.max():.1f}; k-means++ reaching it: {(fp <= tol).sum()}/30, worst {fp.max():.1f}")
print("iterations: random median", int(np.median([len(r[2]) for r in rand])), " ++ median", int(np.median([len(r[2]) for r in pp])))
ks = range(1, 11); W = [best_of(B, kk, 10, rng)[2][-1] for kk in ks]
print("elbow:", [round(w) for w in W])
f, ax = plt.subplots(1, 3, figsize=(10.4, 3.0), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
lab, C, _ = best_of(B, 6, 10, rng)
ax[0].scatter(*B.T, c=np.array(COLS.tolist() + ["#D53F8C", "#718096"])[lab], s=6, lw=0)
ax[0].scatter(*C.T, marker="X", s=80, c="white", edgecolor=INK, lw=1.2); ax[0].set_aspect("equal"); noticks(ax[0])
ax[0].set_title("(a) 480 points, best of 10 with $k=6$")
for runs, c, l in [(rand, ORANGE, "random data points"), (pp, BLUE, "k-means++")]:
    for i, r in enumerate(runs): ax[1].plot(range(1, len(r[2])+1), r[2], color=c, alpha=.45, lw=1, label=l if i == 0 else None)
ax[1].axhline(best, color=INK, ls=":", lw=1); ax[1].set_yscale("log"); ax[1].set_xlabel("Lloyd iteration")
ax[1].set_ylabel("loss $L$"); ax[1].set_title("(b) 30 starts each: every curve only goes down"); ax[1].legend(frameon=False)
ax[2].plot(list(ks), W, "o-", color=BLUE, ms=4); ax[2].axvline(6, color=ORANGE, ls="--", lw=1)
ax[2].annotate("elbow at $k=6$", (6, W[5]), (6.8, W[1]*.55), color=ORANGE, arrowprops=dict(arrowstyle="->", color=ORANGE))
ax[2].set_xlabel("number of clusters $k$"); ax[2].set_ylabel("best loss $L$"); ax[2].set_title("(c) scree plot")
ax[2].set_xticks(list(ks))
save("kmeans_starts")
