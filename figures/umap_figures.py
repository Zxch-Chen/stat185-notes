# Generates figures/umap_compare.png and prints the numbers quoted in the UMAP section.
import math, numpy as np, umap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.metrics import pairwise_distances

def sigma(d, target):
    g = np.maximum(0, np.array(d) - min(d)); lo, hi = 1e-9, 1e3
    for _ in range(200):
        m = (lo + hi) / 2
        if np.exp(-g / m).sum() < target: lo = m
        else: hi = m
    return m

dense = np.array([.10, .12, .15, .20]); sparse = dense * 10
for d in (dense, sparse):
    s = sigma(d, math.log2(4)); w = np.exp(-np.maximum(0, d - d.min()) / s)
    print("sigma", round(s, 4), "w", w.round(3), "sum", round(w.sum(), 3))
    print("  global RBF sigma=.5", np.exp(-d**2 / .25).round(3), round(np.exp(-d**2 / .25).sum(), 3))

# a, b fit for min_dist (least squares on the same target curve umap-learn uses)
def ab(min_dist, spread=1.0):
    x = np.linspace(0, spread * 3, 300)
    y = np.where(x <= min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    from scipy.optimize import curve_fit
    return curve_fit(lambda x, a, b: 1 / (1 + a * x ** (2 * b)), x, y)[0]
for md in (0.001, 0.1, 0.5): print("min_dist", md, "a,b", ab(md).round(3))

# numeric gradient check, a=b=1
yi, yj, w = np.array([.3, .4]), np.array([1.1, -.2]), .6
def C(yi):
    v = 1 / (1 + ((yi - yj)**2).sum()); return -w*math.log(v) - (1-w)*math.log(1-v)
d2 = ((yi - yj)**2).sum()
an = 2*(yi-yj)*(w/(1+d2) - (1-w)/(d2*(1+d2)))
nu = np.array([(C(yi+1e-6*e)-C(yi-1e-6*e))/2e-6 for e in np.eye(2)])
print("grad analytic", an, "numeric", nu)

rng = np.random.default_rng(0)
X1, c1 = make_swiss_roll(3000, noise=.05, random_state=0)
# three clusters in R^10: A and B close, C far; C much sparser
C_ = np.zeros((3, 10)); C_[1, 0] = 6; C_[2, 0] = 40
n = [150, 150, 150]; s = [.5, .5, 3.0]
X2 = np.vstack([C_[i] + s[i]*rng.standard_normal((n[i], 10)) for i in range(3)])
c2 = np.repeat([0, 1, 2], n)

k = 15
def embed(X):
    return [("Isomap", Isomap(n_neighbors=k, n_components=2).fit_transform(X)),
            ("Laplacian eigenmaps", SpectralEmbedding(2, affinity="nearest_neighbors", n_neighbors=k, random_state=0).fit_transform(X)),
            ("UMAP, seed 1", umap.UMAP(n_neighbors=k, random_state=1).fit_transform(X)),
            ("UMAP, seed 2", umap.UMAP(n_neighbors=k, random_state=2).fit_transform(X))]
E1, E2 = embed(X1), embed(X2)
cols = np.array(["#2563a8", "#d9651f", "#2e8b57"])
fig, ax = plt.subplots(2, 4, figsize=(10, 4.9))
for j, (name, Y) in enumerate(E1):
    ax[0, j].scatter(*Y.T, c=c1, cmap="viridis", s=1.5); ax[0, j].set_title(name, fontsize=10)
for j, (name, Y) in enumerate(E2):
    ax[1, j].scatter(*Y.T, c=cols[c2], s=3)
    m = np.array([Y[c2 == i].mean(0) for i in range(3)])
    print(name, "centroid dist AB, AC", round(np.linalg.norm(m[0]-m[1]), 2), round(np.linalg.norm(m[0]-m[2]), 2),
          "ratio AC/AB", round(np.linalg.norm(m[0]-m[2])/np.linalg.norm(m[0]-m[1]), 2),
          "spread A, C", round(Y[c2==0].std(0).mean(), 2), round(Y[c2==2].std(0).mean(), 2))
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
ax[0, 0].set_ylabel("Swiss roll", fontsize=10); ax[1, 0].set_ylabel("Three clusters", fontsize=10)
plt.tight_layout(); plt.savefig("figures/umap_compare.png", dpi=220)
m = np.array([X2[c2 == i].mean(0) for i in range(3)])
print("true AB, AC, ratio", np.linalg.norm(m[0]-m[1]).round(2), np.linalg.norm(m[0]-m[2]).round(2), (np.linalg.norm(m[0]-m[2])/np.linalg.norm(m[0]-m[1])).round(2))
print("true spread A, C", X2[c2==0].std(0).mean().round(2), X2[c2==2].std(0).mean().round(2))
