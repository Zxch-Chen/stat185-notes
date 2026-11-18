# The "zoom test" for local SVD: how sigma_2/sigma_1 (and sigma_3/sigma_2) change with the neighborhood size k
# on a clean curve, a flat sheet, a noisy curve and a thin ribbon, all sitting in R^10.
# Prints every number quoted in the notes.
# Run from the notes folder: uv run --python 3.12 --with numpy --with scipy --with matplotlib python figures/zoom_figures.py
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
BLUE, ORANGE, GREEN, PURPLE, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#1F2933", "#A0AEC0"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
rng = np.random.default_rng(0)
N, D = 4000, 10
def embed(P):
    X = np.zeros((len(P), D)); X[:, :P.shape[1]] = P; return X
t = rng.uniform(0, 2 * np.pi, N)
circle = np.c_[np.cos(t), np.sin(t)]
shapes = {
    "clean curve (1-D, bent)": embed(circle),
    "flat sheet (2-D)": embed(rng.uniform(0, 1, (N, 2))),
    "noisy curve (1-D + noise)": embed(circle) + rng.normal(0, 0.004, (N, D)),
    "ribbon (1-D + thin real width)": embed(np.c_[circle, rng.uniform(0, 0.03, N)]),
}
cols = dict(zip(shapes, [BLUE, GREEN, ORANGE, PURPLE]))
ks = np.unique(np.round(np.geomspace(3, 300, 18)).astype(int))
def ratios(X, n_base=150):
    tree = cKDTree(X); base = rng.choice(len(X), n_base, replace=False)
    r21, r32 = [], []
    for k in ks:
        _, idx = tree.query(X[base], k + 1)
        s = np.array([np.linalg.svd(X[nb[1:]] - X[i], compute_uv=False)[:3] for i, nb in zip(base, idx)])
        r21.append(np.median(s[:, 1] / s[:, 0])); r32.append(np.median(s[:, 2] / s[:, 1]))
    return np.array(r21), np.array(r32)
R = {n: ratios(X) for n, X in shapes.items()}
for n, (a, b) in R.items():
    print(f"{n:32s} s2/s1:", " ".join(f"{k}:{v:.2f}" for k, v in zip(ks, a)))
    print(f"{'':32s} s3/s2:", " ".join(f"{k}:{v:.2f}" for k, v in zip(ks, b)))
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
for n, (a, b) in R.items():
    ax[0].plot(ks, a, "-o", ms=3, lw=1.6, color=cols[n], label=n)
for n in ["noisy curve (1-D + noise)", "ribbon (1-D + thin real width)"]:
    ax[1].plot(ks, R[n][1], "-o", ms=3, lw=1.6, color=cols[n], label=n)
for a in ax: a.set_xscale("log"); a.set_xlabel("neighbors $k$  (small = zoomed in)"); a.set_xticks([3, 10, 30, 100, 300]); a.set_xticklabels(["3", "10", "30", "100", "300"])
ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\sigma_2/\sigma_1$  (width / length)")
ax[0].set_title(r"(a) the zoom test: $\sigma_2/\sigma_1$ against $k$")
ax[1].set_ylabel(r"$\sigma_3/\sigma_2$"); ax[1].set_ylim(0, 1.05)
ax[1].set_title(r"(b) noise or a real thin direction? $\sigma_3/\sigma_2$")
ax[0].legend(fontsize=7, frameon=False, loc="lower right")
ax[1].legend(fontsize=7, frameon=False, loc="upper center")
plt.tight_layout(); plt.savefig("figures/zoom_test.png", dpi=220); print("saved zoom_test")
