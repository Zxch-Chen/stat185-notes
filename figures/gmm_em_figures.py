# Figures for the EM derivation and the covariance-model family (Nov 12 lecture),
# and every number quoted in their captions and examples.
# Run from the notes folder: uv run --with numpy,scipy,scikit-learn,matplotlib figures/gmm_em_figures.py
import warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, PURPLE, RED, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#C53030", "#1F2933", "#A0AEC0"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)

# ------------------------------------------------- EM as climbing a lower bound (1-D, only the weight p is unknown)
rng = np.random.default_rng(0)
N, ptrue = 200, 0.3
z = rng.random(N) < ptrue
x = np.where(z, rng.normal(0, 1, N), rng.normal(1.5, 1, N))
f1, f2 = norm(0, 1).pdf(x), norm(1.5, 1).pdf(x)
def ell(p): return np.log(p*f1 + (1 - p)*f2).sum()
def resp(p): return p*f1/(p*f1 + (1 - p)*f2)
def Q(p, pt):  # expected complete-data log-likelihood, dropping terms free of p
    t = resp(pt); return (t*np.log(p) + (1 - t)*np.log(1 - p)).sum()
def bound(p, pt): return ell(pt) + Q(p, pt) - Q(pt, pt)
grid = np.linspace(0.02, 0.98, 400)
L = np.array([ell(p) for p in grid])
pts = [0.9]
for _ in range(60): pts.append(resp(pts[-1]).mean())
print("EM iterates for p:", [round(v, 3) for v in pts[:5]], "... ->", round(pts[-1], 4), " grid argmax", round(grid[L.argmax()], 3))
print("log-lik along iterates:", [round(ell(v), 2) for v in pts[:5]], " never decreased:", bool(np.all(np.diff([ell(v) for v in pts]) > -1e-9)))
for pt in pts[:3]:
    gapmin = min(ell(p) - bound(p, pt) for p in grid)
    print(f"bound at p_t={pt:.3f}: min over grid of l - B = {gapmin:.2e} (>= 0), touches: {abs(ell(pt) - bound(pt, pt)):.1e}")
f, ax = plt.subplots(1, 2, figsize=(10.2, 3.1), gridspec_kw={"width_ratios": [1.5, 1]})
a = ax[0]
a.plot(grid, L, c=INK, lw=2, label=r"log-likelihood $\ell(p)$")
for pt, pn, col in [(pts[0], pts[1], BLUE), (pts[1], pts[2], ORANGE)]:
    B = np.array([bound(p, pt) for p in grid])
    a.plot(grid, B, c=col, lw=1.3, ls="--", label=fr"lower bound at $p^{{(t)}}={pt:.2f}$")
    a.scatter([pt], [ell(pt)], c=col, s=30, zorder=5, edgecolor=INK, lw=.6)
    a.scatter([pn], [bound(pn, pt)], c=col, s=30, marker="^", zorder=5, edgecolor=INK, lw=.6)
    a.annotate("", (pn, ell(pn)), (pn, bound(pn, pt)), arrowprops=dict(arrowstyle="->", color=col, lw=1))
a.set_ylim(L.max() - 82, L.max() + 6); a.set_xlim(0, 1); a.set_xlabel("mixing weight $p$")
a.set_title("(a) each E-step builds a bound that touches $\\ell$ at $p^{(t)}$;\nthe M-step jumps to the bound's top (triangle)")
a.legend(frameon=False, fontsize=7.5, loc="upper right")
a = ax[1]
a.plot(range(len(pts[:12])), [ell(v) for v in pts[:12]], "o-", c=INK, ms=3.5)
a.set_xlabel("EM iteration $t$"); a.set_ylabel(r"$\ell(p^{(t)})$")
a.set_title("(b) so $\\ell$ never goes down")
save("em_bound")

# ------------------------------------------------- the covariance-model family Sigma_j = lambda_j D_j A_j D_j^T
def rot(t): return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
def shape(a): return np.diag([np.sqrt(a), 1/np.sqrt(a)])  # det(A) = 1, so lambda is the volume
centers = [(-2.6, -0.4), (0.4, 1.9), (2.6, -1.0)]
base = dict(lam=[1, 1, 1], A=[3.5]*3, D=[0.5]*3)
vary = dict(lam=[0.45, 1.0, 2.1], A=[1.6, 7, 3.5], D=[0.1, 1.2, -0.6])
models = ["EII", "VII", "EEI", "VEI", "EVI", "VVI", "EEE", "VEE", "EVE", "VVE", "EEV", "VEV", "EVV", "VVV"]
def sig(code, j):
    lam = (vary if code[0] == "V" else base)["lam"][j]
    A = np.eye(2) if code[1] == "I" else shape((vary if code[1] == "V" else base)["A"][j])
    D = np.eye(2) if code[2] == "I" else rot((vary if code[2] == "V" else base)["D"][j])
    return lam*D @ A @ D.T
def ncov(code, d, k):
    v = {"E": 1, "V": k}[code[0]]
    s = {"E": d - 1, "V": k*(d - 1), "I": 0}[code[1]]
    o = {"E": d*(d - 1)//2, "V": k*d*(d - 1)//2, "I": 0}[code[2]]
    return v + s + o
for d in (2, 10):
    print(f"covariance parameters, d={d}, k=3:", {m: ncov(m, d, 3) for m in models})
# cross-check four families against scikit-learn's own parameter counts (cov part = total - (k-1) - kd)
for m, ct in [("VII", "spherical"), ("VVI", "diag"), ("EEE", "tied"), ("VVV", "full")]:
    g = GaussianMixture(3, covariance_type=ct).fit(rng.normal(size=(60, 10)))
    print(m, ct, "sklearn cov params", g._n_parameters() - 2 - 30, "formula", ncov(m, 10, 3))
f, ax = plt.subplots(2, 7, figsize=(10.4, 3.5))
for a, m in zip(ax.ravel(), models):
    for j, (c, col) in enumerate(zip(centers, [BLUE, ORANGE, GREEN])):
        w, V = np.linalg.eigh(sig(m, j)); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        a.add_patch(Ellipse(c, 2*np.sqrt(w[1]), 2*np.sqrt(w[0]), angle=ang, fc=col, alpha=.25, ec=col, lw=1.2))
    a.set_xlim(-4.6, 4.6); a.set_ylim(-3.4, 3.8); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"{m}  ({ncov(m, 2, 3)})", fontsize=9, fontfamily="monospace")
save("gmm_covmodels")

# ------------------------------------------------- choosing k and the covariance family together, by BIC
rng = np.random.default_rng(3)  # same stretched data as clustering_figures.py
rng.normal(size=(450, 2)); R = rot(.7)
X = np.vstack([(rng.normal(size=(200, 2))*[2.2, .22]) @ R.T + m for m in [(0, 0), (0, 1.6)]])
fam = [("VII", "spherical"), ("VVI", "diag"), ("EEE", "tied"), ("VVV", "full")]
fulls = {k: GaussianMixture(k, covariance_type="full", n_init=20, init_params="random_from_data", random_state=0).fit(X) for k in range(1, 6)}
tab = {}
for m, ct in fam:
    rand = [GaussianMixture(k, covariance_type=ct, n_init=50, init_params="random_from_data", random_state=0).fit(X).bic(X) for k in range(1, 6)]
    seed = [GaussianMixture(k, covariance_type=ct, means_init=fulls[k].means_).fit(X).bic(X) for k in range(1, 6)]
    tab[m] = [min(a, b) for a, b in zip(rand, seed)]
    print(f"BIC {m:>3} ({ct:>9}): 50 random starts {[round(v) for v in rand]}; best incl. start from full-model means {[round(v, 1) for v in tab[m]]}")
best = min(((v, m, k + 1) for m in tab for k, v in enumerate(tab[m])))
print("best:", best[1], "k =", best[2], "BIC", round(best[0], 1))
