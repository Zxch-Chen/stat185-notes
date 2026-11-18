# Generates the figures for choosing k (gap statistic, silhouettes) and Gaussian mixtures,
# and prints every number quoted in their captions and examples.
# Run from the notes folder: uv run --with numpy,scipy,scikit-learn,matplotlib figures/clustering_figures.py
import warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, PURPLE, RED, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#C53030", "#1F2933", "#A0AEC0"
COLS = np.array([BLUE, ORANGE, GREEN, PURPLE, "#D53F8C", "#718096", "#D69E2E", "#319795"])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])
def kmL(X, k, seed=0): return KMeans(k, n_init=5, random_state=seed).fit(X).inertia_

# ------------------------------------------------- the gap statistic, on clustered and on structureless data
rng = np.random.default_rng(2)
blobs = np.vstack([rng.normal(m, .55, (100, 2)) for m in [(0, 0), (4, .5), (1.8, 3.4)]])
flat = rng.uniform(0, 4, (300, 2))
B, ks = 50, np.arange(1, 9)
def gap(X):
    lo, hi = X.min(0), X.max(0)
    logL = np.log([kmL(X, k) for k in ks])
    ref = np.array([[np.log(kmL(rng.uniform(lo, hi, X.shape), k, b)) for k in ks] for b in range(B)])
    G = ref.mean(0) - logL
    s = ref.std(0)*np.sqrt(1 + 1/B)
    pick = next(k for i, k in enumerate(ks[:-1]) if G[i] >= G[i+1] - s[i+1])
    return logL, ref.mean(0), G, s, pick
res = {}
for name, X in [("blobs", blobs), ("flat", flat)]:
    res[name] = gap(X)
    logL, mref, G, s, pick = res[name]
    print(f"gap {name}: Gap(k) {G.round(3).tolist()} s {s.round(3).tolist()} -> chosen k = {pick}; argmax k = {ks[G.argmax()]}")
f, ax = plt.subplots(1, 4, figsize=(10.4, 2.9), gridspec_kw={"width_ratios": [1, 1.15, 1.15, 1.15]})
lo, hi = blobs.min(0), blobs.max(0); ref1 = rng.uniform(lo, hi, blobs.shape)
ax[0].scatter(*ref1.T, s=5, c=GRAY, lw=0, label="one reference set")
ax[0].scatter(*blobs.T, s=5, c=BLUE, lw=0, label="data")
ax[0].add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec=INK, lw=1, ls="--"))
ax[0].set_aspect("equal"); noticks(ax[0]); ax[0].set_title("(a) data and a uniform reference")
logL, mref, G, s, pick = res["blobs"]
ax[1].plot(ks, logL, "o-", color=BLUE, ms=3.5, label="$\\log L_{\\mathrm{data}}(k)$")
ax[1].plot(ks, mref, "o-", color=GRAY, ms=3.5, label="mean $\\log L_{\\mathrm{ref}}(k)$")
ax[1].annotate("", (3, logL[2]), (3, mref[2]), arrowprops=dict(arrowstyle="<->", color=ORANGE))
ax[1].text(3.2, (logL[2]+mref[2])/2, "Gap(3)", color=ORANGE, fontsize=8)
ax[1].set_xlabel("$k$"); ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_title("(b) clustered data: the two curves")
for a, name, t in [(ax[2], "blobs", "(c) clustered data: Gap$(k)$"), (ax[3], "flat", "(d) uniform data: Gap$(k)$")]:
    logL, mref, G, s, pick = res[name]
    a.errorbar(ks, G, yerr=s, fmt="o-", color=BLUE, ms=3.5, capsize=2, lw=1.2)
    a.scatter([pick], [G[pick-1]], s=110, facecolor="none", edgecolor=ORANGE, lw=1.8, zorder=5)
    a.text(pick + .25, G[pick-1] + .16, f"chosen $k={pick}$", color=ORANGE, fontsize=8, va="center")
    a.set_xlabel("$k$"); a.set_title(t)
ax[2].set_ylim(-.25, 1.6); ax[3].set_ylim(-.25, 1.6); ax[3].axhline(0, color=GRAY, lw=.6)
save("gap_stat")

# ------------------------------------------------- silhouette scores
rng = np.random.default_rng(5)
S3 = np.vstack([rng.normal(m, s_, (80, 2)) for m, s_ in [((0, 0), .5), ((3.2, 0), .5), ((1.6, 2.8), .8)]])
mean_s = {}
for k in range(2, 9):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(S3); mean_s[k] = silhouette_samples(S3, lab).mean()
print("mean silhouette by k:", {k: round(v, 3) for k, v in mean_s.items()})
f, ax = plt.subplots(1, 4, figsize=(10.4, 3.1), gridspec_kw={"width_ratios": [1.05, 1, 1, 1]})
lab3 = KMeans(3, n_init=10, random_state=0).fit_predict(S3)
i0 = np.argmin(((S3 - [1.55, 1.05])**2).sum(1))       # a point between clusters, to draw a_i and b_i
own = lab3[i0]; others = [l for l in range(3) if l != own]
avg = {l: np.linalg.norm(S3[lab3 == l] - S3[i0], axis=1).mean() for l in range(3)}
avg[own] = np.linalg.norm(S3[lab3 == own] - S3[i0], axis=1).sum()/((lab3 == own).sum() - 1)
rival = min(others, key=lambda l: avg[l])
si = (avg[rival] - avg[own])/max(avg[own], avg[rival])
print(f"example point own color {COLS[own]} rival {COLS[rival]}: a_i = {avg[own]:.2f}, b_i = {avg[rival]:.2f} (other cluster: {avg[[l for l in others if l != rival][0]]:.2f}), s_i = {si:.2f}")
a = ax[0]
for l, c in [(own, COLS[own]), (rival, COLS[rival])]:
    for p in S3[lab3 == l][::4]: a.plot(*np.c_[S3[i0], p], color=c, lw=.4, alpha=.6, zorder=1)
a.scatter(*S3.T, c=COLS[lab3], s=6, lw=0, zorder=2)
a.scatter(*S3[i0], s=70, c="white", edgecolor=INK, lw=1.4, zorder=3)
a.set_title(f"(a) $a_i={avg[own]:.2f}$, $b_i={avg[rival]:.2f}$, $s_i={si:.2f}$")
a.set_aspect("equal"); noticks(a)
def silplot(a, k, t):
    lab = KMeans(k, n_init=10, random_state=0).fit_predict(S3); sv = silhouette_samples(S3, lab)
    y = 0
    for l in np.argsort([-sv[lab == l].mean() for l in range(k)]):
        v = np.sort(sv[lab == l]); a.barh(np.arange(y, y + len(v)), v, height=1, color=COLS[l], lw=0); y += len(v) + 8
    a.axvline(sv.mean(), color=RED, ls="--", lw=1); a.axvline(0, color=INK, lw=.6)
    a.set_yticks([]); a.set_xlim(-.25, 1); a.set_xlabel("$s_i$"); a.set_title(t + f": mean {sv.mean():.2f}")
    print(f"silhouette k={k}: negatives {(sv < 0).sum()}, mean {sv.mean():.3f}")
silplot(ax[1], 3, "(b) $k=3$"); silplot(ax[2], 5, "(c) $k=5$")
ax[3].plot(list(mean_s), list(mean_s.values()), "o-", color=BLUE, ms=4)
best = max(mean_s, key=mean_s.get)
ax[3].scatter([best], [mean_s[best]], s=110, facecolor="none", edgecolor=ORANGE, lw=1.8)
ax[3].set_xlabel("$k$"); ax[3].set_ylabel("mean silhouette"); ax[3].set_title(f"(d) highest at $k={best}$")
save("silhouette")

# ------------------------------------------------- a mixture density, and the two-step sampling process
p, mu, sd = np.array([.5, .35, .15]), np.array([-2, 1.5, 4.]), np.array([.8, .6, .4])
xs = np.linspace(-5, 6, 600)
rng = np.random.default_rng(0)
z = rng.choice(3, 3000, p=p); xdraw = rng.normal(mu[z], sd[z])
print("mixture: fraction of draws with each label", np.bincount(z)/len(z), " integral of f_mix", round(np.trapezoid(sum(p[j]*norm.pdf(xs, mu[j], sd[j]) for j in range(3)), xs), 4))
f, ax = plt.subplots(1, 2, figsize=(9.4, 2.8))
fmix = sum(p[j]*norm.pdf(xs, mu[j], sd[j]) for j in range(3))
for j in range(3): ax[0].plot(xs, p[j]*norm.pdf(xs, mu[j], sd[j]), color=COLS[j], ls="--", lw=1.2, label=f"$p_{j+1}f_{j+1}$, $p_{j+1}={p[j]}$")
ax[0].plot(xs, fmix, color=INK, lw=2, label="$f_{\\mathrm{mix}}=\\sum_j p_jf_j$")
ax[0].legend(frameon=False, fontsize=7.5); ax[0].set_yticks([]); ax[0].set_title("(a) the mixture is a weighted sum of bumps")
bins = np.linspace(-5, 6, 70)
ax[1].hist([xdraw[z == j] for j in range(3)], bins=bins, stacked=True, density=True, color=COLS[:3], alpha=.85)
ax[1].plot(xs, fmix, color=INK, lw=1.6); ax[1].set_yticks([])
ax[1].set_title("(b) 3000 draws: pick $z_i$, then $x_i\\sim f_{z_i}$ (colored by $z_i$)")
save("gmm_mixture")

# ------------------------------------------------- EM on the stretched clusters that k-means cut across
rng = np.random.default_rng(3)
rng.normal(size=(450, 2)); R = np.array([[np.cos(.7), -np.sin(.7)], [np.sin(.7), np.cos(.7)]])
X = np.vstack([(rng.normal(size=(200, 2))*[2.2, .22]) @ R.T + m for m in [(0, 0), (0, 1.6)]])
truth = np.repeat([0, 1], 200)
km = KMeans(2, n_init=10, random_state=0).fit(X)
def agree(lab): return max((lab == truth).mean(), (lab != truth).mean())
P, M, S = np.array([.5, .5]), km.cluster_centers_.copy(), np.array([np.eye(2)*X.var(0).mean()]*2)
snaps, ll = [], []
for it in range(200):
    dens = np.c_[[P[j]*multivariate_normal(M[j], S[j]).pdf(X) for j in range(2)]].T
    ll.append(np.log(dens.sum(1)).sum()); Rr = dens/dens.sum(1, keepdims=True)
    snaps.append((M.copy(), S.copy(), Rr.copy(), ll[-1]))
    Nj = Rr.sum(0); P = Nj/len(X); M = (Rr.T @ X)/Nj[:, None]
    S = np.array([((Rr[:, j, None]*(X - M[j])).T @ (X - M[j]))/Nj[j] + 1e-6*np.eye(2) for j in range(2)])
    if it > 2 and abs(ll[-1] - ll[-2]) < 1e-6: break
print(f"k-means agreement {agree(km.labels_):.3f}; EM iterations {len(ll)}; GMM agreement {agree(snaps[-1][2].argmax(1)):.3f}")
print("EM log-likelihood (first, then last):", [round(v, 1) for v in ll[:4]], round(ll[-1], 1), " never decreased:", bool(np.all(np.diff(ll) > -1e-6)))
def ell(a, m, C, col):
    w, V = np.linalg.eigh(C); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    for r in (1, 2): a.add_patch(Ellipse(m, 2*r*np.sqrt(w[1]), 2*r*np.sqrt(w[0]), angle=ang, fill=False, ec=col, lw=1.4 if r == 1 else .8))
show = [0, len(snaps)//3, 2*len(snaps)//3, len(snaps) - 1]
f, ax = plt.subplots(1, 4, figsize=(10.4, 2.9))
for a, k in zip(ax, show):
    M_, S_, Rr, L_ = snaps[k]
    c = Rr[:, [0]]*matplotlib.colors.to_rgba_array(BLUE)[0] + Rr[:, [1]]*matplotlib.colors.to_rgba_array(ORANGE)[0]
    a.scatter(*X.T, c=np.clip(c, 0, 1), s=5, lw=0)
    for j, col in enumerate([BLUE, ORANGE]): ell(a, M_[j], S_[j], col); a.scatter(*M_[j], marker="X", s=70, c="white", edgecolor=INK, lw=1)
    a.set_title(("start (from $k$-means)" if k == 0 else f"EM iteration {k}" if k != show[-1] else f"converged (iteration {k})") + f"\nlog-likelihood {L_:.0f}", fontsize=8.5)
    a.set_aspect("equal"); a.set_xlim(-5.2, 5.2); a.set_ylim(-3.6, 4.6); noticks(a)
save("gmm_em")
bic = {k: GaussianMixture(k, n_init=5, random_state=0).fit(X).bic(X) for k in range(1, 7)}
print("BIC, default k-means initialization:", {k: round(v, 1) for k, v in bic.items()})
bic = {k: GaussianMixture(k, n_init=20, init_params="random_from_data", random_state=0).fit(X).bic(X) for k in range(1, 7)}
print("BIC on stretched data:", {k: round(v, 1) for k, v in bic.items()}, "-> min at", min(bic, key=bic.get))
bicb = {k: GaussianMixture(k, n_init=5, random_state=0).fit(blobs).bic(blobs) for k in range(1, 7)}
print("BIC on three blobs:", {k: round(v, 1) for k, v in bicb.items()}, "-> min at", min(bicb, key=bicb.get))

# ------------------------------------------------- Gaussian mass sits on a shell in high dimensions
rng = np.random.default_rng(0)
f, ax = plt.subplots(1, 1, figsize=(6.0, 2.6))
for d, col in zip([1, 10, 100, 1000], [GRAY, GREEN, BLUE, PURPLE]):
    r = np.linalg.norm(rng.normal(size=(20000, d)), axis=1)/np.sqrt(d)
    ax.hist(r, bins=np.linspace(0, 2.2, 120), density=True, histtype="step", color=col, lw=1.5, label=f"$d={d}$")
    print(f"d={d}: radius/sqrt(d) mean {r.mean():.3f} sd {r.std():.3f}; fraction within half the typical radius {np.mean(r < .5):.4f}")
ax.set_xlabel("distance from the mean, divided by $\\sqrt{d}$"); ax.set_yticks([]); ax.legend(frameon=False)
ax.set_title("draws from $\\mathcal{N}(\\mathbf{0},I_d)$: almost none land near the mean")
save("gauss_shell")
