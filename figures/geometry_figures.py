# Generates the geometric pictures added to the Laplacian eigenmaps, autoencoder and k-means sections,
# and prints every number quoted in their captions.
# Run from the notes folder: uv run --with numpy,scipy,scikit-learn,matplotlib figures/geometry_figures.py
import warnings, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, PURPLE, RED, INK, GRAY = "#3182CE", "#DD6B20", "#38A169", "#805AD5", "#C53030", "#1F2933", "#A0AEC0"
COLS = np.array([BLUE, ORANGE, GREEN, PURPLE])
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])

# ------------------------------------------------- what each Laplacian-eigenmaps constraint pins down
nr, nc = 3, 4
idx = lambda r, c: r*nc + c
A = np.zeros((nr*nc, nr*nc))
for r in range(nr):
    for c in range(nc):
        if c+1 < nc: A[idx(r, c), idx(r, c+1)] = A[idx(r, c+1), idx(r, c)] = 1
        if r+1 < nr: A[idx(r, c), idx(r+1, c)] = A[idx(r+1, c), idx(r, c)] = 1
Dg = A.sum(1); D = np.diag(Dg); L = D - A
w, U = eigh(L, D)                       # generalized problem, U.T @ D @ U = I
Y = U[:, 1:3]
Y = Y*np.sign(Y[0])                     # fix the sign so the picture is reproducible
E = [(i, j) for i in range(len(A)) for j in range(i) if A[i, j]]
def stats(Y):
    return np.trace(Y.T @ L @ Y), (Dg[:, None]*Y).sum(0)/Dg.sum(), np.trace(Y.T @ D @ Y)
th = np.deg2rad(35); Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
cases = [("(a) the eigenmaps answer", Y), ("(b) shifted", Y + [0.25, 0.18]), ("(c) rotated $35^\\circ$", Y @ Q.T),
         ("(d) shrunk by half", 0.5*Y)]
f, ax = plt.subplots(1, 4, figsize=(10.4, 3.0))
lim = 0.62
for a, (t, Z) in zip(ax, cases):
    loss, cen, bud = stats(Z)
    for i, j in E: a.plot(*Z[[i, j]].T, color=GRAY, lw=.9, zorder=1)
    a.scatter(*Z.T, s=18 + 10*Dg, c=BLUE, edgecolor=INK, lw=.5, zorder=2)
    a.axhline(0, color=GRAY, lw=.5, ls=":"); a.axvline(0, color=GRAY, lw=.5, ls=":")
    a.scatter(*cen, marker="X", s=80, c=ORANGE, edgecolor=INK, lw=.8, zorder=3)
    ok = lambda b: "✓" if b else "✗"
    a.set_title(t); a.set_xlim(-lim, lim+.2); a.set_ylim(-lim, lim+.2); a.set_aspect("equal"); noticks(a)
    a.text(.03, .03, f"loss {loss:.3f}\ncentered {ok(np.allclose(cen, 0))}   budget {bud:.2f} {ok(np.isclose(bud, 2))}",
           transform=a.transAxes, fontsize=8, va="bottom")
    print(t, "loss", round(loss, 4), "centroid", cen.round(3), "budget", round(bud, 3))
print("grid eigenvalues", w[:4].round(4))
save("le_blindspots")

# ------------------------------------------------- Ly = lambda Dy as tangency, and the D^(1/2) change of variables
M = np.array([[1.6, 0.7], [0.7, 1.4]]); Dd = np.diag([1.0, 4.0])
lam, V = eigh(M, Dd)                         # columns D-orthonormal
Ms = np.diag(1/np.sqrt(np.diag(Dd))) @ M @ np.diag(1/np.sqrt(np.diag(Dd)))
lam_s, Vs = np.linalg.eigh(Ms)
print("generalized eigenvalues", lam.round(4), "same as L_sym:", lam_s.round(4))
print("Euclidean angle between the two solutions on the left:", round(np.degrees(np.arccos(abs(V[:, 0] @ V[:, 1])/np.linalg.norm(V[:, 0])/np.linalg.norm(V[:, 1]))), 1),
      "deg; D-inner product", round(V[:, 0] @ Dd @ V[:, 1], 6))
f, ax = plt.subplots(1, 2, figsize=(8.6, 4.0))
g = np.linspace(-1.4, 1.4, 400); G1, G2 = np.meshgrid(g, g)
for a, (Mat, Con, vecs, vals, ttl) in zip(ax, [(M, Dd, V, lam, "(a) $y$: constraint $y^{\\top}Dy=1$ is an ellipse"),
                                               (Ms, np.eye(2), Vs, lam_s, "(b) $\\tilde y=D^{1/2}y$: it becomes a circle")]):
    Fq = Mat[0, 0]*G1**2 + 2*Mat[0, 1]*G1*G2 + Mat[1, 1]*G2**2
    a.contour(G1, G2, Fq, levels=np.linspace(.08, 1.6, 9), colors=GRAY, linewidths=.6)
    a.contour(G1, G2, Fq, levels=[vals[0]], colors=BLUE, linewidths=1.4)
    a.contour(G1, G2, Fq, levels=[vals[1]], colors=ORANGE, linewidths=1.4)
    tt = np.linspace(0, 2*np.pi, 300); C = np.c_[np.cos(tt), np.sin(tt)]/np.sqrt(np.diag(Con))
    a.plot(*C.T, color=INK, lw=2)
    for k, col in zip(range(2), [BLUE, ORANGE]):
        v = vecs[:, k]/np.sqrt(vecs[:, k] @ Con @ vecs[:, k])
        a.plot([0, v[0]], [0, v[1]], color=col, lw=1.2, ls="--")
        a.scatter(*v, s=40, c=col, edgecolor=INK, zorder=5)
        gl, gc = Mat @ v, Con @ v
        gl, gc = .38*gl/np.linalg.norm(gl), .38*gc/np.linalg.norm(gc)
        a.annotate("", v + gl, v, arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
        a.annotate("", v + .7*gc, v, arrowprops=dict(arrowstyle="->", color=INK, lw=1, ls="-"))
    a.set_aspect("equal"); a.set_xlim(-1.4, 1.4); a.set_ylim(-1.4, 1.4); noticks(a); a.set_title(ttl)
ax[0].text(-1.33, -1.33, f"min: loss $\\lambda_1={lam[0]:.2f}$ (blue)\nmax: loss $\\lambda_2={lam[1]:.2f}$ (orange)\n"
           f"dashed directions meet at {np.degrees(np.arccos(abs(V[:,0]@V[:,1])/np.linalg.norm(V[:,0])/np.linalg.norm(V[:,1]))):.0f}$^\\circ$", fontsize=8)
ax[1].text(-1.33, 1.2, "same losses; directions now meet at $90^\\circ$\n(eigenvectors of the symmetric $L_{\\mathrm{sym}}$)", fontsize=8,
           bbox=dict(fc="white", ec="none", alpha=.9))
save("le_tangency")

# ------------------------------------------------- plain versus denoising autoencoder, as a vector field
rng = np.random.default_rng(1)
tt = rng.uniform(.15*np.pi, .85*np.pi, 300)
Xc = np.c_[np.cos(tt), np.sin(tt)]
def fit(Xin, Xout, seed):
    return MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=4000, tol=1e-7, random_state=seed).fit(Xin, Xout)
plain = fit(Xc, Xc, 0)
reps = 15; sig = .15
Xn = np.repeat(Xc, reps, 0) + rng.normal(0, sig, (len(Xc)*reps, 2))
den = fit(Xn, np.repeat(Xc, reps, 0), 0)
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 11), np.linspace(-.05, 1.4, 7)); P = np.c_[gx.ravel(), gy.ravel()]
def offcurve(Z):  # distance to the arc (the unit circle, for points within its angular range)
    return np.abs(np.linalg.norm(Z, axis=1) - 1)
test = Xc[::3] + rng.normal(0, sig, (100, 2))
print(f"noisy test points: mean distance to curve {offcurve(test).mean():.3f}; after plain AE {offcurve(plain.predict(test)).mean():.3f};"
      f" after denoising AE {offcurve(den.predict(test)).mean():.3f}")
print(f"plain AE on grid: median |f(x)-x| {np.median(np.linalg.norm(plain.predict(P)-P,axis=1)):.3f}; "
      f"denoising: {np.median(np.linalg.norm(den.predict(P)-P,axis=1)):.3f}")
f, ax = plt.subplots(1, 2, figsize=(9.0, 3.3))
arc = np.linspace(.15*np.pi, .85*np.pi, 200)
for a, (net, t) in zip(ax, [(plain, "(a) plain autoencoder, no bottleneck: $f(x)\\approx x$"),
                            (den, "(b) denoising autoencoder, same network")]):
    Fv = net.predict(P) - P
    a.plot(np.cos(arc), np.sin(arc), color=INK, lw=2.2, zorder=1)
    a.quiver(*P.T, *Fv.T, angles="xy", scale_units="xy", scale=1, color=BLUE, width=.005, alpha=.85, zorder=2)
    a.set_aspect("equal"); noticks(a); a.set_title(t); a.set_xlim(-1.35, 1.35); a.set_ylim(-.2, 1.5)
save("ae_denoise")

# ------------------------------------------------- convolution is shift-equivariant, a dense layer is not
rng = np.random.default_rng(4)
n = 60; s = 14
x = np.zeros(n); x[10:16] = 1; x[30] = .6; x[31] = 1; x[32] = .6
xs = np.roll(x, s)
filt = np.array([1, 0, -1.])
conv = lambda z: np.convolve(z, filt, mode="same")
Wd = rng.normal(0, .3, (n, n)); dense = lambda z: Wd @ z
print("conv shift error", np.abs(conv(xs) - np.roll(conv(x), s)).max(), " dense shift error", round(np.abs(dense(xs) - np.roll(dense(x), s)).max(), 3))
f, ax = plt.subplots(3, 1, figsize=(7.4, 4.2), sharex=True)
for a, (fn, t) in zip(ax, [(lambda z: z, "input signal, and the same signal shifted right by 14"),
                           (conv, "after a convolution (filter $[1,0,-1]$): the output shifts by exactly 14"),
                           (dense, "after a dense layer with random weights: no simple relation")]):
    a.plot(fn(x), color=BLUE, lw=1.4, label="original"); a.plot(fn(xs), color=ORANGE, lw=1.4, ls="--", label="shifted")
    a.set_title(t, loc="left"); a.set_yticks([]); [sp.set_visible(False) for k, sp in a.spines.items() if k in ("top", "right")]
ax[0].legend(frameon=False, loc="upper right", ncol=2); ax[2].set_xlabel("position")
save("cnn_equiv")

# ------------------------------------------------- Lloyd's algorithm, step by step, with the nearest-center regions
rng = np.random.default_rng(11)
X = np.vstack([rng.normal(m, .5, (60, 2)) for m in [(0, 0), (3.2, .3), (1.4, 2.8)]])
C = np.array([[-0.9, -0.6], [-0.2, 0.7], [0.6, -0.4]])
gx, gy = np.meshgrid(np.linspace(-1.8, 4.9, 300), np.linspace(-1.8, 4.9, 300)); G = np.c_[gx.ravel(), gy.ravel()]
light = np.array(["#BEE3F8", "#FEEBC8", "#C6F6D5"])
snaps = []
for it in range(20):
    lab = ((X[:, None] - C[None])**2).sum(2).argmin(1)
    newC = np.array([X[lab == l].mean(0) for l in range(3)])
    snaps.append((C.copy(), lab, newC, ((X - C[lab])**2).sum()))
    if np.allclose(newC, C): break
    C = newC
print("Lloyd iterations:", len(snaps), "losses", [round(s[3], 1) for s in snaps])
show = [0, 1, 2, len(snaps)-1]
f, ax = plt.subplots(1, 4, figsize=(10.4, 2.9))
for a, k in zip(ax, show):
    C0, lab, C1, Lk = snaps[k]
    reg = ((G[:, None] - C0[None])**2).sum(2).argmin(1)
    a.imshow(reg.reshape(gx.shape), extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin="lower",
             cmap=matplotlib.colors.ListedColormap(light), interpolation="nearest", alpha=.9)
    a.contour(gx, gy, reg.reshape(gx.shape), levels=[.5, 1.5], colors=INK, linewidths=.8)
    a.scatter(*X.T, c=COLS[lab], s=7, lw=0)
    a.scatter(*C0.T, marker="X", s=90, c="white", edgecolor=INK, lw=1.2, zorder=4)
    if k != show[-1]:
        for p, q in zip(C0, C1): a.annotate("", q, p, arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    a.set_title(f"iteration {k+1}: $L={Lk:.0f}$" if k != show[-1] else f"stopped (iteration {k+1}): $L={Lk:.0f}$")
    a.set_xlim(gx.min(), gx.max()); a.set_ylim(gy.min(), gy.max()); a.set_aspect("equal"); noticks(a)
save("kmeans_lloyd")

# ------------------------------------------------- the kernel lift that makes rings separable by a flat cut
rng = np.random.default_rng(3)
th = rng.uniform(0, 2*np.pi, 500); r = np.where(np.arange(500) < 200, 1, 3) + rng.normal(0, .12, 500)
R = np.c_[r*np.cos(th), r*np.sin(th)]; truth = (np.arange(500) >= 200).astype(int)
Phi = np.c_[R, (R**2).sum(1)]
best = None
for seed in range(10):
    rr = np.random.default_rng(seed); Cc = Phi[rr.choice(500, 2, replace=False)]
    for _ in range(100):
        lab = ((Phi[:, None] - Cc[None])**2).sum(2).argmin(1)
        Cn = np.array([Phi[lab == l].mean(0) for l in range(2)])
        if np.allclose(Cn, Cc): break
        Cc = Cn
    Lk = ((Phi - Cc[lab])**2).sum()
    if best is None or Lk < best[0]: best = (Lk, lab, Cc)
agree = max((best[1] == truth).mean(), (best[1] != truth).mean())
print("k-means on phi = (x1, x2, x1^2+x2^2): agreement with rings", round(agree, 3), "centers z", best[2][:, 2].round(2))
mid = best[2][:, 2].mean()
f = plt.figure(figsize=(9.0, 3.6))
a = f.add_subplot(1, 2, 1)
a.scatter(*R.T, c=COLS[best[1]], s=6, lw=0)
tt = np.linspace(0, 2*np.pi, 200)
a.set_aspect("equal"); noticks(a); a.set_title("(a) the rings in $\\mathbb{R}^2$")
b = f.add_subplot(1, 2, 2, projection="3d")
b.scatter(*Phi.T, c=COLS[best[1]], s=4, lw=0, depthshade=False)
pp = np.linspace(-3.5, 3.5, 2); PX, PY = np.meshgrid(pp, pp)
b.plot_surface(PX, PY, np.full_like(PX, mid), color=GRAY, alpha=.25)
b.set_title("(b) lifted by $\\varphi(x)=(x_1,x_2,x_1^2+x_2^2)$"); b.set_xticks([]); b.set_yticks([]); b.set_zticks([])
b.set_zlabel("$x_1^2+x_2^2$", labelpad=-10); b.view_init(elev=14, azim=-60)
save("kmeans_lift")
