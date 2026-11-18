# Generates the data-driven figures for the graph Laplacian, Isomap, MDS, kernel PCA,
# UMAP and autoencoder sections, and prints every number quoted in their captions.
import warnings, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sla
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path, connected_components
from sklearn.neighbors import kneighbors_graph
warnings.filterwarnings("ignore")
BLUE, ORANGE, GREEN, INK = "#3182CE", "#DD6B20", "#38A169", "#1F2933"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "mathtext.fontset": "dejavuserif"})
def save(n): plt.tight_layout(); plt.savefig(f"figures/{n}.png", dpi=220); plt.close(); print("saved", n)
def noticks(a): a.set_xticks([]); a.set_yticks([])
def lap(A): return np.diag(A.sum(1)) - A
def lsym(A):
    d = A.sum(1); s = 1/np.sqrt(d); return np.eye(len(A)) - s[:, None]*A*s[None, :]
def drawg(ax, P, A, c=None, cmap="RdBu_r", vm=None, s=70):
    for i in range(len(P)):
        for j in range(i):
            if A[i, j] > 0: ax.plot(*P[[i, j]].T, color="#A0AEC0", lw=.9, zorder=1)
    ax.scatter(*P.T, c=BLUE if c is None else c, cmap=None if c is None else cmap, vmin=None if vm is None else -vm, vmax=vm,
               s=s, edgecolor=INK, linewidth=.5, zorder=2, clip_on=False)
    ax.margins(.1); noticks(ax); ax.set_aspect("equal"); [x.set_visible(False) for x in ax.spines.values()]

# ------------------------------------------------- roughness of values on one graph
edges = [(0,1),(0,2),(1,2),(2,3),(1,3),(3,4),(4,5),(4,6),(5,6),(5,7),(6,7)]
P = np.array([[0,1],[0,0],[1,1.2],[1.2,0],[2.2,.5],[3.2,1.1],[3.2,-.1],[4.2,.5]], float)
A = np.zeros((8, 8))
for i, j in edges: A[i, j] = A[j, i] = 1
w, V = np.linalg.eigh(lap(A))
r = np.random.default_rng(3).standard_normal(8); r -= r.mean(); r /= np.linalg.norm(r)
print("graph8 eigs", w.round(3), "random roughness", round(r@lap(A)@r, 3))
fig, ax = plt.subplots(1, 4, figsize=(10, 2.1))
for a, y, t in zip(ax, [np.ones(8)/np.sqrt(8), V[:, 1]*np.sign(V[0, 1]), r, V[:, -1]], ["constant", "smoothest non-constant ($\\mathbf{v}_2$)", "random", "roughest ($\\mathbf{v}_8$)"]):
    drawg(a, P, A, y, vm=.75); a.set_title(f"{t}\n$\\mathbf{{y}}^\\top L\\mathbf{{y}}={y@lap(A)@y:.2f}$")
save("lap_roughness")

# ------------------------------------------------- eigenvectors as vibration modes
def path(n):
    A = np.zeros((n, n)); i = np.arange(n-1); A[i, i+1] = A[i+1, i] = 1; return A
def ring(n):
    A = path(n); A[0, -1] = A[-1, 0] = 1; return A
fig = plt.figure(figsize=(10, 6.4)); gs = fig.add_gridspec(3, 4)
w, V = np.linalg.eigh(lap(path(60)))
for k in range(4):
    a = fig.add_subplot(gs[0, k]); a.plot(V[:, k+1]*np.sign(V[0, k+1]), color=BLUE, lw=1.6); a.axhline(0, color="#A0AEC0", lw=.7)
    a.set_title(f"path: $\\mathbf{{v}}_{k+2}$, $\\lambda={w[k+1]:.4f}$"); noticks(a)
nx, ny = 40, 25
Ax, Ay = path(nx), path(ny); G = np.kron(np.eye(ny), Ax) + np.kron(Ay, np.eye(nx))
w, V = np.linalg.eigh(lap(G))
for k in range(4):
    a = fig.add_subplot(gs[1, k]); a.imshow(V[:, k+1].reshape(ny, nx), cmap="RdBu_r", aspect="equal", vmin=-abs(V[:, k+1]).max(), vmax=abs(V[:, k+1]).max())
    a.set_title(f"$40\\times25$ grid: $\\mathbf{{v}}_{k+2}$, $\\lambda={w[k+1]:.4f}$"); noticks(a)
n = 60; w, V = np.linalg.eigh(lap(ring(n)))
print("ring eigs", w[:5].round(4))
a = fig.add_subplot(gs[2, 0]); a.plot(V[:, 1], color=BLUE, lw=1.6, label="$\\mathbf{v}_2$"); a.plot(V[:, 2], color=ORANGE, lw=1.6, label="$\\mathbf{v}_3$")
a.set_title(f"ring: $\\lambda_2=\\lambda_3={w[1]:.4f}$"); noticks(a); a.legend(frameon=False, fontsize=7.5, loc="upper right")
a = fig.add_subplot(gs[2, 1]); a.scatter(V[:, 1], V[:, 2], c=np.arange(n), cmap="twilight", s=16); a.set_aspect("equal"); noticks(a); a.set_title("ring embedded by $(\\mathbf{v}_2,\\mathbf{v}_3)$")
a = fig.add_subplot(gs[2, 2]); a.plot(V[:, 3], color=BLUE, lw=1.6); a.plot(V[:, 4], color=ORANGE, lw=1.6); a.set_title(f"ring: $\\lambda_4=\\lambda_5={w[3]:.4f}$"); noticks(a)
a = fig.add_subplot(gs[2, 3]); a.scatter(V[:, 3], V[:, 4], c=np.arange(n), cmap="twilight", s=16); a.set_aspect("equal"); noticks(a); a.set_title("$(\\mathbf{v}_4,\\mathbf{v}_5)$ wraps twice")
save("lap_modes")

# ------------------------------------------------- bridges: zero eigenvalues and the Fiedler vector
rng = np.random.default_rng(1)
Q = np.vstack([rng.normal(0, .55, (12, 2)), rng.normal(0, .55, (12, 2)) + [4, 0]])
base = np.zeros((24, 24))
for off in (0, 12):
    B = kneighbors_graph(Q[off:off+12], 3).toarray(); base[off:off+12, off:off+12] = np.maximum(B, B.T)
def bridges(m):
    A = base.copy(); D = np.linalg.norm(Q[:12, None]-Q[12:][None], axis=2)
    for f in np.argsort(D, axis=None)[:m]:
        i, j = np.unravel_index(f, D.shape); A[i, 12+j] = A[12+j, i] = 1
    return A
fig, ax = plt.subplots(2, 3, figsize=(10, 4.6), gridspec_kw={"height_ratios": [1.25, 1]})
for c, (m, t) in enumerate([(0, "no bridge"), (1, "one bridge edge"), (40, "40 bridge edges")]):
    A = bridges(m); w, V = np.linalg.eigh(lsym(A)); y = V[:, 1]/np.sqrt(A.sum(1))
    if m == 0:
        y = np.r_[np.ones(12), -np.ones(12)]/np.sqrt(A.sum()); 
    print(t, "comps", connected_components(A)[0], "eigs", w[:4].round(4))
    drawg(ax[0, c], Q, A, y, vm=abs(y).max(), s=55); ax[0, c].set_title(f"{t}: $\\lambda_2={abs(w[1]):.3f}$")
    ax[1, c].bar(range(1, 9), np.abs(w[:8]), color=[ORANGE]*2 + [BLUE]*6); ax[1, c].set_ylim(0, 1.05); ax[1, c].set_xlabel("$k$"); ax[1, c].set_ylabel("$\\lambda_k$ of $L_{\\mathrm{sym}}$")
    ax[1, c].spines[["top", "right"]].set_visible(False)
save("lap_clusters")

# ------------------------------------------------- what D^{-1/2} does: v -> y on an uneven graph
E = [(0,1),(0,2),(0,3),(0,4),(1,2),(4,5),(5,6),(6,7),(7,8)]
A = np.zeros((9, 9))
for i, j in E: A[i, j] = A[j, i] = 1
PG = np.array([[0,0],[-.8,.7],[-.8,-.7],[-.2,1.05],[1,0],[2,0],[3,0],[4,0],[5,0]], float)
d = A.sum(1); w, V = np.linalg.eigh(lsym(A)); v1 = V[:, 0]*np.sign(V[0, 0]); v2 = V[:, 1]*np.sign(V[8, 1])
y1, y2 = v1/np.sqrt(d), v2/np.sqrt(d)
print("deg", d, "lam", w[:3].round(3), "\nv1", v1.round(3), "\ny1", y1.round(3), "\nv2", v2.round(3), "\ny2", y2.round(3),
      "\ncenter", round(d@y2, 6), "scale", round(d@y2**2, 6), "obj", round(y2@lap(A)@y2, 4))
fig, ax = plt.subplots(1, 4, figsize=(10, 2.4), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
drawg(ax[0], PG, A, s=30+22*d); ax[0].set_title("graph (dot size $\\propto$ degree)")
for i in range(9): ax[0].annotate(str(i+1), PG[i], ha="center", va="center", fontsize=6.5, color="white", zorder=3)
xs = np.arange(1, 10); W = .4
ax[1].bar(xs-W/2, v1, W, color=BLUE, label="$\\mathbf{v}_1\\propto\\sqrt{D_{ii}}$"); ax[1].bar(xs+W/2, y1, W, color=ORANGE, label="$D^{-1/2}\\mathbf{v}_1$: constant"); ax[1].set_title("skipped eigenvector"); ax[1].set_ylim(0, .78); ax[1].legend(frameon=False, fontsize=6.8, loc="upper right")
ax[2].bar(xs-W/2, v2, W, color=BLUE, label="$\\mathbf{v}_2$"); ax[2].bar(xs+W/2, y2, W, color=ORANGE, label="$y_i=v_2(i)/\\sqrt{D_{ii}}$"); ax[2].set_title("kept eigenvector"); ax[2].legend(frameon=False, fontsize=6.8, loc="upper left")
for a in ax[1:3]: a.set_xticks(xs); a.tick_params(labelsize=7); a.set_xlabel("vertex"); a.axhline(0, color="#A0AEC0", lw=.7); a.spines[["top", "right"]].set_visible(False)
ax[3].axhline(0, color="#A0AEC0", lw=.8); ax[3].scatter(y2, np.zeros(9), s=25+20*d, color=BLUE, edgecolor=INK, linewidth=.4, zorder=2); ax[3].scatter([0], [0], marker="+", s=90, color=ORANGE, zorder=3)
for lab, x, h in [("1", y2[0], .05), ("2,3", y2[1], .02), ("4", y2[3], .035)] + [(str(i+1), y2[i], .02) for i in range(4, 9)]:
    ax[3].annotate(lab, (x, h), ha="center", fontsize=6.5)
ax[3].set_ylim(-.04, .1); ax[3].set_yticks([]); ax[3].set_xlabel("$y_i$ (+: origin $=$ $D$-weighted mean)"); ax[3].set_title("1-D embedding"); ax[3].spines[["top", "right", "left"]].set_visible(False); ax[3].tick_params(labelsize=7)
save("lap_change")

# ------------------------------------------------- spiral: short-circuits and heat
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.manifold import Isomap, SpectralEmbedding
import umap
n = 500; tt = np.linspace(1.5*np.pi, 4.5*np.pi, 4000)
arc = np.r_[0, np.cumsum(np.sqrt(1+tt[:-1]**2)*np.diff(tt))]
tq = np.interp(np.linspace(0, arc[-1], n), arc, tt); S = np.c_[tq*np.cos(tq), tq*np.sin(tq)]
true = arc[-1]; straight = np.linalg.norm(S[0]-S[-1])
fig, ax = plt.subplots(1, 3, figsize=(10, 3.3))
ax[0].plot(*S.T, color=BLUE, lw=2); ax[0].plot(*S[[0, -1]].T, "--", color=ORANGE, lw=1.6); ax[0].scatter(*S[[0, -1]].T, color=INK, s=22, zorder=3)
ax[0].set_title(f"true geodesic ${true:.1f}$, straight line ${straight:.1f}$")
for a, k in zip(ax[1:], (6, 100)):
    G = kneighbors_graph(S, k, mode="distance"); G = G.maximum(G.T); d, pr = shortest_path(G, return_predecessors=True, indices=0)
    a.scatter(*S.T, s=2, color="#A0AEC0"); j = n-1; pth = [j]
    while pr[j] >= 0: j = pr[j]; pth.append(j)
    a.plot(*S[pth].T, color=ORANGE, lw=1.8); a.scatter(*S[[0, -1]].T, color=INK, s=22, zorder=3)
    print("spiral k", k, "graph geod", round(d[-1], 2), "true", round(true, 2), "straight", round(straight, 2), "longest hop", round(max(np.linalg.norm(S[pth[i]]-S[pth[i+1]]) for i in range(len(pth)-1)), 2))
    a.set_title(f"$k={k}$: graph distance ${d[-1]:.1f}$")
for a in ax: a.set_aspect("equal"); noticks(a)
save("iso_shortcircuit")

G = kneighbors_graph(S, 6).toarray(); G = np.maximum(G, G.T); w, V = np.linalg.eigh(lap(G)); src = 250
fig, ax = plt.subplots(2, 4, figsize=(10, 4.9))
for a, t in zip(ax[0], (0.5, 20, 200, 1500)):
    u = V@(np.exp(-w*t)*V[src]); a.scatter(*S.T, c=u/u.max(), cmap="inferno", s=6, vmin=0, vmax=1); a.set_title(f"along the graph, $s={t}$")
for a, t in zip(ax[1], (0.25, 2, 8, 25)):
    xx, yy = np.meshgrid(np.linspace(-16, 16, 240), np.linspace(-16, 16, 240))
    a.imshow(np.exp(-((xx-S[src, 0])**2+(yy-S[src, 1])**2)/(4*t)), extent=(-16, 16, -16, 16), origin="lower", cmap="inferno", alpha=.75, vmin=0, vmax=1)
    a.plot(*S.T, color="#A0AEC0", lw=.8); a.set_title(f"through the plane, $s={t}$")
for a in ax.ravel(): a.set_aspect("equal"); noticks(a); a.set_xlim(-16, 16); a.set_ylim(-16, 16); a.set_facecolor("#111")
save("lap_heat")

# ------------------------------------------------- Swiss roll eigenvectors: why the arch
X, t = make_swiss_roll(2500, noise=.05, random_state=0); h = X[:, 1]
G = kneighbors_graph(X, 12, mode="distance"); G = G.maximum(G.T).toarray(); sg = np.median(G[G > 0])
A = np.where(G > 0, np.exp(-G**2/sg**2), 0.); d = A.sum(1); print("roll comps", connected_components(A)[0])
w, V = np.linalg.eigh(lsym(A)); Y = V/np.sqrt(d)[:, None]
cor = [abs(np.corrcoef(Y[:, k], h)[0, 1]) for k in range(1, 12)]; print("corr with height, y2..y12", np.round(cor, 2))
kh = int(np.argmax(cor))+1; print("height eigenvector index", kh+1, "lambda", w[1:6].round(5))
fig = plt.figure(figsize=(10, 4.7)); gs = fig.add_gridspec(2, 3)
for c, k in enumerate((1, 2, kh)):
    a = fig.add_subplot(gs[0, c], projection="3d"); a.scatter(*X.T, c=Y[:, k], cmap="RdBu_r", s=3); a.view_init(12, -70); a.set_axis_off(); a.set_title(f"roll colored by $\\mathbf{{y}}_{{{k+1}}}$")
    a = fig.add_subplot(gs[1, c]); xv = h if k == kh else t
    a.scatter(xv, Y[:, k], c=t, cmap="viridis", s=2); a.set_xlabel("height along the roll's width" if k == kh else "position along the roll"); a.set_ylabel(f"$\\mathbf{{y}}_{{{k+1}}}$"); a.set_yticks([]); a.set_xticks([]); a.spines[["top", "right"]].set_visible(False)
save("lap_roll")

# ------------------------------------------------- choosing k
X, t = make_swiss_roll(1500, noise=.05, random_state=0)
ks = (5, 12, 150); fig, ax = plt.subplots(3, 3, figsize=(8.4, 7))
for c, k in enumerate(ks):
    G = kneighbors_graph(X, k); print("k", k, "components", connected_components(G.maximum(G.T))[0])
    ems = [Isomap(n_neighbors=k).fit_transform(X), SpectralEmbedding(2, affinity="nearest_neighbors", n_neighbors=k, random_state=0).fit_transform(X), umap.UMAP(n_neighbors=k, random_state=1).fit_transform(X)]
    for r, Yk in enumerate(ems): ax[r, c].scatter(*Yk.T, c=t, cmap="viridis", s=2); noticks(ax[r, c])
    ax[0, c].set_title(f"$k={k}$")
for r, n_ in enumerate(("Isomap", "Laplacian eigenmaps", "UMAP")): ax[r, 0].set_ylabel(n_)
save("k_effect")

# ------------------------------------------------- classical MDS on cities
C = {"NYC": (40.71,-74.01), "Boston": (42.36,-71.06), "Chicago": (41.88,-87.63), "Miami": (25.76,-80.19), "Dallas": (32.78,-96.80), "Denver": (39.74,-104.99), "Seattle": (47.61,-122.33), "LA": (34.05,-118.24), "SF": (37.77,-122.42), "Atlanta": (33.75,-84.39)}
nm = list(C); ll = np.radians(np.array([C[c] for c in nm])); N = len(nm)
def hav(a, b):
    return 2*3959*np.arcsin(np.sqrt(np.sin((a[0]-b[0])/2)**2+np.cos(a[0])*np.cos(b[0])*np.sin((a[1]-b[1])/2)**2))
D = np.array([[hav(a, b) for b in ll] for a in ll]); H = np.eye(N)-np.ones((N, N))/N; B = -.5*H@(D**2)@H
ev, U = np.linalg.eigh(B); o = np.argsort(-ev); ev, U = ev[o], U[:, o]; Yc = U[:, :2]*np.sqrt(ev[:2])
print("cities eig", ev.round(0)); print("frac", (ev[:2].sum()/np.abs(ev).sum()).round(4))
Dh = np.linalg.norm(Yc[:, None]-Yc[None], axis=2); print("median rel err", np.median(np.abs(Dh-D)[np.triu_indices(N, 1)]/D[np.triu_indices(N, 1)]).round(4), "NYC-LA", D[0, 7].round(0), Dh[0, 7].round(0))
geo = np.c_[np.degrees(ll[:, 1])*np.cos(np.radians(38)), np.degrees(ll[:, 0])]; geo = (geo-geo.mean(0))
u_, _, vt = np.linalg.svd(geo.T@Yc); Rm = vt.T@u_.T; print("det R", np.linalg.det(Rm).round(2)); Ya = Yc@Rm
fig, ax = plt.subplots(1, 4, figsize=(10, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.15, 1.15]})
im = ax[0].imshow(D, cmap="Blues"); ax[0].set_title("input: distances $D$ (miles)"); im2 = ax[1].imshow(B/1e6, cmap="RdBu_r", vmin=-abs(B).max()/1e6, vmax=abs(B).max()/1e6); ax[1].set_title("$B=-\\frac{1}{2}H(D\\odot D)H$")
for a in ax[:2]: a.set_xticks(range(N)); a.set_yticks(range(N)); a.set_xticklabels(nm, rotation=90, fontsize=5.5); a.set_yticklabels(nm, fontsize=5.5); a.tick_params(length=0)
for a, Yp, t_ in ((ax[2], Yc, "output: top 2 eigenvectors"), (ax[3], Ya, "same map, rotated/reflected")):
    a.scatter(*Yp.T, color=BLUE, s=14)
    for i, c in enumerate(nm): a.annotate(c, Yp[i], fontsize=6, xytext=(2, -8) if (c == "NYC" and Yp is Ya) else (2, 3), textcoords="offset points")
    a.set_aspect("equal"); noticks(a); a.set_title(t_); a.margins(.17)
save("mds_cities")
nc = 40; th = 2*np.pi*np.arange(nc)/nc; X2 = np.c_[np.cos(th), np.sin(th)]
Dc = np.abs(th[:, None]-th[None]); Dc = np.minimum(Dc, 2*np.pi-Dc); Hc = np.eye(nc)-1/nc
eg = np.sort(np.linalg.eigvalsh(-.5*Hc@(Dc**2)@Hc))[::-1]; eu = np.sort(np.linalg.eigvalsh(-.5*Hc@(np.linalg.norm(X2[:, None]-X2[None], axis=2)**2)@Hc))[::-1]
print("circle straight eig", eu[:4].round(3), eu[-2:].round(3), "geod eig", eg[:7].round(3), eg[-4:].round(3))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.15), sharey=True)
for a, e, t_ in ((ax[0], eu, "straight-line distances on a circle (Euclidean)"), (ax[1], eg, "around-the-circle distances (not Euclidean)")):
    a.bar(range(1, nc+1), e, color=[BLUE if x >= 0 else ORANGE for x in e]); a.axhline(0, color=INK, lw=.6); a.set_title(t_, fontsize=8.5); a.set_xlabel("eigenvalue number"); a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("eigenvalue of $B$"); save("mds_negative")

# ------------------------------------------------- kernel PCA: sigma as a zoom knob
rng = np.random.default_rng(0); m = 100; ang = rng.uniform(0, 2*np.pi, 2*m); rad = np.r_[np.ones(m), 3*np.ones(m)]+rng.normal(0, .08, 2*m)
R = np.c_[rad*np.cos(ang), rad*np.sin(ang)]; lab = np.r_[np.zeros(m), np.ones(m)]; o = np.argsort(rad.round(0)*100+ang); 
Dr = np.linalg.norm(R[:, None]-R[None], axis=2)**2; Hr = np.eye(2*m)-1/(2*m)
fig, ax = plt.subplots(2, 4, figsize=(10, 4.9)); cols = np.array([BLUE, ORANGE])[lab.astype(int)]
for c, sg in enumerate((.05, 1, 3, 50)):
    K = np.exp(-Dr/sg**2); ax[0, c].imshow(K[np.ix_(o, o)], cmap="viridis", vmin=0, vmax=1); ax[0, c].set_title(f"$K$, $\\sigma={sg}$"); noticks(ax[0, c])
    e_, u_ = np.linalg.eigh(Hr@K@Hr); e_, u_ = e_[::-1], u_[:, ::-1]; Z = u_[:, :2]*np.sqrt(np.maximum(e_[:2], 0))
    print("kpca sigma", sg, "offdiag mean", round((K.sum()-2*m)/(4*m*m-2*m), 4), "top5 frac", (e_[:5]/e_.sum()).round(3)); ax[1, c].scatter(*Z.T, c=cols, s=5); noticks(ax[1, c]); ax[1, c].set_title("first two kernel PC scores")
save("kpca_sigma")

# ------------------------------------------------- UMAP on real data: 8x8 digit images
dg = load_digits(); Xd, yd = dg.data/16, dg.target
from sklearn.decomposition import PCA
epochs = [0, 2, 10, 50, 200]; m_ = umap.UMAP(n_neighbors=15, random_state=1, n_epochs=epochs); Yf = m_.fit_transform(Xd); L_ = m_.embedding_list_
print("digits N, d", Xd.shape, "snapshots", len(L_))
fig, ax = plt.subplots(1, 5, figsize=(10, 2.25))
for a, Y_, ep in zip(ax, L_, epochs): a.scatter(*np.asarray(Y_).T, c=yd, cmap="tab10", s=1.3); a.set_title(f"epoch {ep}" if ep else "start: eigenmaps"); noticks(a)
save("umap_snapshots")
fig, ax = plt.subplots(3, 3, figsize=(8, 7))
for r, k in enumerate((5, 15, 100)):
    for c, md in enumerate((0., .1, .8)):
        Y_ = umap.UMAP(n_neighbors=k, min_dist=md, random_state=1).fit_transform(Xd); ax[r, c].scatter(*Y_.T, c=yd, cmap="tab10", s=1.3); noticks(ax[r, c])
        if r == 0: ax[r, c].set_title(f"min_dist $={md}$")
    ax[r, 0].set_ylabel(f"$k={k}$")
save("umap_grid")

# ------------------------------------------------- autoencoders actually trained
from sklearn.neural_network import MLPRegressor
def fwd(net, x, a, b):
    for i in range(a, b):
        x = x@net.coefs_[i]+net.intercepts_[i]
        if i < net.n_layers_-2: x = np.tanh(x)
    return x
def fit_best(X, hid, **kw):
    ms = [MLPRegressor(hidden_layer_sizes=hid, activation="tanh", random_state=s, **kw).fit(X, X) for s in range(5)]
    return min(ms, key=lambda q: ((q.predict(X)-X)**2).sum())
rng = np.random.default_rng(0)
ta = rng.uniform(.15*np.pi, 1.65*np.pi, 300); Xa = np.c_[np.cos(ta), np.sin(ta)]+rng.normal(0, .015, (300, 2))
ae = fit_best(Xa, (32, 1, 32), solver="lbfgs", max_iter=8000, tol=1e-7); ra = ae.predict(Xa)
pa = PCA(1).fit(Xa); rp = pa.inverse_transform(pa.transform(Xa))
print("arc MSE AE", ((ra-Xa)**2).sum(1).mean().round(5), "PCA", ((rp-Xa)**2).sum(1).mean().round(4))
tc = rng.uniform(0, 2*np.pi, 500); Xc = np.c_[np.cos(tc), np.sin(tc)]; ac = fit_best(Xc, (64, 1, 64), solver="lbfgs", max_iter=20000, tol=1e-8); rc = ac.predict(Xc); zc = fwd(ac, Xc, 0, 2).ravel(); ec = np.linalg.norm(rc-Xc, axis=1)
gr = np.linspace(0, 2*np.pi, 800, endpoint=False); Xg = np.c_[np.cos(gr), np.sin(gr)]; zg = fwd(ac, Xg, 0, 2).ravel(); eg_ = np.linalg.norm(ac.predict(Xg)-Xg, axis=1)
print("circle AE MSE", (ec**2).mean().round(5), "PCA share lost .5; max err", ec.max().round(3), "angle of max err", np.degrees(gr[eg_.argmax()]).round(0), "max |dz| between adjacent angles", np.abs(np.diff(np.r_[zg, zg[0]])).max().round(3), "z range", zg.min().round(3), zg.max().round(3))
fig, ax = plt.subplots(1, 4, figsize=(10, 3.0))
ax[0].scatter(*Xa.T, s=3, color="#A0AEC0"); ax[0].scatter(*rp.T, s=3, color=ORANGE); ax[0].set_title("arc: PCA with $t=1$")
ax[1].scatter(*Xa.T, s=3, color="#A0AEC0"); ax[1].scatter(*ra.T, s=3, color=BLUE)
i1, i2 = ta.argmin(), ta.argmax(); z1, z2 = fwd(ae, Xa[[i1]], 0, 2), fwd(ae, Xa[[i2]], 0, 2); al = np.linspace(0, 1, 11)[:, None]
ax[1].set_title("arc: autoencoder with $t=1$")
ax[2].scatter(*Xa.T, s=2, color="#E2E8F0"); li = (1-al)*Xa[i1]+al*Xa[i2]; dz = fwd(ae, (1-al)*z1+al*z2, 2, ae.n_layers_-1)
ax[2].plot(*li.T, "o-", ms=3, lw=1, color=ORANGE, label="straight in $\\mathbb{R}^2$"); ax[2].plot(*dz.T, "o-", ms=3, lw=1, color=BLUE, label="straight in latent"); ax[2].legend(frameon=False, fontsize=6.5, loc="center"); ax[2].set_title("interpolating between the ends")
print("interp: straight midpoint radius", np.linalg.norm(li[5]).round(3), "latent path radius range", np.linalg.norm(dz, axis=1).min().round(3), np.linalg.norm(dz, axis=1).max().round(3))
for a in ax[:3]: a.set_aspect("equal", adjustable="box"); noticks(a); a.set_xlim(-1.3, 1.3); a.set_ylim(-1.3, 1.3)
sc = ax[3].scatter(*Xc.T, s=3, color="#E2E8F0"); ax[3].scatter(*rc.T, c=zc, cmap="viridis", s=4); ax[3].set_aspect("equal"); noticks(ax[3]); ax[3].set_xlim(-1.3, 1.3); ax[3].set_ylim(-1.3, 1.3); ax[3].set_title("circle: the seam (color: code $z$)")
save("ae_arc")
aed = MLPRegressor(hidden_layer_sizes=(128, 32, 2, 32, 128), activation="tanh", solver="adam", max_iter=3000, tol=1e-6, n_iter_no_change=60, random_state=0, learning_rate_init=2e-3).fit(Xd, Xd)
Zd = fwd(aed, Xd, 0, 3); Rd = aed.predict(Xd); pd_ = PCA(2).fit(Xd); Zp = pd_.transform(Xd); Rp = pd_.inverse_transform(Zp)
print("digits MSE per pixel (pixels 0..1) AE", ((Rd-Xd)**2).mean().round(4), "PCA", ((Rp-Xd)**2).mean().round(4), "iters", aed.n_iter_)
fig = plt.figure(figsize=(10, 4.4)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
a = fig.add_subplot(gs[0]); a.scatter(*Zp.T, c=yd, cmap="tab10", s=1.5); noticks(a); a.set_title("PCA codes, $t=2$")
a = fig.add_subplot(gs[1]); a.scatter(*Zd.T, c=yd, cmap="tab10", s=1.5); noticks(a); a.set_title("autoencoder codes, $t=2$")
for i in range(10): a.annotate(str(i), np.median(Zd[yd == i], axis=0), fontsize=9, fontweight="bold", ha="center", bbox=dict(boxstyle="round,pad=.1", fc="white", ec="none", alpha=.7))
a = fig.add_subplot(gs[2]); g = 12; lo, hi = np.percentile(Zd, 1, axis=0)-.02, np.percentile(Zd, 99, axis=0)+.02; ca = np.ones((g*9, g*9))
for r, yy in enumerate(np.linspace(hi[1], lo[1], g)):
    for c, xx in enumerate(np.linspace(lo[0], hi[0], g)):
        ca[r*9:r*9+8, c*9:c*9+8] = 1-np.clip(fwd(aed, np.array([[xx, yy]]), 3, aed.n_layers_-1).reshape(8, 8), 0, 1)
a.imshow(ca, cmap="gray", vmin=0, vmax=1); noticks(a); a.set_title("decoder on a grid of new codes")
save("ae_digits")
fig, ax = plt.subplots(3, 10, figsize=(10, 3.1)); pick = [np.where(yd == d_)[0][3] for d_ in range(10)]
for j, i in enumerate(pick):
    for r, im_ in enumerate((Xd[i], Rd[i], Rp[i])): ax[r, j].imshow(1-np.clip(im_.reshape(8, 8), 0, 1), cmap="gray", vmin=0, vmax=1); noticks(ax[r, j])
for r, l_ in enumerate(("original", "autoencoder", "PCA")): ax[r, 0].set_ylabel(l_, fontsize=8)
save("ae_recon")
ia, ib = np.where(yd == 0)[0][3], np.where(yd == 1)[0][3]; al = np.linspace(0, 1, 9)
fig, ax = plt.subplots(2, 9, figsize=(8.2, 2))
for j, q in enumerate(al):
    ax[0, j].imshow(1-((1-q)*Xd[ia]+q*Xd[ib]).reshape(8, 8), cmap="gray", vmin=0, vmax=1); ax[1, j].imshow(1-np.clip(fwd(aed, ((1-q)*Zd[ia]+q*Zd[ib])[None], 3, aed.n_layers_-1).reshape(8, 8), 0, 1), cmap="gray", vmin=0, vmax=1); noticks(ax[0, j]); noticks(ax[1, j])
ax[0, 0].set_ylabel("pixels", fontsize=8); ax[1, 0].set_ylabel("latent", fontsize=8)
save("ae_interp")
