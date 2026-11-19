# STAT 185 Notes: Dimension Reduction and Clustering

Typeset lecture notes for STAT 185, covering linear algebra background, PCA, the SVD, NMF, MDS, kernel PCA, manifold learning (Isomap, local PCA, Laplacian eigenmaps, UMAP), autoencoders, and clustering (k-means, Gaussian mixtures and EM, spectral clustering).

- **`stat185_notes.pdf`**: the compiled notes. Start here.
- **`stat185_notes.tex`**: the LaTeX source.
- **`figures/`**: every figure, plus the Python scripts that generate them. Each script prints the numbers quoted in the text.
- **`transcripts/`**: automatic transcripts of eight lectures (Oct 22 to Nov 17), lightly cleaned. See its README for what was edited.

## Building

The PDF builds with [Tectonic](https://tectonic-typesetting.github.io/):

```bash
tectonic stat185_notes.tex
```

To regenerate a figure, run its script from this folder, for example:

```bash
uv run --python 3.12 --with numpy --with scipy --with scikit-learn --with matplotlib python figures/spectral_figures.py
```

Some scripts also need `umap-learn`.
