import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from lib.utils import load_params


P = load_params()
N = P["normalise"]
F = Path(P["paths"]["features_dir"])
VOWELS = set(P["acoustic"]["french_oral_vowels"])


def lobanov(df, formant):
    """Per-speaker z-score using vowel tokens to estimate (mean, sd)."""
    is_vowel = df["phoneme"].isin(VOWELS)
    sp_stats = df[is_vowel].groupby("speaker")[formant].agg(["mean", "std"])
    mu = df["speaker"].map(sp_stats["mean"])
    sd = df["speaker"].map(sp_stats["std"])
    return (df[formant] - mu) / sd


def reduce_neural(npz_path, primary_layer, out_path):
    z = np.load(npz_path)
    X = z[f"layer_{primary_layer}"]

    out = {"token_idx": z["token_idx"]}
    rs = N["random_state"]
    n_clust = min(N["pca_dim_clust"], X.shape[1], X.shape[0])
    out[f"layer_{primary_layer}_pca50"] = PCA(n_clust, random_state=rs).fit_transform(X)
    out[f"layer_{primary_layer}_pca2"] = PCA(2, random_state=rs).fit_transform(X)

    try:
        import umap
        out[f"layer_{primary_layer}_umap2"] = umap.UMAP(
            n_neighbors=N["umap_neighbors"],
            min_dist=N["umap_min_dist"],
            random_state=rs,
        ).fit_transform(X)
    except Exception as e:
        print(f"UMAP skipped for {npz_path.name}: {e}")

    np.savez(out_path, **out)


def main():
    df = pd.read_csv(F / "features_acoustic.csv")
    for f in ["F1", "F2", "F3"]:
        df[f"{f}_norm"] = lobanov(df, f)
    df.to_csv(F / "features_acoustic_norm.csv", index=False)

    neural = [
        ("features_whisper.npz", P["neural"]["whisper_primary_layer"], "features_whisper_pca.npz"),
        ("features_xlsr.npz",    P["neural"]["xlsr_primary_layer"],    "features_xlsr_pca.npz"),
    ]
    for src, layer, dst in neural:
        if (F / src).exists():
            reduce_neural(F / src, layer, F / dst)
        else:
            print(f"SKIP {src} — drop the file from Kaggle into {F}/ to enable")
    print("Normalisation complete.")


if __name__ == "__main__":
    main()
