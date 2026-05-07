"""Statistical analyses for the phonetics project.

Reads outputs of `normalise.py` and writes figures/tables to ``results/``.
Each section is isolated in its own function and called from ``main()``.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import cosine, pdist, squareform
from sklearn.metrics import (adjusted_rand_score, confusion_matrix, f1_score,
                             silhouette_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import NearestCentroid
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from lib.utils import load_params


# --- setup -------------------------------------------------------------------

P = load_params()
F = Path(P["paths"]["features_dir"])
R = Path(P["paths"]["results_dir"]); R.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(P["normalise"]["random_state"])

W_LAYER = P["neural"]["whisper_primary_layer"]
X_LAYER = P["neural"]["xlsr_primary_layer"]


def load_data():
    ac = pd.read_csv(F / "features_acoustic_norm.csv")
    ac["l1"] = ac["l1"].map({"L1_FR": "L1", "L1_RU": "L2"}).fillna(ac["l1"])

    def load_neural(name, layer):
        path = F / f"features_{name}_pca.npz"
        if not path.exists():
            print(f"WARN: {path.name} not found — {name} analyses skipped")
            return None
        z = np.load(path)
        return z[f"layer_{layer}_pca50"], z[f"layer_{layer}_pca2"]

    wh = load_neural("whisper", W_LAYER)
    xl = load_neural("xlsr", X_LAYER)
    return ac, wh, xl


def restrict_to_vowels(ac, wh, xl):
    vowels = [v for v in P["acoustic"]["french_oral_vowels"] if v in ac["phoneme"].unique()]
    mask = ac["phoneme"].isin(vowels).values
    ac_v = ac[mask].reset_index(drop=True)
    wh_v = (wh[0][mask], wh[1][mask]) if wh else None
    xl_v = (xl[0][mask], xl[1][mask]) if xl else None
    return vowels, ac_v, wh_v, xl_v


# --- helpers -----------------------------------------------------------------

def cosine_dist_centroids(X, mask_a, mask_b):
    return cosine(X[mask_a].mean(0), X[mask_b].mean(0))


def phoneme_centroids(X, labels, vowels):
    return np.stack([X[labels == v].mean(0) for v in vowels])


def between_class_ratio(X, labels):
    mu = X.mean(0)
    bc = sum(((X[labels == c].mean(0) - mu) ** 2).sum() * (labels == c).sum()
             for c in np.unique(labels))
    return bc / ((X - mu) ** 2).sum()


def cosine_within_between_ratio(X, labels):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    same = np.zeros_like(S, dtype=bool)
    for c in np.unique(labels):
        m = labels == c
        same |= np.outer(m, m)
    np.fill_diagonal(same, False)
    diff = ~same
    np.fill_diagonal(diff, False)
    return S[same].mean() / (S[diff].mean() + 1e-9)


def mantel(D1, D2, B=999):
    iu = np.triu_indices_from(D1, k=1)
    r = stats.spearmanr(D1[iu], D2[iu]).correlation
    null = [
        stats.spearmanr(D1[np.ix_(perm, perm)][iu], D2[iu]).correlation
        for perm in (RNG.permutation(D1.shape[0]) for _ in range(B))
    ]
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (B + 1)
    return r, p


def fdr(pvals):
    return multipletests(pvals, method=P["stats"]["fdr_method"])[1]


# --- §5 descriptive ----------------------------------------------------------

def section_descriptive(ac_v, wh_v, xl_v, vowels):
    desc = ac_v.groupby(["phoneme", "l1", "gender"]).agg(
        F1_mean=("F1_norm", "mean"), F1_sd=("F1_norm", "std"),
        F1_iqr=("F1_norm", lambda x: x.quantile(.75) - x.quantile(.25)),
        F2_mean=("F2_norm", "mean"), F2_sd=("F2_norm", "std"),
        F2_iqr=("F2_norm", lambda x: x.quantile(.75) - x.quantile(.25)),
        n=("F1_norm", "size"),
    ).reset_index()
    desc["F1_cv"] = desc["F1_sd"] / desc["F1_mean"].abs()
    desc.to_csv(R / "descriptive_acoustic.csv", index=False)

    rows = []
    for v in vowels:
        sub = ac_v[ac_v["phoneme"] == v].dropna(subset=["F1_norm"])
        if len(sub) < 5: continue
        try:
            m = smf.mixedlm("F1_norm ~ 1", sub, groups=sub["speaker"]).fit(reml=True)
            su, se = m.cov_re.iloc[0, 0], m.scale
            rows.append({"phoneme": v, "var_inter": su, "var_resid": se,
                         "icc": su / (su + se)})
        except Exception:
            pass
    pd.DataFrame(rows).to_csv(R / "variance_decomposition.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    for v in vowels:
        sub = ac_v[ac_v["phoneme"] == v]
        x, y = sub["F2_norm"].mean(), sub["F1_norm"].mean()
        ax.scatter(x, y, s=200)
        ax.annotate(v, (x, y), fontsize=14)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("F2 (Lobanov)"); ax.set_ylabel("F1 (Lobanov)"); ax.set_title("Vowel chart")
    fig.savefig(R / "vowel_chart.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, fn in enumerate(["F1_norm", "F2_norm"]):
        ax[i].boxplot([ac_v[ac_v["phoneme"] == v][fn].dropna() for v in vowels],
                      labels=vowels)
        ax[i].set_title(fn)
    fig.savefig(R / "formant_boxplots.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        X2 = neural[1]
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for j, key in enumerate(["phoneme", "l1", "gender"]):
            for cat in ac_v[key].unique():
                m = (ac_v[key] == cat).values
                ax[j].scatter(X2[m, 0], X2[m, 1], s=10, label=str(cat), alpha=0.5)
            ax[j].set_title(f"{name} PCA2 by {key}"); ax[j].legend(fontsize=7)
        fig.savefig(R / f"neural_pca2_{name}.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    bcr, cos_r = {}, {}
    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        bcr[name + "_pca2"] = between_class_ratio(neural[1], ac_v["phoneme"].values)
        cos_r[name] = cosine_within_between_ratio(neural[0], ac_v["phoneme"].values)
    pd.DataFrame([bcr]).to_csv(R / "between_class_ratio.csv", index=False)
    pd.DataFrame([cos_r]).to_csv(R / "cosine_similarity_ratios.csv", index=False)

    ac_cen = phoneme_centroids(ac_v[["F1_norm", "F2_norm"]].fillna(0).values,
                               ac_v["phoneme"].values, vowels)
    D_ac = squareform(pdist(ac_cen, "euclidean"))
    centroids = {"acoustic": (ac_cen, D_ac)}
    mantel_res = {}
    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        cen = phoneme_centroids(neural[0], ac_v["phoneme"].values, vowels)
        D = squareform(pdist(cen, "cosine"))
        centroids[name] = (cen, D)
        mantel_res[f"ac_vs_{name}"] = mantel(D_ac, D)
    if "whisper" in centroids and "xlsr" in centroids:
        mantel_res["wh_vs_xl"] = mantel(centroids["whisper"][1], centroids["xlsr"][1])
    pd.DataFrame(
        [{k: v[0] for k, v in mantel_res.items()},
         {k: v[1] for k, v in mantel_res.items()}],
        index=["mantel_r", "p_value"],
    ).to_csv(R / "mantel_results.csv")
    return centroids


# --- §6 statistical tests ----------------------------------------------------

def two_sample_test(a, b):
    if len(a) < 3 or len(b) < 3: return None
    norm_p = min(stats.shapiro(a)[1] if len(a) < 5000 else 1.0,
                 stats.shapiro(b)[1] if len(b) < 5000 else 1.0)
    if norm_p > 0.05:
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        return t, pv, "t"
    t, pv = stats.mannwhitneyu(a, b)
    return t, pv, "mw"


def section_tests(ac_v, wh_v, xl_v, vowels):
    rows = []
    for v in vowels:
        for f in ["F1_norm", "F2_norm"]:
            l1g = ac_v[(ac_v["phoneme"] == v) & (ac_v["l1"] == "L1")][f].dropna()
            l2g = ac_v[(ac_v["phoneme"] == v) & (ac_v["l1"] == "L2")][f].dropna()
            res = two_sample_test(l1g, l2g)
            if res is None: continue
            t, pv, test = res
            rows.append({"phoneme": v, "feature": f, "stat": t, "p": pv, "test": test,
                         "diff": l1g.mean() - l2g.mean(),
                         "n_l1": len(l1g), "n_l2": len(l2g)})
    res_t = pd.DataFrame(rows)
    if len(res_t):
        res_t["p_fdr"] = fdr(res_t["p"])
    res_t.to_csv(R / "tests_L1_L2_acoustic.csv", index=False)

    def perm_centroid(X, l1_lab, B=P["stats"]["permutation_B"]):
        m1, m2 = l1_lab == "L1", l1_lab == "L2"
        if m1.sum() < 2 or m2.sum() < 2: return np.nan, np.nan
        obs = cosine(X[m1].mean(0), X[m2].mean(0))
        null = []
        for _ in range(B):
            perm = RNG.permutation(l1_lab)
            null.append(cosine(X[perm == "L1"].mean(0), X[perm == "L2"].mean(0)))
        return obs, (np.sum(np.array(null) >= obs) + 1) / (B + 1)

    rows = []
    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        X = neural[0]
        for v in vowels:
            mv = (ac_v["phoneme"] == v).values
            if mv.sum() < 6: continue
            d, pv = perm_centroid(X[mv], ac_v.loc[mv, "l1"].values)
            rows.append({"phoneme": v, "model": name, "cos_dist": d, "p": pv})
    res_perm = pd.DataFrame(rows)
    if len(res_perm):
        res_perm["p_fdr"] = res_perm.groupby("model")["p"].transform(fdr)
    res_perm.to_csv(R / "tests_L1_L2_neural.csv", index=False)
    return res_t, res_perm


def section_classification(ac_v, wh_v, xl_v, vowels):
    def loso_nc(X, y, groups):
        preds = np.empty_like(y, dtype=object)
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            try:
                preds[te] = NearestCentroid().fit(X[tr], y[tr]).predict(X[te])
            except Exception:
                preds[te] = y[tr][0]
        return preds

    reps = {"acoustic": ac_v[["F1_norm", "F2_norm"]].fillna(0).values}
    if wh_v is not None: reps["whisper"] = wh_v[0]
    if xl_v is not None: reps["xlsr"] = xl_v[0]

    y, groups = ac_v["phoneme"].values, ac_v["speaker"].values
    results = {}
    for name, X in reps.items():
        preds = loso_nc(X, y, groups)
        results[name] = {
            "accuracy": (preds == y).mean(),
            "f1_macro": f1_score(y, preds, average="macro"),
            "predictions": preds,
        }
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y, preds, labels=vowels)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(vowels))); ax.set_yticks(range(len(vowels)))
        ax.set_xticklabels(vowels); ax.set_yticklabels(vowels)
        ax.set_title(f"Confusion: {name}")
        fig.savefig(R / f"confusion_{name}.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame({k: {"accuracy": v["accuracy"], "f1_macro": v["f1_macro"]}
                  for k, v in results.items()}).T.to_csv(R / "classification_accuracy.csv")

    rows = []
    keys = list(results)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pa = results[a]["predictions"] == y
            pb = results[b]["predictions"] == y
            tab = np.array([[(pa & pb).sum(), (pa & ~pb).sum()],
                            [(~pa & pb).sum(), (~pa & ~pb).sum()]])
            r = mcnemar(tab, exact=False)
            rows.append({"pair": f"{a}_vs_{b}", "stat": r.statistic, "p": r.pvalue})
    pd.DataFrame(rows).to_csv(R / "mcnemar.csv", index=False)


# --- §7 LME ------------------------------------------------------------------

def _fit_null(formula, sub):
    m = smf.mixedlm(formula, sub, groups=sub["speaker"]).fit(reml=False)
    su, se = m.cov_re.iloc[0, 0], m.scale
    return {"null_loglik": m.llf, "icc": su / (su + se)}


def _fit_main(formula, sub):
    m = smf.mixedlm(formula, sub, groups=sub["speaker"]).fit(reml=False)
    return {
        "main_loglik": m.llf,
        "beta_L2": m.params.get("L2", np.nan),
        "se_L2": m.bse.get("L2", np.nan),
        "p_L2": m.pvalues.get("L2", np.nan),
        "beta_Male": m.params.get("Male", np.nan),
        "p_Male": m.pvalues.get("Male", np.nan),
    }


def _fit_full(formula, sub):
    m = smf.mixedlm(formula, sub, groups=sub["speaker"]).fit(reml=False)
    var_f = np.var(m.fittedvalues)
    var_r = m.cov_re.iloc[0, 0]
    var_e = m.scale
    total = var_f + var_r + var_e
    return {
        "full_loglik": m.llf,
        "beta_int": m.params.get("L2:Male", np.nan),
        "p_int": m.pvalues.get("L2:Male", np.nan),
        "aic": m.aic, "bic": m.bic,
        "R2_marg": var_f / total,
        "R2_cond": (var_f + var_r) / total,
    }


def fit_lme_sequence(df, response):
    sub = df.dropna(subset=[response]).copy()
    sub["L2"] = (sub["l1"] == "L2").astype(int)
    sub["Male"] = sub["gender"].astype(str).str.upper().str.startswith("M").astype(int)

    out = {}
    for fitter, formula in [
        (_fit_null, f"{response} ~ 1"),
        (_fit_main, f"{response} ~ L2 + Male"),
        (_fit_full, f"{response} ~ L2 * Male"),
    ]:
        try:
            out.update(fitter(formula, sub))
        except Exception:
            pass
    return out


def section_lme(ac_v, wh_v, xl_v, vowels):
    rows = []
    for v in vowels:
        sub = ac_v[ac_v["phoneme"] == v]
        if len(sub) < 20: continue
        rows.append({**fit_lme_sequence(sub, "F1_norm"), "phoneme": v, "response": "F1"})

    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        for v in vowels:
            mv = (ac_v["phoneme"] == v).values
            if mv.sum() < 20: continue
            sub = ac_v[mv].copy()
            sub[f"{name}_PC1"] = neural[0][mv, 0]
            rows.append({**fit_lme_sequence(sub, f"{name}_PC1"),
                         "phoneme": v, "response": f"{name}_PC1"})
    pd.DataFrame(rows).to_csv(R / "lme_results.csv", index=False)


# --- §8 ROPE -----------------------------------------------------------------

def intra_speaker_cos(X, ph, sp):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    out = []
    for s in np.unique(sp):
        for v in np.unique(ph):
            idx = np.where((sp == s) & (ph == v))[0]
            if len(idx) >= 2:
                Xi = Xn[idx]
                out += list(1 - (Xi @ Xi.T)[np.triu_indices(len(idx), 1)])
    return np.mean(out) if out else 0.05


def classify(ci_lo, ci_hi, lo, hi):
    if ci_hi < lo or ci_lo > hi: return "non-equivalent"
    if ci_lo >= lo and ci_hi <= hi: return "equivalent"
    return "indeterminate"


def section_rope(ac_v, wh_v, xl_v, vowels, res_t):
    rope_lo, rope_hi = P["stats"]["rope_acoustic_lobanov"]
    rows = []

    for _, row in res_t[res_t["feature"] == "F1_norm"].iterrows():
        diff = row["diff"]
        if row["p"] > 0:
            se = abs(diff) / max(abs(stats.norm.ppf(row["p"] / 2)), 1e-3)
        else:
            se = np.nan
        ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
        rows.append({
            "phoneme": row["phoneme"], "rep": "acoustic", "diff": diff,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "rope_class": classify(ci_lo, ci_hi, rope_lo, rope_hi),
            "p": row["p"],
        })

    def boot_neural_ci(X, l1_lab, sp, B=P["stats"]["bootstrap_B"]):
        speakers = np.unique(sp); diffs = []
        for _ in range(B):
            s = RNG.choice(speakers, len(speakers), replace=True)
            m = np.isin(sp, s)
            if m.sum() < 4: continue
            try:
                diffs.append(cosine(X[m & (l1_lab == "L1")].mean(0),
                                    X[m & (l1_lab == "L2")].mean(0)))
            except Exception: pass
        return (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)) if diffs else (np.nan, np.nan)

    for name, neural in [("whisper", wh_v), ("xlsr", xl_v)]:
        if neural is None: continue
        X = neural[0]
        delta0 = intra_speaker_cos(X, ac_v["phoneme"].values, ac_v["speaker"].values)
        for v in vowels:
            mv = (ac_v["phoneme"] == v).values
            if mv.sum() < 6: continue
            ci_lo, ci_hi = boot_neural_ci(X[mv], ac_v.loc[mv, "l1"].values,
                                          ac_v.loc[mv, "speaker"].values)
            if np.isnan(ci_lo): continue
            rows.append({
                "phoneme": v, "rep": name, "diff": (ci_lo + ci_hi) / 2,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
                "rope_class": classify(ci_lo, ci_hi, 0, delta0),
                "delta0": delta0,
            })
    pd.DataFrame(rows).to_csv(R / "rope_classification.csv", index=False)


# --- §9 clustering -----------------------------------------------------------

FRONT = {"i", "e", "E", "a", "y", "@", "2", "9"}
HIGH = {"i", "y", "u"}


def gt_membership(vowels, group_set):
    return np.array([1 if v in group_set else 0 for v in vowels])


def best_cluster_ari(centroids, vowels, metric, gt, k_max):
    Z = linkage(pdist(centroids, metric), method=P["clustering"]["linkage"])
    best = (-2, 2)
    for k in range(2, min(k_max, len(vowels))):
        ari = adjusted_rand_score(gt, fcluster(Z, t=k, criterion="maxclust"))
        if ari > best[0]: best = (ari, k)
    return best, Z


def section_clustering_vowels(centroids_dict, vowels):
    rows = []
    k_max = P["clustering"]["k_range"][1]
    for name, (cen, _) in centroids_dict.items():
        metric = "euclidean" if name == "acoustic" else "cosine"
        for gt_name, gt_set in [("front_back", FRONT), ("high", HIGH)]:
            (ari, k), Z = best_cluster_ari(cen, vowels, metric,
                                           gt_membership(vowels, gt_set), k_max)
            rows.append({"rep": name, "ground_truth": gt_name, "best_ari": ari, "best_k": k})
            if gt_name == "front_back":
                fig, ax = plt.subplots(figsize=(8, 4))
                dendrogram(Z, labels=vowels, ax=ax); ax.set_title(f"Vowel dendrogram: {name}")
                fig.savefig(R / f"dendrogram_{name}.png", dpi=120, bbox_inches="tight")
                plt.close(fig)
    pd.DataFrame(rows).to_csv(R / "clustering_vowels.csv", index=False)


def section_clustering_speakers(ac_v, wh_v, xl_v, vowels):
    reps = {"acoustic": ac_v[["F1_norm", "F2_norm"]].fillna(0).values}
    if wh_v is not None: reps["whisper"] = wh_v[0]
    if xl_v is not None: reps["xlsr"] = xl_v[0]

    rows = []
    for name, X in reps.items():
        sp_vecs, sp_meta = [], []
        for sp in ac_v["speaker"].unique():
            ms = (ac_v["speaker"] == sp).values
            per_v = []
            for v in vowels:
                mv = ms & (ac_v["phoneme"] == v).values
                per_v.append(X[mv].mean(0) if mv.sum() else np.zeros(X.shape[1]))
            sp_vecs.append(np.concatenate(per_v))
            sp_meta.append({"speaker": sp,
                            "l1": ac_v.loc[ms, "l1"].iloc[0],
                            "gender": ac_v.loc[ms, "gender"].iloc[0]})
        sp_vecs = np.stack(sp_vecs)
        meta_df = pd.DataFrame(sp_meta)
        metric = "euclidean" if name == "acoustic" else "cosine"
        Z = linkage(pdist(sp_vecs, metric), method=P["clustering"]["linkage"])

        for gt_col in ["l1", "gender"]:
            gt = pd.factorize(meta_df[gt_col])[0]
            best = (-2, 2)
            for k in range(2, min(8, len(sp_vecs))):
                ari = adjusted_rand_score(gt, fcluster(Z, t=k, criterion="maxclust"))
                if ari > best[0]: best = (ari, k)
            sil = (silhouette_score(sp_vecs, fcluster(Z, t=2, criterion="maxclust"), metric=metric)
                   if len(sp_vecs) > 2 else np.nan)
            rows.append({"rep": name, "gt": gt_col, "ari": best[0], "k": best[1],
                         "silhouette_k2": sil})
    pd.DataFrame(rows).to_csv(R / "clustering_speakers.csv", index=False)


# --- main --------------------------------------------------------------------

def main():
    ac, wh, xl = load_data()
    vowels, ac_v, wh_v, xl_v = restrict_to_vowels(ac, wh, xl)

    centroids = section_descriptive(ac_v, wh_v, xl_v, vowels)
    res_t, _ = section_tests(ac_v, wh_v, xl_v, vowels)
    section_classification(ac_v, wh_v, xl_v, vowels)
    section_lme(ac_v, wh_v, xl_v, vowels)
    section_rope(ac_v, wh_v, xl_v, vowels, res_t)
    section_clustering_vowels(centroids, vowels)
    section_clustering_speakers(ac_v, wh_v, xl_v, vowels)
    print(f"Analysis complete. See {R}")


if __name__ == "__main__":
    main()
