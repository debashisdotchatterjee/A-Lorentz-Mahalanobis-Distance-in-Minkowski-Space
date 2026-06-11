
"""
Clean real-data Lorentz--Mahalanobis workflow
=============================================

This script is a trimmed and more manuscript-focused version of the real-data
workflow. It deliberately omits outputs that are not informative for the paper,
for example raw Tyler LM score magnitudes, whose absolute scale is arbitrary.

Main goals:
  1. Load a real/physics jet dataset through EnergyFlow.
  2. Construct genuine four-vectors (E, px, py, pz).
  3. Fit classical and robust Lorentz--Mahalanobis models.
  4. Report rest-frame estimates and interpretable score summaries.
  5. Profile high robust-LM jets.
  6. Verify score/ranking stability under transported Lorentz boosts.
  7. Perform a real-data central-vs-forward two-sample LM test.
  8. Save only manuscript-useful figures/tables and create a ZIP.

Usage in Colab:
    from lm_realdata_analysis_clean import CLEAN_CONFIG, run_clean_realdata_analysis
    results = run_clean_realdata_analysis(CLEAN_CONFIG, make_zip=True)
"""

import os
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from scipy.stats import chi2

try:
    from IPython.display import display, Image as IPImage
    IPYTHON_AVAILABLE = True
except Exception:
    IPYTHON_AVAILABLE = False

try:
    from google.colab import files
    COLAB_AVAILABLE = True
except Exception:
    COLAB_AVAILABLE = False

# Import the earlier full-methodology code as a backend.
# This file is included in the same package.
import lm_realdata_analysis_colab as lm

CLEAN_CONFIG = lm.CONFIG.copy()
CLEAN_CONFIG.update({
    "outdir": "lm_realdata_clean_outputs",
    "top_n_print": 15,
    "top_fraction": 0.01,
    "permutation_B": 999,
    "boost_etas": [0.0, 0.5, 1.0, 1.5, 2.0],
})

COLORS = {
    "Classical LM": "#D55E00",
    "MCD robust LM": "#009E73",
    "Naive Euclidean": "#CC79A7",
    "Minkowski interval": "#A6761D",
}

MARKERS = {
    "Classical LM": "o",
    "MCD robust LM": "s",
    "Naive Euclidean": "^",
    "Minkowski interval": "D",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def make_dirs(outdir):
    outdir = Path(outdir)
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    for d in [outdir, figdir, tabdir]:
        d.mkdir(parents=True, exist_ok=True)
    return outdir, figdir, tabdir


def save_dataframe(df, name, tabdir, index=False):
    csv_path = tabdir / f"{name}.csv"
    tex_path = tabdir / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=index, escape=False, float_format="%.6g"))
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


def save_figure(fig, name, figdir):
    png_path = figdir / f"{name}.png"
    pdf_path = figdir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def show_table(df, title, max_rows=30):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if len(df) > max_rows:
        df_show = df.head(max_rows)
        print(f"Showing first {max_rows} of {len(df)} rows.")
    else:
        df_show = df
    if IPYTHON_AVAILABLE:
        display(df_show)
    else:
        print(df_show.to_string(index=False))


def qsummary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q90": float(np.quantile(x, 0.90)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def compact_kinematic_summary(df):
    rows = []
    names = [
        ("pt", r"$p_T$"),
        ("abs_rapidity", r"$|y|$"),
        ("mass", r"mass"),
        ("E", r"energy"),
    ]
    if "npv" in df.columns:
        names.append(("npv", r"NPV"))
    for col, label in names:
        s = qsummary(df[col].values)
        s.update({"quantity": label})
        rows.append(s)
    return pd.DataFrame(rows)[["quantity", "median", "q75", "q90", "q95", "q99", "max"]]


def source_metadata(df, actual_source):
    m2 = df["m2_from_fourvector"].values
    return pd.DataFrame([{
        "actual_source": actual_source,
        "n_jets_used": len(df),
        "pt_min": float(df["pt"].min()),
        "pt_max": float(df["pt"].max()),
        "rapidity_min": float(df["rapidity"].min()),
        "rapidity_max": float(df["rapidity"].max()),
        "mass_min": float(df["mass"].min()),
        "mass_max": float(df["mass"].max()),
        "median_abs_fourvector_mass_error": float(np.median(np.abs(m2 - np.maximum(df["mass"].values, 0.0)**2))),
    }])


def plot_kinematics_clean(df, figdir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    specs = [
        (np.log10(df["pt"].values), r"$\log_{10}(p_T)$", "Jet transverse momentum"),
        (df["rapidity"].values, r"rapidity $y$", "Jet rapidity"),
        (np.log10(np.maximum(df["mass"].values, 1e-8) + 1.0), r"$\log_{10}(1+m)$", "Jet mass"),
        (np.log10(np.maximum(df["E"].values, 1e-8)), r"$\log_{10}(E)$", "Jet energy"),
    ]

    for ax, (x, xlabel, title) in zip(axes, specs):
        ax.hist(x[np.isfinite(x)], bins=70, color="#56B4E9", alpha=0.85, edgecolor="white", linewidth=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(title)

    save_figure(fig, "fig01_realdata_clean_kinematics", figdir)
    plt.show()


def fit_clean_models(X, config):
    models = {}
    models["classical"] = lm.fit_lm_model(X, method="classical", config=config)
    if lm.SKLEARN_AVAILABLE:
        models["mcd"] = lm.fit_lm_model(X, method="mcd", config=config)
    else:
        warnings.warn("MCD unavailable; robust results will be skipped.")
    return models


def fitted_model_summary(models):
    rows = []
    for key, model in models.items():
        u = model["u"]
        beta = model["beta"]
        eigs = np.linalg.eigvalsh(model["Sigma_s"])
        rows.append({
            "method": model["label"],
            "u0": u[0],
            "u1": u[1],
            "u2": u[2],
            "u3": u[3],
            "beta_norm": float(np.linalg.norm(beta)),
            "rapidity_norm": model["rapidity_norm"],
            "sigma_tau2": model["sigma_tau2"],
            "spatial_condition": float(eigs.max() / max(eigs.min(), 1e-12)),
        })
    return pd.DataFrame(rows)


def score_summary(scores):
    rows = []
    for name, sc in scores.items():
        s = qsummary(sc)
        s.update({"score": name})
        rows.append(s)
    return pd.DataFrame(rows)[["score", "median", "q75", "q90", "q95", "q99", "max"]]


def plot_score_tail(scores, figdir):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    for name, sc in scores.items():
        x = np.sort(np.asarray(sc, dtype=float))
        x = x[np.isfinite(x)]
        # upper tail survival curve
        q = np.linspace(0.50, 0.999, 350)
        vals = np.quantile(x, q)
        surv = 1 - q
        ax.plot(vals, surv, linewidth=2.1, color=COLORS.get(name), label=name)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("score quantile")
    ax.set_ylabel("upper-tail probability")
    ax.set_title("Real-data score tail behaviour")
    ax.legend(frameon=True)
    save_figure(fig, "fig02_realdata_clean_score_tails", figdir)
    plt.show()


def plot_robust_score_map(df, robust_score, figdir):
    cutoff = np.quantile(robust_score, 0.99)
    high = robust_score >= cutoff

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    axes[0].scatter(df["pt"], df["mass"], s=5, alpha=0.20, color="#999999", label="all jets")
    axes[0].scatter(df.loc[high, "pt"], df.loc[high, "mass"], s=14, alpha=0.85, color="#D55E00", label="top 1% robust LM")
    axes[0].set_xlabel(r"$p_T$")
    axes[0].set_ylabel("mass")
    axes[0].set_title("High-score jets in $(p_T,m)$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")

    axes[1].scatter(df["abs_rapidity"], df["mass"], s=5, alpha=0.20, color="#999999")
    axes[1].scatter(df.loc[high, "abs_rapidity"], df.loc[high, "mass"], s=14, alpha=0.85, color="#D55E00")
    axes[1].set_xlabel(r"$|y|$")
    axes[1].set_ylabel("mass")
    axes[1].set_title("High-score jets versus rapidity")
    axes[1].set_yscale("log")

    axes[2].scatter(df["abs_rapidity"], df["pt"], s=5, alpha=0.20, color="#999999")
    axes[2].scatter(df.loc[high, "abs_rapidity"], df.loc[high, "pt"], s=14, alpha=0.85, color="#D55E00")
    axes[2].set_xlabel(r"$|y|$")
    axes[2].set_ylabel(r"$p_T$")
    axes[2].set_title("High-score jets versus transverse momentum")
    axes[2].set_yscale("log")

    axes[0].legend(frameon=True)
    save_figure(fig, "fig03_realdata_clean_high_score_map", figdir)
    plt.show()


def high_score_profile(df, score, temporal, spatial, frac=0.01):
    cutoff = float(np.quantile(score, 1 - frac))
    top = score >= cutoff

    quantities = ["pt", "abs_rapidity", "mass", "E"]
    if "npv" in df.columns:
        quantities.append("npv")
    if "multiplicity" in df.columns:
        quantities.append("multiplicity")

    rows = []
    for q in quantities:
        full_med = float(np.median(df[q].values))
        top_med = float(np.median(df.loc[top, q].values))
        rows.append({
            "quantity": q,
            "full_median": full_med,
            "top1pct_median": top_med,
            "median_ratio_top_to_full": top_med / full_med if abs(full_med) > 1e-12 else np.nan,
            "full_q95": float(np.quantile(df[q].values, 0.95)),
            "top1pct_q95": float(np.quantile(df.loc[top, q].values, 0.95)),
        })

    frac_temporal_full = temporal / np.maximum(score, 1e-12)
    frac_temporal_top = frac_temporal_full[top]
    extra = pd.DataFrame([{
        "quantity": "temporal_fraction_of_score",
        "full_median": float(np.median(frac_temporal_full)),
        "top1pct_median": float(np.median(frac_temporal_top)),
        "median_ratio_top_to_full": float(np.median(frac_temporal_top) / max(np.median(frac_temporal_full), 1e-12)),
        "full_q95": float(np.quantile(frac_temporal_full, 0.95)),
        "top1pct_q95": float(np.quantile(frac_temporal_top, 0.95)),
    }])

    tab = pd.concat([pd.DataFrame(rows), extra], ignore_index=True)
    tab.insert(0, "top_fraction", frac)
    tab.insert(1, "score_cutoff", cutoff)
    return tab


def top_jets_table(df, score, temporal, spatial, top_n=15):
    idx = np.argsort(score)[::-1][:top_n]
    cols = ["jet_index", "pt", "rapidity", "mass", "E"]
    if "npv" in df.columns:
        cols.append("npv")
    if "quality" in df.columns:
        cols.append("quality")
    if "multiplicity" in df.columns:
        cols.append("multiplicity")
    if "qg_label" in df.columns:
        cols.append("qg_label")

    out = df.iloc[idx][cols].copy()
    out["robust_LM_score"] = score[idx]
    out["temporal_component"] = temporal[idx]
    out["spatial_component"] = spatial[idx]
    out["temporal_fraction"] = temporal[idx] / np.maximum(score[idx], 1e-12)
    return out


def boost_stability_table(X, model, baseline_scores, etas, top_frac=0.01):
    baseline_lm = baseline_scores["MCD robust LM"]
    baseline_naive = baseline_scores["Naive Euclidean"]
    baseline_mink = baseline_scores["Minkowski interval"]

    k = max(1, int(np.floor(top_frac * len(X))))
    top_lm0 = set(np.argsort(baseline_lm)[-k:])
    top_naive0 = set(np.argsort(baseline_naive)[-k:])
    top_mink0 = set(np.argsort(baseline_mink)[-k:])

    rows = []

    for eta in etas:
        B = lm.lorentz_boost_x(float(eta))
        Xb = X @ B.T

        mb = lm.transport_model(model, B)
        lm_b = lm.score_lm_model(Xb, mb)

        naive_b = lm.score_naive_euclidean(Xb)
        mink_b = lm.score_minkowski_interval_centered(Xb)

        entries = [
            ("MCD robust LM", baseline_lm, lm_b, top_lm0),
            ("Naive Euclidean", baseline_naive, naive_b, top_naive0),
            ("Minkowski interval", baseline_mink, mink_b, top_mink0),
        ]

        for name, s0, s1, top0 in entries:
            rel = np.abs(s1 - s0) / np.maximum(np.abs(s0), 1e-12)
            rho = spearmanr(s0, s1).correlation
            top1 = set(np.argsort(s1)[-k:])
            overlap = len(top0.intersection(top1)) / k
            rows.append({
                "eta": eta,
                "score": name,
                "median_relative_change": float(np.median(rel)),
                "q95_relative_change": float(np.quantile(rel, 0.95)),
                "spearman_rank_correlation": float(rho),
                "top1pct_overlap": float(overlap),
            })

    return pd.DataFrame(rows)


def plot_boost_stability(tab, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for score, gdf in tab.groupby("score"):
        gdf = gdf.sort_values("eta")
        axes[0].plot(
            gdf["eta"], gdf["top1pct_overlap"],
            marker=MARKERS.get(score, "o"),
            linewidth=2.2,
            color=COLORS.get(score),
            label=score
        )
        axes[1].plot(
            gdf["eta"], gdf["median_relative_change"],
            marker=MARKERS.get(score, "o"),
            linewidth=2.2,
            color=COLORS.get(score),
            label=score
        )

    axes[0].set_xlabel(r"boost rapidity $\eta$")
    axes[0].set_ylabel("top 1% overlap with unboosted ranking")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title("Real-data ranking stability")

    axes[1].set_xlabel(r"boost rapidity $\eta$")
    axes[1].set_ylabel("median relative score change")
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].set_title("Real-data score stability")

    axes[0].legend(frameon=True)
    save_figure(fig, "fig04_realdata_clean_boost_stability", figdir)
    plt.show()


def plot_two_sample_permutation(two_res, figdir):
    if two_res is None:
        return
    perm = np.asarray(two_res["perm_T2"], dtype=float)
    obs = float(two_res["T2"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(perm, bins=45, color="#56B4E9", alpha=0.85, edgecolor="white", linewidth=0.25)
    ax.axvline(obs, color="#D55E00", linestyle="--", linewidth=2.0, label="observed")
    ax.set_xlabel(r"permuted $T^2$")
    ax.set_ylabel("count")
    ax.set_title("Central--forward LM two-sample permutation test")
    ax.legend(frameon=True)
    save_figure(fig, "fig05_realdata_clean_two_sample_permutation", figdir)
    plt.show()


def run_clean_realdata_analysis(config=CLEAN_CONFIG, make_zip=True):
    outdir, figdir, tabdir = make_dirs(config["outdir"])
    print("Starting CLEAN Lorentz--Mahalanobis real-data analysis.")
    print("This version suppresses non-informative outputs and keeps only manuscript-useful diagnostics.")
    print(json.dumps(config, indent=2))

    # ------------------------------------------------------------------
    # 1. Load and validate data.
    # ------------------------------------------------------------------
    df, actual_source = lm.load_dataset(config)
    X = df[["E", "px", "py", "pz"]].values

    meta = source_metadata(df, actual_source)
    kin = compact_kinematic_summary(df)

    save_dataframe(meta, "table01_source_metadata", tabdir)
    save_dataframe(kin, "table02_kinematic_summary", tabdir)
    show_table(meta, "Table 1: data-source metadata")
    show_table(kin, "Table 2: compact kinematic summary")
    plot_kinematics_clean(df, figdir)

    # ------------------------------------------------------------------
    # 2. Fit LM models.
    # ------------------------------------------------------------------
    models = fit_clean_models(X, config)
    model_tab = fitted_model_summary(models)
    save_dataframe(model_tab, "table03_fitted_lm_models", tabdir)
    show_table(model_tab, "Table 3: fitted Lorentz--Mahalanobis rest-frame models")

    # ------------------------------------------------------------------
    # 3. Scores: keep interpretable scores only.
    # ------------------------------------------------------------------
    scores = {}
    components = {}

    for key, model in models.items():
        sc, temporal, spatial, Y = lm.score_lm_model(X, model, return_components=True)
        scores[model["label"]] = sc
        components[model["label"]] = {"temporal": temporal, "spatial": spatial}

    robust_label = "MCD robust LM" if "MCD robust LM" in scores else "Classical LM"

    scores["Naive Euclidean"] = lm.score_naive_euclidean(X)
    scores["Minkowski interval"] = lm.score_minkowski_interval_centered(X)

    score_tab = score_summary(scores)
    save_dataframe(score_tab, "table04_score_tail_summary", tabdir)
    show_table(score_tab, "Table 4: interpretable score-tail summary")
    plot_score_tail(scores, figdir)

    # ------------------------------------------------------------------
    # 4. High-score profiling.
    # ------------------------------------------------------------------
    robust_score = scores[robust_label]
    temporal = components[robust_label]["temporal"]
    spatial = components[robust_label]["spatial"]

    profile = high_score_profile(
        df, robust_score, temporal, spatial,
        frac=config["top_fraction"]
    )
    profile.insert(0, "score_used", robust_label)
    save_dataframe(profile, "table05_high_score_profile", tabdir)
    show_table(profile, "Table 5: high robust-LM score profile")

    top_tab = top_jets_table(
        df, robust_score, temporal, spatial,
        top_n=config["top_n_print"]
    )
    save_dataframe(top_tab, "table06_top_robust_lm_jets", tabdir)
    show_table(top_tab, f"Table 6: top {config['top_n_print']} jets by {robust_label}", max_rows=config["top_n_print"])

    plot_robust_score_map(df, robust_score, figdir)

    # ------------------------------------------------------------------
    # 5. Real-data coordinate stability under transported boosts.
    # ------------------------------------------------------------------
    stability = boost_stability_table(
        X,
        models["mcd"] if "mcd" in models else models["classical"],
        scores,
        config["boost_etas"],
        top_frac=config["top_fraction"]
    )
    save_dataframe(stability, "table07_boost_ranking_stability", tabdir)
    show_table(stability, "Table 7: real-data boost/ranking stability")
    plot_boost_stability(stability, figdir)

    # ------------------------------------------------------------------
    # 6. Central-vs-forward two-sample LM test.
    # ------------------------------------------------------------------
    try:
        two_tab, two_res = lm.two_sample_analysis(
            df,
            models["mcd"] if "mcd" in models else models["classical"],
            config
        )
        save_dataframe(two_tab, "table08_central_forward_two_sample", tabdir)
        show_table(two_tab, "Table 8: central versus forward LM two-sample test")
        plot_two_sample_permutation(two_res, figdir)
    except Exception as e:
        warnings.warn(f"Two-sample analysis failed: {repr(e)}")
        two_tab, two_res = None, None

    # ------------------------------------------------------------------
    # 7. Output inventory and metadata.
    # ------------------------------------------------------------------
    metadata = {
        "actual_source": actual_source,
        "n_jets_used": int(len(df)),
        "robust_label_used": robust_label,
        "omitted_outputs": [
            "Raw Tyler-score magnitudes omitted because Tyler shape is scale-free.",
            "Raw all-score density overlays omitted because scales are incomparable.",
            "Four-vector m2 histogram omitted; only median reconstruction error is retained.",
        ],
        "configuration": config,
    }
    with open(outdir / "metadata_clean.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""
Clean real-data Lorentz--Mahalanobis outputs
============================================

Dataset source used: {actual_source}
Number of jets used: {len(df)}
Robust score used for profiling: {robust_label}

Main tables:
  table01_source_metadata
  table02_kinematic_summary
  table03_fitted_lm_models
  table04_score_tail_summary
  table05_high_score_profile
  table06_top_robust_lm_jets
  table07_boost_ranking_stability
  table08_central_forward_two_sample

Main figures:
  fig01_realdata_clean_kinematics
  fig02_realdata_clean_score_tails
  fig03_realdata_clean_high_score_map
  fig04_realdata_clean_boost_stability
  fig05_realdata_clean_two_sample_permutation

Omitted by design:
  - raw Tyler LM score magnitudes, because Tyler shape is scale-free;
  - visually cluttered score-density overlays across incomparable scales;
  - four-vector m^2 histogram, replaced by a scalar reconstruction check.
"""
    with open(outdir / "README_clean_outputs.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    if make_zip:
        zip_path = Path(f"{config['outdir']}.zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in outdir.rglob("*"):
                zf.write(file, arcname=str(file.relative_to(outdir.parent)))
        print(f"\nCreated ZIP archive: {zip_path.resolve()}")

        if COLAB_AVAILABLE:
            try:
                files.download(str(zip_path))
            except Exception:
                print("Colab automatic download did not start; ZIP was still created.")

    return {
        "data": df,
        "models": models,
        "scores": scores,
        "components": components,
        "metadata": meta,
        "kinematics": kin,
        "model_table": model_tab,
        "score_summary": score_tab,
        "high_score_profile": profile,
        "top_jets": top_tab,
        "boost_stability": stability,
        "two_sample": two_tab,
        "two_sample_result": two_res,
        "outdir": str(outdir),
    }


if __name__ == "__main__":
    run_clean_realdata_analysis(CLEAN_CONFIG, make_zip=True)
