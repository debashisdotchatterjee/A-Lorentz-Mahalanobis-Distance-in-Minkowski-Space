"""
Lorentz--Mahalanobis real-data methodology demonstration
========================================================

Colab-ready analysis package for the Lorentz--Mahalanobis paper.

Primary source:
    CMS Open Data jets through EnergyFlow MOD HDF5 interface, collection CMS2011AJets.

Fallback/source option:
    EnergyFlow quark/gluon jets, generated physics benchmark. The script records
    the actual source used in outputs/metadata.json, so do not describe fallback
    output as CMS Open Data if the fallback is used.

The workflow is intentionally different from the simulation verification:
    1. Load a physics jet dataset from a Python package.
    2. Construct genuine four-vectors (E, px, py, pz).
    3. Fit classical, MCD, and Tyler Lorentz--Mahalanobis rest-frame models.
    4. Rank and profile high-score jets.
    5. Compare classical and robust LM scores.
    6. Verify parameter-transport equivariance on the real data.
    7. Run a real-data two-sample LM test: central vs forward jets, with permutation calibration.
    8. Save all plots/tables and create a downloadable ZIP.

Author-facing note:
    This is real-data methodology code, not a simulation study. The boost step
    uses the observed real jet four-vectors and transported fitted parameters
    to verify implementation-level Lorentz equivariance.
"""

# ============================================================
# Imports and optional installation helpers
# ============================================================

import os
import sys
import json
import math
import zipfile
import warnings
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import pinv
from scipy.stats import chi2, f as f_dist

try:
    from sklearn.covariance import MinCovDet
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

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

# ============================================================
# User configuration
# ============================================================

CONFIG = {
    # Primary recommendation: "cms_mod". Fallback option: "qg_jets".
    # If cms_mod download fails and fallback_to_qg is True, qg_jets is used.
    "source": "cms_mod",
    "fallback_to_qg": True,

    # CMS MOD HDF5 loading. IMPORTANT: amount=1 loads one file; amount=1.0 loads the whole dataset.
    "cms_amount": 1,
    "cms_cut": "corr_jet_pts > 400 & abs_jet_eta < 2.4",
    "cms_store_pfcs": False,
    "cms_validate_files": False,

    # qg_jets fallback/loading.
    "qg_num_data": 60000,
    "qg_generator": "pythia",
    "qg_with_bc": False,

    # Analysis size controls.
    "max_jets_analysis": 60000,
    "mcd_fit_max_n": 15000,
    "tyler_fit_max_n": 25000,
    "two_sample_max_per_group": 3500,
    "permutation_B": 999,

    # Reproducibility and outputs.
    "seed": 20260611,
    "outdir": "lm_realdata_outputs",

    # Real-data artificial boost rapidities for parameter-transport verification.
    "boost_etas": [0.0, 0.5, 1.0, 1.5, 2.0],

    # High-score profile settings.
    "top_fraction": 0.01,
    "top_n_print": 25,
}

# ============================================================
# Global constants
# ============================================================

p = 4
G = np.diag([1.0, -1.0, -1.0, -1.0])
EPS = 1e-10
rng = np.random.default_rng(CONFIG["seed"])

OUTDIR = Path(CONFIG["outdir"])
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"
for folder in [OUTDIR, FIGDIR, TABDIR]:
    folder.mkdir(parents=True, exist_ok=True)

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

COLORS = {
    "Classical LM": "#D55E00",
    "MCD robust LM": "#009E73",
    "Tyler shape LM": "#0072B2",
    "Naive Euclidean": "#CC79A7",
    "Minkowski interval": "#A6761D",
}

# ============================================================
# Basic IO helpers
# ============================================================

def show_table(df, title=None, max_rows=30):
    if title:
        print("\n" + "=" * 110)
        print(title)
        print("=" * 110)
    if len(df) > max_rows:
        shown = df.head(max_rows)
        print(f"Showing first {max_rows} of {len(df)} rows.")
    else:
        shown = df
    if IPYTHON_AVAILABLE:
        display(shown)
    else:
        print(shown.to_string(index=False))


def save_dataframe(df, name, index=False):
    csv_path = TABDIR / f"{name}.csv"
    tex_path = TABDIR / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=index, escape=False, float_format="%.6g"))
    print(f"Saved table: {csv_path}")
    print(f"Saved LaTeX: {tex_path}")


def save_figure(fig, name):
    png_path = FIGDIR / f"{name}.png"
    pdf_path = FIGDIR / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def robust_quantile_summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": np.nan, "sd": np.nan, "q05": np.nan, "q25": np.nan, "median": np.nan,
                "q75": np.nan, "q95": np.nan, "min": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(x)),
        "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.quantile(x, 0.50)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }

# ============================================================
# Lorentz geometry helpers
# ============================================================

def lorentz_boost_beta(beta):
    """
    Lorentz boost matrix for arbitrary 3-velocity beta under signature (+---).
    Column convention: x' = B beta x.
    """
    beta = np.asarray(beta, dtype=float)
    b2 = float(np.dot(beta, beta))
    if b2 < 1e-15:
        return np.eye(4)
    if b2 >= 1.0:
        beta = beta / np.sqrt(b2) * (1.0 - 1e-12)
        b2 = float(np.dot(beta, beta))
    gamma = 1.0 / np.sqrt(1.0 - b2)
    B = np.eye(4)
    B[0, 0] = gamma
    B[0, 1:] = gamma * beta
    B[1:, 0] = gamma * beta
    B[1:, 1:] += ((gamma - 1.0) / b2) * np.outer(beta, beta)
    return B


def lorentz_boost_x(eta):
    c, s = np.cosh(eta), np.sinh(eta)
    B = np.eye(4)
    B[0, 0] = c
    B[0, 1] = s
    B[1, 0] = s
    B[1, 1] = c
    return B


def check_lorentz(B, tol=1e-8):
    return float(np.max(np.abs(B.T @ G @ B - G))) < tol


def minkowski_inner_rows(X, Y):
    return np.sum((X @ G) * Y, axis=1)


def normalize_future_timelike(u):
    u = np.asarray(u, dtype=float)
    guu = float(u.T @ G @ u)
    if guu <= 0:
        raise ValueError("Vector is not timelike with positive norm.")
    u = u / np.sqrt(guu)
    if u[0] < 0:
        u = -u
    return u


def future_unit_timelike_eigenvector_from_scatter(M):
    """
    Given an ambient scatter/shape matrix M, use C=M G and extract the future
    unit timelike eigenvector associated with the positive temporal direction.
    """
    M = 0.5 * (M + M.T)
    C = M @ G
    vals, vecs = np.linalg.eig(C)
    vals = np.real_if_close(vals, tol=1000).real
    vecs = np.real_if_close(vecs, tol=1000).real
    gnorms = np.array([vecs[:, j].T @ G @ vecs[:, j] for j in range(4)], dtype=float)
    candidates = np.where(gnorms > 1e-9)[0]
    if len(candidates) == 0:
        idx = int(np.argmax(vals))
    else:
        idx = int(candidates[np.argmax(vals[candidates])])
    u = vecs[:, idx]
    return normalize_future_timelike(u)


def rest_transform_from_u(u):
    """
    Return R such that y = R x maps the estimated timelike direction u to e0.
    If u=(gamma,gamma beta), then R=B(-beta).
    """
    u = normalize_future_timelike(u)
    beta = u[1:] / max(u[0], EPS)
    return lorentz_boost_beta(-beta)


def rapidity_norm_from_u(u):
    u = normalize_future_timelike(u)
    return float(np.arccosh(max(u[0], 1.0)))


def rapidity_error(u1, u2):
    u1 = normalize_future_timelike(u1)
    u2 = normalize_future_timelike(u2)
    val = float(u1.T @ G @ u2)
    val = max(val, 1.0)
    return float(np.arccosh(val))

# ============================================================
# Four-vector construction
# ============================================================

def pt_y_phi_m_to_epxpypz(pt, y, phi, m):
    """
    Convert jet coordinates (pt, rapidity, phi, mass) to four-vector
    (E, px, py, pz), with signature E^2 - |p|^2 = m^2.
    """
    pt = np.asarray(pt, dtype=float)
    y = np.asarray(y, dtype=float)
    phi = np.asarray(phi, dtype=float)
    m = np.asarray(m, dtype=float)
    m = np.maximum(m, 0.0)
    mt = np.sqrt(np.maximum(m * m + pt * pt, 0.0))
    E = mt * np.cosh(y)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = mt * np.sinh(y)
    return np.column_stack([E, px, py, pz])


def particles_pt_y_phi_to_jet_fourvectors(X_particles):
    """
    For qg_jets fallback: massless constituent approximation.
    X has columns (pt, y, phi, pid). Padded zero particles have pt=0.
    """
    pt = X_particles[:, :, 0]
    y = X_particles[:, :, 1]
    phi = X_particles[:, :, 2]
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(y)
    E = pt * np.cosh(y)
    P = np.column_stack([E.sum(axis=1), px.sum(axis=1), py.sum(axis=1), pz.sum(axis=1)])
    mult = (pt > 0).sum(axis=1)
    return P, mult

# ============================================================
# Data loading
# ============================================================

def _get_mod_column(modds, colname):
    """Robustly get a MODDataset jet column by attribute or jets_f_cols."""
    candidates = [
        colname,
        colname + "s",
        colname.replace("jet_", "jet_") + "s",
    ]
    if colname == "corr_jet_pt":
        candidates = ["corr_jet_pts", "corr_jet_pt"] + candidates
    if colname == "jet_pt":
        candidates = ["jet_pts", "jet_pt"] + candidates
    if colname == "jet_y":
        candidates = ["jet_ys", "jet_y"] + candidates
    if colname == "jet_phi":
        candidates = ["jet_phis", "jet_phi"] + candidates
    if colname == "jet_m":
        candidates = ["jet_ms", "jet_m", "m"] + candidates
    if colname == "jet_eta":
        candidates = ["jet_etas", "jet_eta"] + candidates
    if colname == "npv":
        candidates = ["npvs", "npv"] + candidates
    if colname == "quality":
        candidates = ["qualities", "quality"] + candidates

    for cand in candidates:
        if hasattr(modds, cand):
            val = getattr(modds, cand)
            try:
                arr = np.asarray(val)
                if arr.ndim >= 1 and len(arr) == len(modds):
                    return arr
            except Exception:
                pass

    if hasattr(modds, "jets_f_cols") and hasattr(modds, "jets_f"):
        cols = list(modds.jets_f_cols)
        if colname in cols:
            return np.asarray(modds.jets_f[:, cols.index(colname)])
        if colname == "corr_jet_pt":
            # Try pt * jec.
            if "jet_pt" in cols and "jec" in cols:
                return np.asarray(modds.jets_f[:, cols.index("jet_pt")] * modds.jets_f[:, cols.index("jec")])

    if hasattr(modds, "jets_i_cols") and hasattr(modds, "jets_i"):
        cols = list(modds.jets_i_cols)
        if colname in cols:
            return np.asarray(modds.jets_i[:, cols.index(colname)])

    raise KeyError(f"Could not find MODDataset column {colname}.")


def load_cms_mod_dataset(config):
    import energyflow as ef

    print("Loading CMS Open Data jets through EnergyFlow MOD interface...")
    print("Important: amount=1 means one file; amount=1.0 means full dataset.")

    modds = ef.mod.load(
        config["cms_cut"],
        amount=config["cms_amount"],
        dataset="cms",
        collection="CMS2011AJets",
        validate_files=config["cms_validate_files"],
        store_pfcs=config["cms_store_pfcs"],
        store_gens=False,
        verbose=1,
    )

    n = len(modds)
    print(f"Loaded MODDataset with {n} jets after cuts.")

    pt = _get_mod_column(modds, "corr_jet_pt")
    y = _get_mod_column(modds, "jet_y")
    phi = _get_mod_column(modds, "jet_phi")
    m = _get_mod_column(modds, "jet_m")

    df = pd.DataFrame({
        "pt": pt,
        "rapidity": y,
        "phi": phi,
        "mass": m,
    })

    optional_cols = {
        "eta": "jet_eta",
        "npv": "npv",
        "quality": "quality",
    }
    for outname, col in optional_cols.items():
        try:
            df[outname] = _get_mod_column(modds, col)
        except Exception:
            pass

    P = pt_y_phi_m_to_epxpypz(df["pt"].values, df["rapidity"].values, df["phi"].values, df["mass"].values)
    df["E"] = P[:, 0]
    df["px"] = P[:, 1]
    df["py"] = P[:, 2]
    df["pz"] = P[:, 3]
    df["source_label"] = "CMS2011AJets_CMS"
    df["dataset_source"] = "cms_mod"

    return df


def load_qg_jets_dataset(config):
    from energyflow.datasets import qg_jets

    print("Loading EnergyFlow qg_jets fallback/benchmark dataset...")
    X, y_label = qg_jets.load(
        num_data=config["qg_num_data"],
        pad=True,
        ncol=4,
        generator=config["qg_generator"],
        with_bc=config["qg_with_bc"],
    )

    P, mult = particles_pt_y_phi_to_jet_fourvectors(X)
    pt = np.sqrt(P[:, 1] ** 2 + P[:, 2] ** 2)
    mass2 = minkowski_inner_rows(P, P)
    mass = np.sqrt(np.maximum(mass2, 0.0))
    rapidity = 0.5 * np.log(np.maximum((P[:, 0] + P[:, 3]) / np.maximum(P[:, 0] - P[:, 3], EPS), EPS))
    phi = np.arctan2(P[:, 2], P[:, 1])

    df = pd.DataFrame({
        "pt": pt,
        "rapidity": rapidity,
        "phi": phi,
        "mass": mass,
        "multiplicity": mult,
        "qg_label": y_label.astype(int),
        "E": P[:, 0],
        "px": P[:, 1],
        "py": P[:, 2],
        "pz": P[:, 3],
        "source_label": np.where(y_label == 1, "quark", "gluon"),
        "dataset_source": "qg_jets",
    })
    return df


def load_dataset(config=CONFIG):
    actual_source = None
    try:
        if config["source"] == "cms_mod":
            df = load_cms_mod_dataset(config)
            actual_source = "cms_mod"
        elif config["source"] == "qg_jets":
            df = load_qg_jets_dataset(config)
            actual_source = "qg_jets"
        else:
            raise ValueError("Unknown source. Use 'cms_mod' or 'qg_jets'.")
    except Exception as e:
        if config.get("fallback_to_qg", False):
            warnings.warn(f"Primary source failed: {repr(e)}. Falling back to qg_jets.")
            df = load_qg_jets_dataset(config)
            actual_source = "qg_jets_fallback"
        else:
            raise

    # Clean finite, physically valid rows.
    numeric_cols = ["pt", "rapidity", "phi", "mass", "E", "px", "py", "pz"]
    mask = np.ones(len(df), dtype=bool)
    for col in numeric_cols:
        mask &= np.isfinite(df[col].values)
    mask &= df["pt"].values > 0
    mask &= df["E"].values > 0
    df = df.loc[mask].reset_index(drop=True)

    if len(df) > config["max_jets_analysis"]:
        df = df.sample(config["max_jets_analysis"], random_state=config["seed"]).reset_index(drop=True)

    df.insert(0, "jet_index", np.arange(len(df)))
    df["abs_rapidity"] = np.abs(df["rapidity"])
    df["m2_from_fourvector"] = minkowski_inner_rows(df[["E", "px", "py", "pz"]].values,
                                                    df[["E", "px", "py", "pz"]].values)
    df["log_pt"] = np.log(np.maximum(df["pt"], EPS))
    df["log_mass_plus1"] = np.log1p(np.maximum(df["mass"], 0.0))

    return df, actual_source

# ============================================================
# Lorentz--Mahalanobis model fitting/scoring
# ============================================================

def sample_for_fit(X, max_n, seed):
    if len(X) <= max_n:
        return X
    local_rng = np.random.default_rng(seed)
    idx = local_rng.choice(len(X), size=max_n, replace=False)
    return X[idx]


def fit_classical_scatter(X):
    mu = X.mean(axis=0)
    Z = X - mu
    M = (Z.T @ Z) / len(X)
    M = 0.5 * (M + M.T) + 1e-8 * np.eye(4)
    return mu, M


def fit_mcd_scatter(X, max_n, seed):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn MinCovDet unavailable.")
    X_fit = sample_for_fit(X, max_n=max_n, seed=seed)
    mcd = MinCovDet(random_state=seed, support_fraction=0.75).fit(X_fit)
    mu = mcd.location_
    M = 0.5 * (mcd.covariance_ + mcd.covariance_.T) + 1e-8 * np.eye(4)
    return mu, M


def spatial_median(X, max_iter=500, tol=1e-7):
    m = np.median(X, axis=0)
    for _ in range(max_iter):
        R = X - m
        dist = np.linalg.norm(R, axis=1)
        dist = np.maximum(dist, 1e-12)
        w = 1.0 / dist
        m_new = np.sum(X * w[:, None], axis=0) / np.sum(w)
        if np.linalg.norm(m_new - m) < tol * (1 + np.linalg.norm(m)):
            return m_new
        m = m_new
    return m


def tyler_shape(X_centered, max_iter=800, tol=1e-7):
    n, p_local = X_centered.shape
    S = np.cov(X_centered, rowvar=False, bias=True)
    S = S + 1e-6 * np.eye(p_local)
    S = S * (p_local / np.trace(S))
    for _ in range(max_iter):
        Sinv = pinv(S)
        q = np.sum((X_centered @ Sinv) * X_centered, axis=1)
        q = np.maximum(q, 1e-12)
        S_new = (p_local / n) * ((X_centered / q[:, None]).T @ X_centered)
        S_new = 0.5 * (S_new + S_new.T)
        S_new = S_new * (p_local / np.trace(S_new))
        rel = np.linalg.norm(S_new - S, ord="fro") / (1 + np.linalg.norm(S, ord="fro"))
        S = S_new
        if rel < tol:
            break
    return S


def fit_tyler_scatter(X, max_n, seed):
    X_fit = sample_for_fit(X, max_n=max_n, seed=seed)
    mu = spatial_median(X_fit)
    Z = X_fit - mu
    M = tyler_shape(Z)
    M = 0.5 * (M + M.T) + 1e-8 * np.eye(4)
    return mu, M


def fit_lm_model(X, method="classical", config=CONFIG):
    if method == "classical":
        mu, M = fit_classical_scatter(X)
        label = "Classical LM"
    elif method == "mcd":
        mu, M = fit_mcd_scatter(X, max_n=config["mcd_fit_max_n"], seed=config["seed"])
        label = "MCD robust LM"
    elif method == "tyler":
        mu, M = fit_tyler_scatter(X, max_n=config["tyler_fit_max_n"], seed=config["seed"])
        label = "Tyler shape LM"
    else:
        raise ValueError("method must be 'classical', 'mcd', or 'tyler'.")

    u = future_unit_timelike_eigenvector_from_scatter(M)
    R = rest_transform_from_u(u)
    M_rest = R @ M @ R.T
    M_rest = 0.5 * (M_rest + M_rest.T)

    sigma_tau2 = float(max(M_rest[0, 0], 1e-10))
    Sigma_s = M_rest[1:, 1:]
    Sigma_s = 0.5 * (Sigma_s + Sigma_s.T)
    # Ensure numerical positive definiteness.
    eigmin = float(np.linalg.eigvalsh(Sigma_s).min())
    if eigmin <= 1e-8:
        Sigma_s = Sigma_s + (1e-8 - eigmin + 1e-8) * np.eye(3)

    beta = u[1:] / max(u[0], EPS)
    return {
        "label": label,
        "method": method,
        "mu": mu,
        "M": M,
        "u": u,
        "beta": beta,
        "R_to_rest": R,
        "M_rest": M_rest,
        "sigma_tau2": sigma_tau2,
        "Sigma_s": Sigma_s,
        "Sigma_s_inv": pinv(Sigma_s),
        "rapidity_norm": rapidity_norm_from_u(u),
    }


def score_lm_model(X, model, return_components=False):
    Z = X - model["mu"]
    Y = Z @ model["R_to_rest"].T
    tau = Y[:, 0]
    S = Y[:, 1:]
    temporal = tau ** 2 / model["sigma_tau2"]
    spatial = np.sum((S @ model["Sigma_s_inv"]) * S, axis=1)
    score = temporal + spatial
    if return_components:
        return score, temporal, spatial, Y
    return score


def score_naive_euclidean(X):
    mu = X.mean(axis=0)
    Sdiag = np.var(X - mu, axis=0) + 1e-8
    Z = X - mu
    return np.sum((Z ** 2) / Sdiag[None, :], axis=1)


def score_minkowski_interval_centered(X):
    mu = X.mean(axis=0)
    Z = X - mu
    return np.abs(minkowski_inner_rows(Z, Z))


def transport_model(model, B):
    """Transport fitted parameters under x' = B x."""
    mu_p = B @ model["mu"]
    M_p = B @ model["M"] @ B.T
    u_p = B @ model["u"]
    u_p = normalize_future_timelike(u_p)
    R_p = rest_transform_from_u(u_p)
    M_rest_p = R_p @ M_p @ R_p.T
    M_rest_p = 0.5 * (M_rest_p + M_rest_p.T)
    sigma_tau2 = float(max(M_rest_p[0, 0], 1e-10))
    Sigma_s = M_rest_p[1:, 1:]
    Sigma_s = 0.5 * (Sigma_s + Sigma_s.T)
    eigmin = float(np.linalg.eigvalsh(Sigma_s).min())
    if eigmin <= 1e-8:
        Sigma_s = Sigma_s + (1e-8 - eigmin + 1e-8) * np.eye(3)
    return {
        **model,
        "mu": mu_p,
        "M": M_p,
        "u": u_p,
        "beta": u_p[1:] / max(u_p[0], EPS),
        "R_to_rest": R_p,
        "M_rest": M_rest_p,
        "sigma_tau2": sigma_tau2,
        "Sigma_s": Sigma_s,
        "Sigma_s_inv": pinv(Sigma_s),
        "rapidity_norm": rapidity_norm_from_u(u_p),
    }

# ============================================================
# Analyses
# ============================================================

def build_dataset_summary(df, actual_source):
    rows = []
    for col in ["pt", "rapidity", "abs_rapidity", "mass", "E", "m2_from_fourvector"]:
        stats = robust_quantile_summary(df[col].values)
        stats.update({"quantity": col})
        rows.append(stats)
    summary = pd.DataFrame(rows)
    meta = pd.DataFrame([{
        "actual_source": actual_source,
        "n_jets_used": len(df),
        "pt_min": float(df["pt"].min()),
        "pt_max": float(df["pt"].max()),
        "rapidity_min": float(df["rapidity"].min()),
        "rapidity_max": float(df["rapidity"].max()),
        "mass_min": float(df["mass"].min()),
        "mass_max": float(df["mass"].max()),
    }])
    return summary, meta


def build_model_tables(models):
    rows = []
    for key, model in models.items():
        u = model["u"]
        beta = model["beta"]
        eig_spatial = np.linalg.eigvalsh(model["Sigma_s"])
        rows.append({
            "method": model["label"],
            "u0": u[0],
            "u1": u[1],
            "u2": u[2],
            "u3": u[3],
            "beta_norm": float(np.linalg.norm(beta)),
            "rapidity_norm": model["rapidity_norm"],
            "sigma_tau2": model["sigma_tau2"],
            "spatial_eig_min": float(eig_spatial.min()),
            "spatial_eig_max": float(eig_spatial.max()),
            "spatial_condition": float(eig_spatial.max() / max(eig_spatial.min(), EPS)),
        })
    return pd.DataFrame(rows)


def build_score_tables(df, scores):
    rows = []
    for name, sc in scores.items():
        stats = robust_quantile_summary(sc)
        stats.update({"score": name})
        stats["q99"] = float(np.quantile(sc, 0.99))
        stats["q995"] = float(np.quantile(sc, 0.995))
        rows.append(stats)
    return pd.DataFrame(rows)


def top_outlier_table(df, scores, score_name="MCD robust LM", top_n=25):
    sc = scores[score_name]
    idx = np.argsort(sc)[::-1][:top_n]
    cols = ["jet_index", "pt", "rapidity", "phi", "mass", "E", "m2_from_fourvector"]
    for optional in ["npv", "quality", "multiplicity", "qg_label"]:
        if optional in df.columns:
            cols.append(optional)
    out = df.iloc[idx][cols].copy()
    for name, vals in scores.items():
        out[name] = vals[idx]
    return out.reset_index(drop=True)


def high_score_profile(df, score, score_name="MCD robust LM", frac=0.01):
    cutoff = float(np.quantile(score, 1 - frac))
    flag = score >= cutoff
    quantities = ["pt", "abs_rapidity", "mass", "E"]
    if "multiplicity" in df.columns:
        quantities.append("multiplicity")
    if "npv" in df.columns:
        quantities.append("npv")
    rows = []
    for q in quantities:
        full = robust_quantile_summary(df[q].values)
        top = robust_quantile_summary(df.loc[flag, q].values)
        rows.append({
            "quantity": q,
            "full_median": full["median"],
            "top1pct_median": top["median"],
            "full_q95": full["q95"],
            "top1pct_q95": top["q95"],
            "median_ratio_top_to_full": top["median"] / max(abs(full["median"]), EPS),
        })
    return pd.DataFrame(rows), cutoff


def boost_equivariance_realdata(X, models, etas):
    rows = []
    # Naive baseline computed in original coordinates.
    naive0 = score_naive_euclidean(X)
    interval0 = score_minkowski_interval_centered(X)

    for eta in etas:
        B = lorentz_boost_x(float(eta))
        Xb = X @ B.T
        for key, model in models.items():
            sc0 = score_lm_model(X, model)
            model_b = transport_model(model, B)
            scb = score_lm_model(Xb, model_b)
            diff = np.abs(scb - sc0)
            rows.append({
                "eta": eta,
                "score": model["label"],
                "mean_abs_change": float(np.mean(diff)),
                "median_abs_change": float(np.median(diff)),
                "q95_abs_change": float(np.quantile(diff, 0.95)),
                "max_abs_change": float(np.max(diff)),
            })

        # Naive Euclidean recalculated in boosted coordinate axes.
        naive_b = score_naive_euclidean(Xb)
        diff_n = np.abs(naive_b - naive0)
        rows.append({
            "eta": eta,
            "score": "Naive Euclidean",
            "mean_abs_change": float(np.mean(diff_n)),
            "median_abs_change": float(np.median(diff_n)),
            "q95_abs_change": float(np.quantile(diff_n, 0.95)),
            "max_abs_change": float(np.max(diff_n)),
        })

        interval_b = score_minkowski_interval_centered(Xb)
        diff_i = np.abs(interval_b - interval0)
        rows.append({
            "eta": eta,
            "score": "Minkowski interval",
            "mean_abs_change": float(np.mean(diff_i)),
            "median_abs_change": float(np.median(diff_i)),
            "q95_abs_change": float(np.quantile(diff_i, 0.95)),
            "max_abs_change": float(np.max(diff_i)),
        })
    return pd.DataFrame(rows)


def hotelling_two_sample(Y, group, B=999, seed=123):
    """Permutation-calibrated two-sample Hotelling T^2 in adapted coordinates."""
    local_rng = np.random.default_rng(seed)
    group = np.asarray(group).astype(int)
    Y0 = Y[group == 0]
    Y1 = Y[group == 1]
    n0, n1 = len(Y0), len(Y1)
    p_local = Y.shape[1]

    def T2_stat(Y0_, Y1_):
        n0_, n1_ = len(Y0_), len(Y1_)
        mean0 = Y0_.mean(axis=0)
        mean1 = Y1_.mean(axis=0)
        S0 = np.cov(Y0_, rowvar=False, bias=False)
        S1 = np.cov(Y1_, rowvar=False, bias=False)
        Sp = ((n0_ - 1) * S0 + (n1_ - 1) * S1) / max(n0_ + n1_ - 2, 1)
        D = mean0 - mean1
        return float(D @ pinv((1.0 / n0_ + 1.0 / n1_) * Sp) @ D), float(D @ pinv(Sp) @ D)

    T_obs, lm_mean_sep = T2_stat(Y0, Y1)
    F_stat = ((n0 + n1 - p_local - 1) / (p_local * (n0 + n1 - 2))) * T_obs
    p_f = float(1.0 - f_dist.cdf(F_stat, p_local, n0 + n1 - p_local - 1))

    T_perm = np.empty(B)
    idx = np.arange(len(Y))
    n_zero = int(np.sum(group == 0))
    for b in range(B):
        perm = local_rng.permutation(idx)
        gperm = np.zeros(len(Y), dtype=int)
        gperm[perm[n_zero:]] = 1
        T_perm[b], _ = T2_stat(Y[gperm == 0], Y[gperm == 1])
    p_perm = float((1.0 + np.sum(T_perm >= T_obs)) / (B + 1.0))
    return {
        "n_group0": n0,
        "n_group1": n1,
        "T2": T_obs,
        "F_statistic": F_stat,
        "F_approx_pvalue": p_f,
        "permutation_pvalue": p_perm,
        "LM_mean_separation": lm_mean_sep,
        "T_perm": T_perm,
    }


def two_sample_analysis(df, model, config):
    X = df[["E", "px", "py", "pz"]].values
    _, _, _, Y = score_lm_model(X, model, return_components=True)
    # central vs forward split by median absolute rapidity.
    abs_y = df["abs_rapidity"].values
    cutoff = float(np.median(abs_y))
    group_all = (abs_y > cutoff).astype(int)  # 0 central, 1 forward

    # Balanced subsample to keep permutation analysis fast and interpretable.
    idx0 = np.where(group_all == 0)[0]
    idx1 = np.where(group_all == 1)[0]
    m = min(len(idx0), len(idx1), config["two_sample_max_per_group"])
    idx0 = rng.choice(idx0, size=m, replace=False)
    idx1 = rng.choice(idx1, size=m, replace=False)
    idx = np.r_[idx0, idx1]
    group = np.r_[np.zeros(m, dtype=int), np.ones(m, dtype=int)]
    Y_sub = Y[idx]

    res = hotelling_two_sample(Y_sub, group, B=config["permutation_B"], seed=config["seed"])
    table = pd.DataFrame([{
        "comparison": "central vs forward jets",
        "group0": f"|rapidity| <= median ({cutoff:.4g})",
        "group1": f"|rapidity| > median ({cutoff:.4g})",
        "n_group0": res["n_group0"],
        "n_group1": res["n_group1"],
        "T2": res["T2"],
        "F_statistic": res["F_statistic"],
        "F_approx_pvalue": res["F_approx_pvalue"],
        "permutation_pvalue": res["permutation_pvalue"],
        "LM_mean_separation": res["LM_mean_separation"],
    }])
    return table, res

# ============================================================
# Plotting
# ============================================================

def plot_kinematic_overview(df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    data_specs = [
        ("pt", "Jet transverse momentum", "pT"),
        ("rapidity", "Jet rapidity", "y"),
        ("mass", "Jet mass", "m"),
        ("m2_from_fourvector", r"Four-vector interval $p^\mu p_\mu$", r"$m^2$ check"),
    ]
    for ax, (col, title, xlabel) in zip(axes, data_specs):
        x = df[col].values
        x = x[np.isfinite(x)]
        ax.hist(x, bins=70, color="#56B4E9", alpha=0.82, edgecolor="white", linewidth=0.2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
    save_figure(fig, "fig01_realdata_kinematic_overview")
    plt.show()


def plot_score_histograms(scores):
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for name, sc in scores.items():
        x = np.asarray(sc, dtype=float)
        upper = np.quantile(x[np.isfinite(x)], 0.995)
        bins = np.linspace(0, upper, 100)
        ax.hist(x, bins=bins, density=True, histtype="step", linewidth=2.0,
                label=name, color=COLORS.get(name, None))
    xs = np.linspace(0, chi2.ppf(0.995, 4), 400)
    ax.plot(xs, chi2.pdf(xs, df=4), "k--", linewidth=1.5, label=r"$\chi^2_4$ reference")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title("Real-data LM score distributions and Gaussian reference")
    ax.legend(frameon=True)
    save_figure(fig, "fig02_realdata_score_histograms")
    plt.show()


def plot_decomposition(df, score, temporal, spatial):
    qbins = pd.qcut(score, q=10, labels=False, duplicates="drop")
    tmp = pd.DataFrame({"decile": qbins, "temporal": temporal, "spatial": spatial})
    tab = tmp.groupby("decile", as_index=False).mean()
    tab["decile"] = tab["decile"].astype(int) + 1
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.bar(tab["decile"], tab["temporal"], label="temporal contribution", color="#56B4E9")
    ax.bar(tab["decile"], tab["spatial"], bottom=tab["temporal"], label="spatial contribution", color="#E69F00")
    ax.set_xlabel("LM score decile")
    ax.set_ylabel("mean contribution")
    ax.set_title("Temporal--spatial decomposition of robust LM scores")
    ax.legend(frameon=True)
    save_figure(fig, "fig03_realdata_temporal_spatial_decomposition")
    plt.show()
    return tab


def plot_lm_vs_kinematics(df, robust_score):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    q = np.quantile(robust_score, 0.995)
    cvals = np.minimum(robust_score, q)
    specs = [
        ("pt", "mass", "pT", "mass"),
        ("abs_rapidity", "mass", "|rapidity|", "mass"),
        ("pt", "abs_rapidity", "pT", "|rapidity|"),
    ]
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, specs):
        sc = ax.scatter(df[xcol], df[ycol], c=cvals, s=6, alpha=0.45, cmap="viridis")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("robust LM score, clipped at 99.5% quantile")
    fig.suptitle("Real jets colored by robust Lorentz--Mahalanobis score", y=1.03)
    save_figure(fig, "fig04_realdata_lm_score_vs_kinematics")
    plt.show()


def plot_robust_vs_classical(scores):
    x = scores["Classical LM"]
    y = scores["MCD robust LM"] if "MCD robust LM" in scores else scores["Tyler shape LM"]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    upper_x = np.quantile(x, 0.995)
    upper_y = np.quantile(y, 0.995)
    ax.scatter(np.minimum(x, upper_x), np.minimum(y, upper_y), s=7, alpha=0.35, color="#0072B2")
    lim = max(upper_x, upper_y)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2)
    ax.set_xlabel("classical LM score, clipped")
    ax.set_ylabel("robust LM score, clipped")
    ax.set_title("Classical versus robust LM event rankings")
    save_figure(fig, "fig05_realdata_robust_vs_classical")
    plt.show()


def plot_boost_equivariance(eqtab):
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for name, gdf in eqtab.groupby("score"):
        gdf = gdf.sort_values("eta")
        ax.plot(gdf["eta"], gdf["mean_abs_change"], marker="o", linewidth=2.0,
                color=COLORS.get(name, None), label=name)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel(r"artificial boost rapidity $\eta$")
    ax.set_ylabel("mean absolute score change")
    ax.set_title("Parameter-transport equivariance on the observed jet sample")
    ax.legend(frameon=True)
    save_figure(fig, "fig06_realdata_boost_equivariance")
    plt.show()


def plot_two_sample_permutation(res):
    T_perm = res["T_perm"]
    T_obs = res["T2"]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.hist(T_perm, bins=60, color="#56B4E9", alpha=0.82, edgecolor="white", linewidth=0.2)
    ax.axvline(T_obs, color="#D55E00", linewidth=2.4, label="observed T^2")
    ax.set_xlabel(r"permuted Hotelling $T^2$")
    ax.set_ylabel("count")
    ax.set_title("Permutation calibration: central vs forward jets")
    ax.legend(frameon=True)
    save_figure(fig, "fig07_realdata_two_sample_permutation")
    plt.show()


def plot_top_outlier_map(df, robust_score):
    cutoff = np.quantile(robust_score, 0.99)
    flag = robust_score >= cutoff
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(df["rapidity"], df["phi"], s=4, alpha=0.25, color="#999999", label="all jets")
    ax.scatter(df.loc[flag, "rapidity"], df.loc[flag, "phi"], s=12, alpha=0.85,
               color="#D55E00", label="top 1% robust LM")
    ax.set_xlabel("rapidity")
    ax.set_ylabel(r"azimuth $\phi$")
    ax.set_title("Location of high robust-LM jets in rapidity--azimuth space")
    ax.legend(frameon=True)
    save_figure(fig, "fig08_realdata_top_outlier_map")
    plt.show()

# ============================================================
# Main workflow
# ============================================================

def run_realdata_analysis(config=CONFIG, make_zip=True):
    print("Starting Lorentz--Mahalanobis real-data methodology analysis.")
    print("This is not a simulation verification; it is a real/physics dataset workflow.")
    print("Configuration:")
    print(json.dumps(config, indent=2))

    # 1. Load data.
    df, actual_source = load_dataset(config)
    X = df[["E", "px", "py", "pz"]].values

    summary, meta = build_dataset_summary(df, actual_source)
    save_dataframe(meta, "table01_dataset_meta")
    save_dataframe(summary, "table02_dataset_summary")
    show_table(meta, "Dataset metadata")
    show_table(summary, "Dataset kinematic summary")
    plot_kinematic_overview(df)

    # 2. Fit LM models.
    print("\nFitting Lorentz--Mahalanobis models...")
    models = {}
    models["classical"] = fit_lm_model(X, method="classical", config=config)
    if SKLEARN_AVAILABLE:
        models["mcd"] = fit_lm_model(X, method="mcd", config=config)
    else:
        warnings.warn("MCD robust fit skipped because scikit-learn is unavailable.")
    models["tyler"] = fit_lm_model(X, method="tyler", config=config)

    model_table = build_model_tables(models)
    save_dataframe(model_table, "table03_fitted_rest_frames")
    show_table(model_table, "Fitted rest-frame and scatter summaries")

    # 3. Compute scores.
    scores = {}
    components = {}
    coords = {}
    for key, model in models.items():
        sc, temp, spat, Y = score_lm_model(X, model, return_components=True)
        scores[model["label"]] = sc
        components[model["label"]] = {"temporal": temp, "spatial": spat}
        coords[model["label"]] = Y

    scores["Naive Euclidean"] = score_naive_euclidean(X)
    scores["Minkowski interval"] = score_minkowski_interval_centered(X)

    score_table = build_score_tables(df, scores)
    save_dataframe(score_table, "table04_score_quantiles")
    show_table(score_table, "Score distribution summaries")

    plot_score_histograms(scores)

    robust_label = "MCD robust LM" if "MCD robust LM" in scores else "Tyler shape LM"
    decile_tab = plot_decomposition(
        df,
        scores[robust_label],
        components[robust_label]["temporal"],
        components[robust_label]["spatial"],
    )
    save_dataframe(decile_tab, "table05_temporal_spatial_deciles")
    show_table(decile_tab, "Temporal--spatial score contribution by robust LM decile")

    plot_lm_vs_kinematics(df, scores[robust_label])
    plot_robust_vs_classical(scores)

    # 4. Outlier ranking and profile.
    top_tab = top_outlier_table(df, scores, score_name=robust_label, top_n=config["top_n_print"])
    save_dataframe(top_tab, "table06_top_robust_lm_jets")
    show_table(top_tab, f"Top {config['top_n_print']} jets by {robust_label}", max_rows=config["top_n_print"])

    profile_tab, top_cutoff = high_score_profile(
        df,
        scores[robust_label],
        score_name=robust_label,
        frac=config["top_fraction"],
    )
    profile_tab.insert(0, "score_used", robust_label)
    profile_tab.insert(1, "top_fraction", config["top_fraction"])
    profile_tab.insert(2, "score_cutoff", top_cutoff)
    save_dataframe(profile_tab, "table07_high_score_profile")
    show_table(profile_tab, "High-score jet profile: top fraction versus full sample")
    plot_top_outlier_map(df, scores[robust_label])

    # 5. Real-data boost equivariance check with transported parameters.
    print("\nRunning real-data parameter-transport equivariance check...")
    eqtab = boost_equivariance_realdata(X, models, config["boost_etas"])
    save_dataframe(eqtab, "table08_boost_equivariance_realdata")
    show_table(eqtab, "Real-data boost equivariance table")
    plot_boost_equivariance(eqtab)

    # 6. Real-data two-sample test: central vs forward jets.
    print("\nRunning real-data central-vs-forward two-sample LM test...")
    two_tab, two_res = two_sample_analysis(df, models["mcd"] if "mcd" in models else models["tyler"], config)
    save_dataframe(two_tab, "table09_two_sample_central_forward")
    show_table(two_tab, "Two-sample LM test: central versus forward jets")
    plot_two_sample_permutation(two_res)

    # 7. Save analysis metadata.
    metadata = {
        "actual_source": actual_source,
        "n_jets_used": int(len(df)),
        "config": config,
        "sklearn_available": SKLEARN_AVAILABLE,
        "robust_label_used_for_outlier_profile": robust_label,
        "output_directory": str(OUTDIR),
        "note": "If actual_source is qg_jets_fallback, do not describe the output as CMS Open Data.",
    }
    with open(OUTDIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    readme = f"""
Lorentz--Mahalanobis real-data methodology outputs
=================================================

Actual source used: {actual_source}
Number of jets used: {len(df)}
Robust score used for outlier profile: {robust_label}

Main figures:
  fig01_realdata_kinematic_overview.pdf
  fig02_realdata_score_histograms.pdf
  fig03_realdata_temporal_spatial_decomposition.pdf
  fig04_realdata_lm_score_vs_kinematics.pdf
  fig05_realdata_robust_vs_classical.pdf
  fig06_realdata_boost_equivariance.pdf
  fig07_realdata_two_sample_permutation.pdf
  fig08_realdata_top_outlier_map.pdf

Main tables:
  table01_dataset_meta.csv/.tex
  table02_dataset_summary.csv/.tex
  table03_fitted_rest_frames.csv/.tex
  table04_score_quantiles.csv/.tex
  table05_temporal_spatial_deciles.csv/.tex
  table06_top_robust_lm_jets.csv/.tex
  table07_high_score_profile.csv/.tex
  table08_boost_equivariance_realdata.csv/.tex
  table09_two_sample_central_forward.csv/.tex

Scientific interpretation:
  This workflow demonstrates the proposed method on actual physics four-vector data.
  It does not repeat the Monte Carlo simulation verification. It studies real-data
  scoring, high-score event profiles, robust-vs-classical rankings, transported
  parameter equivariance, and a central-vs-forward real-data two-sample test.

Warning:
  If actual_source is qg_jets_fallback, the code fell back to a labelled generated
  physics benchmark. Do not describe that run as CMS Open Data.
"""
    with open(OUTDIR / "README_outputs.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    # 8. Zip outputs.
    zip_path = Path("lm_realdata_outputs.zip")
    if make_zip:
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in OUTDIR.rglob("*"):
                zf.write(file, arcname=str(file.relative_to(OUTDIR.parent)))
        print(f"\nCreated ZIP archive: {zip_path.resolve()}")
        if COLAB_AVAILABLE:
            print("Attempting Colab download...")
            files.download(str(zip_path))

    return {
        "df": df,
        "models": models,
        "scores": scores,
        "dataset_meta": meta,
        "dataset_summary": summary,
        "model_table": model_table,
        "score_table": score_table,
        "top_outliers": top_tab,
        "high_score_profile": profile_tab,
        "boost_equivariance": eqtab,
        "two_sample_table": two_tab,
        "two_sample_result": two_res,
        "zip_path": str(zip_path),
    }

# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    results = run_realdata_analysis(CONFIG, make_zip=True)
