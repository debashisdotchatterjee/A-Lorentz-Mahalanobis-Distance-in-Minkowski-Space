# ============================================================
# Advanced Lorentz--Mahalanobis Simulation Suite
# Colab-ready: figures, tables, LaTeX tables, and ZIP output
#
# How to use in Colab:
#   1. Upload this file or open the accompanying .ipynb notebook.
#   2. Run:
#          results = run_all_simulations(quick=True, make_zip=True)
#      to test.
#   3. For final manuscript-quality output, run:
#          results = run_all_simulations(quick=False, make_zip=True)
#
# Output folder:
#   lm_advanced_simulation_outputs/
# Output ZIP:
#   lm_advanced_simulation_outputs.zip
# ============================================================

import os
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, norm
from scipy.linalg import pinv

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
# Global settings
# ============================================================

SEED = 20260611
rng = np.random.default_rng(SEED)

OUTDIR = Path("lm_advanced_simulation_outputs")
FIGDIR = OUTDIR / "figures"
TABDIR = OUTDIR / "tables"

for folder in [OUTDIR, FIGDIR, TABDIR]:
    folder.mkdir(parents=True, exist_ok=True)

p = 4
d = p - 1

# Lorentz metric with signature (+---)
G = np.diag([1.0, -1.0, -1.0, -1.0])

# Rest-frame covariance/shape.
# The anisotropy is deliberate: it makes covariance-standardization nontrivial.
Theta = np.diag([1.00, 1.40, 0.70, 0.35])
Theta_inv = np.linalg.inv(Theta)

ALPHA = 0.05
CHI2_CUTOFF = chi2.ppf(1 - ALPHA, df=p)

COLORS = {
    "LM": "#0072B2",
    "Euclidean oracle": "#009E73",
    "Euclidean naive": "#D55E00",
    "Minkowski interval": "#CC79A7",
    "Classical covariance": "#D55E00",
    "MCD robust": "#009E73",
    "Tyler shape": "#0072B2",
    "LM-LDA": "#0072B2",
    "Full Euclidean LDA": "#009E73",
    "Diagonal Euclidean LDA": "#D55E00",
    "Minkowski nearest-centroid": "#CC79A7",
}

MARKERS = {
    "LM": "o",
    "Euclidean oracle": "s",
    "Euclidean naive": "^",
    "Minkowski interval": "D",
    "Classical covariance": "o",
    "MCD robust": "s",
    "Tyler shape": "^",
    "LM-LDA": "o",
    "Full Euclidean LDA": "s",
    "Diagonal Euclidean LDA": "^",
    "Minkowski nearest-centroid": "D",
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
    "axes.facecolor": "white",
    "figure.facecolor": "white",
})


# ============================================================
# Basic Lorentz utilities
# ============================================================

def lorentz_boost_x(eta, p=4):
    """
    Proper orthochronous Lorentz boost in the first spatial direction
    under signature (+---).

    Column convention:
        t'  = cosh(eta) t + sinh(eta) x
        x'  = sinh(eta) t + cosh(eta) x

    For row arrays, use X @ Lambda.T.
    """
    L = np.eye(p)
    c, s = np.cosh(eta), np.sinh(eta)
    L[0, 0] = c
    L[0, 1] = s
    L[1, 0] = s
    L[1, 1] = c
    return L


def check_lorentz(L, tol=1e-9):
    """Check Lambda^T G Lambda = G."""
    return np.max(np.abs(L.T @ G @ L - G)) < tol


def minkowski_inner_rows(X, Y):
    """Row-wise Minkowski inner product x^T G y."""
    return np.sum((X @ G) * Y, axis=1)


def simulate_rest_gaussian(n, mean=None, covariance=None, rng=None):
    """Generate rest-frame Lorentz--Gaussian coordinates Y=(tau,s)."""
    if rng is None:
        rng = np.random.default_rng()
    if mean is None:
        mean = np.zeros(p)
    if covariance is None:
        covariance = Theta
    return rng.multivariate_normal(mean, covariance, size=n)


def transform_rest_to_observed(Y_rest, eta, mu_rest=None):
    """
    Generate boosted observed coordinates X_obs = Lambda(Y_rest + mu_rest).
    """
    if mu_rest is None:
        mu_rest = np.zeros(p)
    L = lorentz_boost_x(eta, p=p)
    X_obs = (Y_rest + mu_rest) @ L.T
    mu_obs = L @ mu_rest
    return X_obs, L, mu_obs


def transform_observed_to_rest(X_obs, eta, mu_rest=None):
    """Known-frame inverse boost back to the rest coordinates."""
    if mu_rest is None:
        mu_rest = np.zeros(p)
    L = lorentz_boost_x(eta, p=p)
    mu_obs = L @ mu_rest
    return (X_obs - mu_obs) @ np.linalg.inv(L).T


# ============================================================
# Competing scores
# ============================================================

def score_lm_known(X_obs, eta, mu_rest=None):
    """
    Lorentz--Mahalanobis score using the correct u-adapted frame.
    """
    Y = transform_observed_to_rest(X_obs, eta, mu_rest=mu_rest)
    return np.sum((Y @ Theta_inv) * Y, axis=1)


def score_euclidean_oracle(X_obs, eta, mu_rest=None):
    """
    Ordinary Euclidean Mahalanobis score using the correctly transformed
    covariance. This is an affine-equivariant oracle benchmark.
    """
    if mu_rest is None:
        mu_rest = np.zeros(p)
    L = lorentz_boost_x(eta, p=p)
    mu_obs = L @ mu_rest
    Cov_obs = L @ Theta @ L.T
    Inv_obs = np.linalg.inv(Cov_obs)
    Z = X_obs - mu_obs
    return np.sum((Z @ Inv_obs) * Z, axis=1)


def score_euclidean_naive(X_obs, eta, mu_rest=None):
    """
    Naive coordinate Euclidean score. It incorrectly treats the boosted axes
    as if the rest-frame diagonal covariance were still valid.
    """
    if mu_rest is None:
        mu_rest = np.zeros(p)
    L = lorentz_boost_x(eta, p=p)
    mu_obs = L @ mu_rest
    Z = X_obs - mu_obs
    return np.sum((Z @ Theta_inv) * Z, axis=1)


def score_minkowski_interval(X_obs, eta, mu_rest=None):
    """
    Absolute Lorentzian interval. It is Lorentz invariant but not
    covariance-standardized.
    """
    if mu_rest is None:
        mu_rest = np.zeros(p)
    L = lorentz_boost_x(eta, p=p)
    mu_obs = L @ mu_rest
    Z = X_obs - mu_obs
    return np.abs(minkowski_inner_rows(Z, Z))


def all_scores(X_obs, eta, mu_rest=None):
    return {
        "LM": score_lm_known(X_obs, eta, mu_rest=mu_rest),
        "Euclidean oracle": score_euclidean_oracle(X_obs, eta, mu_rest=mu_rest),
        "Euclidean naive": score_euclidean_naive(X_obs, eta, mu_rest=mu_rest),
        "Minkowski interval": score_minkowski_interval(X_obs, eta, mu_rest=mu_rest),
    }


# ============================================================
# Saving and display utilities
# ============================================================

def save_dataframe(df, name, index=False):
    csv_path = TABDIR / f"{name}.csv"
    tex_path = TABDIR / f"{name}.tex"

    df.to_csv(csv_path, index=index)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=index, escape=False, float_format="%.4f"))

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")


def save_figure(fig, name):
    png_path = FIGDIR / f"{name}.png"
    pdf_path = FIGDIR / f"{name}.pdf"

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def show_table(df, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    if IPYTHON_AVAILABLE:
        display(df)
    else:
        print(df.to_string(index=False))


# ============================================================
# Robust estimators
# ============================================================

def spatial_median(X, max_iter=400, tol=1e-7):
    """
    Weiszfeld spatial median.
    """
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


def tyler_shape(X_centered, max_iter=700, tol=1e-7):
    """
    Tyler shape estimator with trace normalization tr(S)=p.
    It estimates shape, not absolute covariance scale.
    """
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


def fit_classical_covariance(X):
    mu = X.mean(axis=0)
    Z = X - mu
    M = (Z.T @ Z) / X.shape[0]
    M = 0.5 * (M + M.T) + 1e-8 * np.eye(X.shape[1])
    return mu, M


def fit_mcd_covariance(X):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is unavailable.")
    mcd = MinCovDet(random_state=SEED, support_fraction=0.75).fit(X)
    mu = mcd.location_
    M = mcd.covariance_ + 1e-8 * np.eye(X.shape[1])
    M = 0.5 * (M + M.T)
    return mu, M


def fit_tyler_shape(X):
    mu = spatial_median(X)
    Z = X - mu
    M = tyler_shape(Z)
    M = 0.5 * (M + M.T) + 1e-8 * np.eye(X.shape[1])
    return mu, M


def mahal_score_estimated(X, mu, M):
    Z = X - mu
    Minv = pinv(M)
    return np.sum((Z @ Minv) * Z, axis=1)


def future_unit_timelike_eigenvector_from_scatter(M):
    """
    Estimate u as the future-directed unit timelike eigenvector of C = M G
    associated with the unique positive eigenvalue.
    """
    C = M @ G
    vals, vecs = np.linalg.eig(C)

    vals = np.real_if_close(vals, tol=1000).real
    vecs = np.real_if_close(vecs, tol=1000).real

    gnorms = np.array([vecs[:, j].T @ G @ vecs[:, j] for j in range(vecs.shape[1])])
    candidates = np.where(gnorms > 1e-9)[0]

    if len(candidates) == 0:
        idx = np.argmax(vals)
    else:
        idx = candidates[np.argmax(vals[candidates])]

    u = vecs[:, idx]
    guu = float(u.T @ G @ u)

    if guu <= 0:
        return np.full(p, np.nan)

    u = u / np.sqrt(guu)

    if u[0] < 0:
        u = -u

    return u


def rapidity_error(u_hat, u_true):
    if np.any(~np.isfinite(u_hat)):
        return np.nan
    val = float(u_hat.T @ G @ u_true)
    val = max(val, 1.0)
    return float(np.arccosh(val))


# ============================================================
# Experiment 0: numerical equivariance diagnostic
# ============================================================

def experiment_score_equivariance(etas, n=5000):
    """
    Use the same rest-frame sample and show which scores remain invariant
    under Lorentz boosts.
    """
    Y = simulate_rest_gaussian(n, rng=rng)

    X0, _, _ = transform_rest_to_observed(Y, eta=0.0)
    base_scores = all_scores(X0, eta=0.0)

    rows = []

    for eta in etas:
        X, _, _ = transform_rest_to_observed(Y, eta=eta)
        scores = all_scores(X, eta=eta)

        for method in scores:
            diff = np.abs(scores[method] - base_scores[method])
            rows.append({
                "eta": eta,
                "method": method,
                "max_abs_score_change": float(np.max(diff)),
                "mean_abs_score_change": float(np.mean(diff)),
                "median_abs_score_change": float(np.median(diff)),
            })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 1: type-I error under boosts
# ============================================================

def experiment_size_under_boost(etas, n_mc=60000):
    """
    Empirical size under nominal thresholds.
    """
    Y_cal = simulate_rest_gaussian(n_mc, rng=rng)
    X_cal, _, _ = transform_rest_to_observed(Y_cal, eta=0.0)
    mink_cut = np.quantile(score_minkowski_interval(X_cal, eta=0.0), 1 - ALPHA)

    rows = []

    for eta in etas:
        Y = simulate_rest_gaussian(n_mc, rng=rng)
        X, _, _ = transform_rest_to_observed(Y, eta=eta)
        scores = all_scores(X, eta=eta)

        cutoffs = {
            "LM": CHI2_CUTOFF,
            "Euclidean oracle": CHI2_CUTOFF,
            "Euclidean naive": CHI2_CUTOFF,
            "Minkowski interval": mink_cut,
        }

        for method, sc in scores.items():
            rej = float(np.mean(sc > cutoffs[method]))
            se = float(np.sqrt(max(rej * (1 - rej), 1e-15) / n_mc))

            rows.append({
                "eta": eta,
                "method": method,
                "empirical_size": rej,
                "monte_carlo_se": se,
                "nominal_alpha": ALPHA,
                "cutoff_used": float(cutoffs[method]),
            })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 2: outlier power, nominal and size-adjusted
# ============================================================

def outlier_shift_library():
    """
    Outlier alternatives in rest-frame coordinates.

    The near-null mean shift has large statistical displacement but nearly zero
    Lorentzian interval.
    """
    return {
        "temporal": np.array([3.0, 0.0, 0.0, 0.0]),
        "spatial": np.array([0.0, 3.0 * np.sqrt(Theta[1, 1]), 0.0, 0.0]),
        "mixed": np.array([2.0, 2.0, 1.0, 0.0]),
        "near-null": np.array([4.0, 4.0, 0.0, 0.0]),
    }


def calibrate_cutoffs_by_eta(etas, n_cal=70000):
    """
    Empirical null cutoffs for size-adjusted power.
    This prevents an invalid method with inflated size from appearing powerful.
    """
    rows = []

    for eta in etas:
        Y = simulate_rest_gaussian(n_cal, rng=rng)
        X, _, _ = transform_rest_to_observed(Y, eta=eta)
        scores = all_scores(X, eta=eta)

        for method, sc in scores.items():
            if method in ["LM", "Euclidean oracle"]:
                cutoff = CHI2_CUTOFF
            else:
                cutoff = float(np.quantile(sc, 1 - ALPHA))

            rows.append({
                "eta": eta,
                "method": method,
                "size_adjusted_cutoff": cutoff,
            })

    return pd.DataFrame(rows)


def experiment_outlier_power(etas, n_mc=50000, n_cal=70000):
    cut_df = calibrate_cutoffs_by_eta(etas, n_cal=n_cal)

    Y_cal0 = simulate_rest_gaussian(n_cal, rng=rng)
    X_cal0, _, _ = transform_rest_to_observed(Y_cal0, eta=0.0)
    mink_cut0 = float(np.quantile(score_minkowski_interval(X_cal0, eta=0.0), 1 - ALPHA))

    nominal_cutoffs = {
        "LM": CHI2_CUTOFF,
        "Euclidean oracle": CHI2_CUTOFF,
        "Euclidean naive": CHI2_CUTOFF,
        "Minkowski interval": mink_cut0,
    }

    shifts = outlier_shift_library()
    rows = []

    for alt_name, delta in shifts.items():
        lorentz_interval_of_shift = float(delta.T @ G @ delta)
        lm_separation = float(delta.T @ Theta_inv @ delta)

        for eta in etas:
            Y = simulate_rest_gaussian(n_mc, mean=delta, rng=rng)
            X, _, _ = transform_rest_to_observed(Y, eta=eta)
            scores = all_scores(X, eta=eta)

            for method, sc in scores.items():
                size_cut = float(
                    cut_df.query("eta == @eta and method == @method")["size_adjusted_cutoff"].iloc[0]
                )

                rows.append({
                    "outlier_type": alt_name,
                    "eta": eta,
                    "method": method,
                    "power_nominal": float(np.mean(sc > nominal_cutoffs[method])),
                    "power_size_adjusted": float(np.mean(sc > size_cut)),
                    "size_adjusted_cutoff": size_cut,
                    "lorentz_interval_of_mean_shift": lorentz_interval_of_shift,
                    "lm_separation_of_mean_shift": lm_separation,
                })

    return pd.DataFrame(rows), cut_df


# ============================================================
# Experiment 3: robust contamination study
# ============================================================

def contaminate_rest_sample(Y_clean, epsilon, rng=None):
    """
    Replace epsilon fraction of the training sample by a mixture of high-leverage
    and near-null contaminating points.
    """
    if rng is None:
        rng = np.random.default_rng()

    Y = Y_clean.copy()
    n = Y.shape[0]
    m = int(np.floor(epsilon * n))

    if m == 0:
        return Y

    idx = rng.choice(n, size=m, replace=False)

    C = np.zeros((m, p))
    m1 = m // 2

    C[:m1] = rng.normal(
        loc=np.array([7.0, 7.0, 0.0, 0.0]),
        scale=np.array([0.35, 0.35, 0.20, 0.20]),
        size=(m1, p)
    )

    C[m1:] = rng.normal(
        loc=np.array([0.0, 8.0, -5.0, 2.5]),
        scale=np.array([0.50, 0.50, 0.50, 0.50]),
        size=(m - m1, p)
    )

    Y[idx] = C
    return Y


def experiment_robustness(eps_grid, eta=1.25, n_rep=180, n_train=240, n_cal=900, n_test=900):
    methods = ["Classical covariance", "Tyler shape"]
    if SKLEARN_AVAILABLE:
        methods.insert(1, "MCD robust")

    fitters = {
        "Classical covariance": fit_classical_covariance,
        "Tyler shape": fit_tyler_shape,
    }

    if SKLEARN_AVAILABLE:
        fitters["MCD robust"] = fit_mcd_covariance

    u_true = lorentz_boost_x(eta, p=p) @ np.array([1.0, 0.0, 0.0, 0.0])
    delta_out = np.array([4.0, 4.0, 0.0, 0.0])

    rows = []

    for eps in eps_grid:
        store = {method: {"fpr": [], "power": [], "u_error": []} for method in methods}

        for _ in range(n_rep):
            Y_train_clean = simulate_rest_gaussian(n_train, rng=rng)
            Y_train = contaminate_rest_sample(Y_train_clean, eps, rng=rng)
            X_train, _, _ = transform_rest_to_observed(Y_train, eta=eta)

            # Clean validation sample for empirical calibration, ensuring fair size comparison.
            Y_cal = simulate_rest_gaussian(n_cal, rng=rng)
            X_cal, _, _ = transform_rest_to_observed(Y_cal, eta=eta)

            Y_null = simulate_rest_gaussian(n_test, rng=rng)
            X_null, _, _ = transform_rest_to_observed(Y_null, eta=eta)

            Y_out = simulate_rest_gaussian(n_test, mean=delta_out, rng=rng)
            X_out, _, _ = transform_rest_to_observed(Y_out, eta=eta)

            for method in methods:
                try:
                    mu_hat, M_hat = fitters[method](X_train)

                    sc_cal = mahal_score_estimated(X_cal, mu_hat, M_hat)
                    cutoff = np.quantile(sc_cal, 1 - ALPHA)

                    sc_null = mahal_score_estimated(X_null, mu_hat, M_hat)
                    sc_out = mahal_score_estimated(X_out, mu_hat, M_hat)

                    u_hat = future_unit_timelike_eigenvector_from_scatter(M_hat)

                    store[method]["fpr"].append(float(np.mean(sc_null > cutoff)))
                    store[method]["power"].append(float(np.mean(sc_out > cutoff)))
                    store[method]["u_error"].append(rapidity_error(u_hat, u_true))

                except Exception as e:
                    warnings.warn(f"{method} failed at epsilon={eps}: {e}")

        for method in methods:
            for metric in ["fpr", "power", "u_error"]:
                vals = np.array(store[method][metric], dtype=float)
                vals = vals[np.isfinite(vals)]

                if len(vals) == 0:
                    mean = np.nan
                    se = np.nan
                else:
                    mean = float(np.mean(vals))
                    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

                rows.append({
                    "epsilon": eps,
                    "method": method,
                    "metric": metric,
                    "mean": mean,
                    "monte_carlo_se": se,
                    "n_success": len(vals),
                })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 4: classification
# ============================================================

def choose_classification_delta():
    """
    A contrast chosen so that LM/full LDA has stable separation, while diagonal
    Euclidean LDA loses information under boosts because it ignores induced
    time-space covariance.
    """
    delta = np.array([1.235, -1.198, 0.0, 0.0])
    target_sep = 1.60
    current_sep = np.sqrt(delta @ Theta_inv @ delta)
    return delta * (target_sep / current_sep)


def discriminant_efficiency_diagonal(delta, eta):
    """
    Effective separation of diagonal Euclidean LDA under a wrong diagonal
    covariance approximation, compared with full Bayes separation.
    """
    L = lorentz_boost_x(eta, p=p)
    Cov_obs = L @ Theta @ L.T
    delta_obs = L @ delta

    w_diag = delta_obs / np.diag(Cov_obs)

    eff_diag = float((w_diag @ delta_obs) / np.sqrt(w_diag @ Cov_obs @ w_diag))
    eff_full = float(np.sqrt(delta @ Theta_inv @ delta))

    return eff_diag, eff_full


def lda_predict(X_train, y_train, X_test, diagonal=False):
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    S0 = np.cov(X0, rowvar=False, bias=False)
    S1 = np.cov(X1, rowvar=False, bias=False)

    Sp = ((len(X0) - 1) * S0 + (len(X1) - 1) * S1) / (len(X0) + len(X1) - 2)

    if diagonal:
        Sp = np.diag(np.diag(Sp))

    Sp = Sp + 1e-8 * np.eye(Sp.shape[0])
    Sinv = pinv(Sp)

    d0 = np.sum(((X_test - mu0) @ Sinv) * (X_test - mu0), axis=1)
    d1 = np.sum(((X_test - mu1) @ Sinv) * (X_test - mu1), axis=1)

    return (d1 < d0).astype(int)


def minkowski_nearest_centroid_predict(X_train, y_train, X_test):
    mu0 = X_train[y_train == 0].mean(axis=0)
    mu1 = X_train[y_train == 1].mean(axis=0)

    Z0 = X_test - mu0
    Z1 = X_test - mu1

    d0 = np.abs(minkowski_inner_rows(Z0, Z0))
    d1 = np.abs(minkowski_inner_rows(Z1, Z1))

    return (d1 < d0).astype(int)


def experiment_classification(etas, n_rep=180, n_train_per_class=180, n_test_per_class=1400):
    delta = choose_classification_delta()

    mu0 = -0.5 * delta
    mu1 =  0.5 * delta

    delta2 = float(delta @ Theta_inv @ delta)
    bayes_acc = float(norm.cdf(0.5 * np.sqrt(delta2)))

    rows = []
    theory_rows = []

    for eta in etas:
        eff_diag, eff_full = discriminant_efficiency_diagonal(delta, eta)

        theory_rows.append({
            "eta": eta,
            "full_lm_separation": eff_full,
            "diagonal_wrong_separation": eff_diag,
            "bayes_accuracy": float(norm.cdf(0.5 * eff_full)),
            "diagonal_asymptotic_accuracy": float(norm.cdf(0.5 * eff_diag)),
            "separation_ratio_diag_to_full": eff_diag / eff_full,
        })

        for _ in range(n_rep):
            Y0_tr = simulate_rest_gaussian(n_train_per_class, mean=mu0, rng=rng)
            Y1_tr = simulate_rest_gaussian(n_train_per_class, mean=mu1, rng=rng)

            Y0_te = simulate_rest_gaussian(n_test_per_class, mean=mu0, rng=rng)
            Y1_te = simulate_rest_gaussian(n_test_per_class, mean=mu1, rng=rng)

            X0_tr, _, _ = transform_rest_to_observed(Y0_tr, eta=eta)
            X1_tr, _, _ = transform_rest_to_observed(Y1_tr, eta=eta)

            X0_te, _, _ = transform_rest_to_observed(Y0_te, eta=eta)
            X1_te, _, _ = transform_rest_to_observed(Y1_te, eta=eta)

            Xtr = np.vstack([X0_tr, X1_tr])
            ytr = np.r_[np.zeros(n_train_per_class, dtype=int),
                         np.ones(n_train_per_class, dtype=int)]

            Xte = np.vstack([X0_te, X1_te])
            yte = np.r_[np.zeros(n_test_per_class, dtype=int),
                         np.ones(n_test_per_class, dtype=int)]

            # LM-LDA uses correct rest-frame coordinates.
            Ytr_adapted = transform_observed_to_rest(Xtr, eta=eta)
            Yte_adapted = transform_observed_to_rest(Xte, eta=eta)

            predictions = {
                "LM-LDA": lda_predict(Ytr_adapted, ytr, Yte_adapted, diagonal=False),
                "Full Euclidean LDA": lda_predict(Xtr, ytr, Xte, diagonal=False),
                "Diagonal Euclidean LDA": lda_predict(Xtr, ytr, Xte, diagonal=True),
                "Minkowski nearest-centroid": minkowski_nearest_centroid_predict(Xtr, ytr, Xte),
            }

            for method, pred in predictions.items():
                rows.append({
                    "eta": eta,
                    "method": method,
                    "accuracy": float(np.mean(pred == yte)),
                    "bayes_accuracy": bayes_acc,
                    "delta2": delta2,
                })

    return pd.DataFrame(rows), pd.DataFrame(theory_rows)


# ============================================================
# Plotting
# ============================================================

def plot_equivariance(df):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    for method, gdf in df.groupby("method"):
        gdf = gdf.sort_values("eta")
        ax.plot(
            gdf["eta"],
            gdf["mean_abs_score_change"],
            marker=MARKERS.get(method, "o"),
            linewidth=2.2,
            color=COLORS.get(method),
            label=method
        )

    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel(r"Lorentz boost rapidity $\eta$")
    ax.set_ylabel("Mean absolute score change from rest frame")
    ax.set_title("Numerical equivariance diagnostic")
    ax.legend(frameon=True)

    save_figure(fig, "fig00_equivariance_diagnostic")
    plt.show()


def plot_size(df):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for method, gdf in df.groupby("method"):
        gdf = gdf.sort_values("eta")
        color = COLORS.get(method)
        ax.plot(
            gdf["eta"],
            gdf["empirical_size"],
            marker=MARKERS.get(method, "o"),
            linewidth=2.2,
            color=color,
            label=method
        )
        ax.fill_between(
            gdf["eta"],
            gdf["empirical_size"] - 1.96 * gdf["monte_carlo_se"],
            gdf["empirical_size"] + 1.96 * gdf["monte_carlo_se"],
            color=color,
            alpha=0.14
        )

    ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.4,
               label=r"nominal $\alpha=0.05$")

    ax.set_xlabel(r"Lorentz boost rapidity $\eta$")
    ax.set_ylabel("Empirical type-I error")
    ax.set_title("Size under Lorentz boosts")
    ax.set_ylim(0, max(0.20, 1.10 * df["empirical_size"].max()))
    ax.legend(frameon=True, loc="best")

    save_figure(fig, "fig01_size_under_boost")
    plt.show()


def plot_power(df_power):
    eta_max = df_power["eta"].max()

    for power_col, label, fname in [
        ("power_nominal", "Nominal-threshold power", "fig02a_nominal_power_heatmap"),
        ("power_size_adjusted", "Size-adjusted power", "fig02b_size_adjusted_power_heatmap"),
    ]:
        pivot = (
            df_power[df_power["eta"] == eta_max]
            .pivot(index="method", columns="outlier_type", values=power_col)
            .loc[["LM", "Euclidean oracle", "Euclidean naive", "Minkowski interval"]]
        )

        fig, ax = plt.subplots(figsize=(8.4, 4.6))
        im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=25, ha="right")

        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)

        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val < 0.65 else "black",
                    fontsize=9
                )

        ax.set_title(rf"{label} at $\eta={eta_max}$")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Empirical power")

        save_figure(fig, fname)
        plt.show()

    for alt, adf in df_power.groupby("outlier_type"):
        fig, ax = plt.subplots(figsize=(7.4, 4.7))

        for method, gdf in adf.groupby("method"):
            gdf = gdf.sort_values("eta")
            ax.plot(
                gdf["eta"],
                gdf["power_size_adjusted"],
                marker=MARKERS.get(method, "o"),
                linewidth=2.0,
                color=COLORS.get(method),
                label=method
            )

        ax.set_xlabel(r"Lorentz boost rapidity $\eta$")
        ax.set_ylabel("Size-adjusted outlier power")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Size-adjusted power: {alt} outlier")
        ax.legend(frameon=True, loc="best")

        safe_alt = alt.replace(" ", "_").replace("-", "_")
        save_figure(fig, f"fig03_size_adjusted_power_{safe_alt}")
        plt.show()


def plot_robust(df):
    specs = [
        ("fpr", "False positive rate", "Contaminated training: false positive rate", "fig04_robust_fpr"),
        ("power", "Outlier power", "Contaminated training: outlier power", "fig05_robust_power"),
        ("u_error", r"Mean rapidity error of $\hat u$", "Contaminated training: rest-frame error", "fig06_robust_u_error"),
    ]

    for metric, ylab, title, fname in specs:
        sdf = df[df["metric"] == metric]

        fig, ax = plt.subplots(figsize=(7.4, 4.8))

        for method, gdf in sdf.groupby("method"):
            gdf = gdf.sort_values("epsilon")
            color = COLORS.get(method)
            ax.plot(
                gdf["epsilon"],
                gdf["mean"],
                marker=MARKERS.get(method, "o"),
                linewidth=2.2,
                color=color,
                label=method
            )
            ax.fill_between(
                gdf["epsilon"],
                gdf["mean"] - 1.96 * gdf["monte_carlo_se"],
                gdf["mean"] + 1.96 * gdf["monte_carlo_se"],
                color=color,
                alpha=0.14
            )

        if metric == "fpr":
            ax.axhline(ALPHA, color="black", linestyle="--", linewidth=1.4,
                       label=r"target $\alpha=0.05$")
            ax.set_ylim(0, max(0.20, 1.15 * sdf["mean"].max()))
        elif metric == "power":
            ax.set_ylim(0, 1.02)

        ax.set_xlabel(r"Contamination fraction $\epsilon$")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(frameon=True, loc="best")

        save_figure(fig, fname)
        plt.show()


def plot_classification(df_class, df_theory):
    agg = (
        df_class
        .groupby(["eta", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_se=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            bayes_accuracy=("bayes_accuracy", "first")
        )
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    for method, gdf in agg.groupby("method"):
        gdf = gdf.sort_values("eta")
        color = COLORS.get(method)

        ax.plot(
            gdf["eta"],
            gdf["accuracy_mean"],
            marker=MARKERS.get(method, "o"),
            linewidth=2.2,
            color=color,
            label=method
        )
        ax.fill_between(
            gdf["eta"],
            gdf["accuracy_mean"] - 1.96 * gdf["accuracy_se"],
            gdf["accuracy_mean"] + 1.96 * gdf["accuracy_se"],
            color=color,
            alpha=0.14
        )

    ax.plot(
        df_theory["eta"],
        df_theory["diagonal_asymptotic_accuracy"],
        color="#A6761D",
        linestyle=":",
        linewidth=2.5,
        label="Diagonal theoretical accuracy"
    )

    ax.axhline(
        agg["bayes_accuracy"].iloc[0],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Bayes accuracy"
    )

    ax.set_xlabel(r"Lorentz boost rapidity $\eta$")
    ax.set_ylabel("Classification accuracy")
    ax.set_ylim(0.48, 0.90)
    ax.set_title("Classification stability under Lorentz boosts")
    ax.legend(frameon=True, loc="best")

    save_figure(fig, "fig07_classification_stability")
    plt.show()

    fig, ax = plt.subplots(figsize=(7.4, 4.5))

    ax.plot(
        df_theory["eta"],
        df_theory["full_lm_separation"],
        marker="o",
        linewidth=2.2,
        color=COLORS["LM-LDA"],
        label="LM/full separation"
    )
    ax.plot(
        df_theory["eta"],
        df_theory["diagonal_wrong_separation"],
        marker="^",
        linewidth=2.2,
        color=COLORS["Diagonal Euclidean LDA"],
        label="Diagonal wrong separation"
    )

    ax.set_xlabel(r"Lorentz boost rapidity $\eta$")
    ax.set_ylabel("Effective discriminant separation")
    ax.set_title("Why diagonal Euclidean LDA fails after boosts")
    ax.legend(frameon=True)

    save_figure(fig, "fig08_classification_effective_separation")
    plt.show()

    return agg


def plot_score_distributions(eta=1.6, n=30000):
    delta = outlier_shift_library()["near-null"]

    Y0 = simulate_rest_gaussian(n, rng=rng)
    Y1 = simulate_rest_gaussian(n, mean=delta, rng=rng)

    X0, _, _ = transform_rest_to_observed(Y0, eta=eta)
    X1, _, _ = transform_rest_to_observed(Y1, eta=eta)

    methods_to_show = ["LM", "Euclidean naive", "Minkowski interval"]

    for method in methods_to_show:
        s0 = all_scores(X0, eta=eta)[method]
        s1 = all_scores(X1, eta=eta)[method]

        upper = np.quantile(np.r_[s0, s1], 0.995)
        bins = np.linspace(0, upper, 80)

        fig, ax = plt.subplots(figsize=(7.4, 4.6))

        ax.hist(s0, bins=bins, density=True, alpha=0.55,
                color="#56B4E9", label="null")
        ax.hist(s1, bins=bins, density=True, alpha=0.55,
                color="#E69F00", label="near-null outlier")

        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title(rf"Score distribution at $\eta={eta}$: {method}")
        ax.legend(frameon=True)

        safe = method.lower().replace(" ", "_").replace("/", "_")
        save_figure(fig, f"fig09_score_distribution_{safe}")
        plt.show()


# ============================================================
# Main runner
# ============================================================

def run_all_simulations(quick=True, make_zip=True):
    """
    Run the full simulation suite.

    quick=True:
        Fast test run.

    quick=False:
        Manuscript-quality run. This can take longer, especially with MCD.
    """
    etas = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0])
    eps_grid = np.array([0.00, 0.05, 0.10, 0.15, 0.20])

    if quick:
        n_equiv = 3000
        n_size = 10000
        n_power = 10000
        n_power_cal = 12000
        n_rep_robust = 35
        n_train_robust = 150
        n_cal_robust = 400
        n_test_robust = 400
        n_rep_class = 35
        n_train_class = 90
        n_test_class = 500
    else:
        n_equiv = 8000
        n_size = 70000
        n_power = 60000
        n_power_cal = 80000
        n_rep_robust = 180
        n_train_robust = 240
        n_cal_robust = 900
        n_test_robust = 900
        n_rep_class = 180
        n_train_class = 180
        n_test_class = 1400

    print("Checking Lorentz boost matrices...")
    for eta in etas:
        assert check_lorentz(lorentz_boost_x(float(eta))), f"Lorentz check failed at eta={eta}"
    print("All boost matrices preserve G up to numerical tolerance.")

    params = {
        "seed": SEED,
        "p": p,
        "d": d,
        "G": G.tolist(),
        "Theta_rest": Theta.tolist(),
        "alpha": ALPHA,
        "chi2_cutoff": float(CHI2_CUTOFF),
        "etas": etas.tolist(),
        "eps_grid": eps_grid.tolist(),
        "quick": quick,
        "sklearn_available": SKLEARN_AVAILABLE,
        "classification_delta": choose_classification_delta().tolist(),
    }

    with open(OUTDIR / "simulation_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("\nExperiment 0: numerical equivariance diagnostic...")
    df_equiv = experiment_score_equivariance(etas, n=n_equiv)
    save_dataframe(df_equiv, "table00_equivariance_diagnostic")
    show_table(df_equiv, "Table 0: Score equivariance diagnostic")
    plot_equivariance(df_equiv)

    print("\nExperiment 1: size under Lorentz boosts...")
    df_size = experiment_size_under_boost(etas, n_mc=n_size)
    save_dataframe(df_size, "table01_size_under_boost")
    show_table(df_size, "Table 1: Empirical type-I error under Lorentz boosts")
    plot_size(df_size)

    print("\nExperiment 2: outlier power with nominal and size-adjusted thresholds...")
    df_power, df_cutoffs = experiment_outlier_power(etas, n_mc=n_power, n_cal=n_power_cal)

    save_dataframe(df_cutoffs, "table02_size_adjusted_cutoffs")
    save_dataframe(df_power, "table03_outlier_power_long")

    eta_max = df_power["eta"].max()

    high_nominal = (
        df_power[df_power["eta"] == eta_max]
        .pivot(index="method", columns="outlier_type", values="power_nominal")
        .reset_index()
    )

    high_adjusted = (
        df_power[df_power["eta"] == eta_max]
        .pivot(index="method", columns="outlier_type", values="power_size_adjusted")
        .reset_index()
    )

    save_dataframe(high_nominal, "table04_nominal_power_high_boost")
    save_dataframe(high_adjusted, "table05_size_adjusted_power_high_boost")

    show_table(high_nominal, "Table 2A: Nominal-threshold power at largest boost")
    show_table(high_adjusted, "Table 2B: Size-adjusted power at largest boost")

    plot_power(df_power)

    print("\nExperiment 3: robust estimation under contaminated training...")
    df_robust = experiment_robustness(
        eps_grid,
        eta=1.25,
        n_rep=n_rep_robust,
        n_train=n_train_robust,
        n_cal=n_cal_robust,
        n_test=n_test_robust
    )

    save_dataframe(df_robust, "table06_robustness_long")

    robust_pivot = (
        df_robust
        .pivot_table(index=["epsilon", "method"], columns="metric", values="mean")
        .reset_index()
    )

    save_dataframe(robust_pivot, "table07_robustness_pivot")
    show_table(robust_pivot, "Table 3: Robustness under contaminated training")

    plot_robust(df_robust)

    print("\nExperiment 4: classification stability under Lorentz boosts...")
    df_class, df_theory = experiment_classification(
        etas,
        n_rep=n_rep_class,
        n_train_per_class=n_train_class,
        n_test_per_class=n_test_class
    )

    save_dataframe(df_class, "table08_classification_long")
    save_dataframe(df_theory, "table09_classification_theory")

    class_summary = plot_classification(df_class, df_theory)
    save_dataframe(class_summary, "table10_classification_summary")

    show_table(df_theory, "Table 4A: Theoretical discriminant separation")
    show_table(class_summary, "Table 4B: Empirical classification accuracy")

    print("\nMaking score-distribution diagnostic plots...")
    plot_score_distributions(eta=1.6, n=12000 if quick else 35000)

    readme = f"""
Advanced Lorentz--Mahalanobis simulation outputs
================================================

Main figures:
- fig00_equivariance_diagnostic
- fig01_size_under_boost
- fig02a_nominal_power_heatmap
- fig02b_size_adjusted_power_heatmap
- fig03_size_adjusted_power_*
- fig04_robust_fpr
- fig05_robust_power
- fig06_robust_u_error
- fig07_classification_stability
- fig08_classification_effective_separation
- fig09_score_distribution_*

Main tables:
- table00_equivariance_diagnostic
- table01_size_under_boost
- table02_size_adjusted_cutoffs
- table03_outlier_power_long
- table04_nominal_power_high_boost
- table05_size_adjusted_power_high_boost
- table06_robustness_long
- table07_robustness_pivot
- table08_classification_long
- table09_classification_theory
- table10_classification_summary

All figures are saved as PNG and PDF.
All tables are saved as CSV and LaTeX .tex.

quick = {quick}
scikit-learn available = {SKLEARN_AVAILABLE}
"""

    with open(OUTDIR / "README.txt", "w", encoding="utf-8") as f:
        f.write(readme)

    if make_zip:
        zip_path = Path("lm_advanced_simulation_outputs.zip")
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in OUTDIR.rglob("*"):
                zf.write(file, arcname=str(file.relative_to(OUTDIR.parent)))

        print(f"\nCreated ZIP archive: {zip_path.resolve()}")

        if COLAB_AVAILABLE:
            print("Downloading ZIP from Colab...")
            files.download(str(zip_path))

    return {
        "equivariance": df_equiv,
        "size": df_size,
        "power": df_power,
        "power_cutoffs": df_cutoffs,
        "robustness": df_robust,
        "robustness_pivot": robust_pivot,
        "classification_long": df_class,
        "classification_theory": df_theory,
        "classification_summary": class_summary,
    }


# ============================================================
# Run this line in Colab
# ============================================================

# Fast test run:
# results = run_all_simulations(quick=True, make_zip=True)

# Final manuscript-quality run:
# results = run_all_simulations(quick=False, make_zip=True)
