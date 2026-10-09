# -*- coding: utf-8 -*-
"""Calibration layer: per (model, detector), fit a coverage threshold on a validation set and
report an extrapolation score.

Motivation:
  KDE/distance coverage scores rank well but their absolute threshold drifts (the demo shows
  that the 0.5 threshold does not transfer). This module assumes no global threshold: it
  learns one threshold θ per model from a validation set, so that "coverage < θ" best flags
  the unreliable predictions with "error > error_tol", and it also reports a scalar
  extrapolation score (= the AUC with which coverage separates unreliable from reliable
  predictions) for ranking models.

Convention: higher coverage means more reliable (rho in the kde mode); unreliable = low
coverage.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _binarize(errors, error_tol):
    return (np.asarray(errors, dtype=float) > error_tol).astype(int)


def _bootstrap_auc(cov, err, error_tol, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap confidence interval of the extrapolation score (AUC).

    Resample the (coverage, error) pairs with replacement, recompute the AUC and take the
    alpha/2 and 1-alpha/2 quantiles. Returns (low, high); (nan, nan) if the sample contains a
    single class only.
    """
    from sklearn import metrics
    cov = np.asarray(cov, dtype=float)
    err = np.asarray(err, dtype=float)
    n = len(cov)
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(int(n_boot)):
        idx = rng.randint(0, n, n)
        y = _binarize(err[idx], error_tol)
        if len(np.unique(y)) < 2:
            continue  # this bootstrap sample has a single class, skip it
        aucs.append(float(metrics.roc_auc_score(y, -cov[idx])))
    if not aucs:
        return float("nan"), float("nan")
    lo = float(np.percentile(aucs, 100 * alpha / 2))
    hi = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return lo, hi


def calibrate_threshold(coverage, errors, error_tol, metric="f1"):
    """Scan coverage thresholds so that "coverage < θ means unreliable" is optimal on the
    validation set.

    Returns (best_threshold, best_value, scan_table), where each row of scan_table is
    (threshold, f1, precision, recall).
    """
    cov = np.asarray(coverage, dtype=float)
    y = _binarize(errors, error_tol)
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError(
            "the validation set has no positive/negative samples (all errors are <= or all "
            "are > error_tol), cannot calibrate a threshold; please adjust error_tol")
    ts = np.unique(np.quantile(cov, np.linspace(0.01, 0.99, 200)))
    best_t, best_v = None, -1.0
    rows = []
    for t in ts:
        pred = (cov < t).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)
        # maximise F1 by default; Youden's J = recall + specificity - 1 also works
        val = f1 if metric == "f1" else (recall + (1 - precision))
        rows.append((float(t), f1, precision, recall))
        if val > best_v:
            best_v, best_t = val, t
    return float(best_t), float(best_v), np.array(rows)


def evaluate_model(coverage, errors, error_tol, n_boot=1000, seed=0):
    """Produce a "model trust card".

    Returns a dict:
        threshold           recommended coverage threshold (< θ means unreliable)
        best_f1             F1 at that threshold
        auc                 ROC-AUC of coverage separating unreliable/reliable predictions
        auc_ci              95% bootstrap confidence interval of the AUC (low, high)
        aupr                PR-AUC
        spearman            rank correlation between coverage and error
        extrapolation_score extrapolation score (the main scalar = AUC)
    """
    from sklearn import metrics
    cov = np.asarray(coverage, dtype=float)
    err = np.asarray(errors, dtype=float)
    y = _binarize(err, error_tol)
    two_class = len(np.unique(y)) == 2
    # orientation: unreliable = low coverage, so -cov is the "higher = less reliable" score
    auc = float(metrics.roc_auc_score(y, -cov)) if two_class else float("nan")
    aupr = (float(metrics.average_precision_score(y, -cov))
            if two_class else float("nan"))
    spearman = float(spearmanr(cov, err)[0])
    best_t, best_f1, _ = calibrate_threshold(cov, err, error_tol)
    auc_lo, auc_hi = _bootstrap_auc(cov, err, error_tol,
                                    n_boot=n_boot, seed=seed) if two_class else (float("nan"), float("nan"))
    return {
        "threshold": best_t,
        "best_f1": best_f1,
        "auc": auc,
        "auc_ci": (auc_lo, auc_hi),
        "aupr": aupr,
        "spearman": spearman,
        "extrapolation_score": auc,
    }


def compare_models(models, error_tol):
    """models: list of (name, coverage_array, errors_array).
    Returns {name: trust_card}."""
    return {name: evaluate_model(cov, err, error_tol)
            for name, cov, err in models}


# ---------------------------------------------------------------------------
# Conformal coverage guarantee (split conformal + normalised nonconformity / local adaptivity)
# ---------------------------------------------------------------------------
def _fit_monotone_scale(cov, err, floor_quantile=0.1):
    """Fit a non-increasing error-scale function σ(c): higher coverage -> smaller error scale.

    Uses isotonic regression (increasing=False), so σ(c) is monotone non-increasing, and
    applies a **floor**: σ(c) ≥ the floor_quantile quantile of the errors. Rationale: on the
    sparse high-coverage edge, isotonic regression is dragged down by a few low-error points
    (e.g. σ(1) -> 0.8 meV), which makes the normalised nonconformity e/σ explode and
    contaminates the single q̂ of the multi-output joint conformal. The floor keeps σ at or
    above the typical error scale of the easiest batch.
    """
    from sklearn.isotonic import IsotonicRegression
    cov = np.asarray(cov, dtype=float)
    err = np.asarray(err, dtype=float)
    order = np.argsort(cov)
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso.fit(cov[order].reshape(-1, 1), err[order])
    floor = float(np.quantile(err, floor_quantile))

    def sigma(c):
        c = np.asarray(c, dtype=float)
        return np.maximum(iso.predict(c.reshape(-1, 1)), floor)
    return sigma


class ConformalCalibrator:
    """Split conformal: turn coverage c into a half-width δ(c) with a marginal coverage
    guarantee.

    Guarantee (marginal): for a new test point, P( err ≤ q̂·σ(c) ) ≥ 1−α (when the calibration
    set and the test point are exchangeable).
    Note: this is a **marginal** guarantee, not a conditional one that holds at every coverage
    level.

    Parameters:
        alpha: target miscoverage rate (default 0.1 -> 90% coverage)
        split_frac: fraction of the set used to fit σ (disjoint from the quantile computation,
                    to avoid double-dipping)
        seed: random seed for the split
        sigma_floor: floor quantile of σ(c) (default 0.0 = no floor, i.e. standard split
                     conformal; when non-zero, σ(c) ≥ that quantile of the errors. The
                     measured effect of the floor is ≤2%, off by default)
    """

    def __init__(self, alpha=0.1, split_frac=0.5, seed=0, sigma_floor=0.0):
        self.alpha = alpha
        self.split_frac = split_frac
        self.seed = seed
        self.sigma_floor = sigma_floor
        self.sigma_ = None
        self.q_ = None

    def _split_idx(self, n):
        idx = np.random.RandomState(self.seed).permutation(n)
        half = int(n * self.split_frac)
        return idx[:half], idx[half:]

    def _quantile(self, s):
        # finite-sample corrected quantile
        return float(np.quantile(
            s, min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(s)))))

    def fit(self, cov_cal, err_cal):
        cov = np.asarray(cov_cal, dtype=float)
        err = np.asarray(err_cal, dtype=float)
        idx1, idx2 = self._split_idx(len(cov))
        self.sigma_ = _fit_monotone_scale(cov[idx1], err[idx1], self.sigma_floor)
        s = err[idx2] / self.sigma_(cov[idx2])
        self.q_ = self._quantile(s)
        return self

    def fit_joint(self, cov_cal, e_err_cal, f_err_cal):
        """Split conformal for multiple outputs (energy + forces): one difficulty function
        s = max(e/σ_e, f/σ_f).

        - fit the two error-scale functions σ_e(c) and σ_f(c) on I1 (isotonic, non-increasing);
        - compute the joint nonconformity s = max(e/σ_e(c), f/σ_f(c)) on I2 and take
          q̂ = the (1−α) quantile;
        - then for any new point (when exchangeable):
              P( e ≤ q̂·σ_e(c) and f ≤ q̂·σ_f(c) ) ≥ 1−α
          i.e. **one q̂ gives a joint coverage guarantee for energy and forces** (the max
          construction binds the two outputs into a single scalar).

        Note: the force error must be scalarised first (the mean per-atom |F_model−F_ref|);
        this function takes per-structure scalar energy/force error arrays.
        """
        cov = np.asarray(cov_cal, dtype=float)
        e = np.asarray(e_err_cal, dtype=float)
        f = np.asarray(f_err_cal, dtype=float)
        idx1, idx2 = self._split_idx(len(cov))
        self.sigma_e_ = _fit_monotone_scale(cov[idx1], e[idx1], self.sigma_floor)
        self.sigma_f_ = _fit_monotone_scale(cov[idx1], f[idx1], self.sigma_floor)
        s = np.maximum(e[idx2] / self.sigma_e_(cov[idx2]),
                       f[idx2] / self.sigma_f_(cov[idx2]))
        self.q_ = self._quantile(s)
        return self

    def joint_half_widths(self, cov):
        """Joint error-interval half-widths (δ_e, δ_f) = (q̂·σ_e(c), q̂·σ_f(c)).

        A scalar input returns a scalar (float, float); an array input returns (array, array).
        """
        if getattr(self, "sigma_e_", None) is None:
            raise RuntimeError("call fit_joint first")
        cov = np.asarray(cov, dtype=float)
        de = self.q_ * self.sigma_e_(cov)
        df = self.q_ * self.sigma_f_(cov)
        if cov.ndim == 0:
            return float(np.asarray(de).ravel()[0]), float(np.asarray(df).ravel()[0])
        return de, df

    def threshold_joint(self, eps_e, eps_f):
        """Single threshold θ = min{c : q̂·σ_e(c) ≤ ε_e and q̂·σ_f(c) ≤ ε_f}.

        Because σ_e and σ_f are both non-increasing in c, the c that satisfy this form the
        interval [θ, 1], and bisection finds its left end. A coverage below θ means the
        conformal half-width of the energy or of the forces exceeds its tolerance, so the
        prediction is flagged as failed.
        """
        if getattr(self, "sigma_e_", None) is None:
            raise RuntimeError("call fit_joint first")
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            de, df = self.joint_half_widths(mid)
            if de <= eps_e and df <= eps_f:
                hi = mid
            else:
                lo = mid
        return hi

    def joint_coverage(self, cov_test, e_err_test, f_err_test):
        """Joint empirical coverage = fraction of errors inside both δ_e and δ_f (≈ 1−α)."""
        cov = np.asarray(cov_test, dtype=float)
        e = np.asarray(e_err_test, dtype=float)
        f = np.asarray(f_err_test, dtype=float)
        de, df = self.joint_half_widths(cov)
        return float(np.mean((e <= de) & (f <= df)))

    def half_width(self, cov):
        """Error-interval half-width δ(c) = q̂·σ(c)."""
        if self.sigma_ is None:
            raise RuntimeError("call fit first")
        return self.q_ * self.sigma_(cov)

    def threshold(self, eps_tol):
        """Derive the "safe threshold": the smallest c with δ(c) ≤ eps_tol (a coverage below it
        is deemed unreliable).

        Note: the conformal guarantee applies to the error interval; this threshold is a
        derived quantity and carries no guarantee of its own.
        """
        if self.sigma_ is None:
            raise RuntimeError("call fit first")
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if self.q_ * self.sigma_(mid) <= eps_tol:
                hi = mid
            else:
                lo = mid
        return hi

    def coverage(self, cov_test, err_test):
        """Empirical coverage on the test set = fraction of errors inside δ(c) (≈ 1−α)."""
        cov = np.asarray(cov_test, dtype=float)
        err = np.asarray(err_test, dtype=float)
        return float(np.mean(err <= self.half_width(cov)))


def fit_conformal(cov_cal, err_cal, alpha=0.1, split_frac=0.5, seed=0,
                  sigma_floor=0.0):
    """Convenience function: return a fitted ConformalCalibrator."""
    return ConformalCalibrator(alpha=alpha, split_frac=split_frac, seed=seed,
                               sigma_floor=sigma_floor).fit(cov_cal, err_cal)


def fit_conformal_joint(cov_cal, e_err_cal, f_err_cal, alpha=0.1,
                        split_frac=0.5, seed=0, sigma_floor=0.0):
    """Convenience function: return a ConformalCalibrator fitted jointly on energy + forces."""
    return ConformalCalibrator(alpha=alpha, split_frac=split_frac, seed=seed,
                               sigma_floor=sigma_floor).fit_joint(
        cov_cal, e_err_cal, f_err_cal)
