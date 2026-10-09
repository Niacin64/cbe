# -*- coding: utf-8 -*-
"""Benchmark layer: quantify how useful a coverage score is.

Two metric families (both are needed):
  1. Binary separation: can the coverage score tell in-distribution from out-of-distribution
     structures (AUC, AUPR)? Extrapolation is the positive class, so OOD scores should be
     higher.
  2. Rank correlation with the true error: Spearman correlation between the coverage score and
     the actual prediction error. This is the more fundamental property -- a good score should
     mean "the further outside, the larger the error".
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _auc_aupr(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """y_true: 1 = positive class (OOD); scores: the scores. Returns (ROC-AUC, PR-AUC)."""
    from sklearn import metrics
    auc = metrics.roc_auc_score(y_true, scores)
    pr = metrics.average_precision_score(y_true, scores)
    return float(auc), float(pr)


def benchmark_binary(scores_in: np.ndarray, scores_out: np.ndarray):
    """Binary benchmark from two score groups, in and out. Returns dict(auc, aupr, ...)."""
    y = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    s = np.concatenate([scores_in, scores_out])
    auc, aupr = _auc_aupr(y, s)
    return {
        "auc": auc,
        "aupr": aupr,
        "mean_in": float(np.mean(scores_in)),
        "mean_out": float(np.mean(scores_out)),
    }


def spearman_corr(scores: np.ndarray, errors: np.ndarray) -> float:
    """Spearman rank correlation between the coverage score and the true error (1 is ideal)."""
    rho, _ = spearmanr(scores, errors)
    return float(rho)


def evaluate(scores_in, scores_out, scores_all, errors_all):
    """One-call benchmark: returns the full metric dict."""
    res = benchmark_binary(scores_in, scores_out)
    res["spearman_score_vs_error"] = spearman_corr(
        np.asarray(scores_all, dtype=float),
        np.asarray(errors_all, dtype=float),
    )
    return res
