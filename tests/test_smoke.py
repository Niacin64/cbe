# -*- coding: utf-8 -*-
"""Smoke tests: exercise code paths the demos do not reach."""
import numpy as np
from ase.build import bulk

from cbe.descriptors import BehlerDescriptor
from cbe.index import BruteForceIndex, FaissIndex
from cbe.coverage import CoverageModel
from cbe.benchmark import benchmark_binary, spearman_corr


def test_bruteforce_matches_faiss():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 16)).astype("float32")
    q = rng.normal(size=(5, 16)).astype("float32")
    bf = BruteForceIndex(); bf.build(X)
    ff = FaissIndex(exact=True); ff.build(X)
    db, ib = bf.search(q, 5)
    df, _if = ff.search(q, 5)
    assert np.allclose(db, df, atol=1e-4), (db, df)
    print("ok: brute-force and faiss distances agree")


def test_structure_level_descriptor():
    at = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
    d = BehlerDescriptor(r_cut=5.0, radial=(0.5, 1.0), angular=(), structure_average=True)
    f = d.describe(at)
    assert f.shape == (2,), f.shape
    cov = CoverageModel(d, BruteForceIndex(), k=3, scoring="distance", agg="max").fit([at] * 5)
    s = cov.score(at)
    assert np.isfinite(s)
    print("ok: structure-level descriptor + distance coverage:", round(float(s), 4))


def test_benchmark_edges():
    rng = np.random.default_rng(1)
    s_in = rng.normal(1.0, 0.1, 50)
    s_out = rng.normal(5.0, 0.1, 50)
    r = benchmark_binary(s_in, s_out)
    assert r["auc"] > 0.95, r
    assert r["aupr"] > 0.95, r
    rho = spearman_corr(np.arange(10), np.arange(10))
    assert abs(rho - 1.0) < 1e-9, rho
    print("ok: benchmark AUC/PR/Spearman")


def test_kde_formula_and_ranking():
    from cbe.kde import kde_score, estimate_bandwidth
    from cbe.descriptors import LatentDescriptor
    from cbe.coverage import CoverageModel
    from cbe.index import BruteForceIndex
    from ase import Atoms

    # (1) formula correctness
    d = np.array([[0.0], [1.0]])
    rho = kde_score(d, bandwidth=1.0)
    assert np.isclose(rho[0], 1.0)
    assert np.isclose(rho[1], np.exp(-0.5))

    # (2) the estimated bandwidth is positive and finite
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 8)).astype("float32")
    bf = BruteForceIndex(); bf.build(X)
    h = estimate_bandwidth(bf, X, 50)
    assert np.isfinite(h) and h > 0

    # (3) KDE coverage ranking: dense kernel region vs far-away OOD points
    def feat_fn(atoms):
        return np.array([[atoms.info["x"], atoms.info["y"]]], dtype=np.float32)

    desc = LatentDescriptor(feat_fn, dim=2, level="per_atom")

    def mk(x, y):
        a = Atoms("H", positions=[[0, 0, 0]])
        a.info = {"x": x, "y": y}
        return a

    train = [mk(*rng.normal(0, 0.1, 2)) for _ in range(300)] + \
            [mk(*rng.normal(5, 0.3, 2)) for _ in range(20)]
    in_dist = [mk(*rng.normal(0, 0.1, 2)) for _ in range(50)]
    ood = [mk(*rng.normal(20, 0.1, 2)) for _ in range(50)]
    cov = CoverageModel(desc, BruteForceIndex(), k=30, scoring="kde").fit(train)
    s_in = cov.score_many(in_dist)
    s_out = cov.score_many(ood)
    assert s_in.mean() < s_out.mean(), (s_in.mean(), s_out.mean())
    print("ok: KDE formula/bandwidth/ranking  (in-risk=%.3f < out-risk=%.3f)"
          % (s_in.mean(), s_out.mean()))


def test_calibration():
    from cbe.calibration import (calibrate_threshold,
                                            evaluate_model, compare_models)
    rng = np.random.default_rng(0)
    n = 400
    # reliable: high coverage + low error; unreliable: low coverage + high error
    coverage = np.concatenate([rng.uniform(0.5, 1.0, n // 2),
                               rng.uniform(0.0, 0.2, n // 2)])
    errors = np.concatenate([rng.uniform(0.0, 0.05, n // 2),
                             rng.uniform(0.3, 0.5, n // 2)])
    t, f1, _ = calibrate_threshold(coverage, errors, error_tol=0.2)
    assert 0.15 < t < 0.6, t  # threshold should lie between the two coverage groups
    assert f1 > 0.9

    card = evaluate_model(coverage, errors, error_tol=0.2)
    assert card["auc"] > 0.95, card["auc"]
    # the bootstrap confidence interval should contain the point estimate
    lo, hi = evaluate_model(coverage, errors, error_tol=0.2,
                            n_boot=200)["auc_ci"]
    assert lo <= card["auc"] <= hi, (lo, card["auc"], hi)

    # noise-only error (independent of coverage) -> the extrapolation score should drop
    err_noisy = rng.uniform(0.0, 0.5, n)
    card2 = evaluate_model(coverage, err_noisy, error_tol=0.2)
    assert card2["auc"] < card["auc"], (card2["auc"], card["auc"])

    cards = compare_models([("good", coverage, errors), ("noisy", coverage, err_noisy)],
                           error_tol=0.2)
    assert cards["good"]["auc"] > cards["noisy"]["auc"]
    print("ok: calibrated threshold / extrapolation score (good AUC=%.3f > noisy AUC=%.3f)"
          % (card["auc"], card2["auc"]))


def test_conformal():
    from cbe.calibration import ConformalCalibrator
    rng = np.random.default_rng(42)
    n = 1000
    cov = rng.uniform(0.0, 1.0, n)
    # true error: the scale decays exponentially with coverage (low coverage -> large error)
    scale = 0.5 * np.exp(-2.0 * cov) + 0.02
    err = np.abs(rng.normal(0, 1, n)) * scale
    cal = ConformalCalibrator(alpha=0.1, seed=0).fit(cov[:500], err[:500])
    emp = cal.coverage(cov[500:], err[500:])
    assert 0.85 <= emp <= 1.0, emp  # empirical coverage should be about 0.9
    # low coverage -> wider interval
    assert cal.half_width(np.array([0.0])) > cal.half_width(np.array([1.0]))
    t = cal.threshold(0.1)
    assert 0.0 <= t <= 1.0
    print("ok: conformal empirical coverage=%.3f (target 0.9), threshold theta_alpha=%.3f"
          % (emp, t))


def test_ood_protocol():
    from cbe.ood import (perturb_volume, perturb_thermal, substitute,
                         remove_atom, add_interstitial, generate_ood_protocol)
    from ase.build import bulk
    at = bulk("Cu", "fcc", a=3.61, cubic=True).repeat((2, 2, 2))
    n0 = len(at)
    assert len(perturb_volume(at, 0.9)) == n0
    assert len(perturb_thermal(at, 0.2)) == n0
    s = substitute(at, "Ag", n_subs=3)
    assert len(s) == n0 and s.get_chemical_symbols().count("Ag") == 3
    assert len(remove_atom(at, 0)) == n0 - 1
    assert len(add_interstitial(at, "H")) == n0 + 1
    proto = generate_ood_protocol([at])
    assert {"volume", "thermal"} <= set(proto.keys())
    print("ok: OOD perturbation protocol (volume/thermal/composition/defects)")


if __name__ == "__main__":
    test_bruteforce_matches_faiss()
    test_structure_level_descriptor()
    test_benchmark_edges()
    test_kde_formula_and_ranking()
    test_calibration()
    test_conformal()
    test_ood_protocol()
    print("all tests passed")
