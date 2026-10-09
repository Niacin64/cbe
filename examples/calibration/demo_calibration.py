# -*- coding: utf-8 -*-
"""Calibration demo: fit a coverage threshold for each model and score its extrapolation ability.

Demonstrates:
  1. calibrate the threshold θ on a validation set (instead of a fixed 0.5), fixing "0.5 threshold drift";
  2. different models get different θ and different "extrapolation ability scores" (AUC) that can be ranked.

Run:  PYTHONPATH=../pylibs:.. python3 demo_calibration.py
"""
import numpy as np
from ase.calculators.emt import EMT
from ase.build import bulk

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cbe.descriptors import BehlerDescriptor, SOAPDescriptor
from cbe.index import FaissIndex
from cbe.coverage import CoverageModel
from cbe.calibration import evaluate_model, compare_models


def make_cu_cell(scale=1.0, rattle=0.0, a0=3.61, rep=(3, 3, 3)):
    at = bulk("Cu", "fcc", a=a0, cubic=True).repeat(rep)
    at.set_cell(at.get_cell() * scale, scale_atoms=True)
    if rattle > 0:
        at.positions += np.random.normal(0, rattle, at.positions.shape)
    return at


def main():
    rng = np.random.default_rng(0)
    calc = EMT()

    def ref_energy(at):
        at = at.copy(); at.calc = calc
        return at.get_potential_energy()

    def sample(n, slo, shi, rattles):
        out = []
        for _ in range(n):
            s = rng.uniform(slo, shi)
            r = float(rattles[rng.integers(0, len(rattles))])
            out.append(make_cu_cell(scale=s, rattle=r))
        return out

    train = sample(280, 0.85, 1.15, [0.05, 0.25])
    in_dist = sample(100, 0.85, 1.15, [0.05, 0.25])
    ood_vol = sample(30, 0.60, 0.62, [0.10]) + sample(30, 1.38, 1.40, [0.10])
    ood_melt = sample(40, 0.95, 1.05, [0.8])
    all_test = in_dist + ood_vol + ood_melt

    # model input features (Behler radial)
    model_desc = BehlerDescriptor(r_cut=4.5,
                                  radial=(0.1, 0.3, 0.5, 1.0, 2.0, 4.0), angular=())
    def feat(at):
        return model_desc.describe(at).sum(axis=0)

    from sklearn.linear_model import Ridge
    Xtr = np.stack([feat(a) for a in train])
    ytr = np.array([ref_energy(a) for a in train])

    # model A: ordinary ridge regression (errors concentrate on OOD and track coverage -> high score)
    modelA = Ridge(alpha=1e-3).fit(Xtr, ytr)
    # model B: add large "~0.8 eV per atom" noise to A's predictions so that errors are noise-dominated
    # and decoupled from coverage -> low score (108 atoms x 0.8 eV/atom = ~86 eV total noise)
    predA = modelA.predict(np.stack([feat(a) for a in all_test]))
    predB = predA + rng.normal(0, 0.8 * len(train[0]), predA.shape)

    def per_atom_error(pred, at):
        return abs(pred - ref_energy(at)) / len(at)

    errA = np.array([per_atom_error(p, a) for p, a in
                     zip(modelA.predict(np.stack([feat(a) for a in all_test])), all_test)])
    errB = np.array([per_atom_error(p, a) for p, a in zip(predB, all_test)])

    # coverage: SOAP + KDE (structure-level coverage = min over the per-atom rho values)
    soap = SOAPDescriptor(r_cut=4.5, nmax=4, lmax=4, species=["Cu"], average=False)
    cov = CoverageModel(soap, FaissIndex(exact=True), k=100,
                        scoring="kde", pca_dim=16).fit(train)
    coverage = cov.coverage_many(all_test)

    # error tolerance: the median error (about half judged "unreliable"), so both classes are present
    error_tol = float(np.median(errA))
    print("=" * 68)
    print(f"calibration demo: per-model threshold + extrapolation score (error_tol={error_tol:.4f} eV/atom)")
    print("=" * 68)

    cards = compare_models(
        [("model A (clean)", coverage, errA),
         ("model B (noisy)", coverage, errB)],
        error_tol)

    print(f"\n{'model':<16}{'threshold':>12}{'F1@theta':>8}{'AUC':>8}{'AUPR':>8}{'Spearman':>10}")
    print("-" * 68)
    for name, c in cards.items():
        print(f"{name:<16}{c['threshold']:>12.6f}{c['best_f1']:>8.3f}"
              f"{c['auc']:>8.3f}{c['aupr']:>8.3f}{c['spearman']:>10.3f}")

    print("\nNotes:")
    print("- the threshold theta: coverage < theta = unreliable; learned on the validation set (not 0.5).")
    print("- extrapolation score = AUC: model A's failures track coverage well (score ~1, easily caught);")
    print("  model B's errors are noise-dominated and decoupled from coverage (score ~0.5, uncatchable).")

    # comparison: use the learned θ instead of a fixed 0.5 and look at the in-dist misflag rate
    cA = cards["model A (clean)"]
    misflag_05 = np.mean(coverage[:len(in_dist)] < 0.5)
    misflag_th = np.mean(coverage[:len(in_dist)] < cA["threshold"])
    print(f"\nin-dist misflag rate: fixed 0.5={misflag_05:.2f}  vs  learned theta={misflag_th:.2f}")
    print("(theta is calibrated on the error distribution, removing the fixed-0.5 false flags on in-dist)")


if __name__ == "__main__":
    main()
