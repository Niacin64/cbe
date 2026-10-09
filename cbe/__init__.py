# -*- coding: utf-8 -*-
"""cbe: pluggable coverage / extrapolation-detection middleware and benchmark library
for machine-learned interatomic potentials."""

from .descriptors import (Descriptor, SOAPDescriptor, ACEDescriptor, BehlerDescriptor,
                          LatentDescriptor, make_descriptor,
                          mace_latent_descriptor, AVAILABLE_DESCRIPTORS)
from .index import Index, FaissIndex, BruteForceIndex
from .coverage import CoverageModel
from .kde import kde_score, estimate_bandwidth
from .preprocess import PCA
from .calibration import (calibrate_threshold, evaluate_model, compare_models,
                          ConformalCalibrator, fit_conformal)
from .benchmark import benchmark_binary, spearman_corr, evaluate
from .ood import (perturb_volume, perturb_thermal, substitute, remove_atom,
                  add_interstitial, generate_ood_protocol)
from .data import read_structures, filter_elements, sample_subset
from .eval import (energy, forces, energy_error, force_error,
                   evaluate as eval_models, load_calculator)
from . import pretrained
from .vasp import (read_vasp_atoms, read_vasp_results, collect_vasp_results,
                   write_reference_xyz)

__all__ = [
    "Descriptor", "SOAPDescriptor", "ACEDescriptor", "BehlerDescriptor", "LatentDescriptor",
    "make_descriptor", "mace_latent_descriptor", "AVAILABLE_DESCRIPTORS",
    "pretrained",
    "Index", "FaissIndex", "BruteForceIndex",
    "CoverageModel",
    "kde_score", "estimate_bandwidth", "PCA",
    "calibrate_threshold", "evaluate_model", "compare_models",
    "ConformalCalibrator", "fit_conformal",
    "benchmark_binary", "spearman_corr", "evaluate",
    "perturb_volume", "perturb_thermal", "substitute", "remove_atom",
    "add_interstitial", "generate_ood_protocol",
    "read_structures", "filter_elements", "sample_subset",
    "energy", "forces", "energy_error", "force_error", "eval_models",
    "load_calculator",
    "read_vasp_atoms", "read_vasp_results", "collect_vasp_results",
    "write_reference_xyz",
]
__version__ = "0.6.0"
