# -*- coding: utf-8 -*-
"""Prebuilt registry tests: theta table completeness, index metadata, loading API.

These do not require the index binaries.
"""
import json

import pytest

from cbe import pretrained


def test_thetas_registry_complete():
    doc = pretrained.load_thetas()
    models = doc["models"]
    assert doc["n_models"] == len(models) == 18
    for m in models:
        assert 0.0 < m["theta_f"] < 1.0
        assert m["training_corpus"]
        assert m["indexes"] and set(m["indexes"]) <= {"mptrj", "alexandria", "omat24"}
        h = m["held_out"]
        assert 0.0 <= h["flagged_fraction"] <= 1.0
        assert h["mae_above_theta_meV_per_A"] > 0
    # a multi-corpus model must map to several indexes
    mpa = pretrained.model_info("mace-mpa-0")
    assert mpa["indexes"] == ["mptrj", "alexandria"]
    assert pretrained.theta("mace-mpa-0") == pytest.approx(mpa["theta_f"])


def test_index_registry():
    names = pretrained.list_indexes()
    assert {"mptrj", "alexandria"} <= set(names)
    meta = pretrained.index_meta("mptrj")
    assert meta["descriptor"].startswith("mace-latent")
    assert meta["pca_dim"] == 16
    assert meta["n_environments"] == 49_295_660
    assert meta["bandwidth_used"] == pytest.approx(0.2184)


def test_unknown_model_and_index():
    with pytest.raises(KeyError):
        pretrained.theta("no-such-model")
    with pytest.raises(KeyError):
        pretrained.index_meta("no-such-index")


def test_index_dir_requires_binaries_or_reports_clearly():
    try:
        d = pretrained.index_dir("mptrj")
    except FileNotFoundError as e:          # missing binaries must produce an actionable error
        assert "no *.faiss" in str(e)
    else:
        assert list(d.glob("*.faiss")) or (d / "index.faiss").exists()
