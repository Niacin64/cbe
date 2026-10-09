# CBE — Coverage-Based Extrapolation for machine-learned interatomic potentials

**Know before you trust.** CBE turns the training set of a machine-learned interatomic
potential (MLIP) into a *coverage index*, calibrates one warning threshold per model with
split conformal prediction, and then tells you — for any new structure — whether the model
is interpolating or extrapolating. One index serves every model trained on the same corpus,
and a screen costs a single nearest-neighbour query, with no reference calculation and no
ensemble.

```
new structure ──▶ descriptor ──▶ PCA (d = 16) ──▶ one k-NN query into the corpus index
                                                            │
                              coverage  ρ = (1/k) Σᵢ exp(−‖q − xᵢ‖² / 2h²)
                                                            │
                                        ρ < θ ?  ──▶  "outside the training distribution"
```

Two objects, deliberately separated:

| Object | Property of | Cost to obtain | Cost to use |
|---|---|---|---|
| coverage index | the **training corpus** | one descriptor+PCA+index build | one distance query |
| threshold θ | the **model** | one conformal calibration | nothing |

Because the index belongs to the corpus rather than to the model, adding a model to a screen
costs one more calibration run and nothing at deployment time.

## Install

```bash
git clone https://github.com/<account>/cbe && cd cbe
pip install -e .                  # core: numpy, scipy, ase
pip install -e ".[index]"         # + faiss-cpu, for indexes larger than RAM
pip install -e ".[latent]"        # + mace-torch, for the default model-latent descriptor
pip install -e ".[soap]"          # + dscribe, for SOAP / Behler–Parrinello descriptors
pip install -e ".[all]"           # everything above plus matplotlib/pandas/pyarrow
```

Python ≥ 3.9. A GPU is optional: `faiss-cpu` handles indexes up to tens of millions of
environments, and larger corpora are handled with a sharded index
(`examples/index/build_shards_lowmem.py`).

## Quickstart

```python
from ase.io import read
from cbe import mace_latent_descriptor, CoverageModel, ConformalCalibrator
from cbe.data import read_structures

# 1) descriptor -> PCA -> nearest-neighbour index -> coverage model
train = read_structures("training_set.xyz")          # the corpus your MLIP was trained on
desc  = mace_latent_descriptor("small")              # or make_descriptor("soap" | "behler" | "ace")
cov   = CoverageModel(desc, k=30, scoring="kde", coverage_agg="min", pca_dim=16)
cov.fit(train)
cov.save("my_index/")                                # index.faiss + pca.npz + meta.json

# 2) calibrate ONE threshold for YOUR model, from its own force errors
calib  = read_structures("calibration_sample.xyz")   # corpus sample + structures that are hard
errors = [...]                                       # per-structure mean |F_model − F_ref|
theta  = ConformalCalibrator(alpha=0.1).fit(
             cov.coverage_many(calib), errors).threshold(eps_tol=0.5)   # eps_tol in eV/Å

# 3) screen any candidate with that frozen threshold
rho  = cov.coverage_many(read("candidates.xyz", index=":"))
warn = [i for i, r in enumerate(rho) if r < theta]

# reload later (the descriptor is code, so pass it back in)
cov2 = CoverageModel.load("my_index/", desc)
```

To use an index that already exists instead of fitting a new one, load it directly:
`CoverageModel.from_index("index.faiss", "pca.npz", desc, k=30, scoring="kde", bandwidth=0.2184)`,
or let `cbe.pretrained` pick the right one for a model.

`fit` above holds the descriptor matrix in memory, which is fine to about a million
environments. For a real corpus (tens to hundreds of millions) stream the descriptor pass,
fit the PCA on a sample and add to the index incrementally —
`examples/index/build_index.py` does exactly that, including the low-memory sharded variant
for corpora larger than RAM.

The per-atom score `ρ` is aggregated to a structure score by the **worst-covered atom**
(`coverage_agg="min"`): a structure is as trustworthy as its least-represented region.

### Two calibration rules, one calibration record

| Criterion | Definition | Use when |
|---|---|---|
| absolute | `θ = min{c : q̂σ(c) ≤ ε_tol}` | you have a tolerance in mind (e.g. 0.5 eV/Å for triage) |
| relative | `θ_rel = min{c : σ(c) ≤ κ·e_in}`, κ = 3 | the tolerance is tighter than the model's own in-distribution error, where the absolute criterion saturates at θ = 1 |

`e_in` is the model's median error on its own well-covered structures. Both are read off the
same fitted difficulty function `σ(c)`, so a user can store both and choose per task. See
`examples/reproduce/theta_relative.py`.

## Repository layout

```
cbe/                     the library
  descriptors.py         SOAP / Behler–Parrinello / ACE / model-latent descriptors
  preprocess.py          PCA (pure numpy) and descriptor loading
  index.py               exact L2 nearest-neighbour indexes (faiss or brute force)
  coverage.py            per-atom and per-structure coverage
  kde.py                 kernel-density variant of the score
  calibration.py         split conformal calibration (absolute + relative)
  ood.py                 random-cell / perturbation generators for stress tests
  eval.py                model loading and per-structure error evaluation
  pretrained.py          registry of the shipped indexes and thresholds
  vasp.py                VASP input/output helpers
  data.py                structure I/O

pretrained/              ready-to-use indexes (metadata) and per-model thresholds
  indexes/{mptrj,alexandria,omat24}/meta.json
  thetas/thetas.{json,csv}                 θ_f for the 18 models of the paper
  README.md

examples/                runnable scripts, grouped by task
  common/                shared configuration of the figure scripts
  index/                 build, download and verify an index
  eval/                  evaluate a roster of MLIPs, cache per-structure errors
  calibration/           calibrate thresholds
  cathode/               generate and relax Li–TM–O candidates, voltages
  figures/               the figures and tables of the paper
  reproduce/             the three analyses that need their own numbers

tests/                   smoke tests, descriptor tests, pretrained-registry tests
```

## The pipeline, fixed

```
training corpus ──▶ descriptor ──▶ PCA(16) ──▶ k-NN index          (build once per corpus)
candidate       ──▶ same descriptor + PCA ──▶ 1 query ──▶ ρ ──▶ ρ < θ ?   (per structure)
```

Paper defaults: MACE-MP-0 small rotation-invariant node features, 256 → 16 principal
components, `k = 30` neighbours, kernel width `h = 0.2184` shared by all corpora,
`α = 0.1`, force tolerance `ε_f = 0.5 eV/Å`. All are arguments, not constants.

## Pretrained indexes and thresholds

The three indexes of the paper are distributed as a **dataset**, not as code: the binaries
are 2.9–9.4 GB each (17.5 GB in total) and no single file fits in a git repository.
Everything needed to use them ships here — the per-model thresholds, the PCA matrix of each
index, and the index metadata.

```bash
# fetch one corpus; the script resumes, checks sha256 and sanity-checks the result
pip install huggingface_hub
python examples/index/download_pretrained.py --corpus mptrj --dest ~/cbe_data

# prove the artefact is an index before you screen with it
python examples/index/verify_index.py --index ~/cbe_data/mptrj
```

| Corpus | Environments | Download | Training sets in the paper |
|---|---|---|---|
| MPtrj | 49,295,660 | 2.94 GiB (1 file) | MACE-MP-0, CHGNet, SevenNet-0, GRACE-2L-MPtrj, ORB-v2 |
| Alexandria | 158,597,532 | 9.44 GiB (4 shards) | MACE-MPA-0, ORB-v2-MPA |
| OMat24 | 86,066,902 | 5.12 GiB (5 shards) | GRACE-OAM, ORB-v3, MatterSim-v1, SevenNet-MF-ompa |

You only need the corpus your model was trained on. The binaries live on Hugging Face
(`$HF_REPO`, default in the download script) and are archived with a DOI on Zenodo; the
download script, `CHECKSUMS.json` and `examples/index/verify_index.py` are the contract
between the two. See [`pretrained/README.md`](pretrained/README.md) for sizes per file and
for how to rebuild an index from the public corpora instead.

## Reproducing the paper

The scripts below regenerate every figure, table and number of the paper from the cached
data or from public corpora. `examples/figures/` holds the plotting scripts; set `--out` to
write elsewhere.

| Paper item | Script |
|---|---|
| Fig. 1 pipeline | `examples/figures/fig_pipeline.py` |
| Fig. 2 extrapolation test | `examples/figures/fig_test8.py` |
| Fig. 3 cathode screen (force + voltage) | `examples/figures/fig_cathode8.py` |
| Table 1 thresholds and price | `examples/figures/tables8.py` |
| Table 2 absolute vs relative criterion | `examples/reproduce/theta_relative.py` |
| Table 3 per-model cathode performance | `examples/figures/tables8.py` |
| Table 4 candidates matching known chemistries | `examples/cathode/voltage_roster.py` |
| Table 5 voltage tolerance decision | `examples/reproduce/voltage_triage.py` |
| calibration-mixture experiment | `examples/reproduce/theta_transfer.py` |
| ensemble comparison | `examples/reproduce/ensemble_baseline.py` + `ensemble_baseline_analysis.py` |
| the candidate list (SI Table 5/6) | `examples/cathode/voltage_roster.py`, `tables8.py` |

Typical flow: build an index (`index/`), evaluate a roster (`eval/`), calibrate
(`calibration/`), then plot or tabulate (`figures/`).

## What the score does and does not do

The paper documents two failure modes, and they generalise beyond it:

1. **Extrapolation** — the configuration asks about a local environment absent from the
   training data. This is what `ρ` measures and what `θ` bounds.
2. **Reference or spin-state bias** — the configuration is unremarkable and the answer is
   still wrong, because the error lives in the energy zero, in spin energetics, or in
   long-range order that a short-range, rotation-invariant, per-atom descriptor cannot carry.
   Every model trained on the same corpus inherits this bias at the same configurations, so
   a coverage guard is blind to it by construction and model agreement is not evidence.

A flag is a statement about the training data; agreement between models is a statement about
the models.

## Data sources

The training corpora belong to their original authors and are **not** redistributed here:
MPtrj accompanies the CHGNet release (derived from the Materials Project), Alexandria is
distributed through ColabFit under CC-BY-4.0, and OMat24 comes from the Open Materials 2024
release. Please cite those datasets alongside this package.

## License

MIT — see `LICENSE`.

## Citation

If you use CBE, please cite the paper:

> *Coverage-based extrapolation warnings for universal machine-learned interatomic
> potentials*, J. Chem. Theory Comput. (submitted).
