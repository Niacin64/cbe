# Withheld structure files

The structure files behind the calibration and evaluation numbers in the paper are **not
released**. `CHECKSUMS.json` in this directory carries the SHA-256 digest of each one, so
that if the data is disclosed later it can be verified against the published paper.

## Why they are withheld

The extrapolation block and the test set are the objects a coverage score is measured
against, and this work is intended to support a scoring service for universal potentials.
Once such a set is public, a model can be trained specifically to pass it, and no later
analysis recovers the discrimination that is lost. The extrapolation block is in any case
not reconstructible from public sources, since it was built for this work.

We recognise the tension with the reproducibility standard stated in the paper — a
threshold is reproducible only together with the distribution it was read off. The trade is
deliberate. What is released to bound it:

* **The corpus half of the calibration set is regenerable exactly.** 300 structures drawn
  at random (seed 0) from a 50,000-structure MPtrj sample, itself extracted from the MPtrj
  release with cells of at most 16 atoms and seed 0, plus all 240 Alexandria structures of
  the reference set shipped with this code. Anyone can rebuild that half and check the
  thresholds against it. See `examples/calibration/`.
* **The construction recipe** for the extrapolation block and the test set is in the
  paper's Computational Methods (ten deformation modes, parent compounds, cell sizes, and
  the overlap filter that removes 213 of the 836 test structures).
* **The calibrated thresholds and the per-structure model outputs** of the two analyses
  that needed model evaluations of their own (voltage triage, ensemble comparison) are in
  `pretrained/thetas/` and the released analysis outputs, so those sections can be
  recomputed without re-running a model.

## The digests

| Set | n | file | SHA-256 |
|---|---|---|---|
| calibration corpus sample | 540 | `refs_calib.xyz` | see `CHECKSUMS.json` |
| extrapolation block | 768 | `refs_new_ood_clean.xyz` | see `CHECKSUMS.json` |
| test set, part a | 439 | `refs_full.xyz` | see `CHECKSUMS.json` |
| test set, part b | 397 | `refs_ood.xyz` | see `CHECKSUMS.json` |
| cathode candidates | 75 | `candidates.xyz` | see `CHECKSUMS.json` |

Verify a file you hold against the recorded digest with:

```bash
sha256sum your_copy.xyz     # compare with the sha256 field in CHECKSUMS.json
```
