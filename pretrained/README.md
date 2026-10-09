# pretrained — indexes and thresholds, ready to use

Two kinds of object live here, and they are distributed differently on purpose:

| Object | Size | Where it comes from |
|---|---|---|
| **thresholds** per model (`thetas/`) | ~10 KB | in this repository |
| **PCA matrix** per index (`indexes/*/pca.npz`) | 35 KB each | in this repository |
| **index metadata** per corpus (`indexes/*/meta.json`) | ~1 KB | in this repository |
| **index binaries** (`indexes/*/*.faiss`) | 2.9–9.4 GB per corpus | downloaded (see below) |

The binaries are 17.5 GB in total and no single one fits in a git repository, so they are
distributed as a dataset rather than as code. Everything needed to *use* them is here.

## Download

```bash
python examples/index/download_pretrained.py --corpus mptrj --dest ~/cbe_data
python examples/index/download_pretrained.py --corpus alexandria --dest ~/cbe_data
python examples/index/download_pretrained.py --corpus omat24 --dest ~/cbe_data
```

The script downloads into `<dest>/<corpus>/`, **resumes** an interrupted transfer, checks
every file against `CHECKSUMS.json`, and runs a short functional check on what it fetched.

You only need the corpus your model was trained on — the index is matched to the training
set. Downloading all three (17.5 GB) is only necessary to reproduce the corpus comparison
of the paper.

| Corpus | Environments | Download | Index layout |
|---|---|---|---|
| MPtrj | 49,295,660 | 2.94 GiB | single file |
| Alexandria | 158,597,532 | 9.44 GiB | 4 shards (~2.4 GiB each) |
| OMat24 | 86,066,902 | 5.12 GiB | 5 shards (~1.2 GiB each) |

## Verify before you trust it

```bash
python examples/index/verify_index.py --index ~/cbe_data/mptrj
```

A checksum proves the bytes arrived; this proves the artefact is an index. It checks that
`meta.json` and `pca.npz` agree, that every declared file is present, that the environment
count matches the metadata (which catches a missing shard), that a vector taken out of the
index is returned as its own nearest neighbour, and that the coverage score is monotone in
`k` and lies in (0, 1].

Add `--lazy` to open a sharded index the way a low-memory user would (the round-trip checks
are skipped, because they need to reconstruct vectors).

## Use

```python
from cbe import pretrained

print(pretrained.list_indexes())        # ['alexandria', 'mptrj', 'omat24']
print(pretrained.list_models())         # the 18 calibrated models
cov = pretrained.load_index('mptrj')    # needs mace-torch for the default descriptor
rho, flagged, theta = pretrained.screen('MACE-MP-0 medium', structures)
```

`pretrained.screen` picks the index from the model's training corpus, applies the union rule
when a model saw several corpora, and compares against that model's calibrated `theta_f`.

## Layout

```
pretrained/
├── CHECKSUMS.json          sha256 of every distributed artefact
├── indexes/
│   ├── mptrj/              meta.json + pca.npz   (+ index.faiss after download)
│   ├── alexandria/         meta.json + pca.npz   (+ shard_00{0..3}.faiss)
│   └── omat24/             meta.json + pca.npz   (+ shard_00{0..4}.faiss)
└── thetas/
    ├── thetas.json         per-model theta_f with the calibration record
    └── thetas.csv          the same table
```

## Rebuilding an index instead of downloading it

Every index is reproducible from the public corpora with the scripts in `examples/index/`,
given a GPU for the descriptor pass:

| Corpus | Script |
|---|---|
| MPtrj | `examples/index/build_index.py` |
| Alexandria | `examples/index/build_shards_lowmem.py` (sharded, low memory) |
| OMat24 | `examples/index/build_index.py --corpus omat24` |

Rebuilds are not byte-identical to the distribution: the PCA fit, the corpus snapshot and
the faiss version all enter the binary. `meta.json` records the descriptor, dimension,
bandwidth and environment count a rebuild must reproduce, and `verify_index.py` checks them.
