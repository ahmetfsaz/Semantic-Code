# Semantic-Code

Experiments for:

> **On The Theory of Semantic Information and Communication for Logical Inference**
> Ahmet Faruk Saz, Siheng Xiong, Faramarz Fekri — *IEEE WCNC 2026*
> [arXiv:2401.17556](https://arxiv.org/abs/2401.17556)

## Overview

The paper separates two levels of information in a first-order logic message: the **physical
entropy** of the symbols transmitted, and the **calibrated content** of what those symbols
signify. The first is a Shannon quantity over a semantic alphabet; the second measures how much
a message narrows down the true state of the world, calibrated by language granularity so that
content-informativeness is comparable across systems built on different ontologies.

This repository contains the source-coding experiments. Each story is treated as an independent
semantic source. For each one the scripts compute its content-entropy, compress it both
semantically and with a conventional entropy coder, and trace the trade-off between transmission
rate and retained logical content.

## Repository structure

```
Semantic-Code/
├── semantic_compression.py           # Main experiment: content-entropy, compression, rate–distortion
├── content_entropy_min_norm.py       # Min-normalized variant of the content-entropy measure
├── lambda_sweep_yago.py              # Codeword lengths and entropy over YAGO3-10, swept across λ
├── bart_baseline.py                  # Learned-compression baseline on FOLIO
├── dataset_partition.py              # Splits a knowledge graph across nodes with partial views
├── dataset_exp1/                     # Seven stories with their underlying fact triples
├── folio-train.jsonl                 # FOLIO training split
└── folio-validation.jsonl            # FOLIO validation split
```

## Data

**`dataset_exp1/`** holds seven independent semantic sources. Each is a pair:

- `facts_N.txt` — the underlying knowledge as `subject,predicate,object` triples

  ```
  Boo_Young-tae,playsFor,Yangju_Citizen_FC
  Boo_Young-tae,playsFor,Busan_IPark
  ```

- `story_N.txt` — the same content as a natural language narrative

The fact triples give the logical content whose semantic measures are computed; the narratives
give the natural-language form that conventional compression is applied to, so the two coding
schemes are measured against the same underlying information.

**FOLIO** (`folio-*.jsonl`) is used by the learned-compression baseline. Each record carries
premises, their first-order logic translation, a conclusion, and a label.

**YAGO3-10** is required by `lambda_sweep_yago.py` and `dataset_partition.py` but is not
committed here. It is a standard link-prediction benchmark of roughly 1.1M triples over 123k
entities and 37 relations; download it and place `YAGO3-10.txt` in the repository root.

## Requirements

```bash
pip install numpy scipy matplotlib huffman pandas torch transformers
```

`torch`, `transformers` and `pandas` are needed only for the BART baseline.

## Running

Each script runs standalone with no arguments:

```bash
python3 semantic_compression.py          # content-entropy, compression, rate–distortion plot
python3 content_entropy_min_norm.py      # same pipeline, min-normalized measure
python3 lambda_sweep_yago.py             # λ sweep over YAGO3-10 (requires YAGO3-10.txt)
python3 bart_baseline.py                 # BART autoencoder (GPU recommended)
python3 dataset_partition.py             # overlapping partitions (requires YAGO3-10.txt)
```

Parameters are not passed on the command line. Each script opens with a `Configuration` block
of module-level constants — input paths, budgets, sweep ranges — which is the only place that
needs editing to change a run.

## Outputs

| Script | Produces |
| --- | --- |
| `semantic_compression.py` | Per-story codeword lengths, bit totals and entropies on stdout; rate curves saved to `plot.eps` |
| `content_entropy_min_norm.py` | Per-story bit totals and min-normalized content-entropy on stdout |
| `lambda_sweep_yago.py` | Symbol- and character-level coding costs, then entropy at each λ, on stdout |
| `bart_baseline.py` | Trained weights, saved to a filename recording the hyperparameters used |
| `dataset_partition.py` | One tab-separated file per node under `partitions/` |

`semantic_compression.py` also runs a per-story λ sweep, which is verbose; set
`RUN_LAMBDA_SWEEP = False` to skip it, since `lambda_sweep_yago.py` covers the same ground at
knowledge-graph scale.

## What each script computes

### `semantic_compression.py`

The main experiment. For each of the seven stories it computes the calibrated content-entropy,
then compares two compression paths: a conventional Huffman code over the narrative, and a
semantic pipeline that codes the subject, relation and object of every ground fact under
inductive probabilities. It traces the cumulative cost of sending facts cheapest-first against
the share of content delivered, and reports the content-entropies normalized by the largest.

Content-entropy spans thousands of orders of magnitude, so it is evaluated with `Decimal`
arithmetic at high precision.

### `content_entropy_min_norm.py`

The same pipeline under the min-normalized content measure. Under that normalization the
subject, relation and object counts cancel and the measure depends only on the number of ground
facts, so stories with equal fact counts share a value.

### `lambda_sweep_yago.py`

Works at knowledge-graph rather than story scale. It builds Huffman codebooks over the YAGO3-10
relation vocabulary two ways — at the symbol level under inductive probabilities, and at the
character level over the concatenated relation names — and reports the average codeword length
and entropy of each. It then sweeps the inductive prior coefficient λ from 2⁰ to 2³⁰ to show how
shifting weight from the empirical counts toward the prior flattens the induced distribution.

### `bart_baseline.py`

A BART encoder–decoder with a linear bottleneck, trained to reconstruct FOLIO premises. This is
the learned-compression baseline the framework is contrasted against: it compresses effectively
but gives no account of which logical content survives the bottleneck.

### `dataset_partition.py`

Splits knowledge graph triples across a set of nodes and gives each node a fraction of another
node's triples, so their views overlap. This produces the partial, overlapping perspectives used
in the distributed setting, where no single node observes the whole graph.

## Citation

```bibtex
@inproceedings{saz2026theory,
  title     = {On The Theory of Semantic Information and Communication for Logical Inference},
  author    = {Saz, Ahmet Faruk and Xiong, Siheng and Fekri, Faramarz},
  booktitle = {IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2026}
}
```
