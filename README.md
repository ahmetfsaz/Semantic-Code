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
semantic source. For each one the scripts compute its calibrated content-entropy, compress it
both semantically and with a conventional entropy coder, and trace the trade-off between
transmission rate and retained logical content.

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

`torch` and `transformers` are needed only for the BART baseline.

## Running

Each script runs standalone with no arguments:

```bash
python3 semantic_compression.py          # content-entropy, compression ratios, rate–distortion plot
python3 content_entropy_min_norm.py      # same pipeline, min-normalized measure
python3 lambda_sweep_yago.py             # λ sweep over YAGO3-10 (requires YAGO3-10.txt)
python3 bart_baseline.py                 # BART autoencoder (GPU recommended)
```

Parameters are set inside the scripts rather than on the command line.

## What each script computes

### `semantic_compression.py`

The main experiment. For each of the seven stories it computes the calibrated content-entropy
and its normalized efficiency, then compares two compression paths: a conventional Huffman code
over the narrative, and a semantic pipeline that reduces the text to first-order logic and
entropy-codes it using inductive probabilities. It then sweeps the rate constraint to trace
normalized mutual content-information against bit cost.

Content-entropy spans thousands of orders of magnitude, so it is evaluated with `Decimal`
arithmetic and reported under several normalizations.

### `content_entropy_min_norm.py`

The same pipeline under the min-normalized content measure, producing the corresponding column
of the reported results. It shares most of its body with `semantic_compression.py` and differs
only in which normalization it applies.

### `lambda_sweep_yago.py`

Works at knowledge-graph rather than story scale. It builds Huffman and Shannon codebooks over
YAGO3-10 relation frequencies and reports average codeword length and entropy, then sweeps the
inductive prior coefficient λ from 2⁰ to 2³⁰ to show how the weight placed on the prior against
empirical counts shifts the resulting entropy.

### `bart_baseline.py`

A BART encoder–decoder with a linear bottleneck, trained to reconstruct FOLIO premises. This is
the learned-compression baseline the framework is contrasted against: it compresses effectively
but gives no account of which logical content survives the bottleneck.

### `dataset_partition.py`

Splits knowledge graph edges across a given number of partitions with controlled overlap,
producing the partial, overlapping views used in distributed settings.

## Citation

```bibtex
@inproceedings{saz2026theory,
  title     = {On The Theory of Semantic Information and Communication for Logical Inference},
  author    = {Saz, Ahmet Faruk and Xiong, Siheng and Fekri, Faramarz},
  booktitle = {IEEE Wireless Communications and Networking Conference (WCNC)},
  year      = {2026}
}
```
