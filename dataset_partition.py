"""
Split a knowledge graph into overlapping partitions.

Divides the triples of a knowledge graph across a set of nodes, then gives each
node a fraction of another node's triples so their views overlap. This produces
the partial, overlapping perspectives used in the distributed setting, where no
single node observes the whole graph.
"""

import os
import random

import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_PATH = "YAGO3-10.txt"     # tab-separated subject/relation/object triples
OUTPUT_DIR = "partitions"
NUM_PARTITIONS = 5
OVERLAP_FRACTION = 0.1          # share of a donor partition copied into each node
MAX_TRIPLES = None              # cap the input for a quick test; None reads all
SEED = 2024


def split_indices(num_items, num_pieces):
    """Split `range(num_items)` into `num_pieces` contiguous blocks.

    Any remainder from the integer division is folded into the final block, so
    the number of pieces returned always matches `num_pieces`.
    """
    if num_pieces > num_items:
        raise ValueError(
            f"cannot split {num_items} items into {num_pieces} pieces"
        )

    per_piece = num_items // num_pieces
    pieces = [
        list(range(start, min(start + per_piece, num_items)))
        for start in range(0, num_items, per_piece)
    ]

    if len(pieces) > num_pieces:
        tail = [index for piece in pieces[num_pieces - 1:] for index in piece]
        pieces = pieces[: num_pieces - 1] + [tail]
    return pieces


def partition_with_overlap(triples, num_partitions, overlap_fraction, rng):
    """Partition `triples` into overlapping views.

    Each partition receives its own contiguous block of a shuffled ordering,
    plus `overlap_fraction` of the triples belonging to a randomly chosen donor
    partition. The donor is drawn uniformly and may be the partition itself, in
    which case that partition simply carries duplicates of its own rows.
    """
    shuffled = triples.copy()
    rng.shuffle(shuffled)

    index_blocks = split_indices(len(shuffled), num_partitions)
    partitions = []

    for block in index_blocks:
        donor = index_blocks[random.randrange(num_partitions)]
        donor_rows = shuffled[donor].copy()
        rng.shuffle(donor_rows)
        shared = donor_rows[: int(len(donor_rows) * overlap_fraction)]
        partitions.append(np.vstack((shuffled[block], shared)))

    return partitions


def load_triples(path, max_triples=None):
    """Read tab-separated triples into an (N, 3) array of strings."""
    triples = []
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            triples.append(parts)
            if max_triples is not None and len(triples) >= max_triples:
                break
    return np.array(triples)


def main():
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    triples = load_triples(INPUT_PATH, MAX_TRIPLES)
    print(f"Loaded {len(triples)} triples from {INPUT_PATH}")

    partitions = partition_with_overlap(
        triples, NUM_PARTITIONS, OVERLAP_FRACTION, rng
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for index, partition in enumerate(partitions):
        path = f"{OUTPUT_DIR}/partition_{index}.txt"
        np.savetxt(path, partition, fmt="%s", delimiter="\t")
        print(f"Partition {index}: {len(partition)} triples -> {path}")


if __name__ == "__main__":
    main()
