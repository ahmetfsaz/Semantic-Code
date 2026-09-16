"""
Min-normalized content-entropy over the seven-story corpus.

For each story this computes the min-normalized content-entropy of its logical
content, and the cost of transmitting that content two ways: a semantic path
that Huffman-codes the subject, relation and object of every ground fact under
inductive probabilities, and a conventional path that Huffman-codes the
characters of the natural-language narrative.

Companion to semantic_compression.py, which runs the same pipeline under the
calibrated and max-normalized measures.
"""

from collections import Counter
from decimal import Decimal, getcontext

import huffman

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "dataset_exp1"
NUM_STORIES = 7
DECIMAL_PRECISION = 1000

# Weights for the inductive prior. w1 scales lambda, w2 divides the pseudo-count
# spread across categories; both are 1 in the reported runs.
PRIOR_W1 = 1
PRIOR_W2 = 1

getcontext().prec = DECIMAL_PRECISION


def content_entropy_min_normalized(num_subjects, num_relations, num_objects, length):
    """Min-normalized content-entropy of a source.

    Takes the same arguments as the other normalizations so the three are
    interchangeable, but under this normalization the subject, relation and
    object counts cancel and only `length`, the number of ground facts,
    survives.
    """
    return Decimal(2) ** (2 * Decimal(length)) - Decimal(2) ** Decimal(length)


def lambda_w(w):
    """Prior coefficient as a function of the category weight."""
    return w


def calculate_probability(count, total, w1=PRIOR_W1, w2=PRIOR_W2):
    """Inductive probability of a symbol observed `count` times in `total` draws.

    Balances the empirical frequency against a prior pseudo-count, so that
    unseen and rarely seen symbols retain positive probability.
    """
    return (count + 2 * lambda_w(w1) / w2) / (total + lambda_w(w1))


def load_triples(path):
    """Read a facts file into a list of (subject, relation, object) triples."""
    triples = []
    with open(path) as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def code_lengths(counter, total):
    """Huffman-code a symbol counter under inductive probabilities.

    Returns the code length per symbol, and the total bits needed to transmit
    every occurrence of every symbol.
    """
    sorted_counts = sorted(counter.items())
    codebook = huffman.codebook(sorted_counts)
    lengths = {symbol: len(code) for symbol, code in codebook.items()}
    probabilities = {
        symbol: calculate_probability(count, total) for symbol, count in sorted_counts
    }
    total_bits = sum(lengths[symbol] * count for symbol, count in sorted_counts)
    return lengths, probabilities, total_bits


def semantic_cost(triples):
    """Bits needed to transmit the ground facts through the semantic path.

    Also returns the cumulative cost of sending the facts cheapest-first, which
    traces the rate against the amount of logical content delivered.
    """
    subjects = Counter(subject for subject, _, _ in triples)
    relations = Counter(relation for _, relation, _ in triples)
    objects = Counter(obj for _, _, obj in triples)
    total = len(triples)

    subject_lengths, _, subject_bits = code_lengths(subjects, total)
    relation_lengths, _, relation_bits = code_lengths(relations, total)
    object_lengths, _, object_bits = code_lengths(objects, total)

    fact_costs = sorted(
        subject_lengths[s] + relation_lengths[r] + object_lengths[o]
        for s, r, o in triples
    )
    cumulative = []
    running = 0
    for cost in fact_costs:
        running += cost
        cumulative.append(running)

    return subject_bits + relation_bits + object_bits, cumulative


def shannon_cost(text):
    """Bits needed to transmit the narrative as Huffman-coded characters."""
    sorted_counts = sorted(Counter(text).items())
    codebook = huffman.codebook(sorted_counts)
    return sum(len(codebook[char]) * count for char, count in sorted_counts)


def main():
    entropies = []
    semantic_bits = []
    shannon_bits = []

    for index in range(NUM_STORIES):
        triples = load_triples(f"{DATA_DIR}/facts_{index}.txt")
        with open(f"{DATA_DIR}/story_{index}.txt") as handle:
            narrative = handle.read()

        num_subjects = len({s for s, _, _ in triples})
        num_relations = len({r for _, r, _ in triples})
        num_objects = len({o for _, _, o in triples})

        sem_bits, _cumulative = semantic_cost(triples)
        shn_bits = shannon_cost(narrative)
        entropy = content_entropy_min_normalized(
            num_subjects, num_relations, num_objects, len(triples)
        )

        entropies.append(entropy)
        semantic_bits.append(sem_bits)
        shannon_bits.append(shn_bits)

        print(
            f"Story {index}: "
            f"{num_subjects} subjects, {num_relations} relations, "
            f"{num_objects} objects, {len(triples)} facts | "
            f"semantic {sem_bits} bits, Shannon {shn_bits} bits"
        )

    print("\nSemantic bits: ", semantic_bits)
    print("Shannon bits:  ", shannon_bits)
    print("Min-normalized content-entropy:", entropies)


if __name__ == "__main__":
    main()
