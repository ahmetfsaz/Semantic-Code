"""
Effect of the inductive prior coefficient on relation entropy in YAGO3-10.

Builds Huffman codebooks over the relation vocabulary of YAGO3-10 two ways: at
the symbol level, where each relation is coded under its inductive probability,
and at the character level, where the concatenated relation names are coded by
character frequency. Reports the average codeword length and entropy of each.

It then sweeps the prior coefficient lambda across thirty-one powers of two,
showing how shifting weight from the empirical counts toward the prior flattens
the induced distribution and raises its entropy.
"""

from collections import Counter

import huffman
import numpy as np
from scipy.stats import entropy

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_PATH = "YAGO3-10.txt"
LAMBDA_MIN_EXPONENT = 0.0
LAMBDA_MAX_EXPONENT = 30.0
LAMBDA_STEPS = 31


def lambda_w(w):
    """Prior coefficient as a function of the category weight."""
    return w


def calculate_probability(count, total, w1, w2):
    """Inductive probability of a symbol observed `count` times in `total` draws.

    `w1` scales the prior coefficient and `w2` is the number of categories the
    prior pseudo-count is spread across, so larger `w1` pulls the estimate away
    from the empirical frequency and toward a uniform distribution.
    """
    return (count + 2 * lambda_w(w1) / w2) / (total + lambda_w(w1))


def load_counts(path):
    """Read whitespace-separated triples in a single pass.

    Returns counters for subjects, relations and objects, the relation names
    concatenated into one string for character-level coding, and the number of
    triples read.
    """
    subjects, relations, objects = Counter(), Counter(), Counter()
    relation_text = []
    num_triples = 0

    with open(path) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            subject, relation, obj = parts
            subjects[subject] += 1
            relations[relation] += 1
            objects[obj] += 1
            relation_text.append(relation)
            relation_text.append(" ")
            num_triples += 1

    return subjects, relations, objects, "".join(relation_text), num_triples


def huffman_average_length(counts, probabilities):
    """Average codeword length of a Huffman code under a given distribution."""
    codebook = huffman.codebook(sorted(counts.items()))
    return sum(len(codebook[symbol]) * probabilities[symbol] for symbol in counts)


def main():
    subjects, relations, objects, relation_text, num_triples = load_counts(INPUT_PATH)

    print(f"Read {num_triples} triples from {INPUT_PATH}")
    print(
        f"Unique subjects: {len(subjects)}  "
        f"relations: {len(relations)}  "
        f"objects: {len(objects)}"
    )

    # Symbol-level coding: each relation coded under its inductive probability.
    inductive = {
        relation: calculate_probability(count, num_triples, w1=1, w2=1)
        for relation, count in relations.items()
    }
    print(f"\nSymbol level — average codeword length: "
          f"{huffman_average_length(relations, inductive):.4f} bits")
    print(f"Symbol level — entropy: "
          f"{entropy(list(inductive.values()), base=2):.4f} bits")

    # Character-level coding: the concatenated relation names by character.
    char_counts = Counter(relation_text)
    char_probabilities = {
        char: count / len(relation_text) for char, count in char_counts.items()
    }
    print(f"\nCharacter level — {len(relation_text)} characters, "
          f"{len(char_counts)} distinct")
    print(f"Character level — average codeword length: "
          f"{huffman_average_length(char_counts, char_probabilities):.4f} bits")
    print(f"Character level — entropy: "
          f"{entropy(list(char_probabilities.values()), base=2):.4f} bits")

    # Sweep the prior coefficient. The pseudo-count is spread across the
    # relation vocabulary, so w2 is the number of distinct relations.
    print("\nLambda sweep over the relation distribution:")
    lambdas = np.logspace(
        LAMBDA_MIN_EXPONENT, LAMBDA_MAX_EXPONENT, num=LAMBDA_STEPS, base=2.0
    )
    for lam in lambdas:
        probabilities = [
            calculate_probability(count, num_triples, w1=lam, w2=len(relations))
            for count in relations.values()
        ]
        print(f"  lambda = {lam:>12.1f}   entropy = "
              f"{entropy(probabilities, base=2):.4f} bits")


if __name__ == "__main__":
    main()
