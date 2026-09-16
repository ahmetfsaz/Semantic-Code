"""
Semantic versus conventional source coding over the seven-story corpus.

For each story this measures the cost of transmitting its content two ways: a
semantic path that Huffman-codes the subject, relation and object of every
ground fact under inductive probabilities, and a conventional path that
Huffman-codes the characters of the natural-language narrative. It also traces
the rate curve, the cumulative bits needed as facts are sent cheapest-first, and
reports the max-normalized content-entropy of each source.

Companion to content_entropy_min_norm.py, which runs the same pipeline under the
min-normalized measure.
"""

from collections import Counter
from decimal import Decimal, getcontext

import huffman
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import entropy

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "dataset_exp1"
NUM_STORIES = 7
DECIMAL_PRECISION = 1000

PLOT_PATH = "plot.eps"
PLOT_DPI = 1000
SHOW_SHANNON_BASELINE = False   # draw each story's Shannon cost as a dashed line
SHOW_PLOT = False               # open an interactive window as well as saving

RUN_LAMBDA_SWEEP = True         # per-story entropy against the prior coefficient
LAMBDA_MIN_EXPONENT = 0.0
LAMBDA_MAX_EXPONENT = 30.0
LAMBDA_STEPS = 31

# Weights for the inductive prior; both are 1 in the reported runs.
PRIOR_W1 = 1
PRIOR_W2 = 1

getcontext().prec = DECIMAL_PRECISION


def content_entropy_calibrated(num_subjects, num_relations, num_objects, num_facts):
    """Calibrated content-entropy of a source.

    Calibration is by the logical volume of the language, so the measure scales
    with the number of distinct individuals and relations the source can
    describe rather than with the size of its text.
    """
    num_entities = Decimal(num_subjects) + Decimal(num_objects)
    volume = Decimal("0.5") * (
        Decimal(num_relations) * num_entities * (num_entities - Decimal(1))
    )
    return Decimal(2) ** (Decimal(num_facts) - volume) * (
        Decimal(2) ** Decimal(num_facts) - Decimal(1)
    )


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

    Returns the code length per symbol, the inductive probability per symbol,
    the average codeword length, and the total bits needed to transmit every
    occurrence of every symbol.
    """
    sorted_counts = sorted(counter.items())
    codebook = huffman.codebook(sorted_counts)
    lengths = {symbol: len(code) for symbol, code in codebook.items()}
    probabilities = {
        symbol: calculate_probability(count, total) for symbol, count in sorted_counts
    }
    average = sum(lengths[symbol] * probabilities[symbol] for symbol in lengths)
    total_bits = sum(lengths[symbol] * count for symbol, count in sorted_counts)
    return lengths, probabilities, average, total_bits


def semantic_cost(triples, verbose=True):
    """Bits needed to transmit the ground facts through the semantic path.

    Also returns the cumulative cost of sending the facts cheapest-first, which
    traces the rate against the amount of logical content delivered.
    """
    counters = {
        "subject": Counter(s for s, _, _ in triples),
        "relation": Counter(r for _, r, _ in triples),
        "object": Counter(o for _, _, o in triples),
    }
    total = len(triples)

    lengths, bits = {}, {}
    for field, counter in counters.items():
        field_lengths, probabilities, average, field_bits = code_lengths(counter, total)
        lengths[field] = field_lengths
        bits[field] = field_bits
        if verbose:
            print(
                f"  {field:<9} avg codeword {average:7.4f} bits | "
                f"{field_bits:>6} bits total | "
                f"entropy {entropy(list(probabilities.values()), base=2):7.4f}"
            )

    fact_costs = sorted(
        lengths["subject"][s] + lengths["relation"][r] + lengths["object"][o]
        for s, r, o in triples
    )
    cumulative = np.cumsum(fact_costs)

    return sum(bits.values()), cumulative, counters


def shannon_cost(text, verbose=True):
    """Bits needed to transmit the narrative as Huffman-coded characters."""
    sorted_counts = sorted(Counter(text).items())
    codebook = huffman.codebook(sorted_counts)
    probabilities = [count / len(text) for _, count in sorted_counts]
    average = sum(
        len(codebook[char]) * count / len(text) for char, count in sorted_counts
    )
    total_bits = sum(len(codebook[char]) * count for char, count in sorted_counts)
    if verbose:
        print(
            f"  {'shannon':<9} avg codeword {average:7.4f} bits | "
            f"{total_bits:>6} bits total | "
            f"entropy {entropy(probabilities, base=2):7.4f}"
        )
    return total_bits


def lambda_sweep(counters, total):
    """Print how the prior coefficient shifts the entropy of each field."""
    lambdas = np.logspace(
        LAMBDA_MIN_EXPONENT, LAMBDA_MAX_EXPONENT, num=LAMBDA_STEPS, base=2.0
    )
    for field, counter in counters.items():
        for lam in lambdas:
            probabilities = [
                calculate_probability(count, total, w1=lam, w2=PRIOR_W2)
                for count in counter.values()
            ]
            print(
                f"  {field:<9} lambda = {lam:>12.1f}   "
                f"entropy = {entropy(probabilities, base=2):.4f}"
            )


def plot_rate_curves(cumulative_costs, shannon_costs):
    """Plot cumulative transmission cost against relativized cont-information."""
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    for index, cumulative in enumerate(cumulative_costs):
        # One point per fact, so the x axis spans the share of content sent.
        x = np.linspace(0, 1, len(cumulative))
        plt.plot(x, cumulative, color=colors[index], label=f"Story {index + 1}")
        if SHOW_SHANNON_BASELINE:
            plt.plot(
                x,
                np.full(len(cumulative), shannon_costs[index]),
                color=colors[index],
                linestyle="dashed",
            )

    plt.xlabel("Relativized Cont-Information")
    plt.ylabel("Number of Bits")
    plt.legend()
    plt.savefig(PLOT_PATH, format="eps", dpi=PLOT_DPI)
    print(f"\nSaved rate curves to {PLOT_PATH}")
    if SHOW_PLOT:
        plt.show()


def main():
    cumulative_costs = []
    semantic_bits = []
    shannon_bits = []
    entropies = []

    for index in range(NUM_STORIES):
        triples = load_triples(f"{DATA_DIR}/facts_{index}.txt")
        with open(f"{DATA_DIR}/story_{index}.txt") as handle:
            narrative = handle.read()

        num_subjects = len({s for s, _, _ in triples})
        num_relations = len({r for _, r, _ in triples})
        num_objects = len({o for _, _, o in triples})

        print(
            f"\nStory {index}: {num_subjects} subjects, {num_relations} relations, "
            f"{num_objects} objects, {len(triples)} facts"
        )

        sem_bits, cumulative, counters = semantic_cost(triples)
        shn_bits = shannon_cost(narrative)

        if RUN_LAMBDA_SWEEP:
            lambda_sweep(counters, len(triples))

        cumulative_costs.append(cumulative)
        semantic_bits.append(sem_bits)
        shannon_bits.append(shn_bits)
        entropies.append(
            content_entropy_calibrated(
                num_subjects, num_relations, num_objects, len(triples)
            )
        )

    print("\nSemantic bits: ", semantic_bits)
    print("Shannon bits:  ", shannon_bits)

    largest = max(entropies)
    normalized = [value / largest for value in entropies]
    print("Max-normalized content-entropy:")
    for index, value in enumerate(normalized):
        print(f"  Story {index}: {value:.6E}")

    plot_rate_curves(cumulative_costs, shannon_bits)


if __name__ == "__main__":
    main()
