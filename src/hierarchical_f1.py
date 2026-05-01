"""Ancestor-based hierarchical F1 for propaganda techniques (Kiritchenko et al. 2006).

Provides micro hF1 (SemEval official), macro per-label, and per-sample variants.
Root node "Persuasion" is excluded.
"""

import numpy as np
from collections import defaultdict
from typing import List, Set, Dict
from sklearn.metrics import f1_score, precision_score, recall_score

TAXONOMY = {
    # Ethos -> Ad Hominem
    "Name calling/Labeling":       ["Ad Hominem", "Ethos"],
    "Doubt":                       ["Ad Hominem", "Ethos"],
    "Smears":                      ["Ad Hominem", "Ethos"],
    "Reductio ad hitlerum":        ["Ad Hominem", "Ethos"],

    # Pathos -> Justification
    "Bandwagon":                   ["Justification", "Pathos"],
    "Appeal to authority":         ["Justification", "Pathos"],
    "Glittering generalities (Virtue)": ["Justification", "Pathos"],
    "Appeal to (Strong) Emotions": ["Justification", "Pathos"],
    "Exaggeration/Minimisation":   ["Justification", "Pathos"],
    "Loaded Language":             ["Justification", "Pathos"],
    "Flag-waving":                 ["Justification", "Pathos"],
    "Appeal to fear/prejudice":    ["Justification", "Pathos"],
    "Transfer":                    ["Justification", "Pathos"],

    # Pathos -> direct child
    "Slogans":                     ["Pathos"],

    # Logos -> direct children
    "Repetition":                  ["Logos"],
    "Obfuscation, Intentional vagueness, Confusion": ["Logos"],

    # Logos -> Reasoning -> Distraction
    "Misrepresentation of Someone's Position (Straw Man)": ["Distraction", "Reasoning", "Logos"],
    "Presenting Irrelevant Data (Red Herring)":            ["Distraction", "Reasoning", "Logos"],
    "Whataboutism":                ["Distraction", "Reasoning", "Logos"],

    # Logos -> Reasoning -> Simplification
    "Causal Oversimplification":   ["Simplification", "Reasoning", "Logos"],
    "Black-and-white Fallacy/Dictatorship": ["Simplification", "Reasoning", "Logos"],
    "Thought-terminating cliché":  ["Simplification", "Reasoning", "Logos"],
}

TAXONOMY_LOWER = {k.lower(): (k, v) for k, v in TAXONOMY.items()}

# All nodes that can appear in extended sets (leaves + internal, no root)
ALL_NODES = set(TAXONOMY.keys())
for ancestors in TAXONOMY.values():
    ALL_NODES.update(ancestors)
ALL_NODES.add("NO_PROPAGANDA")
ALL_NODES_SORTED = sorted(ALL_NODES)

def get_ancestor_set(technique: str) -> Set[str]:
    """Get the technique + all its ancestors (root excluded)."""
    technique_lower = technique.lower()

    if technique_lower == "no_propaganda":
        return {"NO_PROPAGANDA"}

    if technique_lower in TAXONOMY_LOWER:
        canonical, ancestors = TAXONOMY_LOWER[technique_lower]
        return {canonical} | set(ancestors)

    return {technique}

def get_extended_label_set(labels: Set[str]) -> Set[str]:
    """Extend a set of leaf labels to include all ancestors."""
    extended = set()
    for label in labels:
        extended |= get_ancestor_set(label)
    return extended

def micro_hierarchical_f1(
    gold_labels: List[Set[str]],
    pred_labels: List[Set[str]],
) -> Dict[str, float]:
    """
    Pool extended sets across all samples, then compute P/R/F1.
    This is the official SemEval metric.
    """
    total_intersection = 0
    total_pred_size = 0
    total_gold_size = 0

    for gold, pred in zip(gold_labels, pred_labels):
        ext_gold = get_extended_label_set(gold)
        ext_pred = get_extended_label_set(pred)

        total_intersection += len(ext_gold & ext_pred)
        total_pred_size += len(ext_pred)
        total_gold_size += len(ext_gold)

    p = total_intersection / total_pred_size if total_pred_size > 0 else 0.0
    r = total_intersection / total_gold_size if total_gold_size > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"precision": p, "recall": r, "f1": f1}

def macro_per_label_hierarchical_f1(
    gold_labels: List[Set[str]],
    pred_labels: List[Set[str]],
) -> Dict[str, object]:
    """
    For each node in the hierarchy (leaves + internal nodes),
    compute binary F1 across all samples: "is this node present
    in the extended set or not?" Then average across nodes.

    Returns overall macro P/R/F1 and per-label breakdown.
    """
    n = len(gold_labels)

    # Build binary vectors per node
    node_gold = defaultdict(list)  # node -> [0,1,1,0,...] for each sample
    node_pred = defaultdict(list)

    for i in range(n):
        ext_gold = get_extended_label_set(gold_labels[i])
        ext_pred = get_extended_label_set(pred_labels[i])

        for node in ALL_NODES_SORTED:
            node_gold[node].append(1 if node in ext_gold else 0)
            node_pred[node].append(1 if node in ext_pred else 0)

    # Compute per-label F1
    per_label = {}
    precisions, recalls, f1s = [], [], []

    for node in ALL_NODES_SORTED:
        y_true = node_gold[node]
        y_pred = node_pred[node]

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        per_label[node] = {"precision": p, "recall": r, "f1": f1}

        # Only include labels that appear in gold OR pred for macro average
        if sum(y_true) > 0 or sum(y_pred) > 0:
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

    return {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
        "per_label": per_label,
    }

def per_sample_hierarchical_f1(
    gold_labels: List[Set[str]],
    pred_labels: List[Set[str]],
) -> Dict[str, object]:
    """
    Compute hP, hR, hF1 for each sample individually.
    Returns mean and the full list of per-sample scores.
    """
    sample_scores = []

    for gold, pred in zip(gold_labels, pred_labels):
        ext_gold = get_extended_label_set(gold)
        ext_pred = get_extended_label_set(pred)

        intersection = len(ext_gold & ext_pred)

        hp = intersection / len(ext_pred) if ext_pred else 0.0
        hr = intersection / len(ext_gold) if ext_gold else 0.0
        hf1 = 2 * hp * hr / (hp + hr) if (hp + hr) > 0 else 0.0

        sample_scores.append({
            "precision": hp,
            "recall": hr,
            "f1": hf1,
        })

    mean_p = float(np.mean([s["precision"] for s in sample_scores]))
    mean_r = float(np.mean([s["recall"] for s in sample_scores]))
    mean_f1 = float(np.mean([s["f1"] for s in sample_scores]))

    return {
        "precision": mean_p,
        "recall": mean_r,
        "f1": mean_f1,
        "per_sample": sample_scores,
    }

def hierarchical_f1(
    gold_labels: List[Set[str]],
    pred_labels: List[Set[str]],
) -> Dict:
    """
    Compute all three hierarchical metrics.

    Returns:
        {
            "micro": {P, R, F1},
            "macro_per_label": {P, R, F1, per_label: {node: {P,R,F1}}},
            "per_sample": {P, R, F1, per_sample: [{P,R,F1}, ...]},
        }
    """
    assert len(gold_labels) == len(pred_labels), "Mismatched lengths"

    return {
        "micro": micro_hierarchical_f1(gold_labels, pred_labels),
        "macro_per_label": macro_per_label_hierarchical_f1(gold_labels, pred_labels),
        "per_sample": per_sample_hierarchical_f1(gold_labels, pred_labels),
    }

if __name__ == "__main__":

    gold = [
        {"Loaded Language", "Name calling/Labeling"},
        # {"NO_PROPAGANDA"},
        # {"Appeal to fear/prejudice", "Flag-waving"},
        # {"Causal Oversimplification"},
        # {"Smears", "Doubt", "Loaded Language"},
    ]

    pred = [
        { "Smears"},
        # {"NO_PROPAGANDA"},
        # {"Appeal to (Strong) Emotions", "Flag-waving"},
        # {"Black-and-white Fallacy/Dictatorship"},
        # {"Smears", "Name calling/Labeling", "Loaded Language"},
    ]

    results = hierarchical_f1(gold, pred)

    # --- Micro ---
    print("=" * 60)
    print("  1. MICRO hF1 (SemEval official)")
    print("=" * 60)
    d = results["micro"]
    print(f"     P: {d['precision']:.4f}  |  R: {d['recall']:.4f}  |  F1: {d['f1']:.4f}")

    # --- Macro per-label ---
    print(f"\n{'=' * 60}")
    print("  2. MACRO PER-LABEL hF1")
    print("=" * 60)
    d = results["macro_per_label"]
    print(f"     P: {d['precision']:.4f}  |  R: {d['recall']:.4f}  |  F1: {d['f1']:.4f}")
    print(f"\n     Per-label breakdown:")
    for node, scores in d["per_label"].items():
        if scores["f1"] > 0 or any(
            node in get_extended_label_set(g) or node in get_extended_label_set(p)
            for g, p in zip(gold, pred)
        ):
            print(f"       {node:55s}  F1: {scores['f1']:.4f}  P: {scores['precision']:.4f}  R: {scores['recall']:.4f}")

    # --- Per-sample ---
    print(f"\n{'=' * 60}")
    print("  3. PER-SAMPLE hF1")
    print("=" * 60)
    d = results["per_sample"]
    print(f"     Mean P: {d['precision']:.4f}  |  Mean R: {d['recall']:.4f}  |  Mean F1: {d['f1']:.4f}")
    print(f"\n     Individual samples:")
    for i, s in enumerate(d["per_sample"]):
        g_str = ", ".join(sorted(gold[i]))
        p_str = ", ".join(sorted(pred[i]))
        print(f"       Sample {i}: hF1={s['f1']:.4f}  (gold: {{{g_str}}}  pred: {{{p_str}}})")