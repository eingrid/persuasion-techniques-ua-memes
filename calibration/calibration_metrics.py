"""Calibration metrics for multi-label classification: ECE, MCE, ACE, Brier, log-loss, reliability diagrams."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

def ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error.

    Args:
        confidences:  1-D array of predicted probabilities in [0, 1].
        correctness:  1-D binary array (1 = correct, 0 = wrong).
        n_bins:       Number of bins.
        strategy:     'uniform' (equal-width) or 'adaptive' (equal-mass).

    Returns:
        Scalar ECE value in [0, 1].
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    assert confidences.shape == correctness.shape, "Shape mismatch"

    if len(confidences) == 0:
        return 0.0

    bin_indices = _get_bin_indices(confidences, n_bins, strategy)

    ece_val = 0.0
    n = len(confidences)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece_val)

def mce(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Maximum Calibration Error — worst-case bin gap."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bin_indices = _get_bin_indices(confidences, n_bins, strategy)

    max_gap = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        max_gap = max(max_gap, abs(bin_acc - bin_conf))
    return float(max_gap)

def ace(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Adaptive Calibration Error (equal-mass / quantile bins).

    Each bin has roughly the same number of samples, which avoids
    empty-bin problems common with uniform binning on skewed
    confidence distributions (very common with LLMs).
    """
    return ece(confidences, correctness, n_bins=n_bins, strategy="adaptive")

def brier_score(
    confidences: np.ndarray,
    correctness: np.ndarray,
) -> float:
    """Brier Score = mean( (confidence - correctness)^2 ).

    Decomposes into calibration + refinement + uncertainty.
    Lower is better; 0 = perfect.
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    return float(np.mean((confidences - correctness) ** 2))

def log_loss_score(
    confidences: np.ndarray,
    correctness: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Binary cross-entropy (log loss). Lower is better."""
    confidences = np.clip(np.asarray(confidences, dtype=float), eps, 1 - eps)
    correctness = np.asarray(correctness, dtype=float)
    return float(-np.mean(
        correctness * np.log(confidences) + (1 - correctness) * np.log(1 - confidences)
    ))

@dataclass
class BrierDecomposition:
    """Murphy decomposition of the Brier score."""
    reliability: float   # calibration component (lower = better calibrated)
    resolution: float    # sharpness component (higher = better)
    uncertainty: float   # base rate uncertainty (constant for dataset)
    brier: float         # = reliability - resolution + uncertainty

def brier_decomposition(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Murphy decomposition: Brier = Reliability - Resolution + Uncertainty."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    n = len(confidences)
    base_rate = correctness.mean()
    uncertainty = base_rate * (1 - base_rate)

    bin_indices = _get_bin_indices(confidences, n_bins, "uniform")

    reliability = 0.0
    resolution = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        nb = mask.sum()
        if nb == 0:
            continue
        bin_acc = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        reliability += nb / n * (bin_conf - bin_acc) ** 2
        resolution += nb / n * (bin_acc - base_rate) ** 2

    return BrierDecomposition(
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        brier=float(reliability - resolution + uncertainty),
    )

@dataclass
class ReliabilityBin:
    bin_lower: float
    bin_upper: float
    bin_mid: float
    avg_confidence: float
    avg_accuracy: float
    count: int
    gap: float  # avg_confidence - avg_accuracy

def reliability_bins(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> List[ReliabilityBin]:
    """Return per-bin statistics for reliability diagrams."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    bin_indices = _get_bin_indices(confidences, n_bins, strategy)

    if strategy == "uniform":
        boundaries = np.linspace(0, 1, n_bins + 1)
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        boundaries = np.percentile(confidences, quantiles)
        boundaries[0] = 0.0
        boundaries[-1] = 1.0

    bins = []
    for b in range(n_bins):
        mask = bin_indices == b
        cnt = int(mask.sum())
        lo, hi = float(boundaries[b]), float(boundaries[b + 1])
        if cnt == 0:
            bins.append(ReliabilityBin(lo, hi, (lo + hi) / 2, 0.0, 0.0, 0, 0.0))
        else:
            acc = float(correctness[mask].mean())
            conf = float(confidences[mask].mean())
            bins.append(ReliabilityBin(lo, hi, (lo + hi) / 2, conf, acc, cnt, conf - acc))
    return bins

def reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Reliability Diagram",
    ax=None,
):
    """Plot a reliability diagram + confidence histogram (matplotlib).

    Returns the matplotlib Axes (creates a figure if ax is None).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for reliability_diagram()")

    bins_data = reliability_bins(confidences, correctness, n_bins, strategy)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    mids = [b.bin_mid for b in bins_data]
    accs = [b.avg_accuracy for b in bins_data]
    confs = [b.avg_confidence for b in bins_data]
    counts = [b.count for b in bins_data]
    width = 1.0 / n_bins

    # Gap bars (red = overconfident, blue = underconfident)
    colors = ["#e74c3c" if b.gap > 0 else "#3498db" for b in bins_data]
    ax.bar(mids, accs, width=width * 0.85, alpha=0.7, color=colors,
           edgecolor="white", linewidth=0.5, label="Accuracy")

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Confidence markers
    ax.scatter(mids, confs, marker="x", color="black", s=30, zorder=5,
               label="Avg confidence")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    # Inset histogram of confidence distribution
    inset = ax.inset_axes([0.58, 0.02, 0.4, 0.2])
    inset.bar(mids, counts, width=width * 0.85, color="#95a5a6", alpha=0.6)
    inset.set_xlabel("Conf", fontsize=6)
    inset.set_ylabel("Count", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.set_xlim(0, 1)

    plt.tight_layout()
    return ax

@dataclass
class CalibrationReport:
    ece: float
    mce: float
    ace: float
    brier: float
    log_loss: float
    brier_decomp: BrierDecomposition
    n_samples: int
    mean_confidence: float
    accuracy: float
    overconfidence: float  # mean_confidence - accuracy

    def __str__(self):
        lines = [
            f"{'Calibration Report':=^50}",
            f"  Samples:           {self.n_samples}",
            f"  Accuracy:          {self.accuracy:.4f}",
            f"  Mean confidence:   {self.mean_confidence:.4f}",
            f"  Overconfidence:    {self.overconfidence:+.4f}",
            f"  ─────────────────────────────────",
            f"  ECE (uniform):     {self.ece:.4f}",
            f"  MCE:               {self.mce:.4f}",
            f"  ACE (adaptive):    {self.ace:.4f}",
            f"  Brier Score:       {self.brier:.4f}",
            f"  Log Loss:          {self.log_loss:.4f}",
            f"  ─────────────────────────────────",
            f"  Brier Reliability: {self.brier_decomp.reliability:.4f}  (↓ better)",
            f"  Brier Resolution:  {self.brier_decomp.resolution:.4f}  (↑ better)",
            f"  Brier Uncertainty: {self.brier_decomp.uncertainty:.4f}",
            f"{'':=^50}",
        ]
        return "\n".join(lines)

def calibration_report(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute all calibration metrics in one call."""
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    return CalibrationReport(
        ece=ece(confidences, correctness, n_bins),
        mce=mce(confidences, correctness, n_bins),
        ace=ace(confidences, correctness, n_bins),
        brier=brier_score(confidences, correctness),
        log_loss=log_loss_score(confidences, correctness),
        brier_decomp=brier_decomposition(confidences, correctness, n_bins),
        n_samples=len(confidences),
        mean_confidence=float(confidences.mean()),
        accuracy=float(correctness.mean()),
        overconfidence=float(confidences.mean() - correctness.mean()),
    )

@dataclass
class MultiLabelCalibrationReport:
    """Aggregated calibration across all label classes."""
    per_class: Dict[str, CalibrationReport]
    macro_ece: float
    macro_ace: float
    macro_brier: float
    micro_ece: float
    micro_brier: float
    class_names: List[str]

    def __str__(self):
        lines = [f"{'Multi-Label Calibration Report':=^60}"]
        lines.append(f"  Classes: {len(self.class_names)}")
        lines.append(f"  Macro ECE:   {self.macro_ece:.4f}")
        lines.append(f"  Macro ACE:   {self.macro_ace:.4f}")
        lines.append(f"  Macro Brier: {self.macro_brier:.4f}")
        lines.append(f"  Micro ECE:   {self.micro_ece:.4f}")
        lines.append(f"  Micro Brier: {self.micro_brier:.4f}")
        lines.append(f"  {'─' * 56}")
        lines.append(f"  {'Class':<20} {'ECE':>7} {'ACE':>7} {'Brier':>7} {'OverConf':>9} {'N':>6}")
        for name in self.class_names:
            r = self.per_class[name]
            lines.append(
                f"  {name:<20} {r.ece:>7.4f} {r.ace:>7.4f} "
                f"{r.brier:>7.4f} {r.overconfidence:>+9.4f} {r.n_samples:>6}"
            )
        lines.append(f"{'':=^60}")
        return "\n".join(lines)

def multilabel_calibration_report(
    class_confidences: Dict[str, np.ndarray],
    class_correctness: Dict[str, np.ndarray],
    n_bins: int = 10,
) -> MultiLabelCalibrationReport:
    """Calibration report across multiple label classes.

    Args:
        class_confidences:  {class_name: array of confidences}
        class_correctness:  {class_name: array of 0/1 correctness}

    Both dicts must have the same keys.
    """
    per_class = {}
    all_conf = []
    all_corr = []

    for cls in class_confidences:
        c = np.asarray(class_confidences[cls], dtype=float)
        y = np.asarray(class_correctness[cls], dtype=float)
        per_class[cls] = calibration_report(c, y, n_bins)
        all_conf.append(c)
        all_corr.append(y)

    class_names = sorted(per_class.keys())

    # Macro averages
    macro_ece = np.mean([per_class[c].ece for c in class_names])
    macro_ace = np.mean([per_class[c].ace for c in class_names])
    macro_brier = np.mean([per_class[c].brier for c in class_names])

    # Micro averages (pool all samples)
    pooled_conf = np.concatenate(all_conf)
    pooled_corr = np.concatenate(all_corr)
    micro_ece_val = ece(pooled_conf, pooled_corr, n_bins)
    micro_brier_val = brier_score(pooled_conf, pooled_corr)

    return MultiLabelCalibrationReport(
        per_class=per_class,
        macro_ece=float(macro_ece),
        macro_ace=float(macro_ace),
        macro_brier=float(macro_brier),
        micro_ece=float(micro_ece_val),
        micro_brier=float(micro_brier_val),
        class_names=class_names,
    )

def find_optimal_temperature(
    confidences: np.ndarray,
    correctness: np.ndarray,
    bounds: Tuple[float, float] = (0.1, 10.0),
    n_steps: int = 1000,
) -> float:
    """Find temperature T that minimizes ECE via grid search.

    Rescales confidences as: calibrated = sigmoid(logit(conf) / T)
    """
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)

    eps = 1e-10
    logits = np.log(np.clip(confidences, eps, 1 - eps) / (1 - np.clip(confidences, eps, 1 - eps)))

    best_t = 1.0
    best_ece = float("inf")
    for t in np.linspace(bounds[0], bounds[1], n_steps):
        scaled = 1 / (1 + np.exp(-logits / t))
        e = ece(scaled, correctness)
        if e < best_ece:
            best_ece = e
            best_t = t

    return float(best_t)

def apply_temperature(confidences: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to confidences."""
    eps = 1e-10
    c = np.clip(np.asarray(confidences, dtype=float), eps, 1 - eps)
    logits = np.log(c / (1 - c))
    return 1 / (1 + np.exp(-logits / temperature))

DEFAULT_MARKER_MAP = {
    # High confidence markers
    "definitely": 0.95, "certainly": 0.95, "clearly": 0.93,
    "obviously": 0.92, "without a doubt": 0.95, "i'm confident": 0.90,
    "for sure": 0.92, "undoubtedly": 0.95,
    # Medium confidence
    "likely": 0.75, "probably": 0.75, "i think": 0.70,
    "it seems": 0.65, "appears to be": 0.65, "most likely": 0.80,
    "i believe": 0.72, "seems like": 0.65,
    # Low confidence
    "possibly": 0.40, "might be": 0.35, "could be": 0.35,
    "i'm not sure": 0.30, "perhaps": 0.40, "not certain": 0.30,
    "hard to say": 0.25, "uncertain": 0.25, "i'm guessing": 0.30,
    "maybe": 0.40,
}

def parse_epistemic_confidence(
    text: str,
    marker_map: Optional[Dict[str, float]] = None,
    default_confidence: float = 0.80,
) -> float:
    """Extract confidence from epistemic markers in free text.

    Scans text for hedging phrases and returns the confidence
    associated with the strongest (first-found) marker.
    If no marker is found, returns default_confidence.

    You should customize marker_map for your model/domain.
    """
    if marker_map is None:
        marker_map = DEFAULT_MARKER_MAP

    text_lower = text.lower()

    # Sort by phrase length descending to match longer phrases first
    sorted_markers = sorted(marker_map.keys(), key=len, reverse=True)

    for marker in sorted_markers:
        if marker in text_lower:
            return marker_map[marker]

    return default_confidence

def _get_bin_indices(
    confidences: np.ndarray, n_bins: int, strategy: str
) -> np.ndarray:
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
        bins[-1] += 1e-8  # include 1.0
        indices = np.digitize(confidences, bins) - 1
        indices = np.clip(indices, 0, n_bins - 1)
    elif strategy == "adaptive":
        quantiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(confidences, quantiles)
        bins[0] = 0.0
        bins[-1] += 1e-8
        indices = np.digitize(confidences, bins) - 1
        indices = np.clip(indices, 0, n_bins - 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return indices

def compare_probing_methods(
    correctness: np.ndarray,
    token_probs: Optional[np.ndarray] = None,
    self_reported: Optional[np.ndarray] = None,
    epistemic: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> Dict[str, CalibrationReport]:
    """Compare calibration across probing methods side-by-side.

    Pass whichever probing signals you have; None signals are skipped.
    Returns dict of {method_name: CalibrationReport}.
    """
    correctness = np.asarray(correctness, dtype=float)
    results = {}

    for name, confs in [
        ("token_probability", token_probs),
        ("self_reported", self_reported),
        ("epistemic_markers", epistemic),
    ]:
        if confs is not None:
            results[name] = calibration_report(
                np.asarray(confs, dtype=float), correctness, n_bins
            )

    return results

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate a slightly overconfident model
    n = 500
    true_probs = np.random.beta(2, 5, n)
    correctness = (np.random.rand(n) < true_probs).astype(float)
    # Model's reported confidence is inflated
    overconfident = np.clip(true_probs + np.random.normal(0.15, 0.05, n), 0.01, 0.99)

    print("=" * 50)
    print("DEMO: Single-class calibration")
    print("=" * 50)
    report = calibration_report(overconfident, correctness)
    print(report)

    # Temperature scaling
    T = find_optimal_temperature(overconfident, correctness)
    calibrated = apply_temperature(overconfident, T)
    print(f"\nOptimal temperature: {T:.3f}")
    report_after = calibration_report(calibrated, correctness)
    print(f"ECE before: {report.ece:.4f} → after: {report_after.ece:.4f}")
    print(f"Brier before: {report.brier:.4f} → after: {report_after.brier:.4f}")

    # Multi-label demo
    print("\n" + "=" * 60)
    print("DEMO: Multi-label calibration")
    print("=" * 60)
    classes = ["sarcasm", "political_humor", "wholesome", "dark_humor"]
    class_conf = {}
    class_corr = {}
    for cls in classes:
        n_cls = np.random.randint(100, 300)
        tp = np.random.beta(2, 3, n_cls)
        class_corr[cls] = (np.random.rand(n_cls) < tp).astype(float)
        class_conf[cls] = np.clip(tp + np.random.normal(0.1, 0.1, n_cls), 0.01, 0.99)

    ml_report = multilabel_calibration_report(class_conf, class_corr)
    print(ml_report)

    # Compare probing methods
    print("\n" + "=" * 50)
    print("DEMO: Probing method comparison")
    print("=" * 50)
    comparison = compare_probing_methods(
        correctness=correctness,
        token_probs=overconfident,
        self_reported=np.clip(overconfident + np.random.normal(0.05, 0.1, n), 0.01, 0.99),
        epistemic=np.clip(overconfident - np.random.normal(0.05, 0.05, n), 0.01, 0.99),
    )
    for method, rep in comparison.items():
        print(f"\n  [{method}]")
        print(f"    ECE={rep.ece:.4f}  ACE={rep.ace:.4f}  "
              f"Brier={rep.brier:.4f}  OverConf={rep.overconfidence:+.4f}")

    # Reliability diagram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        reliability_diagram(overconfident, correctness, title="Before temp scaling", ax=axes[0])
        reliability_diagram(calibrated, correctness, title="After temp scaling", ax=axes[1])
        fig.savefig("reliability_diagram.png", dpi=150, bbox_inches="tight")
        print("\nSaved reliability_diagram.png")
    except ImportError:
        print("\nInstall matplotlib for reliability diagrams: pip install matplotlib")