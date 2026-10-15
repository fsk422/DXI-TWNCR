"""Plotting helpers: comparison bars, attribution heatmaps, convergence traces."""
from typing import Optional, List
import os
import numpy as np
import numpy.typing as npt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

_FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "figures")


def _ensure_figure_dir() -> str:
    os.makedirs(_FIGURE_DIR, exist_ok=True)
    return _FIGURE_DIR


def plot_detection_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "auc_roc",
    title: str = "Detection Performance Comparison",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> str:
    """Bar chart comparing detection methods on a given metric.

    Parameters
    ----------
    results : dict
        method_name -> {metric_name: value}
    metric : str
        Which metric to plot.
    title : str
        Plot title.
    save_path : str, optional
        If None, auto-generate path in figures/.
    figsize : tuple
        Figure size.

    Returns
    -------
    save_path : str
        Path to the saved PDF.
    """
    methods = list(results.keys())
    values = [results[m].get(metric, 0) for m in methods]
    errors = [results[m].get(f"{metric}_std", 0) for m in methods]

    colors = sns.color_palette("viridis", len(methods))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(methods, values, yerr=errors, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(metric.upper(), fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_figure_dir(),
                                 f"detection_comparison_{metric}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_attribution_heatmap(
    contributions: npt.NDArray[np.floating],
    feature_names: Optional[list[str]] = None,
    sample_indices: Optional[list[int]] = None,
    title: str = "Feature Attribution Heatmap",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
) -> str:
    """Heatmap of per-feature anomaly contributions.

    Parameters
    ----------
    contributions : np.ndarray of shape (n_samples, n_features)
        Per-feature contribution values.
    feature_names : list of str, optional
    sample_indices : list of int, optional
        Which samples to show.
    title : str
    save_path : str, optional
    figsize : tuple, optional

    Returns
    -------
    save_path : str
    """
    n_samples, n_features = contributions.shape

    if sample_indices is not None:
        contributions = contributions[sample_indices]

    if figsize is None:
        figsize = (max(8, n_features * 1.2), max(4, contributions.shape[0] * 0.3))

    fig, ax = plt.subplots(figsize=figsize)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(contributions, cmap=cmap, center=0, annot=True, fmt=".3f",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Contribution"})

    if feature_names:
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
    else:
        ax.set_xticklabels([f"F{j}" for j in range(n_features)], rotation=45, ha="right")

    ax.set_ylabel("Sample Index")
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_figure_dir(), "attribution_heatmap.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_convergence(
    param_trace: list[dict],
    title: str = "Parameter Convergence",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> str:
    """Plot convergence trace of parameters over incremental updates.

    Handles heterogeneous trace formats from different detectors:
    - IX-WNCROD: "step", "lammda", "delta_mean"
    - IX-IF:     "step", "n_trees"
    - IX-AE:     "step", "loss"
    - IX-LOF:    "step", "mean_lof"
    - IX-OCSVM:  "step", "n_nonzero_alpha", "rho"

    Parameters
    ----------
    param_trace : list of dict
        Each dict contains "step" and detector-specific metrics.
    title : str
    save_path : str, optional
    figsize : tuple

    Returns
    -------
    save_path : str
    """
    if not param_trace:
        return ""

    steps = [t["step"] for t in param_trace]
    available_keys = list(param_trace[0].keys())
    metric_keys = [k for k in available_keys if k != "step" and k != "window_size" and k != "update_time_ms"]

    if not metric_keys:
        return ""

    n_metrics = len(metric_keys)
    if n_metrics == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(n_metrics, 1, figsize=(figsize[0], figsize[1] * n_metrics / 2), sharex=True)

    colors = ["b", "r", "g", "purple", "orange"]
    markers = ["o", "s", "^", "D", "v"]

    for i, key in enumerate(metric_keys):
        vals = [t[key] for t in param_trace]
        ax = axes[i] if n_metrics > 1 else axes[0]
        ax.plot(steps, vals, f"{colors[i % len(colors)]}-{markers[i % len(markers)]}",
                markersize=4, linewidth=1.5, label=key.replace("_", " ").title())
        ax.set_ylabel(key.replace("_", " ").title(), fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    if n_metrics > 1:
        axes[-1].set_xlabel("Update Step", fontsize=13)
    else:
        axes[0].set_xlabel("Update Step", fontsize=13)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_figure_dir(), f"convergence_trace_{title.replace(' ', '_')}.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_attribution_stability(
    js_divergences: npt.NDArray[np.floating],
    title: str = "Attribution Stability (JS Divergence)",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> str:
    """Violin plot of JS divergence under input perturbations.

    Lower JS divergence = more stable attributions.

    Parameters
    ----------
    js_divergences : np.ndarray
        Per-sample JS divergences.
    title : str
    save_path : str, optional
    figsize : tuple

    Returns
    -------
    save_path : str
    """
    fig, ax = plt.subplots(figsize=figsize)

    parts = ax.violinplot(js_divergences[~np.isnan(js_divergences)],
                          positions=[0], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(sns.color_palette("muted")[0])
        pc.set_alpha(0.7)

    ax.set_xticks([])
    ax.set_ylabel("JS Divergence", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    stats_text = (f"Mean: {np.mean(js_divergences):.4f}\n"
                  f"Std:  {np.std(js_divergences):.4f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(_ensure_figure_dir(), "attribution_stability.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
