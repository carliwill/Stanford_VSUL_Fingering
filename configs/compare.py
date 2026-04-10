"""
co2_fingers.compare
===================
Compare multiple ``Experiment`` objects side-by-side on shared axes.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence

from .experiment import Experiment
from .regimes import REGIME_COLORS, REGIME_LABELS


def compare_experiments(
    experiments: Sequence[Experiment],
    save_path: str | None = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Four-panel comparison of N experiments on shared axes.

    Panels:
      (A) Finger count vs. time
      (B) Mean finger width (cm) vs. time
      (C) Merging metric (N × width) vs. time
      (D) Log–log tip depth vs. time

    Onset markers are drawn for each experiment using its detected tₒ.

    Parameters
    ----------
    experiments : sequence of Experiment
        Each must have had ``.run()`` called already.
    save_path : str or None
        If given, save the figure.
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Experiment comparison", fontsize=14, fontweight="bold")
    ax_n, ax_w, ax_m, ax_tip = axes.flat

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, exp in enumerate(experiments):
        col   = colors[i % len(colors)]
        label = exp.name
        t     = exp.times
        n     = exp.n_fingers
        w     = exp.mean_widths_cm
        tip   = exp.tips

        # onset marker
        t_onset = exp.detector.t_onset if exp.detector else None

        ax_n.plot(t, n, color=col, lw=1.8, marker="o", markersize=2, label=label)
        ax_w.plot(t, w, color=col, lw=1.8, label=label)
        ax_m.plot(t, n * w, color=col, lw=1.8, label=label)

        valid = (t > 0) & (tip > 0)
        ax_tip.loglog(t[valid], tip[valid], color=col, lw=1.8, label=label)

        if t_onset is not None:
            for ax in [ax_n, ax_w, ax_m]:
                ax.axvline(t_onset, color=col, lw=1, ls="--", alpha=0.6)
            ax_tip.axvline(t_onset, color=col, lw=1, ls="--", alpha=0.6)

    # Reference lines on log-log
    all_t   = np.concatenate([e.times[(e.times > 0)] for e in experiments])
    all_tip = np.concatenate([e.tips[(e.tips > 0)]   for e in experiments])
    if len(all_t) and len(all_tip):
        t_ref   = np.array([all_t.min(), all_t.max()])
        tip_mid = np.nanmedian(all_tip)
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 0.5,
                      ls="--", color="gray", lw=1, alpha=0.6, label="∝ √t")
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 1.0,
                      ls=":",  color="gray", lw=1, alpha=0.6, label="∝ t")

    ax_n.set(xlabel="Time (s)", ylabel="Finger count N",   title="(A) Finger count")
    ax_w.set(xlabel="Time (s)", ylabel="Mean width (cm)",  title="(B) Mean finger width")
    ax_m.set(xlabel="Time (s)", ylabel="N × width (cm)",   title="(C) Merging metric")
    ax_tip.set(xlabel="Time (log s)", ylabel="Tip depth (px, log)", title="(D) Tip penetration")

    for ax in axes.flat:
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison figure saved → {save_path}")
    plt.show()
    return fig


def compare_regimes(
    experiments: Sequence[Experiment],
    save_path: str | None = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    Horizontal timeline showing detected regime boundaries for each experiment.

    Parameters
    ----------
    experiments : sequence of Experiment
    save_path : str or None
    figsize : tuple

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    regime_ids = [0, 2, 3, 4]   # skip 1 (point marker)

    for row_i, exp in enumerate(experiments):
        if exp.detector is None:
            continue
        rb = exp.detector.regime_boundaries
        for rid in regime_ids:
            ta, tb = rb[rid]
            ax.barh(row_i, tb - ta, left=ta,
                    color=REGIME_COLORS[rid], alpha=0.7, edgecolor="none", height=0.6)
        # onset tick
        ax.axvline(exp.detector.t_onset, ymin=(row_i) / len(experiments),
                   ymax=(row_i + 1) / len(experiments),
                   color=REGIME_COLORS[1], lw=2)

    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels([e.name for e in experiments])
    ax.set_xlabel("Time (s)")
    ax.set_title("Regime timelines")

    # legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.7,
                               label=REGIME_LABELS[r]) for r in regime_ids]
    patches.append(mpatches.Patch(color=REGIME_COLORS[1], label=REGIME_LABELS[1]))
    ax.legend(handles=patches, fontsize=7, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Regime timeline saved → {save_path}")
    plt.show()
    return fig


def summary_table(experiments: Sequence[Experiment]) -> "pd.DataFrame":
    """
    Return a tidy DataFrame with one row per experiment summarizing
    key regime boundaries and statistics.

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd

    rows = []
    for exp in experiments:
        d = exp.detector
        row = {"experiment": exp.name}
        if d is not None:
            rep = d.report()
            row.update({
                "onset_source":          rep["onset_source"],
                "t_onset_min":           rep["t_onset"] / 60 if rep["t_onset"] else None,
                "t_merge_min":           rep["t_merge"] / 60 if rep["t_merge"] else None,
                "t_convective_min":      rep["t_convective"] / 60 if rep["t_convective"] else None,
                "roughness_growth_rate": rep["roughness_growth_rate"],
                "tip_velocity_m_per_t":  rep["tip_velocity_m_per_t"],
            })
        rows.append(row)
    return pd.DataFrame(rows)
