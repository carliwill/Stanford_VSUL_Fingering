"""
co2_fingers.regimes
===================
Detect the five physical time regimes of CO₂ convective fingering
(Riaz et al. 2006) from per-frame time-series measurements.

This module contains the ``TimeRegimeDetector`` class, which is a
cleaned-up version of Cell 4 (and related cells) in
``time_regime_detector_v3``.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# Regime metadata
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    0: "#B5D4F4",   # blue   – diffusive
    1: "#FAC775",   # amber  – onset marker
    2: "#9FE1CB",   # teal   – linear growth
    3: "#F5C4B3",   # coral  – nonlinear / merging
    4: "#C0DD97",   # green  – convective
}

REGIME_LABELS = {
    0: "Diffusive  (∝ √t)",
    1: "Onset  (critical time tₒ)",
    2: "Linear growth",
    3: "Nonlinear / merging",
    4: "Convective  (∝ t)",
}


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TimeRegimeDetector:
    """
    Detect the five CO₂-fingering time regimes from per-frame measurements.

    The five regimes are:

    0. **Diffusive** — flat interface, σ_h at baseline.
    1. **Onset tₒ** — first unstable perturbations appear.
    2. **Linear growth** — finger amplitudes grow exponentially.
    3. **Nonlinear / merging** — finger count drops by *drop_fraction*.
    4. **Convective** — tip advances at nearly constant velocity (log-log slope ≥ threshold).

    Parameters
    ----------
    times : array-like
        Elapsed time per frame (seconds).
    n_fingers : array-like
        Finger count per frame.
    tip_positions : array-like
        Deepest finger-tip depth in pixels per frame.
    sigma_list : array-like
        Interface roughness σ_h per frame (pixels).
    px_per_m : float
        Pixels per metre (default 1.0 — dimensionless).
    Ra : float or None
        Rayleigh number; enables theoretical-onset reference line on plots.

    Examples
    --------
    >>> det = TimeRegimeDetector(times, n, tips, sigmas, px_per_m=PX_PER_METRE)
    >>> det.detect(manual_onset_sec=330 * 60)
    >>> det.plot()
    >>> results = det.report()
    """

    def __init__(
        self,
        times,
        n_fingers,
        tip_positions,
        sigma_list,
        px_per_m: float = 1.0,
        Ra=None,
    ):
        self.t   = np.asarray(times,         dtype=float)
        self.n   = np.asarray(n_fingers,     dtype=float)
        self.tip = np.asarray(tip_positions, dtype=float)
        self.sig = np.asarray(sigma_list,    dtype=float)
        self.px_per_m = float(px_per_m)
        self.Ra       = Ra
        self.regime_boundaries: dict = {}
        self.t_onset = self.t_merge = self.t_convect = None
        self._tip_slope = self._growth_rate = np.nan
        self._slopes    = None
        self._onset_source = "auto"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tip_log_slopes(self, half_window: int = 5) -> np.ndarray:
        """Local log–log slope of tip depth vs. time."""
        log_t   = np.log(np.clip(self.t,   1e-12, None))
        log_tip = np.log(np.clip(self.tip, 1e-12, None))
        n       = len(log_t)
        slopes  = np.full(n, np.nan)
        for i in range(half_window, n - half_window):
            sl, *_ = linregress(
                log_t[i - half_window: i + half_window + 1],
                log_tip[i - half_window: i + half_window + 1],
            )
            slopes[i] = sl
        return slopes

    def _find_onset_auto(self, sigma_threshold: float) -> float:
        nb  = max(3, len(self.sig) // 10)
        exc = np.where(self.sig > sigma_threshold * self.sig[:nb].mean())[0]
        return self.t[exc[0]] if len(exc) else self.t[len(self.t) // 3]

    def _find_merge_onset(self, drop_fraction: float) -> float:
        peak_n = np.nanmax(self.n)
        if peak_n == 0:
            return self.t[len(self.t) // 2]
        pi  = int(np.nanargmax(self.n))
        exc = np.where(self.n[pi:] < peak_n * (1 - drop_fraction))[0]
        return self.t[pi + exc[0]] if len(exc) else self.t[-1]

    def _find_convective_onset(self, slopes: np.ndarray, slope_threshold: float) -> float:
        oi  = np.searchsorted(self.t, self.t_onset)
        exc = np.where(slopes[oi:] >= slope_threshold)[0]
        return self.t[oi + exc[0]] if len(exc) else self.t[-1]

    def _linear_growth_slope(self) -> float:
        mask = (self.t >= self.t_onset) & (self.t <= self.t_merge)
        if mask.sum() < 3:
            return np.nan
        sl, *_ = linregress(self.t[mask], np.log(np.clip(self.sig[mask], 1e-12, None)))
        return sl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        sigma_threshold: float = 2.0,
        slope_threshold: float = 0.80,
        drop_fraction: float = 0.10,
        manual_onset_sec: float | None = None,
    ):
        """
        Identify regime boundaries.

        Parameters
        ----------
        sigma_threshold : float
            σ_h rise factor to call onset automatically (default 2.0).
        slope_threshold : float
            Log–log tip slope above which the convective regime begins
            (default 0.80).
        drop_fraction : float
            Fractional finger-count drop to call merging (default 0.10).
        manual_onset_sec : float or None
            If given (seconds), skip the σ_h threshold and use this value
            directly as tₒ.  Convert from minutes with ``min * 60``.
        """
        slopes = self._tip_log_slopes()

        if manual_onset_sec is not None:
            nearest = int(np.argmin(np.abs(self.t - manual_onset_sec)))
            self.t_onset       = self.t[nearest]
            self._onset_source = "manual"
        else:
            self.t_onset       = self._find_onset_auto(sigma_threshold)
            self._onset_source = "auto"

        self.t_merge   = self._find_merge_onset(drop_fraction)
        self.t_convect = self._find_convective_onset(slopes, slope_threshold)
        self.t_merge   = max(self.t_merge,   self.t_onset)
        self.t_convect = max(self.t_convect, self.t_merge)

        t0, tf = self.t[0], self.t[-1]
        self.regime_boundaries = {
            0: (t0,           self.t_onset),
            1: (self.t_onset, self.t_onset),
            2: (self.t_onset, self.t_merge),
            3: (self.t_merge, self.t_convect),
            4: (self.t_convect, tf),
        }

        conv_mask = self.t >= self.t_convect
        if conv_mask.sum() >= 2:
            sl, *_ = linregress(self.t[conv_mask], self.tip[conv_mask])
            self._tip_slope = sl

        self._growth_rate = self._linear_growth_slope()
        self._slopes      = slopes

        # Print summary
        sep = "─" * 58
        print(sep)
        print("  TIME REGIME DETECTION SUMMARY")
        print(sep)
        print(f"  Onset source  :  "
              f"{'MANUAL (penetration depth)' if self._onset_source == 'manual' else 'AUTO (σ_h threshold)'}")
        print(sep)
        print(f"  {'Regime':<24}  {'t_start':>9}  {'t_end':>9}")
        print(sep)
        t0 = self.t[0]
        tf = self.t[-1]
        print(f"  {'0  Diffusive':<24}  {t0:>9.4g}  {self.t_onset:>9.4g}")
        print(f"  {'1  Onset tₒ':<24}  {self.t_onset:>9.4g}  {'(marker)':>9}")
        print(f"  {'2  Linear growth':<24}  {self.t_onset:>9.4g}  {self.t_merge:>9.4g}")
        if not np.isnan(self._growth_rate):
            print(f"       roughness growth rate σ = {self._growth_rate:.4g} / t-unit")
        print(f"  {'3  Nonlinear / merging':<24}  {self.t_merge:>9.4g}  {self.t_convect:>9.4g}")
        print(f"  {'4  Convective':<24}  {self.t_convect:>9.4g}  {tf:>9.4g}")
        if not np.isnan(self._tip_slope):
            print(f"       tip velocity = {self._tip_slope:.4g} px/t  "
                  f"({self._tip_slope / self.px_per_m:.4g} m/t)")
        if self.Ra:
            print(f"\n  Theoretical tₒ (Riaz 2006, Ra={self.Ra}): {146 / self.Ra:.4g}")
        print(sep)

    def plot(self, figsize=(13, 9), save_path: str | None = None):
        """
        Four-panel diagnostic figure (σ_h, finger count, log–log tip, slope).

        Parameters
        ----------
        figsize : tuple
            Figure size (default ``(13, 9)``).
        save_path : str or None
            If given, save the figure to this path.
        """
        if not self.regime_boundaries:
            raise RuntimeError("Call detect() before plot().")

        fig, axes = plt.subplots(2, 2, figsize=figsize, tight_layout=True)
        onset_label = (f"tₒ manual = {self.t_onset:.3g}"
                       if self._onset_source == "manual"
                       else f"tₒ auto = {self.t_onset:.3g}")
        fig.suptitle(
            f"CO₂ Fingering — Time Regime Analysis  [onset: {self._onset_source}]",
            fontsize=13, fontweight="bold",
        )
        ax_s, ax_n, ax_tip, ax_sl = axes.flat

        def shade(ax):
            for rid, (ta, tb) in self.regime_boundaries.items():
                if rid == 1:
                    continue
                ax.axvspan(ta, tb, alpha=0.18, color=REGIME_COLORS[rid], zorder=0)
            ax.axvline(self.t_onset,   color=REGIME_COLORS[1], lw=1.8, ls="--",
                       label=onset_label,                       zorder=3)
            ax.axvline(self.t_merge,   color=REGIME_COLORS[3], lw=1.2, ls=":",
                       label=f"t_merge = {self.t_merge:.3g}",  zorder=3)
            ax.axvline(self.t_convect, color=REGIME_COLORS[4], lw=1.2, ls="-.",
                       label=f"t_conv = {self.t_convect:.3g}", zorder=3)
            if self.Ra:
                ax.axvline(146 / self.Ra, color="#D85A30", lw=1.0,
                           ls=(0, (3, 1, 1, 1)),
                           label=f"tₒ theory={146/self.Ra:.3g}", zorder=3)

        ax_s.plot(self.t, self.sig, color="#185FA5", lw=1.5, label="σ_h")
        shade(ax_s)
        ax_s.set(xlabel="Time (s)", ylabel="σ_h  (pixels)",
                 title="(A) Interface roughness σ_h")
        ax_s.legend(fontsize=7, loc="upper left")

        ax_n.plot(self.t, self.n, color="#0F6E56", lw=1.5,
                  marker="o", markersize=3, label="N fingers")
        shade(ax_n)
        ax_n.set(xlabel="Time (s)", ylabel="Finger count N",
                 title="(B) Finger count")
        ax_n.legend(fontsize=7, loc="upper right")

        valid = (self.t > 0) & (self.tip > 0)
        ax_tip.loglog(self.t[valid], self.tip[valid],
                      color="#3C3489", lw=1.5, label="Tip depth")
        t_ref   = np.array([self.t[valid].min(), self.t[valid].max()])
        tip_mid = np.nanmedian(self.tip[valid])
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 0.5,
                      ls="--", color="#888780", lw=1, label="slope 0.5 (diffusive)")
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 1.0,
                      ls=":",  color="#639922", lw=1, label="slope 1.0 (convective)")
        ax_tip.axvline(self.t_onset,   color=REGIME_COLORS[1], lw=1.2, ls="--", zorder=3)
        ax_tip.axvline(self.t_convect, color=REGIME_COLORS[4], lw=1.2, ls="-.", zorder=3)
        ax_tip.set(xlabel="Time (log)", ylabel="Tip depth (log)",
                   title="(C) Tip penetration  [log–log]")
        ax_tip.legend(fontsize=7)

        ax_sl.plot(self.t, self._slopes, color="#993C1D", lw=1.5,
                   label="d log(tip) / d log(t)")
        ax_sl.axhline(0.5, color="#888780", ls="--", lw=1, label="0.5 (diffusive)")
        ax_sl.axhline(1.0, color="#639922", ls=":",  lw=1, label="1.0 (convective)")
        ax_sl.axvline(self.t_onset,   color=REGIME_COLORS[1], lw=1.2, ls="--", zorder=3)
        ax_sl.axvline(self.t_convect, color=REGIME_COLORS[4], lw=1.2, ls="-.", zorder=3)
        ax_sl.set_ylim(-0.3, 2.2)
        ax_sl.set(xlabel="Time (s)", ylabel="Local log–log slope",
                  title="(D) Penetration slope  (0.5 → 1.0 transition)")
        ax_sl.legend(fontsize=7)

        patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.55,
                                  label=REGIME_LABELS[r]) for r in [0, 2, 3, 4]]
        fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=8,
                   title="Regime shading", title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))
        plt.subplots_adjust(bottom=0.12)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved → {save_path}")
        plt.show()

    def report(self) -> dict:
        """
        Return a dictionary of all detected regime boundaries and statistics.

        Returns
        -------
        dict with keys:
            ``t_onset``, ``onset_source``, ``t_merge``, ``t_convective``,
            ``roughness_growth_rate``, ``tip_velocity_px_per_t``,
            ``tip_velocity_m_per_t``, ``regime_boundaries``.
        """
        return {
            "t_onset":               self.t_onset,
            "onset_source":          self._onset_source,
            "t_merge":               self.t_merge,
            "t_convective":          self.t_convect,
            "roughness_growth_rate": self._growth_rate,
            "tip_velocity_px_per_t": self._tip_slope,
            "tip_velocity_m_per_t":  (
                self._tip_slope / self.px_per_m if not np.isnan(self._tip_slope) else np.nan
            ),
            "regime_boundaries":     self.regime_boundaries,
        }
