"""
co2_fingers.regimes
===================
Detect the five physical time regimes of CO2 convective fingering
(Riaz et al. 2006) from per-frame time-series measurements.

This module contains the ``TimeRegimeDetector`` class.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

_SECONDS_PER_HOUR = 3600.0

# ---------------------------------------------------------------------------
# Regime metadata
# ---------------------------------------------------------------------------

REGIME_COLORS = {
    0: "#B5D4F4",   # blue   - diffusive
    1: "#FAC775",   # amber  - onset marker
    2: "#9FE1CB",   # teal   - linear growth
    3: "#F5C4B3",   # coral  - nonlinear / merging
    4: "#C0DD97",   # green  - convective
}

REGIME_LABELS = {
    0: "Diffusive  (prop. sqrt(t))",
    1: "Onset  (critical time t_o)",
    2: "Linear growth",
    3: "Nonlinear / merging",
    4: "Convective  (prop. t)",
}

_BOTTOM_COLOR = "#8B0000"   # dark red for "first finger hits bottom"


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class TimeRegimeDetector:
    """
    Detect the five CO2-fingering time regimes from per-frame measurements.

    The five regimes are:

    0. **Diffusive** -- flat interface, sigma_h at baseline.
    1. **Onset t_o** -- first unstable perturbations appear.
    2. **Linear growth** -- finger amplitudes grow exponentially.
    3. **Nonlinear / merging** -- finger count drops by *drop_fraction*.
    4. **Convective** -- tip advances at nearly constant velocity.

    Parameters
    ----------
    times : array-like
        Elapsed time per frame (seconds).
    n_fingers : array-like
        Finger count per frame.
    tip_positions : array-like
        Deepest finger-tip depth in pixels per frame.
    sigma_list : array-like
        Interface roughness sigma_h per frame (pixels).
    px_per_m : float
        Pixels per metre (default 1.0).
    Ra : float or None
        Rayleigh number; enables theoretical-onset reference line on plots.

    Examples
    --------
    >>> det = TimeRegimeDetector(times, n, tips, sigmas, px_per_m=PX_PER_METRE)
    >>> det.detect(y_top=2200, y_bot=3200)
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
        self.t_bottom: float | None = None
        self.bottom_source: str | None = None
        self._tip_slope = self._growth_rate = np.nan
        self._slopes    = None
        self._onset_source = "auto"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tip_log_slopes(self, half_window: int = 5) -> np.ndarray:
        """Local log-log slope of tip depth vs. time."""
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

    def _find_slope_corner(
        self,
        slopes: np.ndarray,
        min_drop: float = 0.3,
        smooth_window: int = 11,
    ) -> float | None:
        """
        Find the abrupt corner where the log-log slope transitions from a
        sustained positive value to a much flatter region.

        Strategy: piecewise two-line fit (breakpoint scan) over the slope
        curve in the region after convective onset.  The breakpoint that
        minimises combined residuals of left and right linear fits is the
        corner.  Accepted only if the drop from left-segment mean to
        right-segment mean exceeds *min_drop*.

        Parameters
        ----------
        slopes : np.ndarray
            Local log-log slopes from :meth:`_tip_log_slopes`.
        min_drop : float
            Minimum required drop in mean slope across the corner
            (default 0.3 — catches a halving from ~0.8 to ~0.5 or bigger).
        smooth_window : int
            Savitzky-Golay window for pre-smoothing the slope signal before
            fitting (odd, default 11).

        Returns
        -------
        float or None
            Time (seconds) of the detected corner, or None if no clear
            corner is found.
        """
        from scipy.signal import savgol_filter as _sg

        oi = np.searchsorted(self.t, self.t_convect) if self.t_convect is not None else 0
        valid = ~np.isnan(slopes)
        mask = valid.copy()
        mask[:oi] = False

        idx = np.where(mask)[0]
        if len(idx) < 12:
            return None

        s = slopes[idx]
        # smooth the slope signal to suppress frame-to-frame noise
        w = min(smooth_window, len(s) if len(s) % 2 == 1 else len(s) - 1)
        if w < 3:
            return None
        s_sm = _sg(s, window_length=w, polyorder=2)

        # scan breakpoints (skip first/last 20% to avoid edge effects)
        margin = max(2, len(s_sm) // 5)
        best_err, best_k = np.inf, None
        for k in range(margin, len(s_sm) - margin):
            left  = s_sm[:k]
            right = s_sm[k:]
            # residual = sum of variances weighted by segment length
            err = np.var(left) * len(left) + np.var(right) * len(right)
            if err < best_err:
                best_err = best_err if err >= best_err else err
                best_k   = k
                best_err = err

        if best_k is None:
            return None

        left_mean  = s_sm[:best_k].mean()
        right_mean = s_sm[best_k:].mean()
        if (left_mean - right_mean) < min_drop:
            return None

        return float(self.t[idx[best_k]])

    def _find_bottom_hit(
        self,
        slopes: np.ndarray,
        min_drop: float = 0.3,
        y_top: int | None = None,
        y_bot: int | None = None,
    ) -> tuple[float | None, str | None]:
        """
        Identify when the first finger hits the bottom of the rig.

        Returns the earliest of:
        - **corner criterion**: abrupt drop in the log-log slope curve after
          convective onset, detected by a piecewise two-line breakpoint scan.
          Does not require the slope to reach any fixed threshold — just that
          it drops significantly relative to its prior sustained value.
        - **boundary criterion**: tip pixel depth >= 99% of crop height
          (only checked when *y_top* and *y_bot* are provided).

        Returns
        -------
        (t_bottom, source) where source is one of
        "corner", "boundary", "corner+boundary", or (None, None) if not found.
        """
        crop_height = (y_bot - y_top) if (y_top is not None and y_bot is not None) else None

        t_corner = self._find_slope_corner(slopes, min_drop=min_drop)

        t_boundary = None
        if crop_height is not None:
            boundary_mask = self.tip >= crop_height * 0.99
            if boundary_mask.any():
                t_boundary = self.t[np.where(boundary_mask)[0][0]]

        if t_corner is not None and t_boundary is not None:
            if abs(t_corner - t_boundary) < 1.0:
                return min(t_corner, t_boundary), "corner+boundary"
            elif t_corner < t_boundary:
                return t_corner, "corner"
            else:
                return t_boundary, "boundary"
        elif t_corner is not None:
            return t_corner, "corner"
        elif t_boundary is not None:
            return t_boundary, "boundary"
        return None, None

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
        y_top: int | None = None,
        y_bot: int | None = None,
        bottom_min_drop: float = 0.3,
    ):
        """
        Identify regime boundaries and the "first finger hits bottom" time.

        Parameters
        ----------
        sigma_threshold : float
            sigma_h rise factor to call onset automatically (default 2.0).
        slope_threshold : float
            Log-log tip slope above which the convective regime begins
            (default 0.80).
        drop_fraction : float
            Fractional finger-count drop to call merging (default 0.10).
        manual_onset_sec : float or None
            If given (seconds), skip the sigma_h threshold and use this value
            directly as t_o.
        y_top : int or None
            Top crop pixel row. Used for physical boundary detection of bottom hit.
        y_bot : int or None
            Bottom crop pixel row. Used for physical boundary detection of bottom hit.
        bottom_min_drop : float
            Minimum drop in mean log-log slope across the detected corner
            for it to count as "first finger hits bottom" (default 0.3).
            Lower values are more sensitive; raise it if noise triggers
            false positives.
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

        self.t_bottom, self.bottom_source = self._find_bottom_hit(
            slopes,
            min_drop=bottom_min_drop,
            y_top=y_top,
            y_bot=y_bot,
        )

        t0, tf = self.t[0], self.t[-1]
        self.regime_boundaries = {
            0: (t0,            self.t_onset),
            1: (self.t_onset,  self.t_onset),
            2: (self.t_onset,  self.t_merge),
            3: (self.t_merge,  self.t_convect),
            4: (self.t_convect, tf),
        }

        conv_mask = self.t >= self.t_convect
        if conv_mask.sum() >= 2:
            sl, *_ = linregress(self.t[conv_mask], self.tip[conv_mask])
            self._tip_slope = sl

        self._growth_rate = self._linear_growth_slope()
        self._slopes      = slopes

        # --- printed summary ---
        def _fmt(t_sec):
            return f"{t_sec / _SECONDS_PER_HOUR:.3f} h  ({t_sec:.1f} s)"

        sep = "-" * 65
        print(sep)
        print("  TIME REGIME DETECTION SUMMARY")
        print(sep)
        print(f"  Onset source  :  "
              f"{'MANUAL (penetration depth)' if self._onset_source == 'manual' else 'AUTO (sigma_h threshold)'}")
        print(sep)
        print(f"  {'Regime':<26}  {'t_start':>18}  {'t_end':>18}")
        print(sep)
        t0 = self.t[0]
        tf = self.t[-1]
        print(f"  {'0  Diffusive':<26}  {_fmt(t0):>18}  {_fmt(self.t_onset):>18}")
        print(f"  {'1  Onset t_o':<26}  {_fmt(self.t_onset):>18}  {'(marker)':>18}")
        print(f"  {'2  Linear growth':<26}  {_fmt(self.t_onset):>18}  {_fmt(self.t_merge):>18}")
        if not np.isnan(self._growth_rate):
            print(f"       roughness growth rate sigma = {self._growth_rate:.4g} / t-unit")
        print(f"  {'3  Nonlinear / merging':<26}  {_fmt(self.t_merge):>18}  {_fmt(self.t_convect):>18}")
        print(f"  {'4  Convective':<26}  {_fmt(self.t_convect):>18}  {_fmt(tf):>18}")
        if not np.isnan(self._tip_slope):
            print(f"       tip velocity = {self._tip_slope:.4g} px/s  "
                  f"({self._tip_slope / self.px_per_m:.4g} m/s)")
        if self.t_bottom is not None:
            print(f"  First finger hits bottom : {_fmt(self.t_bottom)}  "
                  f"[detected via: {self.bottom_source}]")
        if self.Ra:
            print(f"\n  Theoretical t_o (Riaz 2006, Ra={self.Ra}): {146 / self.Ra:.4g} s")
        print(sep)

    def plot(
        self,
        figsize=(13, 9),
        time_unit: str = "h",
        save_path: str | None = None,
    ):
        """
        Four-panel diagnostic figure (sigma_h, finger count, log-log tip, slope).

        Parameters
        ----------
        figsize : tuple
        time_unit : str
            ``'h'`` for hours (default) or ``'s'`` for seconds.
        save_path : str or None
        """
        if not self.regime_boundaries:
            raise RuntimeError("Call detect() before plot().")

        def _t(t_sec):
            return t_sec / _SECONDS_PER_HOUR if time_unit == "h" else t_sec

        xlabel = "Time (h)" if time_unit == "h" else "Time (s)"

        t_plot = self.t / _SECONDS_PER_HOUR if time_unit == "h" else self.t

        fig, axes = plt.subplots(2, 2, figsize=figsize, tight_layout=True)
        onset_label = (f"t_o manual = {_t(self.t_onset):.3g} {time_unit}"
                       if self._onset_source == "manual"
                       else f"t_o auto = {_t(self.t_onset):.3g} {time_unit}")
        fig.suptitle(
            f"CO2 Fingering -- Time Regime Analysis  [onset: {self._onset_source}]",
            fontsize=13, fontweight="bold",
        )
        ax_s, ax_n, ax_tip, ax_sl = axes.flat

        def shade(ax, logscale=False):
            for rid, (ta, tb) in self.regime_boundaries.items():
                if rid == 1:
                    continue
                ax.axvspan(_t(ta), _t(tb), alpha=0.18, color=REGIME_COLORS[rid], zorder=0)
            ax.axvline(_t(self.t_onset), color=REGIME_COLORS[1], lw=1.8, ls="--",
                       label=onset_label, zorder=3)
            ax.axvline(_t(self.t_merge), color=REGIME_COLORS[3], lw=1.2, ls=":",
                       label=f"t_merge = {_t(self.t_merge):.3g} {time_unit}", zorder=3)
            ax.axvline(_t(self.t_convect), color=REGIME_COLORS[4], lw=1.2, ls="-.",
                       label=f"t_conv = {_t(self.t_convect):.3g} {time_unit}", zorder=3)
            if self.Ra:
                ax.axvline(_t(146 / self.Ra), color="#D85A30", lw=1.0,
                           ls=(0, (3, 1, 1, 1)),
                           label=f"t_o theory={_t(146/self.Ra):.3g} {time_unit}", zorder=3)
            # "first finger hits bottom" line on every panel
            if self.t_bottom is not None:
                tb_plot = _t(self.t_bottom)
                ax.axvline(tb_plot, color=_BOTTOM_COLOR, lw=2.0, ls="-", zorder=5)
                # label just above the x-axis using axis-fraction coordinates
                ax.text(
                    tb_plot, 0.02, "first finger\nhits bottom",
                    transform=ax.get_xaxis_transform(),
                    color=_BOTTOM_COLOR, fontsize=6.5, fontweight="bold",
                    ha="left", va="bottom", rotation=0,
                )

        # (A) sigma_h
        ax_s.plot(t_plot, self.sig, color="#185FA5", lw=1.5, label="sigma_h")
        shade(ax_s)
        ax_s.set(xlabel=xlabel, ylabel="sigma_h  (pixels)",
                 title="(A) Interface roughness sigma_h")
        ax_s.legend(fontsize=7, loc="upper left")

        # (B) finger count
        ax_n.plot(t_plot, self.n, color="#0F6E56", lw=1.5,
                  marker="o", markersize=3, label="N fingers")
        shade(ax_n)
        ax_n.set(xlabel=xlabel, ylabel="Finger count N",
                 title="(B) Finger count")
        ax_n.legend(fontsize=7, loc="upper right")

        # (C) log-log tip
        valid = (t_plot > 0) & (self.tip > 0)
        ax_tip.loglog(t_plot[valid], self.tip[valid],
                      color="#3C3489", lw=1.5, label="Tip depth")
        t_ref   = np.array([t_plot[valid].min(), t_plot[valid].max()])
        tip_mid = np.nanmedian(self.tip[valid])
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 0.5,
                      ls="--", color="#888780", lw=1, label="slope 0.5 (diffusive)")
        ax_tip.loglog(t_ref, tip_mid * (t_ref / t_ref.mean()) ** 1.0,
                      ls=":",  color="#639922", lw=1, label="slope 1.0 (convective)")
        ax_tip.axvline(_t(self.t_onset),   color=REGIME_COLORS[1], lw=1.2, ls="--", zorder=3)
        ax_tip.axvline(_t(self.t_convect), color=REGIME_COLORS[4], lw=1.2, ls="-.", zorder=3)
        if self.t_bottom is not None:
            tb_plot = _t(self.t_bottom)
            ax_tip.axvline(tb_plot, color=_BOTTOM_COLOR, lw=2.0, ls="-", zorder=5)
            ax_tip.text(
                tb_plot, 0.02, "first finger\nhits bottom",
                transform=ax_tip.get_xaxis_transform(),
                color=_BOTTOM_COLOR, fontsize=6.5, fontweight="bold",
                ha="left", va="bottom",
            )
        xlabel_log = xlabel.replace("(h)", "(h, log)").replace("(s)", "(s, log)")
        ax_tip.set(xlabel=xlabel_log, ylabel="Tip depth (log)",
                   title="(C) Tip penetration  [log-log]")
        ax_tip.legend(fontsize=7)

        # (D) slope
        ax_sl.plot(t_plot, self._slopes, color="#993C1D", lw=1.5,
                   label="d log(tip) / d log(t)")
        ax_sl.axhline(0.5, color="#888780", ls="--", lw=1, label="0.5 (diffusive)")
        ax_sl.axhline(1.0, color="#639922", ls=":",  lw=1, label="1.0 (convective)")
        ax_sl.axhline(0.01, color=_BOTTOM_COLOR, ls=":", lw=0.8, alpha=0.6,
                      label="0.01 (bottom threshold)")
        ax_sl.axvline(_t(self.t_onset),   color=REGIME_COLORS[1], lw=1.2, ls="--", zorder=3)
        ax_sl.axvline(_t(self.t_convect), color=REGIME_COLORS[4], lw=1.2, ls="-.", zorder=3)
        if self.t_bottom is not None:
            tb_plot = _t(self.t_bottom)
            ax_sl.axvline(tb_plot, color=_BOTTOM_COLOR, lw=2.0, ls="-", zorder=5)
            ax_sl.text(
                tb_plot, 0.02, "first finger\nhits bottom",
                transform=ax_sl.get_xaxis_transform(),
                color=_BOTTOM_COLOR, fontsize=6.5, fontweight="bold",
                ha="left", va="bottom",
            )
        ax_sl.set_ylim(-0.3, 2.2)
        ax_sl.set(xlabel=xlabel, ylabel="Local log-log slope",
                  title="(D) Penetration slope  (0.5 -> 1.0 transition)")
        ax_sl.legend(fontsize=7)

        patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.55,
                                  label=REGIME_LABELS[r]) for r in [0, 2, 3, 4]]
        if self.t_bottom is not None:
            patches.append(mpatches.Patch(color=_BOTTOM_COLOR, alpha=0.8,
                                          label="First finger hits bottom"))
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8,
                   title="Regime shading", title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.03))
        plt.subplots_adjust(bottom=0.12)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved -> {save_path}")
        plt.show()

    def report(self) -> dict:
        """
        Return a dictionary of all detected regime boundaries and statistics.

        Returns
        -------
        dict with keys:
            ``t_onset``, ``onset_source``, ``t_merge``, ``t_convective``,
            ``t_bottom``, ``bottom_source``,
            ``roughness_growth_rate``, ``tip_velocity_px_per_s``,
            ``tip_velocity_m_per_s``, ``regime_boundaries``.
        """
        return {
            "t_onset":               self.t_onset,
            "onset_source":          self._onset_source,
            "t_merge":               self.t_merge,
            "t_convective":          self.t_convect,
            "t_bottom":              self.t_bottom,
            "bottom_source":         self.bottom_source,
            "roughness_growth_rate": self._growth_rate,
            "tip_velocity_px_per_s": self._tip_slope,
            "tip_velocity_m_per_s":  (
                self._tip_slope / self.px_per_m if not np.isnan(self._tip_slope) else np.nan
            ),
            "regime_boundaries":     self.regime_boundaries,
        }
