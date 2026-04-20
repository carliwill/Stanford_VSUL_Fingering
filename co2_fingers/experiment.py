"""
co2_fingers.experiment
======================
``Experiment`` — the main high-level class.

Load a config, process every frame, and expose results.
One ``Experiment`` object = one FluidFlower run.
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .config import ExperimentConfig
from .io import load_timestamps, load_image, fmt_time
from .preprocessing import preprocess
from .interface import raw_outline, gauss_outline
from .baseline import compute_static_baseline, interpolate_baseline
from .fingers import detect_fingers, measure_finger_widths, roughness_sigma, tip_from_valleys
from .regimes import TimeRegimeDetector


@dataclass
class FrameResult:
    """Per-frame measurements."""
    filename: str
    t_sec: float
    n_fingers: int
    tip_px: float
    sigma_h: float
    mean_width_cm: float
    std_width_cm: float
    min_width_cm: float
    max_width_cm: float


class Experiment:
    """
    Process one FluidFlower CO₂ experiment end-to-end.

    Parameters
    ----------
    config : ExperimentConfig
        Loaded via ``co2_fingers.config.load_config()``.

    Examples
    --------
    >>> from co2_fingers.config import load_config
    >>> from co2_fingers.experiment import Experiment
    >>>
    >>> cfg = load_config("configs/C2R5.yaml")
    >>> exp = Experiment(cfg)
    >>> exp.run()
    >>> exp.plot_all()
    """

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.name = config.name

        # filled after run()
        self.baseline: dict | None = None
        self.frame_results: list[FrameResult] = []
        self.detector: TimeRegimeDetector | None = None
        self._arrays: dict = {}

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True):
        """
        Process all frames: baseline → per-frame detection → regime fitting.

        Parameters
        ----------
        verbose : bool
            Print progress (default True).
        """
        cfg = self.cfg
        c   = cfg.crop
        d   = cfg.detection
        os.makedirs(cfg.output_dir, exist_ok=True)

        # 1. Timestamps
        if verbose:
            print(f"[{self.name}] Loading timestamps …")
        df = load_timestamps(cfg.csv_path)
        if verbose:
            print(f"[{self.name}]   {len(df)} frames")

        # 2. Static baseline
        if verbose:
            print(f"[{self.name}] Computing baseline from {cfg.baseline_frame} …")
        baseline_img = load_image(os.path.join(cfg.image_dir, cfg.baseline_frame))
        self.baseline = compute_static_baseline(
            baseline_img,
            y_top=c.y_top, y_bot=c.y_bot, x_left=c.x_left, x_right=c.x_right,
            roughness_k=d.roughness_k,
            contour_epsilon=d.contour_epsilon,
            median_ksize=d.median_ksize,
            gaussian_sigma=d.gaussian_sigma,
        )
        if verbose:
            print(f"[{self.name}]   σ_h = {self.baseline['sigma_h']:.3f} px  "
                  f"prom = {self.baseline['prominence']:.3f} px")

        # 3. Per-frame loop
        if verbose:
            print(f"[{self.name}] Processing frames …")

        results = []
        for _, row in df.iterrows():
            fname    = row["filename"]
            t_sec    = row["t_sec"]
            img_path = os.path.join(cfg.image_dir, fname)

            try:
                img = load_image(img_path)
            except FileNotFoundError:
                if verbose:
                    print(f"[{self.name}]   SKIP {fname} (not found)")
                continue

            mask         = preprocess(img, c.y_top, c.y_bot, c.x_left, c.x_right,
                                      median_ksize=d.median_ksize)
            smooth_cnt, _ = raw_outline(mask, contour_epsilon=d.contour_epsilon)
            x_vals, smooth_iface = gauss_outline(smooth_cnt,
                                                  median_ksize=d.median_ksize,
                                                  gaussian_sigma=d.gaussian_sigma,
                                                  flow_direction=cfg.physical.flow_direction)
            x_vals = np.array(x_vals)

            if len(x_vals) == 0:
                continue

            interp = interpolate_baseline(self.baseline, x_vals)

            v_idx = detect_fingers(
                smooth_iface, self.baseline["residual"], x_vals,
                distance=d.valley_distance,
                prominence_override=30,
                baseline_y_bar=interp["y_bar"],
            )
            v_x = x_vals[v_idx]
            v_y = smooth_iface[v_idx]

            widths = measure_finger_widths(smooth_iface, x_vals, v_idx,
                                           cfg.px_per_metre)
            tip    = tip_from_valleys(v_x, v_y, cfg.physical.flow_direction)
            sigma  = roughness_sigma(smooth_iface, interp["y_bar"])

            results.append(FrameResult(
                filename      = fname,
                t_sec         = t_sec,
                n_fingers     = len(v_idx),
                tip_px        = tip,
                sigma_h       = sigma,
                mean_width_cm = widths["mean_width_cm"],
                std_width_cm  = widths["std_width_cm"],
                min_width_cm  = widths["min_width_cm"],
                max_width_cm  = widths["max_width_cm"],
            ))

        self.frame_results = results
        self._build_arrays()

        if verbose:
            print(f"[{self.name}]   {len(results)} frames processed")

        # 4. Regime detection
        self._fit_regimes(verbose=verbose)

        # 5. Save arrays
        np.savez(
            os.path.join(cfg.output_dir, f"{self.name}_arrays.npz"),
            **self._arrays,
        )
        if verbose:
            print(f"[{self.name}] Arrays saved → {cfg.output_dir}/{self.name}_arrays.npz")

    def _build_arrays(self):
        r = self.frame_results
        self._arrays = {
            "times":          np.array([f.t_sec         for f in r]),
            "tips":           np.array([f.tip_px        for f in r]),
            "sigmas":         np.array([f.sigma_h       for f in r]),
            "n_fingers":      np.array([f.n_fingers     for f in r]),
            "mean_widths_cm": np.array([f.mean_width_cm for f in r]),
            "std_widths_cm":  np.array([f.std_width_cm  for f in r]),
        }

    def _fit_regimes(self, verbose: bool = True):
        a   = self._arrays
        cfg = self.cfg
        d   = cfg.detection

        # Smooth finger count (rolling ±2)
        n = a["n_fingers"]
        n_smooth = np.array([
            int(np.mean(n[max(0, i - 2): min(len(n), i + 2)]))
            for i in range(len(n))
        ])

        self.detector = TimeRegimeDetector(
            times         = a["times"],
            n_fingers     = n_smooth,
            tip_positions = a["tips"],
            sigma_list    = a["sigmas"],
            px_per_m      = cfg.px_per_metre,
        )
        if verbose:
            print(f"[{self.name}] Detecting regimes …")
        self.detector.detect(
            sigma_threshold  = d.sigma_threshold,
            slope_threshold  = d.slope_threshold,
            drop_fraction    = d.drop_fraction,
            manual_onset_sec = d.manual_onset_sec,
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def times(self) -> np.ndarray:
        return self._arrays["times"]

    @property
    def n_fingers(self) -> np.ndarray:
        return self._arrays["n_fingers"]

    @property
    def tips(self) -> np.ndarray:
        return self._arrays["tips"]

    @property
    def sigmas(self) -> np.ndarray:
        return self._arrays["sigmas"]

    @property
    def mean_widths_cm(self) -> np.ndarray:
        return self._arrays["mean_widths_cm"]

    @property
    def results_df(self) -> pd.DataFrame:
        """All frame results as a tidy DataFrame."""
        return pd.DataFrame([vars(r) for r in self.frame_results])

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self, save: bool = True):
        """Generate and save all standard plots."""
        from .plotting import (
            plot_finger_count, plot_finger_widths_per_frame,
            plot_merging_metric, plot_loglog_tip,
        )
        out = self.cfg.output_dir

        def _path(name):
            return os.path.join(out, f"{self.name}_{name}.png") if save else None

        plot_finger_count(self.times, self.n_fingers, save_path=_path("finger_count"))
        plot_finger_widths_per_frame(self.times, self.mean_widths_cm,
                                     self._arrays["std_widths_cm"],
                                     save_path=_path("finger_widths"))
        plot_merging_metric(self.times, self.n_fingers, self.mean_widths_cm,
                            save_path=_path("merging_metric"))
        plot_loglog_tip(self.times, self.tips, save_path=_path("loglog_tip"))

        if self.detector is not None:
            self.detector.plot(save_path=_path("time_regimes"))

    def save_csv(self):
        """Save per-frame results to CSV."""
        path = os.path.join(self.cfg.output_dir, f"{self.name}_results.csv")
        self.results_df.to_csv(path, index=False)
        print(f"[{self.name}] CSV saved → {path}")
