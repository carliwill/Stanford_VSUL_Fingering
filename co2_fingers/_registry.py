"""
co2_fingers._registry
======================
Dictionary of every public function / class in the package.

Each entry contains:
  - ``module``    : dotted module path
  - ``purpose``   : one-sentence description
  - ``inputs``    : list of key parameter names and types
  - ``returns``   : description of return value(s)

Access via ``co2_fingers.FUNCTION_REGISTRY`` or print the
built-in summary with ``co2_fingers.print_registry()``.
"""

FUNCTION_REGISTRY = {

    # ------------------------------------------------------------------ io
    "load_timestamps": {
        "module":  "co2_fingers.io",
        "purpose": "Load the image-timestamp CSV; returns a DataFrame with 't_sec' column.",
        "inputs":  ["csv_path: str", "filename_col: str = 'Filename'",
                    "time_col: str = 'TimeSinceStart(min)'"],
        "returns": "pd.DataFrame",
    },
    "fmt_time": {
        "module":  "co2_fingers.io",
        "purpose": "Convert elapsed seconds to a human-readable 'Xh YYm' string.",
        "inputs":  ["t_sec: float"],
        "returns": "str",
    },
    "load_image": {
        "module":  "co2_fingers.io",
        "purpose": "Load a single BGR image from disk; raises FileNotFoundError if missing.",
        "inputs":  ["path: str"],
        "returns": "np.ndarray (BGR, H×W×3)",
    },
    "load_images": {
        "module":  "co2_fingers.io",
        "purpose": "Bulk-load all images matching a glob pattern in a directory.",
        "inputs":  ["image_dir: str", "pattern: str = '*.JPG'", "sort: bool = True"],
        "returns": "list of (filename, np.ndarray) tuples",
    },

    # -------------------------------------------------------- preprocessing
    "crop_image": {
        "module":  "co2_fingers.preprocessing",
        "purpose": "Crop a full-resolution FluidFlower image to the CO₂-permeable zone.",
        "inputs":  ["image: np.ndarray", "y_top: int", "y_bot: int",
                    "x_left: int = 230", "x_right: int = 5700"],
        "returns": "np.ndarray (cropped BGR)",
    },
    "preprocess": {
        "module":  "co2_fingers.preprocessing",
        "purpose": "Full pipeline: crop → greyscale → Otsu binarize → median blur → fill holes → morph close.",
        "inputs":  ["image: np.ndarray", "y_top: int = 2300", "y_bot: int = 3300",
                    "x_left: int = 230", "x_right: int = 5700",
                    "use_otsu: bool = True", "manual_thresh: int = 127"],
        "returns": "np.ndarray (binary mask, uint8, 0/255)",
    },

    # ----------------------------------------------------------- interface
    "raw_outline": {
        "module":  "co2_fingers.interface",
        "purpose": "Find the largest contour in a binary mask and apply approxPolyDP smoothing.",
        "inputs":  ["mask: np.ndarray", "contour_epsilon: float = 0.0001"],
        "returns": "(smooth_cnt: np.ndarray, trace: np.ndarray)",
    },
    "gauss_outline": {
        "module":  "co2_fingers.interface",
        "purpose": "Convert a contour to a single-valued smoothed interface y(x) using median + Gaussian filtering.",
        "inputs":  ["smooth_cnt: np.ndarray", "median_ksize: int = 9",
                    "gaussian_sigma: float = 1.0", "flow_direction: str = 'down'"],
        "returns": "(x_vals: np.ndarray, smooth_interface: np.ndarray)",
    },

    # ------------------------------------------------------------ baseline
    "compute_static_baseline": {
        "module":  "co2_fingers.baseline",
        "purpose": "Build a static baseline (mean front + roughness band) from a single reference frame.",
        "inputs":  ["baseline_image: np.ndarray", "y_top", "y_bot", "x_left", "x_right",
                    "window_frac: float = 0.6", "y_bar_offset: float = 5.0",
                    "roughness_k: float = 0.5"],
        "returns": "dict {x_base, y_bar_static, smooth_iface, residual, sigma_h, prominence, upper_band, lower_band}",
    },
    "interpolate_baseline": {
        "module":  "co2_fingers.baseline",
        "purpose": "Interpolate a pre-computed baseline onto a new frame's x-coordinate array.",
        "inputs":  ["baseline: dict", "x_new: np.ndarray"],
        "returns": "dict {y_bar, upper_band, lower_band} on the new x-grid",
    },

    # ------------------------------------------------------------- fingers
    "roughness_sigma": {
        "module":  "co2_fingers.fingers",
        "purpose": "Compute interface roughness σ_h = std(interface − y_bar).",
        "inputs":  ["smooth_interface: np.ndarray", "y_bar: np.ndarray"],
        "returns": "float (pixels)",
    },
    "mean_front": {
        "module":  "co2_fingers.fingers",
        "purpose": "Estimate the smooth mean front y_bar(x) using a Savitzky–Golay filter.",
        "inputs":  ["y: np.ndarray", "window_frac: float = 0.6", "polyorder: int = 2"],
        "returns": "np.ndarray (same length as y)",
    },
    "detect_fingers": {
        "module":  "co2_fingers.fingers",
        "purpose": "Detect finger indices along the smoothed interface using find_peaks with prominence = 2.5 × σ.",
        "inputs":  ["smooth_interface: np.ndarray", "residual: np.ndarray",
                    "x_vals: np.ndarray", "distance: int = 20",
                    "prominence_override: float | None = None",
                    "baseline_y_bar: np.ndarray | None = None"],
        "returns": "np.ndarray of integer indices",
    },
    "measure_finger_widths": {
        "module":  "co2_fingers.fingers",
        "purpose": "Measure shoulder-to-shoulder finger widths in pixels and centimetres.",
        "inputs":  ["smooth_interface", "x_vals", "finger_idx", "px_per_metre: float"],
        "returns": "dict {widths_px, widths_cm, left_shoulders, right_shoulders, n_fingers, mean_width_cm, std_width_cm, min_width_cm, max_width_cm}",
    },
    "tip_from_valleys": {
        "module":  "co2_fingers.fingers",
        "purpose": "Return the deepest (or rightmost) finger-tip coordinate from detected valleys.",
        "inputs":  ["v_x: np.ndarray", "v_y: np.ndarray", "flow_direction: str = 'down'"],
        "returns": "float (pixels)",
    },

    # ------------------------------------------------------------- regimes
    "TimeRegimeDetector": {
        "module":  "co2_fingers.regimes",
        "purpose": "Class that detects all five CO₂-fingering time regimes from per-frame time-series data.",
        "inputs":  ["times", "n_fingers", "tip_positions", "sigma_list",
                    "px_per_m: float = 1.0", "Ra = None"],
        "returns": "TimeRegimeDetector instance — call .detect(), .plot(), .report()",
    },
    "TimeRegimeDetector.detect": {
        "module":  "co2_fingers.regimes.TimeRegimeDetector",
        "purpose": "Identify regime boundaries and print summary table.",
        "inputs":  ["sigma_threshold: float = 2.0", "slope_threshold: float = 0.80",
                    "drop_fraction: float = 0.10", "manual_onset_sec: float | None = None"],
        "returns": "None (mutates detector state)",
    },
    "TimeRegimeDetector.plot": {
        "module":  "co2_fingers.regimes.TimeRegimeDetector",
        "purpose": "Four-panel diagnostic figure: σ_h, finger count, log–log tip, local slope.",
        "inputs":  ["figsize=(13,9)", "save_path: str | None = None"],
        "returns": "None (shows figure)",
    },
    "TimeRegimeDetector.report": {
        "module":  "co2_fingers.regimes.TimeRegimeDetector",
        "purpose": "Return a dictionary of all detected regime boundaries and statistics.",
        "inputs":  [],
        "returns": "dict {t_onset, onset_source, t_merge, t_convective, roughness_growth_rate, tip_velocity_px_per_t, tip_velocity_m_per_t, regime_boundaries}",
    },

    # ------------------------------------------------------------- plotting
    "plot_baseline_frame": {
        "module":  "co2_fingers.plotting",
        "purpose": "Two-panel plot: baseline image with interface overlays + 1-D signal.",
        "inputs":  ["image", "baseline: dict", "y_top", "y_bot", "x_left", "x_right",
                    "scale: float = 0.25", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_finger_check": {
        "module":  "co2_fingers.plotting",
        "purpose": "Two-panel finger-detection verification for a single frame.",
        "inputs":  ["image", "smooth_iface", "x_vals", "baseline", "widths_info",
                    "filename", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_finger_widths_per_frame": {
        "module":  "co2_fingers.plotting",
        "purpose": "Line plot of mean finger width (cm) vs. time.",
        "inputs":  ["times_sec", "mean_widths_cm", "std_widths_cm = None", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_finger_count": {
        "module":  "co2_fingers.plotting",
        "purpose": "Line plot of finger count N vs. time.",
        "inputs":  ["times_sec", "n_fingers", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_merging_metric": {
        "module":  "co2_fingers.plotting",
        "purpose": "Plot the merging metric N × mean_width_cm vs. time.",
        "inputs":  ["times_sec", "n_fingers", "mean_widths_cm", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_loglog_tip": {
        "module":  "co2_fingers.plotting",
        "purpose": "Log–log plot of deepest finger tip vs. time with diffusive/convective reference lines.",
        "inputs":  ["times_sec", "tip_depths_px", "save_path"],
        "returns": "plt.Figure",
    },
    "plot_heatmap_overlay": {
        "module":  "co2_fingers.plotting",
        "purpose": "Seismic-colour heatmaps of consecutive grayscale frame differences.",
        "inputs":  ["images: list", "filenames", "y_top", "y_bot", "x_left", "x_right", "save_path"],
        "returns": "None (shows figure)",
    },
    "plot_time_regimes": {
        "module":  "co2_fingers.plotting",
        "purpose": "Convenience wrapper around TimeRegimeDetector.plot().",
        "inputs":  ["detector: TimeRegimeDetector", "save_path"],
        "returns": "None (shows figure)",
    },

    # -------------------------------------------------------------- heatmap
    "build_heatmap_overlay": {
        "module":  "co2_fingers.heatmap",
        "purpose": "Seismic-colour heatmaps of consecutive frame differences (standalone version).",
        "inputs":  ["images: list", "filenames", "y_top", "y_bot", "x_left", "x_right",
                    "cmap='seismic'", "figsize=(20,5)", "save_path"],
        "returns": "None (shows and optionally saves figure)",
    },
    "overlay_interfaces": {
        "module":  "co2_fingers.heatmap",
        "purpose": "Overlay smoothed interfaces from multiple frames on a single background image.",
        "inputs":  ["images: list", "x_vals_list", "interfaces_list", "filenames",
                    "colors", "y_top", "y_bot", "x_left", "x_right", "save_path"],
        "returns": "None (shows and optionally saves figure)",
    },
}


def print_registry():
    """Print a formatted summary of every function in the registry."""
    print("=" * 72)
    print(f"  co2_fingers — Function Registry  ({len(FUNCTION_REGISTRY)} entries)")
    print("=" * 72)
    for name, info in FUNCTION_REGISTRY.items():
        print(f"\n  {name}")
        print(f"    Module  : {info['module']}")
        print(f"    Purpose : {info['purpose']}")
        if info["inputs"]:
            print(f"    Inputs  : {', '.join(info['inputs'])}")
        print(f"    Returns : {info['returns']}")
    print("=" * 72)
