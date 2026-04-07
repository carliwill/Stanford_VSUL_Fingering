# co2_fingers

A Python package for analyzing **CO₂ convective fingering** in FluidFlower experiments.

Extracted and cleaned from Carlin Will's Colab notebooks
(`valley_fingers_single` and `time_regime_detector_v3`).

---

## Package structure

```
co2_fingers/
├── co2_fingers/
│   ├── __init__.py          # Public API + imports
│   ├── io.py                # Time extraction, image loading
│   ├── preprocessing.py     # Crop → Otsu → median → fill → close
│   ├── interface.py         # Contour extraction, Gaussian smoothing
│   ├── baseline.py          # Static baseline from reference frame
│   ├── fingers.py           # Detection, widths, roughness, tip depth
│   ├── regimes.py           # TimeRegimeDetector class
│   ├── plotting.py          # All visualization functions
│   ├── heatmap.py           # Seismic-colour frame-difference overlays
│   └── _registry.py         # FUNCTION_REGISTRY dictionary
├── scripts/
│   └── run_analysis.py      # End-to-end CLI script
└── setup.py
```

---

## Installation

```bash
pip install -e .
```

Or install dependencies manually:
```bash
pip install numpy scipy pandas opencv-python matplotlib
```

---

## Quick start

```python
import co2_fingers as cf

# 1. Load timestamps
df = cf.load_timestamps("/data/image_last_modified.csv")

# 2. Compute static baseline from a flat-front reference frame
baseline_img = cf.load_image("/data/C2R5/IMG_1260.JPG")
baseline = cf.compute_static_baseline(baseline_img)

# 3. Process a single frame
img = cf.load_image("/data/C2R5/IMG_1400.JPG")
mask = cf.preprocess(img)
smooth_cnt, _ = cf.raw_outline(mask)
x_vals, smooth_iface = cf.gauss_outline(smooth_cnt)

# 4. Detect fingers
import numpy as np
from co2_fingers.baseline import interpolate_baseline
interp = interpolate_baseline(baseline, np.array(x_vals))
v_idx = cf.detect_fingers(smooth_iface, baseline["residual"],
                           np.array(x_vals),
                           baseline_y_bar=interp["y_bar"])

widths = cf.measure_finger_widths(smooth_iface, np.array(x_vals),
                                   v_idx, px_per_metre=6727.0)
print(f"  {widths['n_fingers']} fingers, mean width {widths['mean_width_cm']:.2f} cm")

# 5. Run full analysis from the command line
# python scripts/run_analysis.py \
#   --image_dir /data/C2R5 \
#   --csv_path  /data/image_last_modified.csv \
#   --baseline  IMG_1260.JPG \
#   --output    ./results
```

---

## Function registry

```python
import co2_fingers as cf
cf.print_registry()          # prints all functions with purpose, inputs, returns
print(cf.FUNCTION_REGISTRY)  # raw dictionary
```

---

## Module descriptions

| Module | Contents |
|---|---|
| `io` | `load_timestamps`, `load_image`, `load_images`, `fmt_time` |
| `preprocessing` | `crop_image`, `preprocess` |
| `interface` | `raw_outline`, `gauss_outline` |
| `baseline` | `compute_static_baseline`, `interpolate_baseline` |
| `fingers` | `detect_fingers`, `measure_finger_widths`, `roughness_sigma`, `mean_front`, `tip_from_valleys` |
| `regimes` | `TimeRegimeDetector`, `REGIME_COLORS`, `REGIME_LABELS` |
| `plotting` | `plot_baseline_frame`, `plot_finger_check`, `plot_finger_widths_per_frame`, `plot_finger_count`, `plot_merging_metric`, `plot_loglog_tip`, `plot_heatmap_overlay`, `plot_time_regimes` |
| `heatmap` | `build_heatmap_overlay`, `overlay_interfaces` |

---

## Physical parameters (C2R5 defaults)

| Parameter | Value |
|---|---|
| Frame width | 55 cm (0.55 m) |
| Crop x | 230 – 5700 px → 5470 px |
| Crop y | 2300 – 3300 px |
| px/m | 5470 / 0.55 ≈ 9945 |
| Injection rate | 0.5 cm³/min for 3 h 11 min |
| Manual onset | 330 min (5.5 hours) |

---

## Time regimes (Riaz et al. 2006)

| # | Regime | Signal |
|---|---|---|
| 0 | Diffusive | σ_h flat at baseline |
| 1 | Onset tₒ | σ_h first crosses 2× early baseline |
| 2 | Linear growth | σ_h rising exponentially |
| 3 | Nonlinear / merging | Finger count drops 10% from peak |
| 4 | Convective | Log–log tip slope ≥ 0.8 |
