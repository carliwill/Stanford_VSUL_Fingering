"""
scripts/run_analysis.py
=======================
End-to-end analysis script for a single FluidFlower experiment.

Usage (from the repo root):
    python scripts/run_analysis.py \
        --image_dir /path/to/C2R5 \
        --csv_path  /path/to/image_last_modified.csv \
        --baseline  IMG_1260.JPG \
        --output    ./results

All plots are saved to --output.  Edit CONFIG at the top of the script
to match your experiment's crop limits and physical dimensions.
"""

import argparse
import os
import numpy as np
import pandas as pd

import co2_fingers as cf
from co2_fingers.io import fmt_time


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit these to match your experiment)
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = dict(
    y_top=2300,
    y_bot=3300,
    x_left=230,
    x_right=5700,
    frame_width_m=0.55,     # physical width of the cropped frame in metres
    manual_onset_min=330,   # set to None to use automatic σ_h detection
    sigma_threshold=2.0,
    slope_threshold=0.80,
    drop_fraction=0.10,
    valley_distance=20,
    flow_direction="down",
)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CO₂ fingering analysis")
    parser.add_argument("--image_dir", required=True, help="Directory of JPG images")
    parser.add_argument("--csv_path",  required=True, help="Timestamp CSV path")
    parser.add_argument("--baseline",  required=True, help="Baseline frame filename")
    parser.add_argument("--output",    default="./results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    C = CONFIG

    px_per_m = (C["x_right"] - C["x_left"]) / C["frame_width_m"]

    # 1. Load timestamps
    print("Loading timestamps …")
    df = cf.load_timestamps(args.csv_path)
    print(f"  {len(df)} frames in CSV")

    # 2. Load baseline image and compute static baseline
    print(f"\nComputing static baseline from {args.baseline} …")
    baseline_img = cf.load_image(os.path.join(args.image_dir, args.baseline))
    baseline = cf.compute_static_baseline(
        baseline_img,
        y_top=C["y_top"], y_bot=C["y_bot"],
        x_left=C["x_left"], x_right=C["x_right"],
    )
    print(f"  σ_h = {baseline['sigma_h']:.3f} px   prominence = {baseline['prominence']:.3f} px")

    fig = cf.plot_baseline_frame(
        baseline_img, baseline,
        y_top=C["y_top"], y_bot=C["y_bot"],
        x_left=C["x_left"], x_right=C["x_right"],
        save_path=os.path.join(args.output, "01_static_baseline.png"),
    )

    # 3. Process every frame
    print("\nProcessing all frames …")
    times, tips, sigmas, n_fingers_list = [], [], [], []
    mean_widths_cm_list, std_widths_cm_list = [], []

    for _, row in df.iterrows():
        fname = row["filename"]
        img_path = os.path.join(args.image_dir, fname)
        img = cf.load_image(img_path)

        # Preprocess
        mask = cf.preprocess(img, C["y_top"], C["y_bot"], C["x_left"], C["x_right"])
        smooth_cnt, _ = cf.raw_outline(mask)
        x_vals, smooth_iface = cf.gauss_outline(smooth_cnt)
        x_vals = np.array(x_vals)

        # Baseline interpolation
        from co2_fingers.baseline import interpolate_baseline
        interp = interpolate_baseline(baseline, x_vals)

        # Fingers
        v_idx = cf.detect_fingers(
            smooth_iface, baseline["residual"], x_vals,
            distance=C["valley_distance"],
            prominence_override=baseline["prominence"],
            baseline_y_bar=interp["y_bar"],
        )
        v_x = x_vals[v_idx]
        v_y = smooth_iface[v_idx]

        widths = cf.measure_finger_widths(smooth_iface, x_vals, v_idx, px_per_m)
        tip    = cf.tip_from_valleys(v_x, v_y, C["flow_direction"])
        sigma  = cf.roughness_sigma(smooth_iface, interp["y_bar"])

        times.append(row["t_sec"])
        tips.append(tip)
        sigmas.append(sigma)
        n_fingers_list.append(len(v_idx))
        mean_widths_cm_list.append(widths["mean_width_cm"])
        std_widths_cm_list.append(widths["std_width_cm"])

    times     = np.array(times)
    tips      = np.array(tips)
    sigmas    = np.array(sigmas)
    n_fingers = np.array(n_fingers_list)
    mean_w    = np.array(mean_widths_cm_list)
    std_w     = np.array(std_widths_cm_list)

    # Save arrays
    np.savez(os.path.join(args.output, "regime_arrays.npz"),
             times=times, tips=tips, sigmas=sigmas, n_fingers=n_fingers,
             mean_widths_cm=mean_w, std_widths_cm=std_w)
    print(f"  Arrays saved → {args.output}/regime_arrays.npz")

    # 4. Plots
    cf.plot_finger_count(
        times, n_fingers,
        save_path=os.path.join(args.output, "02_finger_count.png"),
    )
    cf.plot_finger_widths_per_frame(
        times, mean_w, std_w,
        save_path=os.path.join(args.output, "03_finger_widths.png"),
    )
    cf.plot_merging_metric(
        times, n_fingers, mean_w,
        save_path=os.path.join(args.output, "04_merging_metric.png"),
    )
    cf.plot_loglog_tip(
        times, tips,
        save_path=os.path.join(args.output, "05_loglog_tip.png"),
    )

    # 5. Time-regime detection
    print("\nDetecting time regimes …")
    manual_sec = C["manual_onset_min"] * 60 if C["manual_onset_min"] else None

    # Smooth n_fingers slightly (rolling mean of ±2 frames)
    n_smooth = np.array([
        int(np.mean(n_fingers[max(0, i - 2): min(len(n_fingers), i + 2)]))
        for i in range(len(n_fingers))
    ])

    detector = cf.TimeRegimeDetector(
        times=times,
        n_fingers=n_smooth,
        tip_positions=tips,
        sigma_list=sigmas,
        px_per_m=px_per_m,
    )
    detector.detect(
        sigma_threshold=C["sigma_threshold"],
        slope_threshold=C["slope_threshold"],
        drop_fraction=C["drop_fraction"],
        manual_onset_sec=manual_sec,
    )
    detector.plot(save_path=os.path.join(args.output, "06_time_regimes.png"))

    results = detector.report()
    print("\nResults:")
    for k, v in results.items():
        if k == "regime_boundaries":
            continue
        if isinstance(v, float) and not np.isnan(v) and "velocity" not in k and "rate" not in k:
            print(f"  {k:<30} = {fmt_time(v)}")
        else:
            print(f"  {k:<30} = {v}")

    print(f"\n✓ Analysis complete.  All outputs in: {args.output}")


if __name__ == "__main__":
    main()
