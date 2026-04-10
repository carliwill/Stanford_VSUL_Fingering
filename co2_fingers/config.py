"""
co2_fingers.config
==================
Load, validate, and provide typed access to a YAML experiment config.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required: pip install pyyaml")


@dataclass
class CropConfig:
    y_top: int   = 2300
    y_bot: int   = 3300
    x_left: int  = 230
    x_right: int = 5700

    @property
    def width_px(self) -> int:
        return self.x_right - self.x_left

    @property
    def height_px(self) -> int:
        return self.y_bot - self.y_top


@dataclass
class PhysicalConfig:
    frame_width_m: float  = 0.55
    flow_direction: str   = "down"

    @property
    def px_per_metre(self) -> float:
        # filled in after CropConfig is known
        return getattr(self, "_px_per_metre", None)


@dataclass
class DetectionConfig:
    manual_onset_min: Optional[float] = None
    sigma_threshold: float  = 2.0
    slope_threshold: float  = 0.80
    drop_fraction: float    = 0.10
    valley_distance: int    = 20
    roughness_k: float      = 0.5
    contour_epsilon: float  = 0.0001
    median_ksize: int       = 9
    gaussian_sigma: float   = 1.0

    @property
    def manual_onset_sec(self) -> Optional[float]:
        return self.manual_onset_min * 60 if self.manual_onset_min is not None else None


@dataclass
class ExperimentConfig:
    name: str
    image_dir: str
    csv_path: str
    baseline_frame: str
    output_dir: str
    crop: CropConfig          = field(default_factory=CropConfig)
    physical: PhysicalConfig  = field(default_factory=PhysicalConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def __post_init__(self):
        # Compute px_per_metre from crop + physical
        self.physical._px_per_metre = self.crop.width_px / self.physical.frame_width_m

    @property
    def px_per_metre(self) -> float:
        return self.physical._px_per_metre


def load_config(path: str) -> ExperimentConfig:
    """
    Load an experiment config from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    ExperimentConfig

    Example
    -------
    >>> cfg = load_config("configs/C2R5.yaml")
    >>> print(cfg.name, cfg.px_per_metre)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    exp  = raw.get("experiment", {})
    crop = raw.get("crop", {})
    phys = raw.get("physical", {})
    det  = raw.get("detection", {})

    return ExperimentConfig(
        name           = exp["name"],
        image_dir      = exp["image_dir"],
        csv_path       = exp["csv_path"],
        baseline_frame = exp["baseline_frame"],
        output_dir     = exp.get("output_dir", f"./results/{exp['name']}"),
        crop = CropConfig(
            y_top   = crop.get("y_top",   2300),
            y_bot   = crop.get("y_bot",   3300),
            x_left  = crop.get("x_left",  230),
            x_right = crop.get("x_right", 5700),
        ),
        physical = PhysicalConfig(
            frame_width_m  = phys.get("frame_width_m",  0.55),
            flow_direction = phys.get("flow_direction", "down"),
        ),
        detection = DetectionConfig(
            manual_onset_min = det.get("manual_onset_min", None),
            sigma_threshold  = det.get("sigma_threshold",  2.0),
            slope_threshold  = det.get("slope_threshold",  0.80),
            drop_fraction    = det.get("drop_fraction",    0.10),
            valley_distance  = det.get("valley_distance",  20),
            roughness_k      = det.get("roughness_k",      0.5),
            contour_epsilon  = det.get("contour_epsilon",  0.0001),
            median_ksize     = det.get("median_ksize",     9),
            gaussian_sigma   = det.get("gaussian_sigma",   1.0),
        ),
    )


def load_configs(*paths: str) -> list[ExperimentConfig]:
    """
    Load multiple experiment configs at once.

    Parameters
    ----------
    *paths : str
        Any number of YAML config file paths.

    Returns
    -------
    list of ExperimentConfig
    """
    return [load_config(p) for p in paths]
