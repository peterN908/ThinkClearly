from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
from PIL import Image as PILImage


def _load_image_rgb(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = PILImage.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, PILImage.BICUBIC)
    return np.array(img, dtype=np.uint8)


def _absdiff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a16 = a.astype(np.int16)
    b16 = b.astype(np.int16)
    diff = np.abs(a16 - b16)
    if diff.ndim == 3:
        diff_gray = np.mean(diff, axis=2)
    else:
        diff_gray = diff
    denom = max(1.0, float(diff_gray.max()))
    heat = (diff_gray / denom) * 255.0
    return heat.astype(np.uint8)


def _overlay_red(image_rgb: np.ndarray, heat_gray: np.ndarray, max_alpha: float = 0.6) -> np.ndarray:
    imgf = image_rgb.astype(np.float32)
    alpha = (heat_gray.astype(np.float32) / 255.0) * float(max_alpha)
    red = np.zeros_like(imgf)
    red[:, :, 0] = 255.0
    out = imgf * (1.0 - alpha[:, :, None]) + red * alpha[:, :, None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _mse_psnr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    diff = a32 - b32
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return 0.0, float("inf")
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return float(mse), float(psnr)


def _save_png(arr: np.ndarray, path: str) -> None:
    PILImage.fromarray(arr).save(path)


def heatmap_diff_png(
    img_a_path: str,
    img_b_path: str,
    out_dir: Optional[str] = None,
    prefix: str = "ab",
    max_alpha: float = 0.6,
) -> Dict[str, object]:
    """
    Compute an absolute-difference heatmap and an overlay visualization between two images.
    Optionally saves results to out_dir and returns arrays and metrics.
    """
    a_img = _load_image_rgb(img_a_path, size=None)
    b_img = _load_image_rgb(img_b_path, size=None)
    if a_img.shape[:2] != b_img.shape[:2]:
        h, w = a_img.shape[:2]
        b_img = _load_image_rgb(img_b_path, size=(w, h))

    heat = _absdiff_heatmap(a_img, b_img)
    overlay = _overlay_red(a_img, heat, max_alpha=max_alpha)
    mse, psnr = _mse_psnr(a_img, b_img)

    saved: Dict[str, str] = {}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        heat_path = os.path.join(out_dir, f"{prefix}_heat.png")
        overlay_path = os.path.join(out_dir, f"{prefix}_overlay.png")
        _save_png(heat, heat_path)
        _save_png(overlay, overlay_path)
        saved = {"heat": heat_path, "overlay": overlay_path}

    return {
        "heatmap": heat,
        "overlay": overlay,
        "mse": float(mse),
        "psnr": float(psnr),
        "paths": saved,
    }
