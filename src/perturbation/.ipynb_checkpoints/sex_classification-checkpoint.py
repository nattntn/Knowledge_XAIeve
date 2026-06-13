"""
sex_classification.py
---------------------
Perturbation functions for the sex classification task.
Applied via seamlessClone with an elliptical mask.
Both mesiodistal width (scale_x) and total tooth length (scale_y) axes can be perturbed independently.
"""

import cv2
import numpy as np


def adjust_scale(
    img: np.ndarray,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    tag_name: str,          
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Perturb a semantic region via resize + seamlessClone.
    
    Parameters
    ----------
    img : np.ndarray
        Input BGR image (H x W x 3).
    x_center : float
        YOLO normalised x-center of bounding box (0–1).
    y_center : float
        YOLO normalised y-center of bounding box (0–1).
    width : float
        YOLO normalised width of bounding box (0–1).
    height : float
        YOLO normalised height of bounding box (0–1).
    scale_x : float, optional
        Horizontal scale factor. Default 1.0 (no change).
    scale_y : float, optional
        Vertical scale factor. Default 1.0 (no change).

    Returns
    -------
    edited : np.ndarray
        Perturbed image (same shape as input).
    new_bbox : tuple[int, int, int, int]
        New bounding box in pixel coordinates (x1, y1, x2, y2).
    """
    if img is None:
        raise FileNotFoundError("Image is None — check the file path passed to cv2.imread().")

    H, W = img.shape[:2]

    # Convert YOLO normalised → pixel coords
    bw = int(round(width * W))
    bh = int(round(height * H))
    cx = int(round(x_center * W))
    cy = int(round(y_center * H))

    x1 = max(0, cx - bw // 2)
    y1 = max(0, cy - bh // 2)
    x2 = min(W, cx + bw // 2)
    y2 = min(H, cy + bh // 2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box has zero area — check YOLO coordinates.")

    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]

    # Resize ROI
    new_w = max(1, int(round(rw * scale_x)))
    new_h = max(1, int(round(rh * scale_y)))
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Elliptical mask for seamless blending
    mask = np.zeros((new_h, new_w), dtype=np.uint8)
    cv2.ellipse(mask, (new_w // 2, new_h // 2), (new_w // 2, new_h // 2), 0, 0, 360, 255, -1)

    # Clip clone centre to keep ROI within image bounds
    half_w = new_w // 2
    half_h = new_h // 2
    cx_safe = int(np.clip(cx, half_w, W - half_w - 1))
    cy_safe = int(np.clip(cy, half_h, H - half_h - 1))

    edited = cv2.seamlessClone(roi_resized, img, mask, (cx_safe, cy_safe), cv2.NORMAL_CLONE)

    # Compute new bounding box
    x1_new = max(0,     int(round(cx_safe - new_w / 2)))
    y1_new = max(0,     int(round(cy_safe - new_h / 2)))
    x2_new = min(W - 1, int(round(cx_safe + new_w / 2)))
    y2_new = min(H - 1, int(round(cy_safe + new_h / 2)))

    meta = {
        "tag_name":  tag_name,
        "orig_x":    x1,
        "orig_y":    y1,
        "orig_w":    x2 - x1,
        "orig_h":    y2 - y1,
        "scale_x":   scale_x,
        "scale_y":   scale_y,
        "new_x":     x1_new,
        "new_y":     y1_new,
        "new_w":     x2_new - x1_new,
        "new_h":     y2_new - y1_new,
    }

    return edited, meta