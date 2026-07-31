"""
age_estimation.py
-----------------
Perturbation functions for the age estimation task.
so perturbation via inpaint + resize
with separate top-anchor (root) and bottom-anchor (crown) strategies.
"""

import cv2
import numpy as np


def _line_length(pts: list) -> float:
    """Euclidean length of a 2-point polyline.

    Parameters
    ----------
    pts : list
        Two points as [[x0, y0], [x1, y1]].

    Returns
    -------
    float
        Length in pixels.
    """
    a = np.array(pts[0], float)
    b = np.array(pts[1], float)
    return float(np.linalg.norm(b - a))


def adjust_root(
    image: np.ndarray,
    root_box,
    crown_box,
    pts_root: list,
    pts_dist_root: list,
    pts_crown: list,
    pts_dist_crown: list,
    scale: float,
) -> tuple[np.ndarray, dict]:
    """Perturb the root height of a lower third molar.

    Removes the root region via inpainting, then pastes a resized version
    back using a top-anchor strategy (top edge fixed, bottom edge moves).
    Crown metadata is recorded unchanged for ratio computation.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    root_box : row-like
        Annotation row for tooth_root bounding box
        (must have .x_min, .y_min, .width, .height).
    crown_box : row-like
        Annotation row for tooth_crown bounding box.
    pts_root : list
        Root polyline [[x0,y0],[x1,y1]].
    pts_dist_root : list
        Distance-root polyline [[x0,y0],[x1,y1]].
    pts_crown : list
        Crown polyline [[x0,y0],[x1,y1]].
    pts_dist_crown : list
        Distance-crown polyline [[x0,y0],[x1,y1]].
    scale : float
        Height scale factor applied to the root region.

    Returns
    -------
    out : np.ndarray
        Perturbed image.
    meta : dict
        Metadata for both root (before/after) and crown (original only).
    """
    x, y = int(root_box.x_min), int(root_box.y_min)
    w, h = int(root_box.width), int(root_box.height)
    orig_root_h   = h
    orig_root_len = _line_length(pts_root)
    orig_dist_len = _line_length(pts_dist_root)

    # Inpaint root region
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cleaned = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    base = cv2.inpaint(cleaned, mask, 2, cv2.INPAINT_TELEA)

    # Resize — top-anchor (top edge fixed, bottom moves)
    new_root_h = int(h * scale)
    roi = image[y:y + h, x:x + w]
    resized = cv2.resize(roi, (w, new_root_h), interpolation=cv2.INTER_LANCZOS4)
    out = base.copy()
    out[y:y + new_root_h, x:x + w] = resized

    # Recalculate root polyline
    p0, p1 = np.array(pts_root[0]), np.array(pts_root[1])
    top_pt = p0 if p0[1] < p1[1] else p1
    vec = (p1 - p0) / np.linalg.norm(p1 - p0)
    target_y = y + new_root_h
    t = (target_y - top_pt[1]) / vec[1]
    end_root = (top_pt + vec * t).astype(int)
    new_root_line = [top_pt.tolist(), end_root.tolist()]
    new_root_len = float(np.linalg.norm(end_root - top_pt))

    # Recalculate distance-root polyline
    end_dist = end_root + np.array([0, orig_dist_len], int)
    new_dist_line = [end_root.tolist(), end_dist.tolist()]

    meta = {
        # Root — before & after
        "orig_tooth_root_box":  [x, y, w, h],
        "orig_tooth_root_h":    orig_root_h,
        "new_tooth_root_h":     new_root_h,
        "orig_root_pts":        pts_root,
        "new_root_pts":         new_root_line,
        "orig_root_len":        orig_root_len,
        "new_root_len":         new_root_len,
        "orig_dist_root_pts":   pts_dist_root,
        "new_dist_root_pts":    new_dist_line,
        "orig_dist_root_len":   orig_dist_len,
        "new_dist_root_len":    orig_dist_len,   # distance line length unchanged
        # Crown — original only (not perturbed)
        "orig_tooth_crown_box": [int(crown_box.x_min), int(crown_box.y_min),
                                 int(crown_box.width), int(crown_box.height)],
        "orig_tooth_crown_h":   int(crown_box.height),
        "orig_crown_pts":       pts_crown,
        "orig_dist_crown_pts":  pts_dist_crown,
        "orig_crown_len":       _line_length(pts_crown),
        "orig_dist_crown_len":  _line_length(pts_dist_crown),
    }
    return out, meta


def adjust_crown(
    image: np.ndarray,
    root_box,
    crown_box,
    pts_root: list,
    pts_dist_root: list,
    pts_cr: list,
    pts_dist_cr: list,
    scale: float,
) -> tuple[np.ndarray, dict]:
    """Perturb the crown height of a lower third molar.

    Removes the crown region via inpainting, then pastes a resized version
    back using a bottom-anchor strategy (bottom edge fixed, top edge moves).
    Root metadata is recorded unchanged for ratio computation.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.
    root_box : row-like
        Annotation row for tooth_root bounding box.
    crown_box : row-like
        Annotation row for tooth_crown bounding box.
    pts_root : list
        Root polyline [[x0,y0],[x1,y1]].
    pts_dist_root : list
        Distance-root polyline [[x0,y0],[x1,y1]].
    pts_cr : list
        Crown polyline [[x0,y0],[x1,y1]].
    pts_dist_cr : list
        Distance-crown polyline [[x0,y0],[x1,y1]].
    scale : float
        Height scale factor applied to the crown region.

    Returns
    -------
    out : np.ndarray
        Perturbed image.
    meta : dict
        Metadata for both crown (before/after) and root (original only).
    """
    # Root metadata (unchanged)
    orig_root_len      = _line_length(pts_root)
    orig_dist_root_len = _line_length(pts_dist_root)

    x, y = int(crown_box.x_min), int(crown_box.y_min)
    w, h = int(crown_box.width), int(crown_box.height)
    orig_cr_h      = h
    orig_cr_len    = _line_length(pts_cr)
    orig_dist_cr_len = _line_length(pts_dist_cr)

    # Inpaint crown region
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    cleared = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    base = cv2.inpaint(cleared, mask, 2, cv2.INPAINT_TELEA)

    # Resize — bottom-anchor (bottom edge fixed, top moves)
    new_cr_h = int(h * scale)
    roi = image[y:y + h, x:x + w]
    roi_rs = cv2.resize(roi, (w, new_cr_h), interpolation=cv2.INTER_LANCZOS4)
    new_y = y + h - new_cr_h
    out = base.copy()
    out[new_y:new_y + new_cr_h, x:x + w] = roi_rs

    # Recalculate crown polyline
    # bottom_pt is the fixed anchor (bottom edge does NOT move — bottom-anchor resize),
    # so it must stay unchanged. The new endpoint is found by intersecting the line
    # with the NEW top edge of the resized crown box (new_y), the same way
    # adjust_root() intersects with the new bottom edge. The previous version
    # incorrectly shifted bottom_pt itself and used a fixed original length,
    # which produced a line that did not match the actual resized box at all.
    p0, p1 = np.array(pts_cr[0]), np.array(pts_cr[1])
    bottom_pt = p0 if p0[1] > p1[1] else p1
    vec = (p1 - p0) / np.linalg.norm(p1 - p0)
    target_y = new_y  # new top edge of the resized crown box
    t = (target_y - bottom_pt[1]) / vec[1]
    end_cr = (bottom_pt + vec * t).astype(int)
    new_cr_line = [bottom_pt.tolist(), end_cr.tolist()]
    new_cr_len = float(np.linalg.norm(end_cr - bottom_pt))

    # Recalculate distance-crown polyline
    end_dist = end_cr + np.array([0, orig_dist_cr_len], int)
    new_dist_line = [end_cr.tolist(), end_dist.tolist()]

    meta = {
        # Crown — before & after
        "orig_tooth_crown_box": [x, y, w, h],
        "orig_tooth_crown_h":   orig_cr_h,
        "new_tooth_crown_h":    new_cr_h,
        "orig_crown_pts":       pts_cr,
        "new_crown_pts":        new_cr_line,
        "orig_crown_len":       orig_cr_len,
        "new_crown_len":        new_cr_len,
        "orig_dist_crown_pts":  pts_dist_cr,
        "new_dist_crown_pts":   new_dist_line,
        "orig_dist_crown_len":  orig_dist_cr_len,
        "new_dist_crown_len":   orig_dist_cr_len,  # distance line length unchanged
        # Root — original only (not perturbed)
        "orig_tooth_root_box":  [int(root_box.x_min), int(root_box.y_min),
                                 int(root_box.width), int(root_box.height)],
        "orig_tooth_root_h":    int(root_box.height),
        "orig_root_pts":        pts_root,
        "orig_dist_root_pts":   pts_dist_root,
        "orig_root_len":        orig_root_len,
        "orig_dist_root_len":   orig_dist_root_len,
    }
    return out, meta