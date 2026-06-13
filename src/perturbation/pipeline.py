"""
pipeline.py
-----------
run iterate over all images and scale factors
"""

import os
import cv2
import pandas as pd

from .scale import DEFAULT_SCALE_FACTORS
from .sex_classification import adjust_scale
from .age_estimation import adjust_root, adjust_crown


def run_sex_perturbation(
    df: pd.DataFrame,
    numbers: list[float],
    out_dir: str,
    axis: str = "x",
) -> pd.DataFrame:
    """Run perturbation pipeline for sex classification.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: Image_Path, tag_name,
        x_center, y_center, width, height (YOLO normalised).
    numbers : list[float]
        Scale factors to apply (e.g. DEFAULT_SCALE_FACTORS).
    out_dir : str
        Directory where perturbed images will be saved.
    axis : str, optional
        Axis to perturb -- "x" (width) or "y" (height). Default "x".

    Returns
    -------
    pd.DataFrame
        One row per (image region x scale factor) with columns:
        tag_name, filename, path_name_ori,
        orig_x, orig_y, orig_w, orig_h,
        scale_x, scale_y,
        path_name_adj, filename_adj,
        new_x, new_y, new_w, new_h.
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got '{axis}'.")

    os.makedirs(out_dir, exist_ok=True)
    rows = []

    for i in range(len(df)):
        row = df.iloc[i]

        img_path = row["Image_Path"]
        tag_name = row["tag_name"]

        img = cv2.imread(img_path)
        if img is None:
            print("skip:", img_path)
            continue

        base = os.path.basename(img_path)
        name, ext = os.path.splitext(base)

        for s in numbers:
            scale_x = float(s) if axis == "x" else 1.0
            scale_y = float(s) if axis == "y" else 1.0

            edited, meta = apply_scale(
                img=img,
                x_center=float(row["x_center"]),
                y_center=float(row["y_center"]),
                width=float(row["width"]),
                height=float(row["height"]),
                tag_name=tag_name,
                scale_x=scale_x,
                scale_y=scale_y,
            )

            out_name = f"{name}_s{axis}{s:.2f}_{tag_name}{ext}"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, edited)

            record = {
                "filename":      base,
                "path_name_ori": img_path,
                "path_name_adj": out_path,
                "filename_adj":  out_name,
            }
            record.update(meta)   # tag_name, orig_x/y/w/h, scale_x/y, new_x/y/w/h
            rows.append(record)

    return pd.DataFrame(rows)


def run_age_perturbation(
    data: pd.DataFrame,
    output_dir: str,
    region: str = "root",
    scale_factors: list[float] | None = None,
) -> pd.DataFrame:
    """Run perturbation pipeline for age estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns: Path_Name, label, type, points,
        and bounding box fields (x_min, y_min, width, height).
    output_dir : str
        Directory where perturbed images will be saved.
    region : str, optional
        Region to perturb -- "root" or "crown". Default "root".
    scale_factors : list[float], optional
        Scale values to apply. Defaults to DEFAULT_SCALE_FACTORS.

    Returns
    -------
    pd.DataFrame
        One row per (image x scale factor) with perturbation metadata.
    """
    if region not in ("root", "crown"):
        raise ValueError(f"region must be 'root' or 'crown', got '{region}'.")

    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for path, grp in data.groupby("Path_Name"):
        img = cv2.imread(path)
        if img is None:
            print("skip:", path)
            continue

        filename = os.path.splitext(os.path.basename(path))[0]

        # Extract annotations
        box_root   = grp.query("label=='tooth_root'     and type=='box'").iloc[0]
        pts_root   = grp.query("label=='root'           and type=='polyline'").points.iloc[0]
        pts_droot  = grp.query("label=='distance_root'  and type=='polyline'").points.iloc[0]
        box_crown  = grp.query("label=='tooth_crown'    and type=='box'").iloc[0]
        pts_crown  = grp.query("label=='crown'          and type=='polyline'").points.iloc[0]
        pts_dcrown = grp.query("label=='distance_crown' and type=='polyline'").points.iloc[0]

        for scale in scale_factors:
            if region == "root":
                img_out, meta = adjust_root(
                    image=img,
                    root_box=box_root,   crown_box=box_crown,
                    pts_root=pts_root,   pts_dist_root=pts_droot,
                    pts_crown=pts_crown, pts_dist_crown=pts_dcrown,
                    scale=scale,
                )
            else:
                img_out, meta = adjust_crown(
                    image=img,
                    root_box=box_root,  crown_box=box_crown,
                    pts_root=pts_root,  pts_dist_root=pts_droot,
                    pts_cr=pts_crown,   pts_dist_cr=pts_dcrown,
                    scale=scale,
                )

            new_name = f"{filename}_{region}_{scale:.2f}"
            new_path = os.path.join(output_dir, new_name + ".png")
            cv2.imwrite(new_path, img_out)

            record = {
                "filename":     filename,
                "path_name_ori":    path,
                "filename_adj": new_name,
                "path_name_adj": new_path,
                "scale":        scale,
                "region":       region,
            }
            record.update(meta)   # orig_*/new_* keys from age_estimation
            records.append(record)

    return pd.DataFrame(records)