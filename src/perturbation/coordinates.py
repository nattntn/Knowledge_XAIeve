from __future__ import annotations

from PIL import Image
import pandas as pd


def to_yolo_normalized(
    regions_table: pd.DataFrame,
    image_path_col: str = "path_name_original",
) -> pd.DataFrame:
    """แปลง xmin/ymin/xmax/ymax (พิกเซล) -> x_center/y_center/width/height (0-1)"""
    x_centers, y_centers, widths, heights = [], [], [], []
    for _, row in regions_table.iterrows():
        with Image.open(row[image_path_col]) as img:
            img_w, img_h = img.size
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]
        x_centers.append(((xmin + xmax) / 2) / img_w)
        y_centers.append(((ymin + ymax) / 2) / img_h)
        widths.append((xmax - xmin) / img_w)
        heights.append((ymax - ymin) / img_h)
    result = regions_table.copy()
    result["x_center"] = x_centers
    result["y_center"] = y_centers
    result["width"] = widths
    result["height"] = heights
    return result