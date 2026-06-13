"""
scale.py
---------------
Generate perturbation scale factors used in these experiments.
"""

import numpy as np
 
 
def generate_scale_factors(
    start_shrink: float,
    stop_shrink: float,
    start_expand: float,
    stop_expand: float,
    step: float,
) -> list[float]:
    
    """Generate a list of scale factors for perturbation experiments.
 
    Produces two ranges joined together:
      - Shrink range : [start_shrink, stop_shrink) stepping downward  (values < 1.0)
      - Expand range : 1.0 (baseline) + [start_expand, stop_expand) stepping downward (values > 1.0)
 
    Parameters
    ----------
    start_shrink : float
        First shrink scale value (e.g. 0.95).
    stop_shrink : float
        Lower bound of shrink range, exclusive (e.g. 0.74 → last value is 0.75).
    start_expand : float
        First expand scale value (e.g. 1.25).
    stop_expand : float
        Lower bound of expand range, exclusive (e.g. 1.04 → last value is 1.05).
    step : float
        Step size between consecutive values (e.g. 0.01).
 
    Returns
    -------
    list[float]
        Combined list: shrink scales + [1.0] + expand scales.

    """
    
    shrink_scales = [
        round(x, 2) for x in np.arange(start_shrink, stop_shrink, -step)
    ]
    expand_scales = [1.0] + [
        round(x, 2) for x in np.arange(start_expand, stop_expand, -step)
    ]
    return shrink_scales + expand_scales
 
 
# ---------------------------------------------------------------------------
# Default scale factors used in the Knowledge XAIeve paper
# (sex classification and age estimation, both tasks share the same range)
# ---------------------------------------------------------------------------
DEFAULT_SCALE_FACTORS = generate_scale_factors(
    start_shrink=0.95,
    stop_shrink=0.74,
    start_expand=1.25,
    stop_expand=1.04,
    step=0.01,
)