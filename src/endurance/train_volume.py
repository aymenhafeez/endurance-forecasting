import pandas as pd
import numpy as np
from .db import connect

FEATURES = [
    "distance_km",
    "time_min",
    "elev_m",
    "trimp",
    "easy_pct",
    "hard_sessions",
    "dist_7d",
    "dist_28d",
    "trimp_7d",
    "trimp_28d",
    "acwr",
    "monotony_7d",
    "strain_7d",
]
