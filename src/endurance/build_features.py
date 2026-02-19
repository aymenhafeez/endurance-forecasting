import numpy as np
import pandas as pd

from .db import connect


def week_start_monday(date_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series)
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.date.astype(str)
