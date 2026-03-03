import pandas as pd


def test_rolling_sum_past_only():
    daily = pd.DataFrame({"trimp": [1, 1, 1, 1, 1, 1, 1, 100]})

    daily["rolling_7"] = daily["trimp"].rolling(7).sum()

    # 6th index should be the sum of the first 7 ones
    assert daily.loc[6, "rolling_7"] == 7
