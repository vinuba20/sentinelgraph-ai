
import numpy as np
import pandas as pd

from sentinelgraph.data.cleaner import clean_dataframe, cleaning_report


def test_clean_dataframe():
    df = pd.DataFrame({
        " Feature ": [1.0, np.inf, 1.0],
        " Label ": ["BENIGN", "DDoS", "BENIGN"],
    })

    cleaned = clean_dataframe(df)

    assert list(cleaned.columns) == ["Feature", "Label"]
    assert len(cleaned) == 2
    assert cleaned["Feature"].isna().sum() == 1


def test_cleaning_report():
    before = pd.DataFrame({"x": [1.0, np.inf, 1.0]})
    after = clean_dataframe(before)

    report = cleaning_report(before, after)

    assert report["rows_before"] == 3
    assert report["rows_after"] == 2
    assert report["duplicates_removed"] == 1
    assert report["infinite_values_after"] == 0
