
import pandas as pd
import pytest

from sentinelgraph.data.labels import normalize_labels, to_binary_labels


def test_normalize_labels():
    labels = pd.Series(["BENIGN", " DDoS ", "PortScan"])

    assert normalize_labels(labels).tolist() == [
        "BENIGN",
        "DDOS",
        "PORTSCAN",
    ]


def test_binary_labels():
    labels = pd.Series(["BENIGN", "DDoS", "PortScan"])

    assert to_binary_labels(labels).tolist() == [0, 1, 1]


def test_binary_labels_reject_missing_values():
    labels = pd.Series(["BENIGN", None])

    with pytest.raises(ValueError):
        to_binary_labels(labels)
