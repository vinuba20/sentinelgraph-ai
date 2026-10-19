import numpy as np
import pandas as pd

from sentinelgraph.evaluation import evaluate_classifier


def test_binary_metrics_include_pr_auc_and_false_positive_rate():
    result = evaluate_classifier(
        model_name="test",
        y_true=pd.Series(["BENIGN", "BENIGN", "ATTACK", "ATTACK"]),
        y_pred=np.array(["BENIGN", "ATTACK", "ATTACK", "ATTACK"]),
        probabilities=np.array([[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]]),
        classes=np.array(["BENIGN", "ATTACK"]),
    )
    assert result.false_positive_rate == 0.5
    assert result.pr_auc == 1.0
    assert result.confusion_matrix == [[1, 1], [0, 2]]


def test_multiclass_metrics_do_not_claim_binary_false_positive_rate():
    result = evaluate_classifier(
        model_name="test", y_true=np.array(["a", "b", "c"]), y_pred=np.array(["a", "b", "a"])
    )
    assert np.isnan(result.false_positive_rate)
    assert result.pr_auc is None
