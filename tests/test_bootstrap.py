"""Tests del bootstrap de F1-macro (examples/cinc2020/bootstrap_ci.py)."""

import numpy as np
from sklearn.metrics import f1_score

from bootstrap_ci import bootstrap_f1, load_predictions, percentile_ci


def test_percentile_ci():
    lo, hi = percentile_ci(np.arange(100, dtype=float))
    assert lo < 50 < hi


def test_bootstrap_contiene_estimador(tmp_path):
    rng = np.random.default_rng(4)
    yt = (rng.random((300, 3)) < 0.4).astype(np.uint8)
    yp = (yt & (rng.random((300, 3)) < 0.8)).astype(np.uint8)
    point, (lo, hi), stats = bootstrap_f1(yt, yp, n_bootstrap=200, seed=0)
    assert point == f1_score(yt, yp, average="macro", zero_division=0)
    assert lo <= point <= hi and len(stats) == 200


def test_load_predictions_columnas(tmp_path):
    import pandas as pd

    df = pd.DataFrame({"record_name": ["a", "b"], "true_X": [1, 0],
                       "prob_X": [0.9, 0.1], "pred_X": [1, 0]})
    path = tmp_path / "p.csv"
    df.to_csv(path, index=False)
    _, yt, yp, classes = load_predictions(path)
    assert classes == ["X"] and yt.tolist() == [[1], [0]] and yp.tolist() == [[1], [0]]
