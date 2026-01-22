import numpy as np
import pandas as pd

from helper import preprocess, run_kmeans


def test_pipeline_smoke():
    df = pd.DataFrame(
        {
            "q1": ["Yes", "No", "Yes", "No"],
            "q2": ["1-2", "3-4", "1-2", "5-6"],
            "numeric": [1, 2, np.nan, 4],
            "category": ["A", "B", "C", "A"],
        }
    )

    X, feature_names = preprocess(df)

    assert X.shape == (4, 4)
    assert feature_names == ["q1", "q2", "numeric", "category"]
    assert np.isfinite(X).all()

    km, labels, inertia = run_kmeans(X, n_clusters=2, random_state=0)

    assert set(labels) <= {0, 1}
    assert inertia >= 0.0
