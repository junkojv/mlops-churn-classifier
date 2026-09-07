import pandas as pd

from src.pipeline import build_pipeline


def test_build_pipeline_structure():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    named_steps = dict(pipe.named_steps)
    assert "pre" in named_steps
    assert "model" in named_steps


def test_build_pipeline_random_forest():
    pipe = build_pipeline(["num"], ["cat"], "random_forest")
    assert pipe.named_steps["model"].__class__.__name__ == "RandomForestClassifier"


def test_pipeline_fit_predict():
    X = pd.DataFrame({
        "num": [1.0, 2.0, None, 4.0, 5.0],
        "cat": ["A", "B", "A", None, "B"]
    })
    y = [0, 1, 0, 1, 0]

    pipe = build_pipeline(numeric=["num"], categorical=["cat"], model_type="logreg")
    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
