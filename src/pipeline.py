from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric, categorical, model_type="logreg"):
    """
    Build a reproducible scikit-learn Pipeline with ColumnTransformer preprocessing.
    """
    num = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num, numeric),
            ("cat", cat, categorical)
        ]
    )

    if model_type == "logreg":
        model = LogisticRegression(max_iter=500, random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline(steps=[("pre", pre), ("model", model)])
