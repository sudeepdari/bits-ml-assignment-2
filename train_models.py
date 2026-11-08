"""Train and evaluate the five required classifiers on the Adult dataset."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "census-income"
MODEL_DIR = ROOT / "model"
TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    TARGET,
]
NUMERIC_FEATURES = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_FEATURES = [
    column for column in COLUMNS[:-1] if column not in NUMERIC_FEATURES
]
FEATURES = COLUMNS[:-1]


def load_adult_file(path: Path, *, skiprows: int = 0) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values="?",
        skip_blank_lines=True,
    )
    for column in CATEGORICAL_FEATURES:
        values = data[column].astype("string").str.strip()
        data[column] = values.astype(object).where(values.notna(), np.nan)
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[TARGET] = (
        data[TARGET]
        .astype("string")
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )
    data = data[data[TARGET].isin([NEGATIVE_LABEL, POSITIVE_LABEL])].copy()
    return data


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def make_models() -> dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            min_samples_leaf=2,
            random_state=42,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": LinearSVC(
            C=1.0,
            dual="auto",
            max_iter=5000,
            random_state=42,
        ),
    }


def auc_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def main() -> None:
    train_data = load_adult_file(DATA_DIR / "adult.data")
    test_data = load_adult_file(DATA_DIR / "adult.test", skiprows=1)
    if train_data.empty or test_data.empty:
        raise ValueError("Training or test data is empty after parsing.")

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]

    MODEL_DIR.mkdir(exist_ok=True)
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    rows = []

    for name, classifier in make_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        scores = auc_scores(pipeline, X_test)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, scores),
                "Precision": precision_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "F1": f1_score(
                    y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )
        model_path = MODEL_DIR / f"{name.lower().replace(' ', '_').replace('-', '')}.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Saved {name} -> {model_path.relative_to(ROOT)}")

    results = pd.DataFrame(rows)
    results.to_csv(ROOT / "metrics.csv", index=False)
    print(f"Training rows: {len(train_data):,}; test rows: {len(test_data):,}")
    print("\nTest-set metrics (positive class: >50K):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved cleaned labeled test data -> {(ROOT / 'test_data.csv').name}")


if __name__ == "__main__":
    main()