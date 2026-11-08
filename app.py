"""Streamlit dashboard for evaluating saved Adult income classifiers."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"
FEATURES = [
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
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
LABEL_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]


@st.cache_resource
def load_models() -> dict[str, object]:
    return {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_FILES.items()
    }


def clean_upload(data: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES + [TARGET]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    cleaned = data[required].copy()
    cleaned.replace("?", pd.NA, inplace=True)
    for column in FEATURES:
        if column in {"age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"}:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        else:
            values = cleaned[column].astype("string").str.strip()
            cleaned[column] = values.astype(object).where(values.notna(), float("nan"))
    cleaned[TARGET] = (
        cleaned[TARGET]
        .astype("string")
        .str.strip()
        .str.replace(r"\.$", "", regex=True)
    )
    if cleaned[TARGET].isna().any():
        raise ValueError("The target column contains missing labels.")
    invalid_labels = sorted(set(cleaned[TARGET].dropna()) - set(LABEL_ORDER))
    if invalid_labels:
        raise ValueError(
            "Income labels must be '<=50K' or '>50K'. Unexpected labels: "
            + ", ".join(invalid_labels)
        )
    if cleaned[TARGET].nunique() != 2:
        raise ValueError("The uploaded test data must contain both income classes.")
    return cleaned


def get_auc_scores(model: object, features: pd.DataFrame) -> object:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_models(data: pd.DataFrame, models: dict[str, object]) -> pd.DataFrame:
    features = data[FEATURES]
    labels = data[TARGET]
    rows = []
    for name, model in models.items():
        predictions = model.predict(features)
        scores = get_auc_scores(model, features)
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(labels, predictions),
                "AUC": roc_auc_score(labels, scores),
                "Precision": precision_score(
                    labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "Recall": recall_score(
                    labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "F1": f1_score(
                    labels, predictions, pos_label=POSITIVE_LABEL, zero_division=0
                ),
                "MCC": matthews_corrcoef(labels, predictions),
            }
        )
    return pd.DataFrame(rows)


def evaluate_selected_model(data: pd.DataFrame, model: object) -> dict[str, object]:
    labels = data[TARGET]
    predictions = model.predict(data[FEATURES])
    return {
        "matrix": confusion_matrix(labels, predictions, labels=LABEL_ORDER),
        "report": classification_report(
            labels,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        ),
    }


@st.cache_data(show_spinner="Evaluating models on the uploaded test data...")
def cached_evaluate_models(
    data: pd.DataFrame, _models: dict[str, object]
) -> pd.DataFrame:
    return evaluate_models(data, _models)


@st.cache_data(show_spinner=False)
def cached_model_diagnostics(
    data: pd.DataFrame, model_name: str, _model: object
) -> dict[str, object]:
    return evaluate_selected_model(data, _model)


st.set_page_config(page_title="Adult Income Model Lab", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] { background: #f4f7f5; color: #1b302a; }
    [data-testid="stHeader"] { background: rgba(244, 247, 245, 0.96); }
    .block-container { max-width: 1320px; padding: 2.25rem 2rem 3rem; }
    h1, h2, h3 { color: #173d35; }
    h1 { font-size: 2.25rem; line-height: 1.15; }
    h2, h3 { font-size: 1.35rem; }
    p, label { color: #40574f; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d6e2dc;
        border-radius: 5px;
        padding: 1rem 1.1rem;
        box-shadow: 0 2px 7px rgba(27, 48, 42, 0.04);
    }
    div[data-testid="stMetricLabel"] { color: #647a71; }
    div[data-testid="stMetricValue"] { color: #173d35; }
    [data-baseweb="select"] > div { background: #ffffff; border-color: #cbd9d2; }
    [data-testid="stFileUploaderDropzone"] { background: #ffffff; border-color: #a9beb4; }
    [data-testid="stDataFrame"] { background: #ffffff; border: 1px solid #d6e2dc; border-radius: 5px; }
    [data-testid="stAlert"] { background: #e8f2ed; color: #254a3e; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Adult Income Model Lab")
st.caption("Compare five saved classifiers on labeled test data. Models are trained separately; this app never retrains them.")

sample_path = ROOT / "test_data.csv"
uploaded_file = st.file_uploader("Upload labeled test data (CSV)", type=["csv"])

try:
    models = load_models()
except FileNotFoundError:
    st.error("Saved models were not found. Run `python train_models.py` first.")
    st.stop()

try:
    if uploaded_file is not None:
        source_data = pd.read_csv(uploaded_file, skipinitialspace=True)
        source_name = uploaded_file.name
    elif sample_path.exists():
        source_data = pd.read_csv(sample_path, skipinitialspace=True)
        source_name = "bundled test_data.csv"
        st.info("Showing the bundled Adult test split. Upload another labeled CSV to evaluate it instead.")
    else:
        st.warning("Upload a labeled CSV to run the evaluation.")
        st.stop()
    test_data = clean_upload(source_data)
except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not use this CSV: {error}")
    st.stop()

saved_metrics = ROOT / "metrics.csv"
if uploaded_file is None and saved_metrics.exists():
    metrics = pd.read_csv(saved_metrics)
else:
    metrics = cached_evaluate_models(test_data, models)
best_model = metrics.loc[metrics["F1"].idxmax(), "Model"]

st.write(f"Evaluating **{len(test_data):,} rows** from `{source_name}`. Positive class: `>50K`.")
summary_cols = st.columns(3)
summary_cols[0].metric("Test rows", f"{len(test_data):,}")
summary_cols[1].metric("Income classes", "2")
summary_cols[2].metric("Top F1 model", best_model)

st.subheader("Model comparison")
st.dataframe(
    metrics.style.format(
        {column: "{:.4f}" for column in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}
    ),
    hide_index=True,
    width="stretch",
)

st.subheader("Per-model diagnostics")
selected_model = st.selectbox("Choose a model", list(MODEL_FILES))
selected_details = cached_model_diagnostics(
    test_data, selected_model, models[selected_model]
)
matrix_tab, report_tab = st.tabs(["Confusion matrix", "Classification report"])
with matrix_tab:
    st.markdown("**Confusion matrix**")
    st.dataframe(
        pd.DataFrame(
            selected_details["matrix"],
            index=[f"Actual {label}" for label in LABEL_ORDER],
            columns=[f"Predicted {label}" for label in LABEL_ORDER],
        ),
        width="stretch",
    )
with report_tab:
    st.markdown("**Classification report**")
    report = pd.DataFrame(selected_details["report"]).T
    st.dataframe(
        report.style.format({column: "{:.3f}" for column in report.columns if column != "support"}),
        width="stretch",
    )

with st.expander("Preview uploaded test rows"):
    st.dataframe(test_data.head(20), hide_index=True, width="stretch")

st.caption("Educational demonstration only. Census demographic attributes can encode historical inequities; do not use these predictions for consequential decisions.")
