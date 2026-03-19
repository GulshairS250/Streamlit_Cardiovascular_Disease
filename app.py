import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def _iter_estimators(obj, seen=None):
    """Recursively yield all nested estimators (Pipeline, GridSearchCV, etc.)."""
    seen = seen or set()
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, Pipeline):
        for _, step in obj.steps:
            yield step
            yield from _iter_estimators(step, seen)
    elif hasattr(obj, "best_estimator_"):
        yield obj.best_estimator_
        yield from _iter_estimators(obj.best_estimator_, seen)
    elif hasattr(obj, "transformers_"):
        for _, trans, _ in obj.transformers_:
            if trans != "drop":
                yield trans
                yield from _iter_estimators(trans, seen)
    elif hasattr(obj, "estimators_") and isinstance(obj.estimators_, (list, tuple)):
        for est in obj.estimators_:
            yield est
            yield from _iter_estimators(est, seen)


def _patch_simple_imputer(model):
    """Fix SimpleImputer _fill_dtype/_fit_dtype for models from older scikit-learn."""
    for obj in _iter_estimators(model):
        if isinstance(obj, SimpleImputer):
            if hasattr(obj, "statistics_"):
                dt = np.result_type(obj.statistics_.dtype, np.float64)
                if not hasattr(obj, "_fill_dtype"):
                    obj._fill_dtype = dt
                if not hasattr(obj, "_fit_dtype"):
                    obj._fit_dtype = dt


@st.cache_resource
def load_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        st.error(
            f"Model file '{model_path}' not found. Put `best_model.pkl` in the app folder."
        )
        return None
    model = joblib.load(path)
    _patch_simple_imputer(model)
    return model


def _infer_model_features(model):
    """Infer expected model input features."""
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and len(feature_names) > 0:
        return list(feature_names)
    # Fallback to common heart/cardiovascular columns if model has no feature_names_in_
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]


def _default_value_for_feature(col):
    """Reasonable fallback defaults by feature name."""
    defaults = {
        "age": 50,
        "sex": 1,
        "cp": 0,
        "trestbps": 130,
        "ap_hi": 120,
        "ap_lo": 80,
        "chol": 240,
        "cholesterol": 1,
        "gluc": 1,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.0,
        "slope": 1,
        "ca": 0,
        "thal": 2,
        "height": 165,
        "weight": 70,
        "smoke": 0,
        "alco": 0,
        "active": 1,
        "gender": 1,
        "age_years": 50,
        "bmi": 25.0,
        "bp_category_encoded": 1,
    }
    return defaults.get(col, 0.0)


def _build_input_widget(col, default):
    """
    Build a practical input widget from feature name only (no dataset needed).
    Uses number_input for maximum compatibility.
    """
    # binary / small integer coded fields
    binary_like = {"sex", "fbs", "exang", "smoke", "alco", "active", "target", "cardio", "gender"}
    small_cat_like = {"cp", "restecg", "slope", "ca", "thal", "cholesterol", "gluc", "bp_category_encoded"}

    if col in binary_like:
        return st.sidebar.selectbox(col, options=[0, 1], index=int(default) if default in (0, 1) else 0)
    if col in small_cat_like:
        options = [0, 1, 2, 3, 4]
        idx = int(default) if isinstance(default, (int, np.integer)) and 0 <= int(default) <= 4 else 0
        return st.sidebar.selectbox(col, options=options, index=idx)

    if isinstance(default, float) and not float(default).is_integer():
        return st.sidebar.number_input(col, value=float(default), step=0.1, format="%.4f")

    return st.sidebar.number_input(col, value=float(default), step=1.0, format="%.4f")


st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="centered")

st.title("Cardiovascular Disease Risk Predictor")
st.markdown("Model-only app: uses `best_model.pkl` directly (no Kaggle/cloud dataset required).")

app_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
model_path = app_dir / "best_model.pkl"
model = load_model(str(model_path))

if model is None:
    st.stop()

feature_cols = _infer_model_features(model)

with st.sidebar:
    st.header("Patient Inputs")
    st.caption("Enter feature values expected by your model.")
    inputs = {}
    for col in feature_cols:
        d = _default_value_for_feature(col)
        inputs[col] = _build_input_widget(col, d)
    predict_btn = st.button("Predict", use_container_width=True)

if predict_btn:
    input_df = pd.DataFrame([{c: inputs[c] for c in feature_cols}], columns=feature_cols)
    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Your model expects different feature names/order. "
            "Use a model saved from the same training pipeline and rerun."
        )
        st.stop()

    st.subheader("Prediction Result")
    if proba is not None:
        st.metric("Estimated Probability (class 1)", f"{proba:.2%}")

    if pred == 1:
        st.warning("Model predicts **cardiovascular disease present (class 1)**.")
    else:
        st.success("Model predicts **no cardiovascular disease (class 0)**.")

st.caption("Educational use only. Not a substitute for medical advice.")
