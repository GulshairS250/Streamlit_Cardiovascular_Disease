import streamlit as st
import pandas as pd
import numpy as np
import joblib
import kagglehub
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt


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
            f"Model file '{model_path}' not found. Run the notebook to train and save the model in this folder."
        )
        return None
    model = joblib.load(path)
    _patch_simple_imputer(model)
    return model


@st.cache_resource
def load_dataset():
    """Load Kaggle dataset once and compute numeric feature columns."""
    path = kagglehub.dataset_download("colewelkins/cardiovascular-disease")
    csv_files = [f for f in Path(path).iterdir() if f.suffix.lower() == ".csv"]
    data_path = csv_files[0] if csv_files else Path(path)
    df = pd.read_csv(str(data_path))

    if "cardio" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"cardio": "target"})

    # Basic cleaning to match notebook behavior
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)

    if "target" not in df.columns:
        raise ValueError("Dataset does not contain a 'target' column after preprocessing.")

    # Keep only numeric features, exclude target + id (common in this dataset)
    numeric_cols = [
        c
        for c in df.columns
        if c not in ("target", "id") and pd.api.types.is_numeric_dtype(df[c])
    ]

    return df, numeric_cols


def _risk_bar(ax, proba):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    color = "#E94F37" if proba >= 0.5 else "#2E86AB"
    ax.add_patch(plt.Rectangle((0, 0.35), proba, 0.3, color=color, alpha=0.95))
    ax.add_patch(plt.Rectangle((proba, 0.35), 1 - proba, 0.3, color="#C7C7C7", alpha=0.35))
    ax.text(0.02, 0.82, "Risk", fontsize=12, color="#212529")
    ax.text(
        0.02,
        0.48,
        f"{proba:.2%}",
        fontsize=22,
        color=color,
        fontweight="bold",
    )
    ax.text(0.98, 0.82, "0%           100%", fontsize=9, ha="right", color="#6C757D")


def _feature_percentile(df, col, value):
    s = df[col].dropna().to_numpy()
    if len(s) == 0:
        return np.nan
    return float(np.mean(s <= value))


def _hist_with_marker(df, col, value, ax):
    s = df[col].dropna().to_numpy()
    if len(s) == 0:
        ax.set_axis_off()
        return
    p1, p99 = np.nanpercentile(s, [1, 99])
    s2 = s[(s >= p1) & (s <= p99)]
    ax.hist(s2, bins=30, color="#2E86AB", alpha=0.25, edgecolor="none")
    ax.axvline(value, color="#E94F37", linewidth=2)
    ax.set_title(f"{col}", fontsize=11, color="#212529")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")


def _corr_story(df, feature_cols, target_col="target", top_n=8):
    # Pearson correlation is fine for these numeric-coded features
    corrs = []
    for c in feature_cols:
        if df[c].nunique(dropna=True) <= 1:
            continue
        corrs.append((c, df[c].corr(df[target_col])))
    corrs.sort(key=lambda x: abs(x[1]) if x[1] is not None else 0.0, reverse=True)
    return corrs[:top_n]


def _plot_corr_heatmap(df, cols, target_col="target", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    data = df[cols + [target_col]].corr()
    im = ax.imshow(data.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title("Correlation Story (features vs outcome)", fontsize=12, color="#212529")

    # Annotate a smaller matrix for readability
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    return ax


def _what_if_curve(model, input_row, df, col, steps=26):
    base = input_row.copy()
    s = df[col].dropna().to_numpy()
    if len(s) == 0:
        return None, None, None
    p10, p90 = np.nanpercentile(s, [10, 90])
    grid = np.linspace(p10, p90, steps)
    probs = []
    for v in grid:
        base[col] = v
        if hasattr(model, "predict_proba"):
            probs.append(float(model.predict_proba(base)[:, 1][0]))
        else:
            probs.append(float(model.predict(base)[0]))
    # Input marker
    ax = None
    return grid, np.array(probs), float(input_row[col].iloc[0])


def _infer_model_features(model, dataset_feature_cols):
    # Prefer sklearn's fitted feature list if present
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        return dataset_feature_cols
    # Ensure they are in the dataset feature columns
    cols = [c for c in feature_names if c in dataset_feature_cols]
    return cols if cols else dataset_feature_cols


def _input_widget(df, col):
    """Create a dynamic widget for a numeric feature."""
    s = df[col].dropna()
    if len(s) == 0:
        st.sidebar.warning(f"No values found for `{col}`.")
        return 0.0

    unique_vals = np.sort(s.unique())
    is_int_like = np.all(np.isclose(unique_vals, unique_vals.astype(int)))
    is_binary = len(unique_vals) <= 2 and set(unique_vals.tolist()).issubset({0, 1})
    is_small_cat = len(unique_vals) <= 8 and is_int_like

    median = float(np.nanpercentile(s.to_numpy(), 50))
    p1, p99 = np.nanpercentile(s.to_numpy(), [1, 99])

    if is_binary:
        return st.sidebar.selectbox(col, options=[0, 1], index=int(median >= 0.5), key=f"in_{col}")
    if is_small_cat:
        opts = unique_vals.astype(int).tolist()
        default_idx = int(np.argmin(np.abs(np.array(opts) - median)))
        return st.sidebar.selectbox(col, options=opts, index=default_idx, key=f"in_{col}")

    # Continuous numeric
    if is_int_like:
        lo, hi = int(p1), int(p99)
        step = 1
        default = int(np.clip(round(median), lo, hi))
        return st.sidebar.slider(col, min_value=lo, max_value=hi, value=default, step=step, key=f"in_{col}")

    # Float
    lo, hi = float(p1), float(p99)
    default = float(np.clip(median, lo, hi))
    span = max(hi - lo, 1e-9)
    step = span / 200.0
    step = max(step, 0.0001)
    return st.sidebar.slider(col, min_value=lo, max_value=hi, value=default, step=step, key=f"in_{col}")


st.set_page_config(page_title="Cardiovascular Disease Risk Predictor", layout="wide")

plt.rcParams["figure.facecolor"] = "#F8F9FA"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "#DEE2E6"
plt.rcParams["text.color"] = "#212529"
plt.rcParams["font.size"] = 11

st.markdown(
    """
<style>
.header_card {
  padding: 18px 18px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(46,134,171,0.14), rgba(233,79,55,0.10));
  border: 1px solid rgba(0,0,0,0.06);
}
.subtle {
  color: #6C757D;
}
.metric_big {
  font-size: 28px;
  font-weight: 800;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
## Cardiovascular Disease Risk Predictor
This is a storytelling-style dashboard: change patient inputs and the graphs will update to show how your profile sits against the dataset and how the model's risk responds.
""",
)

df, dataset_feature_cols = load_dataset()

_app_dir = Path(__file__).parent if "__file__" in dir() else Path(".")
_pkl_files = list(_app_dir.glob("*.pkl"))
_choices = ["best_model.pkl"] + sorted([f.name for f in _pkl_files if f.name != "best_model.pkl"])
_default_idx = _choices.index("k-NN.pkl") if "k-NN.pkl" in _choices else 0

with st.sidebar:
    st.markdown("### Model & Inputs")
    model_file = st.selectbox("Model file", _choices, index=_default_idx)
    model = load_model(str(_app_dir / model_file))

    st.divider()
    st.markdown("### Patient profile (dynamic)")
    st.caption("Inputs are generated from the Kaggle dataset numeric columns.")

    model_feature_cols = _infer_model_features(model, dataset_feature_cols) if model is not None else dataset_feature_cols

    input_values = {}
    for col in model_feature_cols:
        input_values[col] = _input_widget(df, col)

    run_diagnostics_btn = st.button("Compute model diagnostics (slow)", use_container_width=True)


def _build_input_frame(feature_cols, values):
    row = {c: values[c] for c in feature_cols}
    # Preserve feature order
    return pd.DataFrame([row], columns=feature_cols)


if model is not None:
    input_data = _build_input_frame(model_feature_cols, input_values)

    # Predict
    try:
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_data)[:, 1][0])
        pred = int(model.predict(input_data)[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Tip: retrain the model using the notebook after switching to the Kaggle dataset, so feature columns match the dataset-driven UI."
        )
        st.stop()

    risk_label = "cardiovascular disease present (class 1)" if pred == 1 else "no cardiovascular disease (class 0)"
    risk_color = "#E94F37" if proba is not None and proba >= 0.5 else "#2E86AB"

    # Main story panel
    top_left, top_right = st.columns([2, 1], gap="large")
    with top_left:
        st.markdown('<div class="header_card">', unsafe_allow_html=True)
        st.markdown(f"**Model outcome:** {risk_label}")
        if proba is not None:
            st.markdown(f"<div class='metric_big' style='color:{risk_color};'>{proba:.2%}</div>", unsafe_allow_html=True)
            st.markdown('<div class="subtle">Probability estimate from `predict_proba`.</div>', unsafe_allow_html=True)
        else:
            st.warning("This model does not expose `predict_proba`, so probability is unavailable.")
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        fig, ax = plt.subplots(figsize=(4, 2.8))
        _risk_bar(ax, proba if proba is not None else 0.0)
        st.pyplot(fig, use_container_width=True)

    # Live analytics (updates with inputs)
    # 1) Compare input percentiles
    abs_corr = _corr_story(df, model_feature_cols, top_n=min(8, len(model_feature_cols)))
    top_features = [c for c, _ in abs_corr[:4]]

    st.divider()
    st.markdown("### Input vs Dataset: where your profile lands")
    cols = st.columns(2)
    for i, col in enumerate(top_features[:4]):
        with cols[i % 2]:
            p = _feature_percentile(df, col, float(input_data[col].iloc[0]))
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            _hist_with_marker(df, col, float(input_data[col].iloc[0]), ax=ax)
            ax.set_title(f"{col} (input: {input_data[col].iloc[0]:.3g}, percentile: {p*100:.1f}%)", fontsize=10)
            st.pyplot(fig, use_container_width=True)

    # 2) Correlation story heatmap
    st.markdown("### Correlation Story (model context)")
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _plot_corr_heatmap(df, top_features, target_col="target", ax=ax)
    st.pyplot(fig, use_container_width=True)

    # 3) What-if curves
    st.markdown("### What-if: change one feature, watch risk respond")
    # Choose up to 2 top drivers by abs correlation
    drivers = abs_corr[:2]
    what_if_cols = [c for c, _ in drivers]

    for col in what_if_cols:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        grid, probs, marker = _what_if_curve(model, input_data.iloc[0:1].copy(), df, col, steps=30)
        if grid is None:
            st.warning(f"Could not build what-if curve for {col}.")
            continue
        ax.plot(grid, probs, color="#2E86AB", linewidth=2)
        ax.scatter([marker], [probs[np.argmin(np.abs(grid - marker))]], color="#E94F37", s=40, zorder=3)
        ax.set_title(f"Risk sensitivity to `{col}`", fontsize=12, color="#212529")
        ax.set_xlabel(col)
        ax.set_ylabel("Predicted probability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        st.pyplot(fig, use_container_width=True)

    if not run_diagnostics_btn:
        st.info(
            "What-if sensitivity is live. Click 'Compute model diagnostics (slow)' to also show confusion matrix, ROC, learning curve, and feature importance."
        )
        st.stop()

    # Model diagnostics (optional but asked: confusion + training visuals)
    with st.expander("Model diagnostics (confusion matrix, ROC, learning curve)"):
        # Use a stratified split, but cap size for speed
        SAMPLE_MAX = 14000
        df_eval = df.copy()
        if len(df_eval) > SAMPLE_MAX:
            df_eval = df_eval.sample(n=SAMPLE_MAX, random_state=42)

        X = df_eval[model_feature_cols].copy()
        y = df_eval["target"].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        st.markdown(f"**Test F1 (diagnostics subset):** {f1:.3f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = ["class 0", "class 1"]
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        ax.set_title("Story: Confusion Matrix (diagnostics)", color="#212529")
        st.pyplot(fig, use_container_width=True)

        # ROC curve if proba exists
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], color="#6C757D", linestyle="--", linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("Story: ROC Curve", color="#212529")
            ax.legend()
            ax.grid(True, alpha=0.25)
            st.pyplot(fig, use_container_width=True)

        # Learning curve (slow -> gated)
        show_lc = st.toggle("Show learning curve (slower)", value=False)
        if show_lc:
            st.markdown("Learning curve story: F1 vs training set size (sample-based).")
            # Cap training data size
            MAX_TRAIN = 6000
            if len(X_train) > MAX_TRAIN:
                X_lc = X_train.sample(n=MAX_TRAIN, random_state=42)
                y_lc = y_train.loc[X_lc.index]
            else:
                X_lc = X_train
                y_lc = y_train

            # Use a small training fraction grid
            train_sizes = np.linspace(0.2, 1.0, 6)
            fig, ax = plt.subplots(figsize=(7, 4))
            lc = learning_curve(
                model,
                X_lc,
                y_lc,
                cv=3,
                scoring="f1",
                train_sizes=train_sizes,
                n_jobs=None,
                random_state=42,
            )
            train_sizes_abs, train_scores, val_scores = lc[0], lc[1], lc[2]
            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)
            ax.plot(train_sizes_abs, train_mean, "o-", color="#2E86AB", label="Training F1")
            ax.plot(train_sizes_abs, val_mean, "s-", color="#E94F37", label="Validation F1")
            ax.set_title("Story: Learning Curve (sample-based)", color="#212529")
            ax.set_xlabel("Training set size")
            ax.set_ylabel("F1 score")
            ax.grid(True, alpha=0.25)
            ax.legend()
            st.pyplot(fig, use_container_width=True)

    # Feature importance story (permutation importance)
    st.divider()
    with st.expander("Feature importance (how the model weighs inputs)"):
        MAX_PI = 6000
        df_pi = df.copy()
        if len(df_pi) > MAX_PI:
            df_pi = df_pi.sample(n=MAX_PI, random_state=42)
        X_pi = df_pi[model_feature_cols].copy()
        y_pi = df_pi["target"].copy()

        # Compute permutation importance (may still take some time)
        pi = permutation_importance(
            model,
            X_pi,
            y_pi,
            scoring="f1",
            n_repeats=8,
            random_state=42,
        )
        imp = pd.Series(pi.importances_mean, index=model_feature_cols)
        imp = imp.sort_values(ascending=False).head(10)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(imp.index[::-1], imp.values[::-1], color="#2E86AB", alpha=0.65)
        ax.set_title("Story: Top feature influences (permutation importance)", color="#212529")
        ax.set_xlabel("Mean importance (F1 delta)")
        st.pyplot(fig, use_container_width=True)

    st.caption(
        "Educational tool only. Do not use as a substitute for professional medical advice."
    )
