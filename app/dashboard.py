"""Interactive results dashboard; run `streamlit run app/dashboard.py`."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from sentinelgraph.explainability import explain_prediction, global_feature_importance
from sentinelgraph.models.base import RandomForestIntrusionModel

st.set_page_config(page_title="SentinelGraph AI", page_icon="🛡️", layout="wide")

@st.cache_data
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def load_model(path: str):
    return RandomForestIntrusionModel.load(path)

def main() -> None:
    st.title("SentinelGraph AI")
    st.caption("Experimental intrusion-detection results — not a live security guarantee.")
    root = Path(st.sidebar.text_input("Experiment artifact directory", "artifacts/latest"))
    table_path = root / "tables" / "model_comparison.csv"
    if not table_path.exists():
        st.info("Run the training pipeline first, then enter its output directory.")
        st.code("python -m sentinelgraph.pipeline --input data/processed/flows_with_graph_features.csv")
        return
    comparison = read_csv(str(table_path))
    st.subheader("Fair model comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    metric_names = [x for x in ["precision", "recall", "f1", "macro_f1", "pr_auc"] if x in comparison]
    st.bar_chart(comparison.set_index("model_name")[metric_names])
    choice = st.selectbox("Inspect model", comparison["model_name"].tolist())
    predictions_path = root / "tables" / f"{choice}_predictions.csv"
    model_path = root / "models" / f"{choice}.joblib"
    if predictions_path.exists():
        predictions = read_csv(str(predictions_path))
        c1, c2, c3 = st.columns(3)
        c1.metric("Test flows", len(predictions))
        c2.metric("Flagged as attack", int((predictions["predicted_class"] == "ATTACK").sum()))
        c3.metric("Misclassified", int(predictions["is_error"].sum()))
        st.subheader("Prediction summary")
        st.dataframe(predictions.sort_values("confidence", ascending=False), use_container_width=True, hide_index=True)
    if model_path.exists():
        model = load_model(str(model_path))
        st.subheader("Why the model flags traffic")
        st.bar_chart(global_feature_importance(model).set_index("feature")["importance"])
        uploaded = st.file_uploader("Optional: upload a CSV of flows to explain", type="csv")
        if uploaded is not None:
            flows = pd.read_csv(uploaded)
            index = st.number_input("Row to inspect", min_value=0, max_value=max(0, len(flows) - 1), value=0, step=1)
            row = flows.iloc[[int(index)]]
            st.write(f"**Predicted class:** {model.predict(row)[0]} | **confidence:** {model.predict_proba(row).max():.1%}")
            st.dataframe(explain_prediction(model, row), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
