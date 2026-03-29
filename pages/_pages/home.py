import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
from .utils import set_css
from predictor import get_model


@st.cache_resource
def load_model():
    model, metrics = get_model(0)
    return model, metrics


def _safe_percent(v):
    try:
        return f"{float(v) * 100:.2f}%"
    except Exception:
        return "N/A"


def _safe_float(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except Exception:
        return "N/A"


def _compute_live_threshold_metrics(prob, threshold):
    pred = 1 if prob >= threshold else 0
    return {
        "pred_class": pred,
        "pred_label": "Tumor" if pred == 1 else "No Tumor",
        "confidence": prob if pred == 1 else (1 - prob),
    }


def _render_metric_card(title_txt, value_txt, caption_txt=""):
    st.markdown(
        f"""
        <div class='metric-card'>
          <div class='metric-title'>{title_txt}</div>
          <div class='metric-value'>{value_txt}</div>
          <div class='metric-caption'>{caption_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    set_css("pages/css/streamlit.css")

    st.markdown(
        """
        <div class='hero-banner'>
          <div class='hero-title'>Brain Tumor Detection System</div>
          <div class='hero-sub'>Doctor-friendly AI support for MRI scan triage and interpretation</div>
          <div class='hero-badge'>MRI Clinical Support • CNN Based</div>

          <div style="margin-top:14px;color:#d9ecfa;font-size:0.95rem;max-width:760px;line-height:1.6;">
            Brain tumors are abnormal growths of cells in or around the brain. Early screening on MRI scans
            helps clinicians identify high-risk cases faster. This tool assists with confidence-based predictions,
            preprocessing visuals, and transparent evaluation metrics to support decision-making.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-blue-box">
          <div class="info-blue-title">Model Guidance</div>
          <div class="info-blue-text">
            These are pre-cropped MRI scan samples used to validate the model.
            Use the decision threshold to explore sensitivity/specificity trade-offs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Decision threshold (lower = higher sensitivity, higher = higher specificity)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.01,
    )

    samples = sorted(os.listdir("pages/samples"))
    option = st.selectbox("Select an MRI sample for analysis", range(1, len(samples) + 1))

    for i in range(0, len(samples), 3):
        cols = st.columns(3)
        for idx, col in enumerate(cols):
            if i + idx < len(samples):
                col.image(Image.open(f"pages/samples/{samples[i + idx]}"))
                col.caption(f"Sample {i + idx + 1}")

    if st.button("Analyze Selected Sample"):
        with st.spinner(text="Analyzing..."):
            model, metrics = load_model()
            image_path = f"pages/samples/{samples[option - 1]}"
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[-1]
            img = np.array([cv2.resize(image, (50, 50))])
            prediction = model.predict(img, verbose=0)

            prob = float(prediction[0][0])
            live = _compute_live_threshold_metrics(prob, threshold)

            st.markdown("#### MRI Preprocessing Outputs")
            c1, c2 = st.columns(2)
            with c1:
                st.image(thresh, caption=f"Mask Threshold - Sample {option}")
            with c2:
                st.image(image, caption=f"Original MRI - Sample {option}")

            st.markdown("#### Live Prediction (using selected threshold)")
            st.success(
                f"Prediction: {live['pred_label']} | Threshold: {threshold:.2f} | Confidence: {live['confidence'] * 100:.2f}%"
            )

            st.markdown("### Model Evaluation Dashboard")

            st.markdown("#### Threshold-free metrics")
            st.caption("Independent of decision threshold; reflects ranking/calibration quality.")
            t1, t2, t3 = st.columns(3)
            with t1:
                _render_metric_card("AUC", _safe_percent(metrics.get("auc")), "Discrimination across all thresholds")
            with t2:
                _render_metric_card("Loss", _safe_float(metrics.get("loss"), 4), "Validation objective")
            with t3:
                _render_metric_card("Brier Score", _safe_float(metrics.get("brier"), 4), "Calibration error (lower is better)")

            st.markdown("#### Thresholded @0.50 metrics")
            st.caption("Computed at threshold 0.50 on validation data.")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                _render_metric_card("Accuracy", _safe_percent(metrics.get("accuracy")), "Overall correctness")
            with a2:
                _render_metric_card("Precision", _safe_percent(metrics.get("precision")), "Positive predictive value")
            with a3:
                _render_metric_card("Recall (Sensitivity)", _safe_percent(metrics.get("recall")), "True positive rate")
            with a4:
                _render_metric_card("Specificity", _safe_percent(metrics.get("specificity")), "True negative rate")

            b1, b2 = st.columns(2)
            with b1:
                _render_metric_card("F1 Score", _safe_float(metrics.get("f1"), 4), "Balance of precision and recall")
            with b2:
                _render_metric_card("MCC", _safe_float(metrics.get("mcc"), 4), "Robust under class imbalance")

            st.markdown("#### Confusion Matrix Counts (@0.50)")
            st.markdown(
                f"""
                <div class='metric-card'>
                  <div class='conf-matrix'>
                    <div class='conf-cell'>TP: {metrics.get('tp', 'N/A')}</div>
                    <div class='conf-cell'>FP: {metrics.get('fp', 'N/A')}</div>
                    <div class='conf-cell'>TN: {metrics.get('tn', 'N/A')}</div>
                    <div class='conf-cell'>FN: {metrics.get('fn', 'N/A')}</div>
                  </div>
                  <div class='metric-caption'>Validation samples: {metrics.get('n_eval', 'N/A')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )