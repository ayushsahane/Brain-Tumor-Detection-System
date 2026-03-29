import streamlit as st
import streamlit.components.v1 as html_components
from .utils import set_css
from .components import title


def main():
    set_css("pages/css/streamlit.css")
    

    st.markdown(
        """
        <div class='hero-banner'>
          <div class='hero-title'>Machine Learning Project<br/>Brain Tumor Detection</div>
          <div class='hero-sub'>Doctor-friendly AI support for MRI scan triage and interpretation with clear confidence and evaluation insights.</div>
          <div class='hero-badge'>MRI Clinical Support • CNN Based</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    set_css("pages/css/streamlit.css")
    

    st.markdown(
        "<div class='hero-banner'><div class='hero-title'>Brain Tumor Detection System</div><div class='hero-sub'>Doctor-friendly AI support for MRI scan triage and interpretation</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("## About the Project")
    st.write(
        "This platform provides AI-assisted binary classification of brain MRI scans "
        "(tumor vs. no tumor) using a convolutional neural network (CNN). "
        "It is designed as a clinical-support prototype for educational and research use."
    )

    st.markdown("## Clinical Motivation")
    st.write(
        "- Support quicker preliminary triage during MRI review.\n"
        "- Improve consistency in identifying suspicious scans.\n"
        "- Provide transparent model behavior through threshold-aware predictions and evaluation metrics."
    )

    st.markdown("## What makes this interface doctor/patient friendly?")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class='metric-card'>
              <div class='metric-title'>For Doctors</div>
              <div class='metric-caption'>
                • Threshold slider to tune sensitivity/specificity<br>
                • Structured metric dashboard<br>
                • Confusion matrix counts for auditability
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class='metric-card'>
              <div class='metric-title'>For Patients / Non-technical users</div>
              <div class='metric-caption'>
                • Clean, readable UI with clear labels<br>
                • Confidence-based outputs<br>
                • MRI visuals (original + preprocessing)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Evaluation Framework")
    st.write(
        "The system reports two groups of metrics:\n\n"
        "1. **Threshold-free metrics**: AUC, Loss, Brier Score.\n"
        "2. **Thresholded @0.50 metrics**: Accuracy, Precision, Recall (Sensitivity), Specificity, F1, MCC, TP/FP/TN/FN.\n\n"
        "This separation helps distinguish model ranking quality from decision-threshold behavior."
    )

    st.info(
        "Note: This tool is a decision-support aid and not a standalone diagnostic system."
    )