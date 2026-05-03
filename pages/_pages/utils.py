import streamlit as st
import os
from bs4 import BeautifulSoup

def set_css(css_path):
    """
    Set the CSS file to use.
    """
    with open(css_path, "r") as f:
        css_file = f.read()

    st.markdown(f"<style>{css_file}</style>", unsafe_allow_html=True)

    # --- Force readable blue text for captions, alerts, and notes ---
    st.markdown(
        """
        <style>
        :root{
            --accent-blue: #0b4f7a;
            --accent-blue-2: #0a5f92;
        }

        /* Image captions (e.g., "Original MRI - Sample 1") */
        [data-testid="stImage"] p,
        [data-testid="stImageCaption"] {
            color: var(--accent-blue) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        /* Alerts/warnings/info (e.g., "Please upload an MRI image first.") */
        [data-testid="stAlert"] * {
            color: var(--accent-blue) !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }

        /* Notes and markdown text inside custom boxes */
        .stMarkdown, .stMarkdown * {
            color: #0f2f46;
        }

        /* Make strong text and "Note:" lines blue */
        .stMarkdown strong {
            color: var(--accent-blue) !important;
        }

        /* Links */
        .stMarkdown a {
            color: var(--accent-blue-2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def set_js(content, id='custom-js'):
    html_path = os.path.join(os.path.dirname(os.path.abspath(st.__file__)), 'static', 'index.html')
    soup = BeautifulSoup(open(html_path, 'r'), features="lxml")
    if not soup.find(id=id):
        script_tag = soup.new_tag("script", id=id)
        script_tag.string = content
        soup.head.append(script_tag)
        with open(html_path, 'w') as f:
            f.write(str(soup))