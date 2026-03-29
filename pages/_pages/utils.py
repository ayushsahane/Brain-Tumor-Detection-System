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

def set_js(content, id='custom-js'):
    html_path = os.path.join(os.path.dirname(os.path.abspath(st.__file__)), 'static', 'index.html')
    soup = BeautifulSoup(open(html_path, 'r'), features="lxml")
    if not soup.find(id=id):
        script_tag = soup.new_tag("script", id=id)
        script_tag.string = content
        soup.head.append(script_tag)
        with open(html_path, 'w') as f:
            f.write(str(soup))