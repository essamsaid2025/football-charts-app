import os
import tempfile

import streamlit as st
import pandas as pd

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    THEMES,
)

st.set_page_config(page_title="Football Charts Generator", layout="wide")

st.title("⚽ Football Charts Generator")
st.caption("Upload CSV/Excel. Pass Map filters + Progressive FIX + Debug counts.")

uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])

with st.expander("🎛️ Settings", expanded=True):
    title_text = st.text_input("Title", value="Match Report")
    subtitle_text = st.text_input("Subtitle", value="")

    theme_name = st.selectbox("Theme", list(THEMES.keys()), index
