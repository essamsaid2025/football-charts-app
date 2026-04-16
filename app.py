import os
import tempfile
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
    defensive_regains_map,
    shot_spot_and_direction_map,
    THEMES,
)

st.set_page_config(
    page_title="Football Charts Generator",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# STYLES
# =========================================================
st.markdown(
    """
    <style>
        :root {
            --bg: #0b1220;
            --card: #111827;
            --card-2: #0f172a;
            --border: #243041;
            --text: #f3f4f6;
            --muted: #9ca3af;
            --accent: #38bdf8;
        }

        .stApp {
            background: linear-gradient(180deg, #09111f 0%, #0b1220 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.5rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: var(--text);
        }

        .app-header {
            background: linear-gradient(135deg, rgba(56,189,248,0.16), rgba(16,185,129,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            padding: 20px 22px;
            border-radius: 20px;
            margin-bottom: 18px;
        }

        .app-title {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }

        .app-subtitle {
            color: var(--muted);
            margin-top: 8px;
            font-size: 0.98rem;
        }

        .panel-card {
            background: rgba(17,24,39,0.92);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px 16px 10px 16px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
            margin-bottom: 14px;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: #ffffff;
        }

        .panel-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: -3px;
            margin-bottom: 10px;
        }

        .preview-shell {
            background: rgba(17,24,39,0.92);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 16px;
            min-height: 70vh;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        }

        .preview-placeholder {
            border: 1px dashed #334155;
            background: rgba(15,23,42,0.65);
            border-radius: 16px;
            padding: 28px 20px;
            text-align: center;
            color: #94a3b8;
            margin-top: 10px;
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.12), rgba(255,255,255,0.03));
            margin: 14px 0 14px 0;
            border-radius: 999px;
        }

        div[data-testid="stFileUploader"] section {
            background: rgba(15,23,42,0.85);
            border: 1px dashed #334155;
            border-radius: 14px;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {
            background: #0f172a;
        }

        .stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #2d3b50;
            background: linear-gradient(135deg, #0ea5e9, #2563eb);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #2d3b50;
            background: #162235;
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
        }

        .small-kpi {
            background: rgba(15,23,42,0.85);
            border: 1px solid #243041;
            border-radius: 14px;
            padding: 12px;
            text-align: center;
        }

        .small-kpi .label {
            color: #9ca3af;
            font-size: 0.86rem;
            margin-bottom: 6px;
        }

        .small-kpi .value {
            color: #ffffff;
            font-size: 1.1rem;
            font-weight: 800;
        }

        .stExpander {
            border-radius: 14px !important;
            border: 1px solid #243041 !important;
            background: rgba(15,23,42,0.55) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">⚽ Football Charts Generator</div>
        <div class="app-subtitle">
            Upload CSV / Excel → Match Report / Pizza / Shot Card / Defensive Map / Shot Direction Map
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


def ensure_outcome_column(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    if "outcome" in df.columns:
        return df

    cols_lower = {c.lower().strip(): c for c in df.columns}
    candidates = [
        "event", "event_type", "type",
        "result", "shot_result", "outcome_type",
        "shot_outcome", "final_outcome",
    ]
    for c in candidates:
        if c in cols_lower:
            df["outcome"] = df[cols_lower[c]]
            return df

    df["outcome"] = "unknown"
    return df


def normalize_outcome_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "outcome" not in out.columns:
        return out

    s = out["outcome"].astype(str).str.strip().str.lower()

    mapping = {
        "on target": "ontarget",
        "ontarget": "ontarget",
        "1 on target": "ontarget",
        "shot on target": "ontarget",
        "sot": "ontarget",
        "saved": "ontarget",

        "off target": "off target",
        "offtarget": "off target",
        "shot off target": "off target",
        "miss": "off target",
        "wide": "off target",

        "goal": "goal",
        "scored": "goal",

        "blocked": "blocked",
        "block": "blocked",

        "successful": "successful",
        "success": "successful",
        "complete": "successful",
        "completed": "successful",
        "successfull": "successful",
        "accurate": "successful",
        "true": "successful",
        "yes": "successful",
        "1": "successful",

        "unsuccessful": "unsuccessful",
        "unsuccess": "unsuccessful",
        "incomplete": "unsuccessful",
        "failed": "unsuccessful",
        "unsuccessfull": "unsuccessful",
        "inaccurate": "unsuccessful",
        "false": "unsuccessful",
        "no": "unsuccessful",
        "0": "unsuccessful",

        "key pass": "key pass",
        "keypass": "key pass",
        "kp": "key pass",

        "assist": "assist",
        "a": "assist",

        "touch": "touch",
        "ball touch": "touch",
        "receive": "touch",
        "reception": "touch",
        "received": "touch",
    }

    out["outcome"] = s.map(lambda v: mapping.get(v, v))
    return out


def _percentile_rank(series: pd.Series, value: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s < value).mean() * 100.0)


def build_pizza_df(players_df: pd.DataFrame, player_col: str, player_name: str, metrics: list[str]) -> pd.DataFrame:
    row = players_df.loc[players_df[player_col] == player_name]
    if row.empty:
        raise ValueError("Player not found in file.")

    out = []
    for m in metrics:
        val = pd.to_numeric(row.iloc[0][m], errors="coerce")
        pct = _percentile_rank(players_df[m], val)
        out.append({
            "metric": m,
            "value": "" if pd.isna(val) else round(float(val), 2),
            "percentile": 0 if pd.isna(pct) else round(float(pct), 1),
        })
    return pd.DataFrame(out)


def _lower_cols(df: pd.DataFrame) -> set[str]:
    return set([c.strip().lower() for c in df.columns])


def _missing_for_chart(df_cols_lower: set[str], required_cols: list[str]) -> list[str]:
    miss = []
    for c in required_cols:
        c0 = c.strip().lower()
        if c0.startswith("(optional"):
            continue
        if c0 not in df_cols_lower:
            miss.append(c)
    return miss


CHART_REQUIREMENTS = {
    "Outcome Bar": ["outcome"],
    "Start Heatmap": ["x", "y"],
    "Touch Map (Scatter)": ["x", "y", "(optional) outcome='touch'"],
    "Pass Map": ["outcome", "x", "y", "x2", "y2"],
    "Shot Map": ["outcome", "x", "y", "(optional) x2,y2", "(optional) xg"],
    "Defensive Actions Map": ["x", "y", "outcome"],
    "Shot Spot + Direction Map": ["x", "y", "outcome", "x2", "y2"],
}


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.25)
    buf.seek(0)
    return buf.getvalue()


def read_file_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# =========================================================
# LAYOUT
# =========================================================
left_col, right_col = st.columns([1.05, 1.55], gap="large")

with left_col:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Output & File</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Choose output type",
        [
            "Match Charts",
            "Pizza Chart",
            "Shot Detail Card",
            "Defensive Actions Map",
            "Shot Spot + Direction Map",
        ],
        horizontal=False,
    )

    uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🎛️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-note">All controls stay on the left. Preview and downloads stay on the right.</div>', unsafe_allow_html=True)

    st.markdown("### 🧩 Report Header")
    report_title = st.text_input("Title", value="Match Report")
    report_subtitle = st.text_input("Subtitle", value="")

    header_img = st.file_uploader(
        "Upload header image (Club logo / Player face) - PNG/JPG",
        type=["png", "jpg", "jpeg"],
        key="header_img_uploader",
    )
    header_img_side = st.selectbox("Image position", ["Left", "Right"], index=0)
    header_img_size = st.slider("Image size (as % of figure width)", 5, 18, 10)
    header_img_width_frac = header_img_size / 100.0

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Title / Subtitle Style")
    title_align = st.selectbox("Title align", ["Center", "Left", "Right"], index=0)
    subtitle_align = st.selectbox("Subtitle align", ["Center", "Left", "Right"], index=0)
    title_fontsize = st.slider("Title font size", 12, 28, 16)
    subtitle_fontsize = st.slider("Subtitle font size", 9, 20, 11)
    title_color = st.color_picker("Title color", "#FFFFFF")
    subtitle_color = st.color_picker("Subtitle color", "#A0A7B4")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Theme")
    theme_name = st.selectbox(
        "Choose theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index("The Athletic Dark") if "The Athletic Dark" in THEMES else 0,
    )

    attack_dir_ui = st.selectbox("Attack direction", ["Left → Right", "Right → Left"])
    attack_dir = "ltr" if attack_dir_ui == "Left → Right" else "rtl"

    pitch_orientation_ui = st.selectbox("Pitch orientation", ["Horizontal", "Vertical"], index=0)
    pitch_orientation = "horizontal" if pitch_orientation_ui == "Horizontal" else "vertical"

    flip_y = st.checkbox("Flip Y axis (use this if your Y=0 is at the bottom)", value=False)

    pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular (recommended)", "Square (0-100)"])
    pitch_mode = "rect" if pitch_mode_ui.startswith("Rectangular") else "square"

    pitch_width = st.slider("Rect pitch width", min_value=50.0, max_value=80.0, value=64.0, step=1.0)

    st.markdown("### xG Settings")
    model_file = st.text_input("Model file path", value="xg_pipeline.joblib")
    model_exists = os.path.exists(model_file)

    xg_method_ui = st.radio("xG method", ["Zone", "Model"], index=1 if model_exists else 0)
    xg_method = "model" if xg_method_ui == "Model" else "zone"
    st.caption(f"Model exists: **{model_exists}**")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Report Charts to Include")
    legend_options = ["Passes", "Shots", "Touches", "Defensive Actions"]

    legend_items = st.multiselect(
        "Select legend items to display",
        legend_options,
        default=legend_options,
    )

    all_charts = [
        "Outcome Bar",
        "Start Heatmap",
        "Touch Map (Scatter)",
        "Pass Map",
        "Shot Map",
        "Defensive Actions Map",
        "Shot Spot + Direction Map",
    ]
    selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts[:-1])

    if selected_charts:
        with st.expander("📌 Required columns for selected charts (pre-check)", expanded=False):
            for ch in selected_charts:
                st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS.get(ch, [])))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Pass Map Filters")
    pass_view = st.selectbox(
        "Pass map view",
        [
            "All passes",
            "Into Final Third",
            "Into Penalty Box",
            "Line-breaking",
            "Progressive passes",
        ],
        index=0,
    )

    pass_scope = st.selectbox(
        "Pass result scope",
        ["Attempts (all)", "Successful only", "Unsuccessful only"],
        index=0,
    )

    pass_min_packing = st.slider("Min packing (only used if no line_breaking column)", 1, 3, 1)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Pass colors")
    col_pass_success = st.color_picker("Successful", "#00FF6A")
    col_pass_unsuccess = st.color_picker("Unsuccessful", "#FF4D4D")
    col_pass_key = st.color_picker("Key pass", "#00C2FF")
    col_pass_assist = st.color_picker("Assist", "#FFD400")

    st.markdown("### Shot colors")
    col_shot_off = st.color_picker("Off target", "#7A7A7A")
    col_shot_on = st.color_picker("On target", "#111111")
    col_shot_goal = st.color_picker("Goal", "#FF0000")
    col_shot_blocked = st.color_picker("Blocked", "#AAAAAA")

    st.markdown("### Defensive action colors")
    col_interception = st.color_picker("Interception", "#00C2FF")
    col_tackle = st.color_picker("Tackle", "#FF8A00")
    col_recovery = st.color_picker("Recovery", "#00FF6A")
    col_aerial = st.color_picker("Aerial Duel", "#FFD400")
    col_ground = st.color_picker("Ground Duel", "#FF4D4D")
    col_clearance = st.color_picker("Clearance", "#A78BFA")

    st.markdown("### Outcome bar colors")
    bar_success = st.color_picker("Bar: successful", "#00FF6A")
    bar_unsuccess = st.color_picker("Bar: unsuccessful", "#FF4D4D")
    bar_key = st.color_picker("Bar: key pass", "#00C2FF")
    bar_assist = st.color_picker("Bar: assist", "#FFD400")
    bar_ont = st.color_picker("Bar: ontarget", "#00C2FF")
    bar_off = st.color_picker("Bar: off target", "#FF8A00")
    bar_goal = st.color_picker("Bar: goal", "#00FF6A")
    bar_blocked = st.color_picker("Bar: blocked", "#AAAAAA")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### Event shapes (markers)")
    marker_options = {
        "None": None,
        "Circle (o)": "o",
        "Star (*)": "*",
        "Triangle up (^)": "^",
        "Triangle down (v)": "v",
        "Square (s)": "s",
        "Diamond (D)": "D",
        "Plus (+)": "+",
        "X (x)": "x",
        "Pentagon (p)": "p",
        "Hexagon (h)": "h",
    }
    marker_labels = list(marker_options.keys())

    st.markdown("#### Shot markers")
    mk_shot_off = st.selectbox("Marker: Off target", marker_labels, index=marker_labels.index("X (x)"))
    mk_shot_on = st.selectbox("Marker: On target", marker_labels, index=marker_labels.index("Circle (o)"))
    mk_shot_goal = st.selectbox("Marker: Goal", marker_labels, index=marker_labels.index("Circle (o)"))
    mk_shot_blocked = st.selectbox("Marker: Blocked", marker_labels, index=marker_labels.index("Square (s)"))

    st.markdown("#### Pass start markers")
    mk_pass_success = st.selectbox("Marker: Successful pass", marker_labels, index=marker_labels.index("Circle (o)"))
    mk_pass_unsuccess = st.selectbox("Marker: Unsuccessful pass", marker_labels, index=marker_labels.index("X (x)"))
    mk_pass_key = st.selectbox("Marker: Key pass", marker_labels, index=marker_labels.index("Diamond (D)"))
    mk_pass_assist = st.selectbox("Marker: Assist", marker_labels, index=marker_labels.index("Star (*)"))

    st.markdown("#### Defensive action markers")
    mk_interception = st.selectbox("Marker: Interception", marker_labels, index=marker_labels.index("Circle (o)"))
    mk_tackle = st.selectbox("Marker: Tackle", marker_labels, index=marker_labels.index("Square (s)"))
    mk_recovery = st.selectbox("Marker: Recovery", marker_labels, index=marker_labels.index("Diamond (D)"))
    mk_aerial = st.selectbox("Marker: Aerial Duel", marker_labels, index=marker_labels.index("Triangle up (^)"))
    mk_ground = st.selectbox("Marker: Ground Duel", marker_labels, index=marker_labels.index("X (x)"))
    mk_clearance = st.selectbox("Marker: Clearance", marker_labels, index=marker_labels.index("Star (*)"))

    st.markdown("### Defensive Map settings")
    def_map_title = st.text_input("Defensive map title", value="Ball Regains Map")
    def_show_zone_values = st.checkbox("Show zone values", value=False)
    def_marker_size = st.slider("Defensive marker size", 60, 260, 110)
    def_zone_alpha = st.slider("Zone alpha", 20, 100, 78) / 100.0

    st.markdown("#### Touch map marker")
    touch_marker_label = st.selectbox("Marker: Touch", marker_labels, index=marker_labels.index("Circle (o)"))
    touch_dot_color = st.color_picker("Touch dots color", "#34D5FF")
    touch_dot_edge = st.color_picker("Touch edge color", "#0B0F14")
    touch_dot_size = st.slider("Touch dot size", 60, 520, 220)
    touch_alpha = st.slider("Touch alpha", 20, 100, 95) / 100.0

    st.markdown("### Shot Spot + Direction settings")
    shot_combo_title = st.text_input("Shot combo map title", value="Shot Spot + Direction Map")
    shot_combo_plot_all = st.checkbox("Plot all shots", value=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🍕 Pizza center image")
    pizza_center_img = st.file_uploader(
        "Upload pizza center image (Club logo / Player face) - PNG/JPG",
        type=["png", "jpg", "jpeg"],
        key="pizza_center_uploader",
    )
    pizza_center_scale = st.slider("Center image size (Pizza)", 12, 32, 18) / 100.0
    st.caption("Tip: لو الصورة كبيرة خلّيها 0.16–0.20")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# SETTINGS DICTS
# =========================================================
pass_colors = {
    "successful": col_pass_success,
    "unsuccessful": col_pass_unsuccess,
    "key pass": col_pass_key,
    "assist": col_pass_assist,
}
shot_colors = {
    "off target": col_shot_off,
    "ontarget": col_shot_on,
    "goal": col_shot_goal,
    "blocked": col_shot_blocked,
}
bar_colors = {
    "successful": bar_success,
    "unsuccessful": bar_unsuccess,
    "key pass": bar_key,
    "assist": bar_assist,
    "ontarget": bar_ont,
    "off target": bar_off,
    "goal": bar_goal,
    "blocked": bar_blocked,
}
def_colors = {
    "interception": col_interception,
    "tackle": col_tackle,
    "recovery": col_recovery,
    "aerial_duel": col_aerial,
    "ground_duel": col_ground,
    "clearance": col_clearance,
}

shot_markers = {
    "off target": marker_options[mk_shot_off],
    "ontarget": marker_options[mk_shot_on],
    "goal": marker_options[mk_shot_goal],
    "blocked": marker_options[mk_shot_blocked],
}
pass_markers = {
    "successful": marker_options[mk_pass_success],
    "unsuccessful": marker_options[mk_pass_unsuccess],
    "key pass": marker_options[mk_pass_key],
    "assist": marker_options[mk_pass_assist],
}
def_markers = {
    "interception": marker_options[mk_interception],
    "tackle": marker_options[mk_tackle],
    "recovery": marker_options[mk_recovery],
    "aerial_duel": marker_options[mk_aerial],
    "ground_duel": marker_options[mk_ground],
    "clearance": marker_options[mk_clearance],
}
touch_marker = marker_options[touch_marker_label]

header_img_obj = None
if header_img is not None:
    try:
        header_img_obj = Image.open(header_img).convert("RGBA")
    except Exception:
        header_img_obj = None

pizza_img_obj = None
if pizza_center_img is not None:
    try:
        pizza_img_obj = Image.open(pizza_center_img).convert("RGBA")
    except Exception:
        pizza_img_obj = None

# =========================================================
# SESSION STATE FOR PREVIEW
# =========================================================
if "preview_images" not in st.session_state:
    st.session_state.preview_images = []

if "download_files" not in st.session_state:
    st.session_state.download_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "data_info" not in st.session_state:
    st.session_state.data_info = {}

if "last_mode" not in st.session_state:
    st.session_state.last_mode = mode

if st.session_state.last_mode != mode:
    st.session_state.preview_images = []
    st.session_state.download_files = []
    st.session_state.messages = []
    st.session_state.data_info = {}
    st.session_state.last_mode = mode

# =========================================================
# MAIN PROCESSING
# =========================================================
with right_col:
    st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
    st.markdown("### 📊 Preview & Downloads")

    if uploaded is None:
        st.markdown(
            """
            <div class="preview-placeholder">
                <div style="font-size:1.2rem;font-weight:800;margin-bottom:8px;">No file uploaded yet</div>
                <div>Upload a CSV / Excel file from the left panel, choose your mode, then generate the output.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        # =====================================================
        # PIZZA
        # =====================================================
        if mode == "Pizza Chart":
            dfp = load_data(path)
            st.session_state.data_info = {
                "rows": len(dfp),
                "cols": len(dfp.columns),
                "mode": mode,
            }

            k1, k2, k3 = st.columns(3)
            with k1:
                st.markdown(f'<div class="small-kpi"><div class="label">Mode</div><div class="value">{mode}</div></div>', unsafe_allow_html=True)
            with k2:
                st.markdown(f'<div class="small-kpi"><div class="label">Rows</div><div class="value">{len(dfp)}</div></div>', unsafe_allow_html=True)
            with k3:
                st.markdown(f'<div class="small-kpi"><div class="label">Columns</div><div class="value">{len(dfp.columns)}</div></div>', unsafe_allow_html=True)

            with st.expander("Preview data (first 25 rows)", expanded=False):
                st.write("Columns:", list(dfp.columns))
                st.dataframe(dfp.head(25), use_container_width=True)

            cols_lower = {c.lower(): c for c in dfp.columns}
            player_col = cols_lower.get("player", None) or st.selectbox("Select player column", dfp.columns.tolist())
            players = sorted(dfp[player_col].dropna().astype(str).unique().tolist())
            selected_player = st.selectbox("Choose player", players)

            minutes_col = cols_lower.get("minutes", None)
            dfp_filtered = dfp.copy()
            if minutes_col is not None:
                min_minutes = st.number_input("Min minutes filter (optional)", min_value=0, value=0, step=50)
                dfp_filtered[minutes_col] = pd.to_numeric(dfp_filtered[minutes_col], errors="coerce")
                if min_minutes > 0:
                    dfp_filtered = dfp_filtered[dfp_filtered[minutes_col] >= min_minutes].copy()

            exclude = {player_col}
            if minutes_col is not None:
                exclude.add(minutes_col)
            for maybe in ["team", "position", "pos", "league", "season", "age"]:
                if maybe in cols_lower:
                    exclude.add(cols_lower[maybe])

            metric_cols = [c for c in dfp_filtered.columns if c not in exclude]
            default_n = min(8, len(metric_cols))
            selected_metrics = st.multiselect("Choose metrics", metric_cols, default=metric_cols[:default_n])

            pizza_title = st.text_input("Pizza title", value=selected_player)
            pizza_subtitle = st.text_input("Pizza subtitle", value="Percentile vs peers (per90)")

            def pct_color(p):
                try:
                    p = float(p)
                except Exception:
                    p = 0.0
                if p >= 85:
                    return "#1F77B4"
                elif p >= 70:
                    return "#2ECC71"
                elif p >= 50:
                    return "#F39C12"
                else:
                    return "#E74C3C"

            if st.button("Generate Pizza", key="generate_pizza"):
                pizza_df = build_pizza_df(dfp_filtered, player_col, selected_player, selected_metrics)
                slice_colors = [pct_color(p) for p in pizza_df["percentile"].tolist()]
                center_img = pizza_img_obj if pizza_img_obj is not None else header_img_obj

                fig = pizza_chart(
                    pizza_df,
                    title=pizza_title,
                    subtitle=pizza_subtitle,
                    slice_colors=slice_colors,
                    show_values_legend=False,
                    center_image=center_img,
                    center_img_scale=pizza_center_scale,
                    footer_text="",
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "pizza.png")
                pdf_path = os.path.join(out_dir, "pizza.pdf")

                fig.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.session_state.preview_images = [fig_to_png_bytes(fig)]
                st.session_state.download_files = [
                    ("pizza.pdf", read_file_bytes(pdf_path), "application/pdf"),
                    ("pizza.png", read_file_bytes(png_path), "image/png"),
                ]
                st.session_state.messages = [("success", "Pizza chart generated successfully.")]

            if st.session_state.messages:
                for level, msg in st.session_state.messages:
                    getattr(st, level)(msg)

            if st.session_state.preview_images:
                st.subheader("Preview")
                for img in st.session_state.preview_images:
                    st.image(img, use_container_width=True)

                st.subheader("Downloads")
                for fname, fbytes, mime in st.session_state.download_files:
                    st.download_button(
                        f"⬇️ Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{fname}",
                    )
            else:
                st.markdown(
                    """
                    <div class="preview-placeholder">
                        Choose player + metrics, then click <b>Generate Pizza</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # =====================================================
        # COMMON DATA PREP
        # =====================================================
        df_raw = load_data(path)

        with st.expander("Raw file preview (first 25 rows)", expanded=False):
            st.write("Columns:", list(df_raw.columns))
            st.dataframe(df_raw.head(25), use_container_width=True)

        if "outcome" not in df_raw.columns:
            st.warning("Column `outcome` not found. Trying to derive it automatically.")
        df_raw = ensure_outcome_column(df_raw)
        df_raw = normalize_outcome_values(df_raw)

        cols_lower = _lower_cols(df_raw)
        missing_by_chart = {}
        charts_to_check = selected_charts if mode == "Match Charts" else [mode.replace("Shot Detail Card", "Shot Map").replace("Defensive Actions Map", "Defensive Actions Map")]
        if mode == "Shot Spot + Direction Map":
            charts_to_check = ["Shot Spot + Direction Map"]

        for ch in selected_charts if mode == "Match Charts" else charts_to_check:
            req = CHART_REQUIREMENTS.get(ch, [])
            miss = _missing_for_chart(cols_lower, req)
            if miss:
                missing_by_chart[ch] = miss

        if missing_by_chart and mode == "Match Charts":
            st.error("❌ Missing required columns for selected charts (fix file or unselect chart):")
            for ch, miss in missing_by_chart.items():
                st.write(f"**{ch}** missing → {', '.join(miss)}")
        else:
            if mode != "Match Charts" and missing_by_chart:
                for ch, miss in missing_by_chart.items():
                    st.error(f"❌ {ch} missing → {', '.join(miss)}")
            else:
                st.success("✅ Required columns check passed (or not needed for this mode).")

        can_generate_report = (len(missing_by_chart) == 0)

        model_pipe = None
        if xg_method == "model":
            if os.path.exists(model_file):
                try:
                    model_pipe = joblib.load(model_file)
                    st.success("Model loaded ✅")
                except Exception as e:
                    st.warning(f"Could not load model file. Falling back to Zone. Reason: {e}")
                    model_pipe = None
            else:
                st.warning("Model file not found. Falling back to Zone.")
                model_pipe = None

        df2 = prepare_df_for_charts(
            df_raw,
            attack_direction=attack_dir,
            flip_y=flip_y,
            pitch_mode=pitch_mode,
            pitch_width=pitch_width,
            xg_method=xg_method,
            model_pipe=model_pipe,
        )

        st.session_state.data_info = {
            "rows": len(df2),
            "cols": len(df2.columns),
            "mode": mode,
        }

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="small-kpi"><div class="label">Mode</div><div class="value">{mode}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="small-kpi"><div class="label">Rows</div><div class="value">{len(df2)}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="small-kpi"><div class="label">Columns</div><div class="value">{len(df2.columns)}</div></div>', unsafe_allow_html=True)

        if "xg_source" in df2.columns and len(df2):
            st.info(f"xG source used: **{df2['xg_source'].iloc[0]}**")

        # =====================================================
        # DEFENSIVE MAP
        # =====================================================
        if mode == "Defensive Actions Map":
            if st.button("Generate Defensive Map", key="generate_def_map"):
                fig = defensive_regains_map(
                    df2,
                    title=def_map_title,
                    def_colors=def_colors,
                    def_markers=def_markers,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    theme_name=theme_name,
                    marker_size=def_marker_size,
                    zone_alpha=def_zone_alpha,
                    show_zone_values=def_show_zone_values,
                    orientation=pitch_orientation,
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "defensive_actions_map.png")
                pdf_path = os.path.join(out_dir, "defensive_actions_map.pdf")

                fig.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.session_state.preview_images = [fig_to_png_bytes(fig)]
                st.session_state.download_files = [
                    ("defensive_actions_map.png", read_file_bytes(png_path), "image/png"),
                    ("defensive_actions_map.pdf", read_file_bytes(pdf_path), "application/pdf"),
                ]
                st.session_state.messages = [("success", "Defensive map generated successfully.")]

            if st.session_state.messages:
                for level, msg in st.session_state.messages:
                    getattr(st, level)(msg)

            if st.session_state.preview_images:
                st.subheader("Preview")
                for img in st.session_state.preview_images:
                    st.image(img, use_container_width=True)

                st.subheader("Downloads")
                for fname, fbytes, mime in st.session_state.download_files:
                    st.download_button(
                        f"⬇️ Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{fname}",
                    )
            else:
                st.markdown(
                    """
                    <div class="preview-placeholder">
                        Click <b>Generate Defensive Map</b> to render the chart here.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # =====================================================
        # SHOT DETAIL CARD
        # =====================================================
        if mode == "Shot Detail Card":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            shots_only["label"] = shots_only.apply(
                lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f} | ({_safe_float(r["x"]):.1f},{_safe_float(r["y"]):.1f})',
                axis=1,
            )
            selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
            shot_index = int(selected.split("|")[0].strip()) - 1
            card_title = st.text_input("Card title", value="Shot Detail")

            if st.button("Generate Shot Card", key="generate_shot_card"):
                fig, _ = shot_detail_card(
                    df2,
                    shot_index=int(shot_index),
                    title=card_title,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    theme_name=theme_name,
                    orientation=pitch_orientation,
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "shot_card.png")
                pdf_path = os.path.join(out_dir, "shot_card.pdf")

                fig.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.session_state.preview_images = [fig_to_png_bytes(fig)]
                st.session_state.download_files = [
                    ("shot_card.png", read_file_bytes(png_path), "image/png"),
                    ("shot_card.pdf", read_file_bytes(pdf_path), "application/pdf"),
                ]
                st.session_state.messages = [("success", "Shot detail card generated successfully.")]

            if st.session_state.messages:
                for level, msg in st.session_state.messages:
                    getattr(st, level)(msg)

            if st.session_state.preview_images:
                st.subheader("Preview")
                for img in st.session_state.preview_images:
                    st.image(img, use_container_width=True)

                st.subheader("Downloads")
                for fname, fbytes, mime in st.session_state.download_files:
                    st.download_button(
                        f"⬇️ Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{fname}",
                    )
            else:
                st.markdown(
                    """
                    <div class="preview-placeholder">
                        Select a shot, then click <b>Generate Shot Card</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # =====================================================
        # SHOT SPOT + DIRECTION MAP
        # =====================================================
        if mode == "Shot Spot + Direction Map":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.markdown("</div>", unsafe_allow_html=True)
                st.stop()

            selected_idx = None
            if not shot_combo_plot_all:
                shots_only["label"] = shots_only.apply(
                    lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f} | ({_safe_float(r["x"]):.1f},{_safe_float(r["y"]):.1f})',
                    axis=1,
                )
                selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
                selected_idx = int(selected.split("|")[0].strip()) - 1

            if st.button("Generate Shot Spot + Direction Map", key="generate_shot_combo"):
                fig = shot_spot_and_direction_map(
                    df2,
                    title=shot_combo_title,
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    theme_name=theme_name,
                    orientation=pitch_orientation,
                    plot_all=shot_combo_plot_all,
                    shot_index=selected_idx,
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "shot_spot_direction_map.png")
                pdf_path = os.path.join(out_dir, "shot_spot_direction_map.pdf")

                fig.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.session_state.preview_images = [fig_to_png_bytes(fig)]
                st.session_state.download_files = [
                    ("shot_spot_direction_map.png", read_file_bytes(png_path), "image/png"),
                    ("shot_spot_direction_map.pdf", read_file_bytes(pdf_path), "application/pdf"),
                ]
                st.session_state.messages = [("success", "Shot Spot + Direction map generated successfully.")]

            if st.session_state.messages:
                for level, msg in st.session_state.messages:
                    getattr(st, level)(msg)

            if st.session_state.preview_images:
                st.subheader("Preview")
                for img in st.session_state.preview_images:
                    st.image(img, use_container_width=True)

                st.subheader("Downloads")
                for fname, fbytes, mime in st.session_state.download_files:
                    st.download_button(
                        f"⬇️ Download {fname}",
                        data=fbytes,
                        file_name=fname,
                        mime=mime,
                        key=f"dl_{fname}",
                    )
            else:
                st.markdown(
                    """
                    <div class="preview-placeholder">
                        Click <b>Generate Shot Spot + Direction Map</b>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        # =====================================================
        # MATCH REPORT
        # =====================================================
        generate_clicked = st.button("Generate Report", disabled=not can_generate_report, key="generate_report")

        if not can_generate_report:
            st.info("Generate Report is disabled until required columns exist for all selected charts.")

        if generate_clicked:
            out_dir = os.path.join(tmp, "output")

            pdf_path, png_paths = build_report_from_prepared_df(
                df2,
                out_dir=out_dir,
                title=report_title,
                subtitle=report_subtitle,
                header_image=header_img_obj,
                header_img_side=header_img_side.lower(),
                header_img_width_frac=header_img_width_frac,
                title_align=title_align.lower(),
                subtitle_align=subtitle_align.lower(),
                title_fontsize=title_fontsize,
                subtitle_fontsize=subtitle_fontsize,
                title_color=title_color,
                subtitle_color=subtitle_color,
                theme_name=theme_name,
                pitch_mode=pitch_mode,
                pitch_width=pitch_width,
                pitch_orientation=pitch_orientation,
                pass_colors=pass_colors,
                pass_markers=pass_markers,
                shot_colors=shot_colors,
                shot_markers=shot_markers,
                def_colors=def_colors,
                def_markers=def_markers,
                bar_colors=bar_colors,
                charts_to_include=selected_charts,
                touch_dot_color=touch_dot_color,
                touch_dot_edge=touch_dot_edge,
                touch_dot_size=touch_dot_size,
                touch_alpha=touch_alpha,
                touch_marker=touch_marker,
                pass_view=pass_view,
                pass_result_scope=pass_scope,
                pass_min_packing=pass_min_packing,
                legend_items=legend_items,
            )

            st.session_state.preview_images = [read_file_bytes(p) for p in png_paths]
            st.session_state.download_files = [("report.pdf", read_file_bytes(pdf_path), "application/pdf")]
            for p in png_paths:
                name = os.path.basename(p)
                st.session_state.download_files.append((name, read_file_bytes(p), "image/png"))

            st.session_state.messages = [("success", "Match report generated successfully.")]

        if st.session_state.messages:
            for level, msg in st.session_state.messages:
                getattr(st, level)(msg)

        if st.session_state.preview_images:
            st.subheader("Preview")
            for img in st.session_state.preview_images:
                st.image(img, use_container_width=True)

            st.subheader("Downloads")
            for fname, fbytes, mime in st.session_state.download_files:
                st.download_button(
                    f"⬇️ Download {fname}",
                    data=fbytes,
                    file_name=fname,
                    mime=mime,
                    key=f"dl_{fname}",
                )
        else:
            st.markdown(
                """
                <div class="preview-placeholder">
                    Configure the settings from the left panel, then click <b>Generate Report</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
