"""
Football Analysis Suite — Modular Streamlit Application
Each tool is fully independent: its own inputs, controls, and exports.
"""

import os
import io
import tempfile
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    mpl_pizza_dark,
    athletic_pizza,
    shot_detail_card,
    defensive_regains_map,
    outcome_bar,
    start_location_heatmap,
    touch_map,
    pass_map,
    shot_map,
    defensive_actions_map,
    THEMES,
    make_pitch,
)

try:
    from scouting_tools_v2 import (
        ROLE_TEMPLATES,
        load_player_data,
        standard_columns,
        numeric_metrics,
        coerce_numeric,
        match_template_metrics,
        add_percentiles_and_score,
        player_profile,
        profile_text,
        similar_players,
        auto_dataset_insights,
        comparison_chart,
        radar_chart,
        make_template_csv,
        recommendation_text,
    )
except Exception:
    ROLE_TEMPLATES = None
    recommendation_text = None

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


st.set_page_config(
    page_title="Football Analysis Suite",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --bg: #07111f;
    --surface: #0d1f35;
    --surface2: #112240;
    --border: #1e3a5f;
    --text: #e8f0fe;
    --muted: #6b8cae;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --green: #00e676;
    --red: #ff4060;
    --gold: #ffd060;
}

.stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', -apple-system, sans-serif;
}

.block-container {
    padding: 1rem 1.5rem 2rem;
    max-width: 100%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #050e1c !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #0a1929 !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(13,31,53,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 7px 10px !important;
    margin-bottom: 5px !important;
    transition: background 0.2s;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(30,58,95,0.9) !important;
}

/* Headers */
.suite-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 18px 22px;
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,58,237,0.06));
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 16px;
    margin-bottom: 20px;
}
.suite-header .icon { font-size: 2rem; }
.suite-header .title {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
    line-height: 1.1;
}
.suite-header .sub {
    font-size: 0.88rem;
    color: var(--muted);
    margin-top: 3px;
}

/* Tool card */
.tool-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 18px 14px;
    margin-bottom: 14px;
}
.tool-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

/* KPI chips */
.kpi-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 14px; }
.kpi-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 9px 14px;
    text-align: center;
    flex: 1;
    min-width: 90px;
}
.kpi-chip .kv { font-size: 1.15rem; font-weight: 800; color: var(--text); }
.kpi-chip .kl { font-size: 0.75rem; color: var(--muted); margin-top: 2px; }

/* Tab strip override */
div[data-testid="stTabs"] > div:first-child {
    background: var(--surface);
    border-radius: 12px 12px 0 0;
    border-bottom: 2px solid var(--border);
    gap: 4px;
    padding: 6px 8px 0;
}
button[data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    font-weight: 700;
    padding: 0.55rem 1.1rem;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
.stDownloadButton > button {
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--text);
    font-weight: 600;
    width: 100%;
}

/* File uploader */
div[data-testid="stFileUploader"] section {
    background: var(--surface2);
    border: 1.5px dashed var(--border);
    border-radius: 12px;
}

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 16px 0;
}

/* Empty state */
.empty-state {
    border: 1.5px dashed var(--border);
    background: rgba(13,31,53,0.5);
    border-radius: 14px;
    padding: 40px 24px;
    text-align: center;
    color: var(--muted);
}
.empty-state .es-icon { font-size: 2.5rem; margin-bottom: 10px; }
.empty-state .es-title { font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }

/* Expander */
.stExpander {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Score badge */
.score-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EXTRAS theme
# ─────────────────────────────────────────────────────────────────────────────
THEMES["Opta Analyst Light"] = {
    "bg": "#ECECEC", "panel": "#F5F5F5", "pitch": "#ECECEC", "pitch_stripe": None,
    "text": "#201C2B", "muted": "#7A7584", "lines": "#A7A7A7", "goal": "#8F8F8F",
    "pitch_lines": "#9F9F9F",
}

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MARKER_OPTIONS = {
    "None": None, "Circle (o)": "o", "Star (*)": "*",
    "Triangle ▲": "^", "Triangle ▼": "v", "Square (s)": "s",
    "Diamond (D)": "D", "Plus (+)": "+", "X (x)": "x",
    "Pentagon (p)": "p", "Hexagon (h)": "h",
}
MARKER_LABELS = list(MARKER_OPTIONS.keys())

def fig_bytes(fig, dpi=280):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.2)
    buf.seek(0)
    return buf.getvalue()

def pdf_bytes(fig):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.2)
    buf.seek(0)
    return buf.read()

def clean_numeric(s):
    return pd.to_numeric(
        s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip(),
        errors="coerce"
    )

def kpi(label, value):
    return f'<div class="kpi-chip"><div class="kv">{value}</div><div class="kl">{label}</div></div>'

def header(icon, title, subtitle=""):
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'''
    <div class="suite-header">
        <div class="icon">{icon}</div>
        <div><div class="title">{title}</div>{sub}</div>
    </div>''', unsafe_allow_html=True)

def card(title, content_fn):
    st.markdown(f'<div class="tool-card"><div class="tool-card-title">{title}</div>', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)

def load_event_file(f):
    name = getattr(f, "name", "file")
    ext = name.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        for enc in ["utf-8", "utf-8-sig", "cp1256", "latin1"]:
            try:
                f.seek(0); return pd.read_csv(f, encoding=enc)
            except Exception:
                pass
    return pd.read_excel(f)

def ensure_outcome(df):
    if "outcome" in df.columns:
        return df
    for c in df.columns:
        if c.lower().strip() in ["event","result","type","event_type"]:
            df = df.copy(); df["outcome"] = df[c]; return df
    df = df.copy(); df["outcome"] = "unknown"; return df

def _pct_rank(series, higher=True):
    x = pd.to_numeric(series, errors="coerce")
    p = x.rank(pct=True, method="average") * 100
    return (100 - p if not higher else p).clip(0, 100)

def download_row(fig, base_name):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(f"⬇ PNG", fig_bytes(fig), f"{base_name}.png", "image/png", key=f"dl_png_{base_name}_{id(fig)}")
    with c2:
        st.download_button(f"⬇ PDF", pdf_bytes(fig), f"{base_name}.pdf", "application/pdf", key=f"dl_pdf_{base_name}_{id(fig)}")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — section navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Football Analysis Suite")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    section = st.radio(
        "Navigate to",
        [
            "🏠 Home",
            "⚔️ Attacking Charts",
            "🛡️ Defensive Charts",
            "🔄 Distribution Charts",
            "🍕 Player Radars & Pizza",
            "🧠 Player Scouting",
            "🖱️ Tagging Tool",
        ],
        key="nav_section",
    )

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if section == "🏠 Home":
    header("⚽", "Football Analysis Suite", "Professional-grade charts, scouting & tagging — fully modular")
    cols = st.columns(3)
    tiles = [
        ("⚔️", "Attacking Charts", "Shot maps, xG, touch maps and start heatmaps"),
        ("🛡️", "Defensive Charts", "Defensive actions, ball regains, zone maps"),
        ("🔄", "Distribution Charts", "Pass maps, progressive passes, direction analysis"),
        ("🍕", "Radars & Pizza", "MPL pizza, Athletic style, scatter plots, percentile bars"),
        ("🧠", "Player Scouting", "Role templates, scoring, shortlists, recommendations"),
        ("🖱️", "Tagging Tool", "Click-to-tag events directly on an interactive pitch"),
    ]
    for i, (ic, t, d) in enumerate(tiles):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="tool-card" style="min-height:110px;">
                <div style="font-size:1.6rem;margin-bottom:8px;">{ic}</div>
                <div style="font-weight:800;font-size:1rem;color:var(--text);">{t}</div>
                <div style="font-size:0.83rem;color:var(--muted);margin-top:4px;">{d}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### File formats accepted")
    st.markdown("- **Event data** (CSV/Excel): needs `outcome`, `x`, `y` columns — used by attacking, defensive, and distribution charts.")
    st.markdown("- **Player metrics** (CSV/Excel): needs a `Player` column plus numeric stat columns — used by radar/pizza and scouting.")
    st.markdown("- **Tagging tool**: no file needed — click directly on the pitch to record events.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED EVENT-DATA LOADER (used by attacking / defensive / distribution)
# ─────────────────────────────────────────────────────────────────────────────
def event_data_sidebar(prefix):
    """Returns (df_prepared | None, settings_dict)."""
    with st.sidebar:
        st.markdown(f"### 📂 Event Data")
        f = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], key=f"{prefix}_file")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🎨 Theme & Pitch")
        theme_name = st.selectbox("Theme", list(THEMES.keys()),
            index=list(THEMES.keys()).index("The Athletic Dark"), key=f"{prefix}_theme")
        attack_dir_ui = st.selectbox("Attack direction", ["Left → Right","Right → Left"], key=f"{prefix}_dir")
        attack_dir = "ltr" if "Left" in attack_dir_ui else "rtl"
        flip_y = st.checkbox("Flip Y axis", value=False, key=f"{prefix}_flip")
        pitch_shape = st.selectbox("Pitch shape", ["Rectangular","Square"], key=f"{prefix}_shape")
        pitch_mode = "rect" if pitch_shape == "Rectangular" else "square"
        pitch_width = st.slider("Pitch width (rect only)", 50.0, 80.0, 68.0, 1.0, key=f"{prefix}_pw") if pitch_mode == "rect" else 100.0
        vertical = st.checkbox("Vertical pitch", value=False, key=f"{prefix}_vert")

    settings = dict(theme_name=theme_name, attack_dir=attack_dir, flip_y=flip_y,
                    pitch_mode=pitch_mode, pitch_width=pitch_width, vertical=vertical)

    if f is None:
        return None, settings

    try:
        raw = load_event_file(f)
        raw = ensure_outcome(raw)
        df = prepare_df_for_charts(raw, attack_direction=attack_dir, flip_y=flip_y,
                                   pitch_mode=pitch_mode, pitch_width=pitch_width,
                                   xg_method="zone")
        return df, settings
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return None, settings


def no_file_state():
    st.markdown("""
    <div class="empty-state">
        <div class="es-icon">📂</div>
        <div class="es-title">No file uploaded</div>
        <div>Upload an event data file from the sidebar to get started.</div>
    </div>""", unsafe_allow_html=True)


def preview_and_download(fig, name):
    st.image(fig_bytes(fig, dpi=200), use_container_width=True)
    download_row(fig, name)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR / MARKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def shot_color_marker_controls(prefix):
    st.markdown("##### Shot colours")
    c1,c2 = st.columns(2)
    with c1:
        col_off = st.color_picker("Off target", "#FF8A00", key=f"{prefix}_coff")
        col_on  = st.color_picker("On target",  "#00C2FF", key=f"{prefix}_con")
    with c2:
        col_goal= st.color_picker("Goal",        "#00FF6A", key=f"{prefix}_cgoal")
        col_blk = st.color_picker("Blocked",     "#AAAAAA", key=f"{prefix}_cblk")
    st.markdown("##### Shot markers")
    c3,c4 = st.columns(2)
    with c3:
        mk_off  = st.selectbox("Off target",   MARKER_LABELS, index=MARKER_LABELS.index("Triangle ▲"), key=f"{prefix}_moff")
        mk_on   = st.selectbox("On target",    MARKER_LABELS, index=MARKER_LABELS.index("Diamond (D)"), key=f"{prefix}_mon")
    with c4:
        mk_goal = st.selectbox("Goal",         MARKER_LABELS, index=MARKER_LABELS.index("Star (*)"), key=f"{prefix}_mgoal")
        mk_blk  = st.selectbox("Blocked",      MARKER_LABELS, index=MARKER_LABELS.index("Square (s)"), key=f"{prefix}_mblk")
    colors  = {"off target": col_off, "ontarget": col_on, "goal": col_goal, "blocked": col_blk}
    markers = {"off target": MARKER_OPTIONS[mk_off], "ontarget": MARKER_OPTIONS[mk_on],
               "goal": MARKER_OPTIONS[mk_goal], "blocked": MARKER_OPTIONS[mk_blk]}
    return colors, markers

def pass_color_marker_controls(prefix):
    st.markdown("##### Pass colours")
    c1,c2 = st.columns(2)
    with c1:
        col_s  = st.color_picker("Successful",   "#00FF6A", key=f"{prefix}_cs")
        col_u  = st.color_picker("Unsuccessful", "#FF4D4D", key=f"{prefix}_cu")
    with c2:
        col_kp = st.color_picker("Key pass",     "#00C2FF", key=f"{prefix}_ckp")
        col_a  = st.color_picker("Assist",       "#FFD400", key=f"{prefix}_ca")
    st.markdown("##### Pass markers")
    c3,c4 = st.columns(2)
    with c3:
        mk_s  = st.selectbox("Successful",   MARKER_LABELS, index=MARKER_LABELS.index("Circle (o)"), key=f"{prefix}_ms")
        mk_u  = st.selectbox("Unsuccessful", MARKER_LABELS, index=MARKER_LABELS.index("X (x)"), key=f"{prefix}_mu")
    with c4:
        mk_kp = st.selectbox("Key pass",     MARKER_LABELS, index=MARKER_LABELS.index("Diamond (D)"), key=f"{prefix}_mkp")
        mk_a  = st.selectbox("Assist",       MARKER_LABELS, index=MARKER_LABELS.index("Star (*)"), key=f"{prefix}_ma")
    colors  = {"successful": col_s, "unsuccessful": col_u, "key pass": col_kp, "assist": col_a}
    markers = {"successful": MARKER_OPTIONS[mk_s], "unsuccessful": MARKER_OPTIONS[mk_u],
               "key pass": MARKER_OPTIONS[mk_kp], "assist": MARKER_OPTIONS[mk_a]}
    return colors, markers

def def_color_marker_controls(prefix):
    cols  = {"interception": "#00C2FF","tackle": "#FF8A00","recovery": "#00FF6A",
             "aerial_duel": "#FFD400","ground_duel": "#FF4D4D","clearance": "#A78BFA"}
    mks   = {"interception": "Circle (o)","tackle": "Square (s)","recovery": "Diamond (D)",
             "aerial_duel": "Triangle ▲","ground_duel": "X (x)","clearance": "Star (*)"}
    out_c, out_m = {}, {}
    acts = list(cols.keys())
    c1, c2 = st.columns(2)
    for i, act in enumerate(acts):
        label = act.replace("_"," ").title()
        col = c1 if i % 2 == 0 else c2
        with col:
            out_c[act] = st.color_picker(label, cols[act], key=f"{prefix}_dc_{act}")
            mk_label = st.selectbox(f"{label} marker", MARKER_LABELS,
                index=MARKER_LABELS.index(mks[act]), key=f"{prefix}_dm_{act}")
            out_m[act] = MARKER_OPTIONS[mk_label]
    return out_c, out_m


# ─────────────────────────────────────────────────────────────────────────────
# ⚔️  ATTACKING CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "⚔️ Attacking Charts":
    df, S = event_data_sidebar("atk")
    header("⚔️", "Attacking Charts", "Shot maps, xG overlays, touch maps, start heatmaps")

    tab_shot, tab_touch, tab_heat, tab_card = st.tabs(
        ["Shot Map", "Touch Map", "Start Heatmap", "Shot Detail Card"]
    )

    # ── Shot Map ─────────────────────────────────────────────────────────────
    with tab_shot:
        st.markdown("#### Shot Map")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                chart_title = st.text_input("Chart title", "Shot Map", key="sm_title")
                show_xg = st.checkbox("Show xG labels", value=True, key="sm_xg")
                shot_colors, shot_markers = shot_color_marker_controls("sm")
                gen = st.button("Generate Shot Map", key="sm_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen:
                    fig = shot_map(df, shot_colors=shot_colors, shot_markers=shot_markers,
                                   pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                   show_xg=show_xg, theme_name=S["theme_name"],
                                   vertical_pitch=S["vertical"])
                    fig.axes[0].set_title(chart_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["atk_shot_fig"] = fig
                if "atk_shot_fig" in st.session_state:
                    preview_and_download(st.session_state["atk_shot_fig"], "shot_map")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">🎯</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    # ── Touch Map ────────────────────────────────────────────────────────────
    with tab_touch:
        st.markdown("#### Touch Map")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                tm_title = st.text_input("Chart title", "Touch Map", key="tm_title")
                tm_color = st.color_picker("Dot color", "#34D5FF", key="tm_col")
                tm_edge  = st.color_picker("Edge color", "#0B0F14", key="tm_edge")
                tm_size  = st.slider("Dot size", 60, 500, 220, key="tm_size")
                tm_alpha = st.slider("Opacity", 20, 100, 90, key="tm_alpha") / 100
                tm_mk_l  = st.selectbox("Marker", MARKER_LABELS, index=1, key="tm_mk")
                tm_mk    = MARKER_OPTIONS[tm_mk_l]
                gen_tm   = st.button("Generate Touch Map", key="tm_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_tm:
                    fig = touch_map(df, pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                    theme_name=S["theme_name"], dot_color=tm_color,
                                    edge_color=tm_edge, dot_size=tm_size, alpha=tm_alpha,
                                    marker=tm_mk, vertical_pitch=S["vertical"])
                    fig.axes[0].set_title(tm_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["atk_touch_fig"] = fig
                if "atk_touch_fig" in st.session_state:
                    preview_and_download(st.session_state["atk_touch_fig"], "touch_map")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">👟</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    # ── Start Heatmap ────────────────────────────────────────────────────────
    with tab_heat:
        st.markdown("#### Start Location Heatmap")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                heat_title = st.text_input("Chart title", "Start Location Heatmap", key="ht_title")
                gen_heat   = st.button("Generate Heatmap", key="ht_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_heat:
                    fig = start_location_heatmap(df, pitch_mode=S["pitch_mode"],
                                                 pitch_width=S["pitch_width"],
                                                 theme_name=S["theme_name"],
                                                 vertical_pitch=S["vertical"])
                    fig.axes[0].set_title(heat_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["atk_heat_fig"] = fig
                if "atk_heat_fig" in st.session_state:
                    preview_and_download(st.session_state["atk_heat_fig"], "start_heatmap")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">🌡️</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    # ── Shot Detail Card ─────────────────────────────────────────────────────
    with tab_card:
        st.markdown("#### Shot Detail Card")
        if df is None:
            no_file_state()
        else:
            shots = df[df["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots.empty:
                st.warning("No shots found in this file.")
            else:
                def _safe_f(v):
                    try: return float(v)
                    except: return float("nan")

                shots["label"] = shots.apply(
                    lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_f(r.get("xg")):.2f}', axis=1)
                lcol, rcol = st.columns([1, 2.2])
                with lcol:
                    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                    sel  = st.selectbox("Select shot", shots["label"].tolist(), key="sc_sel")
                    sc_t = st.text_input("Card title", "Shot Detail", key="sc_title")
                    shot_colors_sc, shot_markers_sc = shot_color_marker_controls("sc")
                    gen_sc = st.button("Generate Shot Card", key="sc_gen")
                    st.markdown('</div>', unsafe_allow_html=True)
                with rcol:
                    if gen_sc:
                        idx = int(sel.split("|")[0].strip()) - 1
                        fig, _ = shot_detail_card(df, shot_index=idx, title=sc_t,
                                                  pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                                  shot_colors=shot_colors_sc, shot_markers=shot_markers_sc,
                                                  theme_name=S["theme_name"])
                        st.session_state["atk_card_fig"] = fig
                    if "atk_card_fig" in st.session_state:
                        preview_and_download(st.session_state["atk_card_fig"], "shot_detail_card")
                    else:
                        st.markdown('<div class="empty-state"><div class="es-icon">🃏</div><div class="es-title">Select a shot and generate</div></div>', unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🛡️  DEFENSIVE CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🛡️ Defensive Charts":
    df, S = event_data_sidebar("def")
    header("🛡️", "Defensive Charts", "Defensive actions map, ball regains zone map, outcome distribution")

    tab_act, tab_reg, tab_bar = st.tabs(["Defensive Actions Map", "Ball Regains Map", "Outcome Distribution"])

    # ── Defensive Actions ────────────────────────────────────────────────────
    with tab_act:
        st.markdown("#### Defensive Actions Map")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                da_title = st.text_input("Chart title", "Defensive Actions", key="da_title")
                def_colors, def_markers = def_color_marker_controls("da")
                gen_da = st.button("Generate Map", key="da_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_da:
                    fig = defensive_actions_map(df, def_colors=def_colors, def_markers=def_markers,
                                                pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                                theme_name=S["theme_name"], vertical_pitch=S["vertical"])
                    fig.axes[0].set_title(da_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["def_act_fig"] = fig
                if "def_act_fig" in st.session_state:
                    preview_and_download(st.session_state["def_act_fig"], "defensive_actions_map")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">🛡️</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    # ── Ball Regains ─────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown("#### Ball Regains Heatmap")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                br_title  = st.text_input("Chart title", "Ball Regains Map", key="br_title")
                br_zvals  = st.checkbox("Show zone counts", value=True, key="br_zv")
                br_msize  = st.slider("Marker size", 60, 260, 110, key="br_ms")
                br_alpha  = st.slider("Zone opacity", 20, 100, 78, key="br_alpha") / 100
                def_colors_br, def_markers_br = def_color_marker_controls("br")
                gen_br = st.button("Generate Map", key="br_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_br:
                    fig = defensive_regains_map(df, title=br_title, def_colors=def_colors_br,
                                                def_markers=def_markers_br,
                                                pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                                theme_name=S["theme_name"], vertical_pitch=S["vertical"],
                                                marker_size=br_msize, zone_alpha=br_alpha,
                                                show_zone_values=br_zvals)
                    st.session_state["def_reg_fig"] = fig
                if "def_reg_fig" in st.session_state:
                    preview_and_download(st.session_state["def_reg_fig"], "ball_regains_map")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">🗺️</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    # ── Outcome Bar ──────────────────────────────────────────────────────────
    with tab_bar:
        st.markdown("#### Outcome Distribution")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                ob_title = st.text_input("Chart title", "Outcome Distribution", key="ob_title")
                st.markdown("##### Bar colours")
                bar_colors = {
                    "successful":   st.color_picker("Successful",   "#00FF6A", key="ob_cs"),
                    "unsuccessful": st.color_picker("Unsuccessful", "#FF4D4D", key="ob_cu"),
                    "key pass":     st.color_picker("Key pass",     "#00C2FF", key="ob_ck"),
                    "assist":       st.color_picker("Assist",       "#FFD400", key="ob_ca"),
                    "goal":         st.color_picker("Goal",         "#00FF6A", key="ob_cg"),
                    "ontarget":     st.color_picker("On target",    "#00C2FF", key="ob_co"),
                    "off target":   st.color_picker("Off target",   "#FF8A00", key="ob_cf"),
                    "blocked":      st.color_picker("Blocked",      "#AAAAAA", key="ob_cb"),
                }
                gen_ob = st.button("Generate Chart", key="ob_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_ob:
                    fig = outcome_bar(df, bar_colors=bar_colors, theme_name=S["theme_name"])
                    fig.axes[0].set_title(ob_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["def_bar_fig"] = fig
                if "def_bar_fig" in st.session_state:
                    preview_and_download(st.session_state["def_bar_fig"], "outcome_distribution")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">📊</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🔄  DISTRIBUTION CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🔄 Distribution Charts":
    df, S = event_data_sidebar("dist")
    header("🔄", "Distribution Charts", "Pass maps, pass filters, progressive actions")

    tab_pm, = st.tabs(["Pass Map"])

    with tab_pm:
        st.markdown("#### Pass Map")
        if df is None:
            no_file_state()
        else:
            lcol, rcol = st.columns([1, 2.2])
            with lcol:
                st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                pm_title  = st.text_input("Chart title", "Pass Map", key="pm_title")
                pm_view   = st.selectbox("Pass view", ["All passes","Into Final Third",
                                                        "Into Penalty Box","Line-breaking",
                                                        "Progressive passes"], key="pm_view")
                pm_scope  = st.selectbox("Result scope", ["Attempts (all)","Successful only",
                                                           "Unsuccessful only"], key="pm_scope")
                pm_pack   = st.slider("Min packing", 1, 5, 1, key="pm_pack")
                pass_colors, pass_markers = pass_color_marker_controls("pm")
                gen_pm    = st.button("Generate Pass Map", key="pm_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with rcol:
                if gen_pm:
                    fig = pass_map(df, pass_colors=pass_colors, pass_markers=pass_markers,
                                   pitch_mode=S["pitch_mode"], pitch_width=S["pitch_width"],
                                   theme_name=S["theme_name"], vertical_pitch=S["vertical"],
                                   pass_view=pm_view, result_scope=pm_scope, min_packing=pm_pack)
                    fig.axes[0].set_title(pm_title, color=THEMES[S["theme_name"]]["text"],
                                          fontsize=16, weight="bold")
                    st.session_state["dist_pm_fig"] = fig
                if "dist_pm_fig" in st.session_state:
                    preview_and_download(st.session_state["dist_pm_fig"], "pass_map")
                else:
                    st.markdown('<div class="empty-state"><div class="es-icon">🎯</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🍕  PLAYER RADARS & PIZZA
# ─────────────────────────────────────────────────────────────────────────────
if section == "🍕 Player Radars & Pizza":
    header("🍕", "Player Radars & Pizza", "Pizza charts, percentile bars, scatter plots — all per-player")

    with st.sidebar:
        st.markdown("### 📂 Player Metrics File")
        pf = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], key="pizza_file")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🖼️ Images")
        center_img_f = st.file_uploader("Center image (logo/face)", type=["png","jpg","jpeg"], key="pizza_center")
        center_scale = st.slider("Center image scale", 10, 30, 16, key="pizza_cs") / 100

    tabs = st.tabs(["MPL Pizza (Dark)", "Athletic Pizza (Light)", "Percentile Bar", "Scatter Plot"])

    def _load_pizza_file(f):
        if f is None: return None
        name = getattr(f,"name","x")
        ext = name.lower().rsplit(".",1)[-1]
        if ext == "csv":
            for enc in ["utf-8","utf-8-sig","cp1256","latin1"]:
                try: f.seek(0); return pd.read_csv(f, encoding=enc)
                except: pass
        return pd.read_excel(f)

    def _default_player_col(df):
        low = {str(c).strip().lower(): c for c in df.columns}
        for k in ["player","player name","name"]: 
            if k in low: return low[k]
        return df.columns[0]

    def _metric_cols(df, excl, min_v=2):
        excl = set(excl)
        return [c for c in df.columns if c not in excl and clean_numeric(df[c]).notna().sum() >= min_v]

    def _build_table(df, player_col, player, metrics):
        d = df.copy()
        row = d[d[player_col].astype(str) == str(player)]
        if row.empty: raise ValueError("Player not found")
        r = row.iloc[0]
        rows = []
        for m in metrics:
            s = clean_numeric(d[m]); v = pd.to_numeric(r[m], errors="coerce")
            pct = float((s < float(v)).mean() * 100) if s.notna().any() and not pd.isna(v) else 0.0
            rows.append({"metric": m, "value": round(float(v),2) if not pd.isna(v) else 0,
                         "percentile": round(pct,1), "plot_value": round(pct,1)})
        return pd.DataFrame(rows)

    center_img_obj = None
    if center_img_f is not None:
        try: center_img_obj = Image.open(center_img_f).convert("RGBA")
        except: pass

    dfp = _load_pizza_file(pf)

    for tab_idx, (tab, chart_key) in enumerate(zip(tabs, ["mpl","ath","pbar","scat"])):
        with tab:
            if dfp is None:
                no_file_state()
                continue

            player_col = _default_player_col(dfp)
            meta_skip  = {player_col}
            mc = _metric_cols(dfp, meta_skip, min_v=1)
            if not mc:
                st.error("No numeric columns found."); continue
            players = sorted(dfp[player_col].dropna().astype(str).unique().tolist())

            if chart_key in ["mpl","ath","pbar"]:
                lcol, rcol = st.columns([1, 2])
                with lcol:
                    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                    sel_player = st.selectbox("Player", players, key=f"{chart_key}_player")
                    chart_title = st.text_input("Title", sel_player, key=f"{chart_key}_title")
                    chart_sub   = st.text_input("Subtitle", "Percentile vs peers", key=f"{chart_key}_sub")
                    sel_metrics = st.multiselect("Metrics", mc, default=mc[:min(12,len(mc))], key=f"{chart_key}_metrics")

                    categories = []
                    if chart_key in ["mpl","ath"] and sel_metrics:
                        st.markdown("**Metric categories**")
                        for m in sel_metrics:
                            categories.append(st.selectbox(m, ["Attacking","Possession","Defending"],
                                                            key=f"{chart_key}_cat_{m}"))

                    if chart_key == "mpl":
                        atk_c = st.color_picker("Attacking color", "#1A78CF", key="mpl_atk")
                        pos_c = st.color_picker("Possession color", "#FF9300", key="mpl_pos")
                        def_c = st.color_picker("Defending color", "#D70232", key="mpl_def")
                        bg_c  = st.color_picker("Background", "#222222", key="mpl_bg")
                    elif chart_key == "ath":
                        atk_c = st.color_picker("Attacking color", "#4B78B9", key="ath_atk")
                        pos_c = st.color_picker("Possession color", "#F0C987", key="ath_pos")
                        def_c = st.color_picker("Defending color", "#9E374B", key="ath_def")
                    elif chart_key == "pbar":
                        good_c = st.color_picker("Good (≥70th)", "#00e676", key="pb_good")
                        mid_c  = st.color_picker("Mid (50-70th)", "#ffd060", key="pb_mid")
                        low_c  = st.color_picker("Low (<50th)", "#ff4060", key="pb_low")

                    gen = st.button("Generate", key=f"{chart_key}_gen")
                    st.markdown('</div>', unsafe_allow_html=True)

                with rcol:
                    fkey = f"pizza_{chart_key}_fig"
                    if gen and sel_metrics:
                        try:
                            table = _build_table(dfp, player_col, sel_player, sel_metrics)
                            if chart_key == "mpl":
                                fig = mpl_pizza_dark(table, title=chart_title, subtitle=chart_sub,
                                                     categories=categories or None,
                                                     attacking_color=atk_c, possession_color=pos_c,
                                                     defending_color=def_c, center_image=center_img_obj,
                                                     center_img_scale=center_scale, bg_color=bg_c)
                            elif chart_key == "ath":
                                fig = athletic_pizza(table, title=chart_title, subtitle=chart_sub,
                                                     categories=categories or None,
                                                     attacking_color=atk_c, possession_color=pos_c,
                                                     defending_color=def_c)
                            else:
                                d = table.copy().sort_values("percentile", ascending=True)
                                vals = d["percentile"].tolist()
                                colors = [good_c if v>=70 else mid_c if v>=50 else low_c for v in vals]
                                fig, ax = plt.subplots(figsize=(10, max(4.5, len(d)*0.45)))
                                fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                                ax.barh(d["metric"], vals, color=colors, alpha=0.92)
                                ax.set_xlim(0,100)
                                for i,v in enumerate(vals):
                                    ax.text(min(97,float(v)+1.2), i, f"{v:.0f}", va="center",
                                            color="#e8f0fe", fontsize=9, weight="bold")
                                ax.axvline(50, color="#1e3a5f", lw=1.5, ls="--")
                                ax.axvline(70, color="#1e3a5f", lw=1.5, ls="--")
                                ax.set_title(chart_title, color="#e8f0fe", fontsize=16, weight="bold")
                                ax.set_xlabel("Percentile", color="#6b8cae")
                                ax.tick_params(colors="#6b8cae")
                                for sp in ax.spines.values(): sp.set_color("#1e3a5f")
                            st.session_state[fkey] = fig
                        except Exception as e:
                            st.error(str(e))
                    if fkey in st.session_state:
                        preview_and_download(st.session_state[fkey], chart_key)
                    else:
                        st.markdown('<div class="empty-state"><div class="es-icon">🍕</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

            else:  # Scatter
                lcol, rcol = st.columns([1, 2])
                with lcol:
                    st.markdown('<div class="tool-card">', unsafe_allow_html=True)
                    sc_player = st.selectbox("Highlight player", ["(none)"] + players, key="scat_player")
                    sc_x = st.selectbox("X-axis metric", mc, index=0, key="scat_x")
                    sc_y = st.selectbox("Y-axis metric", mc, index=min(1,len(mc)-1), key="scat_y")
                    sc_title = st.text_input("Title", "Scatter Plot", key="scat_title")
                    sc_dot   = st.color_picker("Dot color", "#00d4ff", key="scat_dot")
                    sc_hi    = st.color_picker("Highlight color", "#ffd060", key="scat_hi")
                    gen_sc   = st.button("Generate Scatter", key="scat_gen")
                    st.markdown('</div>', unsafe_allow_html=True)
                with rcol:
                    if gen_sc:
                        d = dfp.copy()
                        d[sc_x] = clean_numeric(d[sc_x]); d[sc_y] = clean_numeric(d[sc_y])
                        d = d.dropna(subset=[sc_x,sc_y])
                        fig, ax = plt.subplots(figsize=(9,6))
                        fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                        ax.scatter(d[sc_x], d[sc_y], s=75, color=sc_dot, alpha=0.72, edgecolors="white", lw=0.6)
                        if sc_player != "(none)":
                            h = d[d[player_col].astype(str) == str(sc_player)]
                            if not h.empty:
                                ax.scatter(h[sc_x], h[sc_y], s=200, color=sc_hi, edgecolors="white", lw=1.5, zorder=5)
                                for _, r in h.iterrows():
                                    ax.text(float(r[sc_x]), float(r[sc_y]), f"  {r[player_col]}",
                                            color="#e8f0fe", fontsize=10, weight="bold")
                        ax.axvline(d[sc_x].median(), color="#1e3a5f", lw=1.5, ls="--")
                        ax.axhline(d[sc_y].median(), color="#1e3a5f", lw=1.5, ls="--")
                        ax.set_title(sc_title, color="#e8f0fe", fontsize=16, weight="bold")
                        ax.set_xlabel(sc_x, color="#6b8cae"); ax.set_ylabel(sc_y, color="#6b8cae")
                        ax.tick_params(colors="#6b8cae")
                        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
                        st.session_state["pizza_scat_fig"] = fig
                    if "pizza_scat_fig" in st.session_state:
                        preview_and_download(st.session_state["pizza_scat_fig"], "scatter_plot")
                    else:
                        st.markdown('<div class="empty-state"><div class="es-icon">📈</div><div class="es-title">Configure and generate</div></div>', unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🧠  PLAYER SCOUTING
# ─────────────────────────────────────────────────────────────────────────────
if section == "🧠 Player Scouting":
    if ROLE_TEMPLATES is None:
        st.error("scouting_tools_v2.py not found. Make sure it is in the same directory as app.py.")
        st.stop()

    header("🧠", "Player Scouting", "Role templates · percentile scoring · shortlists · recommendations")

    def _clean(x): return str(x).strip().lower().replace("_"," ").replace("-"," ")
    def _safe_numeric(s): return pd.to_numeric(s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False), errors="coerce")
    def _label(score):
        if pd.isna(score): return "No data"
        if score >= 82:  return "🟢 Elite / Priority"
        if score >= 70:  return "🟡 Strong option"
        if score >= 58:  return "🟠 Watchlist"
        if score >= 45:  return "🔴 Average – needs video"
        return "⛔ Risky"

    def _infer_role(pos):
        p = _clean(pos)
        for k, v in {
            "gk":"Goalkeeper","goalkeeper":"Goalkeeper",
            "cb":"Centre Back","rcb":"Centre Back","lcb":"Centre Back",
            "rb":"Full Back / Wing Back","lb":"Full Back / Wing Back",
            "rwb":"Full Back / Wing Back","lwb":"Full Back / Wing Back",
            "dm":"Defensive Midfielder","cdm":"Defensive Midfielder",
            "cm":"Central Midfielder","mc":"Central Midfielder",
            "am":"Attacking Midfielder","cam":"Attacking Midfielder",
            "rw":"Winger","lw":"Winger","winger":"Winger",
            "st":"Striker","cf":"Striker","striker":"Striker",
        }.items():
            if k in p: return v
        return list(ROLE_TEMPLATES.keys())[0]

    def _make_scored(df_in, metrics, lower_better, group_col=None, minutes_col=None, min_floor=900, weights=None):
        out = df_in.copy()
        pct_cols = []; weights = weights or {}
        for m in metrics:
            if m not in out.columns: continue
            out[m] = _safe_numeric(out[m])
            pc = f"pct__{m}"
            higher = m not in lower_better
            if group_col and group_col in out.columns:
                out[pc] = out.groupby(group_col, dropna=False)[m].transform(lambda x: _pct_rank(x, higher))
            else:
                out[pc] = _pct_rank(out[m], higher)
            pct_cols.append(pc)
        if not pct_cols:
            out["Scouting Score"] = np.nan; out["Adjusted Score"] = np.nan; return out
        w = np.array([float(weights.get(c.replace("pct__",""),1.0)) for c in pct_cols], dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        mat = out[pct_cols].astype(float)
        out["Scouting Score"] = mat.mul(w, axis=1).sum(axis=1) / mat.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)
        out["Scouting Score"] = out["Scouting Score"].round(1)
        if minutes_col and minutes_col in out.columns:
            mins = pd.to_numeric(out[minutes_col], errors="coerce").fillna(0)
            out["Reliability"] = (mins / float(max(min_floor,1))).clip(0,1).round(2)
            out["Adjusted Score"] = (out["Scouting Score"] * (0.65 + 0.35 * out["Reliability"])).round(1)
        else:
            out["Reliability"] = 1.0; out["Adjusted Score"] = out["Scouting Score"]
        return out

    # Sidebar
    with st.sidebar:
        st.markdown("### 📂 Scouting File")
        scout_f = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], key="scout_file")
        st.download_button("⬇ Template CSV", data=make_template_csv(),
                           file_name="scouting_template.csv", mime="text/csv", key="scout_template")
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        min_mins_g = st.number_input("Default min minutes", 0, value=300, step=50, key="scout_mmg")
        scope_ui   = st.radio("Compare scope", ["All filtered","Same position"], index=1, key="scout_scope")
        use_rel    = st.checkbox("Reliability adjustment", value=True, key="scout_rel")
        rel_floor  = st.number_input("'Reliable' = minutes ≥", 1, value=900, step=50, key="scout_rf")

    if scout_f is None:
        st.info("Upload a player scouting file to begin. You can download the template from the sidebar.")
        st.stop()

    try:
        df_raw_s = load_player_data(scout_f)
    except Exception as e:
        st.error(f"Could not read file: {e}"); st.stop()

    df_raw_s.columns = [str(c).strip() for c in df_raw_s.columns]
    cols_s = standard_columns(df_raw_s)
    all_cols = [None] + df_raw_s.columns.tolist()

    with st.sidebar:
        st.markdown("### 🧩 Column Mapping")
        player_col   = st.selectbox("Player",   all_cols, index=all_cols.index(cols_s["player"])   if cols_s["player"]        in all_cols else 0, key="sc_pcol")
        team_col     = st.selectbox("Team",     all_cols, index=all_cols.index(cols_s["team"])     if cols_s["team"]          in all_cols else 0, key="sc_tcol")
        position_col = st.selectbox("Position", all_cols, index=all_cols.index(cols_s["position"]) if cols_s["position"]      in all_cols else 0, key="sc_poscol")
        age_col      = st.selectbox("Age",      all_cols, index=all_cols.index(cols_s["age"])      if cols_s["age"]           in all_cols else 0, key="sc_acol")
        minutes_col  = st.selectbox("Minutes",  all_cols, index=all_cols.index(cols_s["minutes"])  if cols_s["minutes"]       in all_cols else 0, key="sc_mcol")
        value_col_s  = st.selectbox("Market Value", all_cols, index=all_cols.index(cols_s["market_value"]) if cols_s["market_value"] in all_cols else 0, key="sc_vcol")

    if not player_col:
        st.error("Select player column."); st.stop()

    base_exclude = [player_col, team_col, position_col, age_col, minutes_col, value_col_s]
    metric_cols_s = numeric_metrics(df_raw_s, base_exclude)
    df_s = coerce_numeric(df_raw_s, metric_cols_s + [c for c in [age_col, minutes_col, value_col_s] if c])

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sel_pos   = st.multiselect("Positions", sorted(df_s[position_col].dropna().astype(str).unique()) if position_col else [], key="sc_fpos")
    with fc2:
        sel_teams = st.multiselect("Teams",     sorted(df_s[team_col].dropna().astype(str).unique())     if team_col     else [], key="sc_fteam")
    with fc3:
        min_m     = st.number_input("Min minutes", 0, value=int(min_mins_g), step=50, key="sc_fm")
    with fc4:
        max_age   = st.number_input("Max age (0=off)", 0, value=0, step=1, key="sc_fa")

    df_f = df_s.copy()
    if sel_pos   and position_col: df_f = df_f[df_f[position_col].astype(str).isin(sel_pos)]
    if sel_teams and team_col:     df_f = df_f[df_f[team_col].astype(str).isin(sel_teams)]
    if minutes_col: df_f = df_f[pd.to_numeric(df_f[minutes_col], errors="coerce").fillna(0) >= min_m]
    if max_age > 0 and age_col: df_f = df_f[pd.to_numeric(df_f[age_col], errors="coerce") <= max_age]
    if df_f.empty: st.warning("No players match filters."); st.stop()

    # Role + metrics
    role_opts = list(ROLE_TEMPLATES.keys())
    auto_role = _infer_role(df_f[position_col].dropna().astype(str).mode().iloc[0]) if position_col and len(df_f[position_col].dropna()) else role_opts[0]
    role = st.selectbox("🎯 Role template", role_opts, index=role_opts.index(auto_role) if auto_role in role_opts else 0, key="sc_role")
    matched, missing_m, mapping = match_template_metrics(df_f, role)
    default_m = matched[:14] if matched else metric_cols_s[:10]
    custom_m = st.multiselect("Metrics for scoring", metric_cols_s, default=default_m, key="sc_metrics")
    if not custom_m: st.error("Choose at least one metric."); st.stop()

    neg_guess = [m for m in custom_m if any(w in _clean(m) for w in ["conceded","foul","error","turnover","dispossessed","card"])]
    lower_better = st.multiselect("Lower = better", custom_m, default=neg_guess, key="sc_lb")

    with st.expander("Metric weights", expanded=False):
        weights_s = {}
        wc = st.columns(3)
        for i, m in enumerate(custom_m):
            with wc[i % 3]:
                weights_s[m] = st.number_input(m, 0.1, 3.0, 1.0, 0.1, key=f"sw_{m}")

    group_col_s = position_col if scope_ui == "Same position" and position_col else None
    score_col   = "Adjusted Score" if use_rel else "Scouting Score"
    df_scored   = _make_scored(df_f, custom_m, lower_better, group_col=group_col_s,
                                minutes_col=minutes_col, min_floor=rel_floor, weights=weights_s)

    # KPIs
    kpis = kpi("Players", len(df_scored)) + kpi("Metrics", len(custom_m)) + \
           kpi(f"Avg {score_col}", f"{df_scored[score_col].mean():.1f}") + \
           kpi("Top player", str(df_scored.sort_values(score_col, ascending=False).iloc[0][player_col]))
    st.markdown(f'<div class="kpi-row">{kpis}</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    players_s = sorted(df_scored[player_col].dropna().astype(str).unique().tolist())

    tab_prof, tab_cmp, tab_rank, tab_ins, tab_sl, tab_check = st.tabs(
        ["👤 Profile", "⚔️ Compare", "🏆 Rankings", "💡 Insights", "⭐ Shortlist", "📋 Data Check"])

    # ── Profile ──────────────────────────────────────────────────────────────
    with tab_prof:
        sel_p = st.selectbox("Player", players_s, key="sc_pp")
        row_p = df_scored[df_scored[player_col].astype(str) == sel_p].iloc[0]
        score_v = row_p.get(score_col, np.nan)
        pairs = sorted([(m, float(row_p[f"pct__{m}"])) for m in custom_m if f"pct__{m}" in df_scored.columns and pd.notna(row_p.get(f"pct__{m}"))], key=lambda x: x[1], reverse=True)
        strengths = pd.DataFrame(pairs[:6], columns=["Metric","Percentile"])
        concerns  = pd.DataFrame(sorted(pairs, key=lambda x: x[1])[:6], columns=["Metric","Percentile"])

        lc, rc = st.columns([1,1.6])
        with lc:
            st.markdown(f"""
            <div class="tool-card">
                <div class="panel-title" style="font-size:1.1rem;font-weight:800;color:var(--accent);">{sel_p}</div>
                <div style="color:var(--muted);font-size:0.85rem;margin-bottom:8px;">Role: {role}</div>
                <div style="font-size:2.4rem;font-weight:900;color:var(--text);">{'NA' if pd.isna(score_v) else f'{float(score_v):.1f}'}<span style="font-size:1rem;color:var(--muted)">/100</span></div>
                <div style="font-size:1rem;font-weight:700;margin-top:4px;">{_label(score_v)}</div>
                <div style="color:var(--muted);font-size:0.8rem;margin-top:8px;">Raw: {row_p.get('Scouting Score','—')} | Reliability: {row_p.get('Reliability','—')}</div>
            </div>""", unsafe_allow_html=True)
            rec = ""
            if recommendation_text:
                try:
                    rec = recommendation_text(df_scored, player_col, sel_p, custom_m,
                                              role=role, team_col=team_col, position_col=position_col,
                                              age_col=age_col, minutes_col=minutes_col)
                except: pass
            if not rec:
                top_txt = ", ".join([f"{m} ({p:.0f}th)" for m,p in pairs[:3]])
                weak_txt= ", ".join([f"{m} ({p:.0f}th)" for m,p in sorted(pairs, key=lambda x:x[1])[:3]])
                rec = f"{sel_p}: {_label(score_v).split(' ',1)[-1]} | Score {score_v:.1f if not pd.isna(score_v) else 'NA'}\nStrengths: {top_txt}.\nConcerns: {weak_txt}."
            st.text_area("Scout recommendation", rec, height=200, key="sc_rec_text")

        with rc:
            st.markdown("**Strengths**")
            st.dataframe(strengths, use_container_width=True, hide_index=True)
            st.markdown("**Weaknesses / Watch-outs**")
            st.dataframe(concerns, use_container_width=True, hide_index=True)
            st.markdown("**Similar Players**")
            try:
                sim = similar_players(df_scored, player_col, sel_p, custom_m, top_n=7)
                show_c = [player_col] + [c for c in [team_col, position_col, score_col, "Similarity Distance"] if c and c in sim.columns and c != player_col]
                st.dataframe(sim[show_c], use_container_width=True, hide_index=True)
            except: st.info("Similar players unavailable.")

    # ── Compare ───────────────────────────────────────────────────────────────
    with tab_cmp:
        cc1, cc2, cc3 = st.columns([1,1,1.2])
        with cc1: p1 = st.selectbox("Player A", players_s, index=0, key="sc_cmp_p1")
        with cc2: p2 = st.selectbox("Player B", players_s, index=min(1,len(players_s)-1), key="sc_cmp_p2")
        with cc3: ctype = st.radio("Chart type", ["Bar","Radar"], horizontal=True, key="sc_cmp_type")
        cmp_m = st.multiselect("Metrics", custom_m, default=custom_m[:8], key="sc_cmp_metrics")
        if p1 == p2: st.warning("Choose two different players.")
        elif cmp_m:
            try:
                fig = comparison_chart(df_scored, player_col, p1, p2, cmp_m, use_percentiles=True) \
                      if ctype == "Bar" else radar_chart(df_scored, player_col, p1, p2, cmp_m)
                preview_and_download(fig, "comparison")
            except Exception as e: st.error(str(e))

    # ── Rankings ──────────────────────────────────────────────────────────────
    with tab_rank:
        show_c = [player_col] + [c for c in [team_col, position_col, age_col, minutes_col,
                                              "Scouting Score","Adjusted Score","Reliability"] if c and c in df_scored.columns and c != player_col]
        st.dataframe(df_scored.sort_values(score_col, ascending=False)[show_c].head(100),
                     use_container_width=True, hide_index=True)
        st.download_button("⬇ Export CSV", data=df_scored.to_csv(index=False).encode("utf-8-sig"),
                           file_name="scored_players.csv", mime="text/csv", key="sc_export")

    # ── Insights ──────────────────────────────────────────────────────────────
    with tab_ins:
        insights = auto_dataset_insights(df_scored, player_col, custom_m,
                                         team_col=team_col, age_col=age_col, minutes_col=minutes_col)
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}.** {ins}")
        if not insights: st.info("Not enough data for insights.")

    # ── Shortlist ─────────────────────────────────────────────────────────────
    with tab_sl:
        if "shortlist" not in st.session_state: st.session_state.shortlist = []
        sc1, sc2, sc3 = st.columns([1,1,1.5])
        with sc1: add_p  = st.selectbox("Player",  players_s, key="sl_player")
        with sc2: add_st = st.selectbox("Status", ["Watch","Follow","Recommend","Reject"], key="sl_status")
        with sc3: add_n  = st.text_input("Note", key="sl_note")
        if st.button("Add to shortlist", key="sl_add"):
            r = df_scored[df_scored[player_col].astype(str) == str(add_p)].iloc[0]
            st.session_state.shortlist.append({
                "Player": add_p, "Score": r.get(score_col,""),
                "Status": add_st, "Note": add_n,
                "Team": r.get(team_col,"") if team_col else "",
                "Position": r.get(position_col,"") if position_col else "",
                "Age": r.get(age_col,"") if age_col else "",
                "Minutes": r.get(minutes_col,"") if minutes_col else "",
            })
            st.success("Added!")
        sl_df = pd.DataFrame(st.session_state.shortlist)
        if not sl_df.empty:
            st.dataframe(sl_df, use_container_width=True, hide_index=True)
            st.download_button("⬇ Export shortlist", sl_df.to_csv(index=False).encode("utf-8-sig"),
                               "shortlist.csv", "text/csv", key="sl_export")
        else: st.info("No players added yet.")

    # ── Data Check ────────────────────────────────────────────────────────────
    with tab_check:
        if mapping:
            st.dataframe(pd.DataFrame([{"Template Metric": k,"File Column": v} for k,v in mapping.items()]),
                         use_container_width=True, hide_index=True)
        if missing_m:
            st.warning("Missing template metrics:")
            st.dataframe(pd.DataFrame({"Missing": missing_m}), use_container_width=True, hide_index=True)
        qrows = []
        for m in custom_m:
            s = _safe_numeric(df_f[m]) if m in df_f.columns else pd.Series(dtype=float)
            qrows.append({"Metric": m, "Valid": int(s.notna().sum()), "Missing": int(s.isna().sum()),
                          "Mean": round(float(s.mean()),2) if s.notna().any() else np.nan,
                          "Lower Better": m in lower_better})
        st.dataframe(pd.DataFrame(qrows), use_container_width=True, hide_index=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🖱️  TAGGING TOOL  (unchanged logic, redesigned UI)
# ─────────────────────────────────────────────────────────────────────────────
if section == "🖱️ Tagging Tool":
    if streamlit_image_coordinates is None:
        header("🖱️","Tagging Tool")
        st.error("Missing dependency: `streamlit-image-coordinates`")
        st.code("streamlit-image-coordinates", language="text")
        st.stop()

    header("🖱️","Interactive Tagging Tool","Click the pitch to tag events — auto-save or manual mode")

    def _tag_y_max(pitch_mode, pitch_width):
        return float(pitch_width if pitch_mode == "rect" else 100.0)

    def _event_color(et):
        return {"pass":"#00C2FF","carry":"#FF9300","dribble":"#A78BFA","cross":"#FFD400",
                "shot":"#00FF6A","touch":"#FFFFFF","defensive action":"#FF4D4D","recovery":"#EF4444"
                }.get(str(et).lower(), "#FFFFFF")

    def _init():
        for k,v in {"tag_events":[],"tag_start":None,"tag_end":None,"tag_last_click":None,
                    "tag_click_counter":0,"tag_last_processed":None,"tag_flash":""}.items():
            if k not in st.session_state: st.session_state[k] = v

    def _pitch_to_img(x, y, w, h, y_max, pad=22):
        iw = max(1,w-2*pad); ih = max(1,h-2*pad)
        return pad + int(round((float(x)/100.0)*iw)), pad + int(round(ih - (float(y)/float(y_max))*ih))

    def _img_to_pitch(px, py, w, h, y_max, pad=22):
        iw = max(1,w-2*pad); ih = max(1,h-2*pad)
        return round(((float(px)-pad)/iw)*100.0, 2), round(y_max - (((float(py)-pad)/ih)*y_max), 2)

    def _draw_arrow(draw, s, e, color, width=5):
        import math
        draw.line([s,e], fill=color, width=width)
        ang = math.atan2(e[1]-s[1], e[0]-s[0]); sz = 14
        for a in (ang+math.pi*0.82, ang-math.pi*0.82):
            draw.polygon([(e[0], e[1]), (e[0]+sz*math.cos(a), e[1]+sz*math.sin(a))], fill=color)

    def _dot(draw, x, y, w, h, y_max, fill, edge="#FFF", r=9, pad=22, marker="o"):
        if marker is None or str(marker).lower() in {"none",""}:  return
        px, py = _pitch_to_img(x, y, w, h, y_max, pad)
        box = [px-r, py-r, px+r, py+r]
        m = str(marker)
        if m == "s":   draw.rectangle(box, fill=fill, outline=edge, width=3)
        elif m == "D": draw.polygon([(px,py-r),(px+r,py),(px,py+r),(px-r,py)], fill=fill); draw.line([(px,py-r),(px+r,py),(px,py+r),(px-r,py),(px,py-r)], fill=edge, width=3)
        elif m in {"^","v"}:
            pts = [(px,py-r),(px+r,py+r),(px-r,py+r)] if m=="^" else [(px-r,py-r),(px+r,py-r),(px,py+r)]
            draw.polygon(pts, fill=fill); draw.line(pts+[pts[0]], fill=edge, width=3)
        elif m == "*":
            for dx,dy in [(r,0),(0,r),(r,r),(-r,r)]:
                draw.line([(px-dx,py-dy),(px+dx,py+dy)], fill=edge, width=3)
            draw.ellipse([px-r//2,py-r//2,px+r//2,py+r//2], fill=fill)
        elif m in {"+","x"}:
            if m=="+": draw.line([(px-r,py),(px+r,py)], fill=fill, width=5); draw.line([(px,py-r),(px,py+r)], fill=fill, width=5)
            else: draw.line([(px-r,py-r),(px+r,py+r)], fill=fill, width=5); draw.line([(px-r,py+r),(px+r,py-r)], fill=fill, width=5)
        else: draw.ellipse(box, fill=fill, outline=edge, width=3)

    def _make_pitch_img(theme_name, pitch_mode, pitch_width, dw=760, thirds=True,
                        cur_mk="o", cur_col="#22C55E", cur_edge="#FFF", cur_sz=9):
        from charts import THEMES
        theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
        y_max = _tag_y_max(pitch_mode, pitch_width); pad = 22
        dh = int(round((dw-2*pad)*y_max/100.0)) + 2*pad
        img = Image.new("RGB", (dw, dh), theme.get("pitch","#1f5f3b"))
        draw = ImageDraw.Draw(img)
        def P(x,y): return _pitch_to_img(x, y, dw, dh, y_max, pad)
        lc = theme.get("pitch_lines","#E6E6E6"); lw = 3
        draw.rectangle([P(0,0), P(100,y_max)], outline=lc, width=lw)
        if thirds:
            tc = theme.get("lines","#446688")
            for tx in (100/3, 200/3): draw.line([P(tx,0), P(tx,y_max)], fill=tc, width=2)
        draw.line([P(50,0), P(50,y_max)], fill=lc, width=lw)
        cx,cy = P(50,y_max/2); rx=int((dw-2*pad)*9.15/105); ry=int((dh-2*pad)*9.15/68)
        draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], outline=lc, width=lw)
        mid=y_max/2; pw=y_max*40.32/68; sw=y_max*18.32/68; pl=16.5/105*100; sl=5.5/105*100
        draw.rectangle([P(0,mid-pw/2), P(pl,mid+pw/2)], outline=lc, width=lw)
        draw.rectangle([P(100-pl,mid-pw/2), P(100,mid+pw/2)], outline=lc, width=lw)
        draw.rectangle([P(0,mid-sw/2), P(sl,mid+sw/2)], outline=lc, width=lw)
        draw.rectangle([P(100-sl,mid-sw/2), P(100,mid+sw/2)], outline=lc, width=lw)
        for evts in st.session_state.get("tag_events",[]):
            try:
                et=str(evts.get("event_type","")).lower(); col=str(evts.get("start_color",_event_color(et)))
                edge=str(evts.get("start_edge","#FFF")); mk=evts.get("start_marker","o")
                sz=int(float(evts.get("start_size",9))); ac=str(evts.get("arrow_color",col))
                x,y = float(evts.get("x",0)), float(evts.get("y",0))
                x2,y2 = evts.get("x2"), evts.get("y2")
                if et in ["pass","carry","dribble","cross"] and pd.notna(x2) and pd.notna(y2):
                    _draw_arrow(draw, P(x,y), P(float(x2),float(y2)), ac, width=5)
                _dot(draw, x, y, dw, dh, y_max, col, edge=edge, r=max(4,sz), pad=pad, marker=mk)
            except: pass
        if st.session_state.get("tag_start"):
            sx,sy = st.session_state.tag_start
            _dot(draw, sx, sy, dw, dh, y_max, cur_col, edge=cur_edge, r=int(cur_sz), pad=pad, marker=cur_mk)
        return img, dw, dh, y_max, pad

    def _save_event(et, outcome, player, team, minute, tag, note, start, end,
                    mk, col, edge, sz, acol):
        nid = len(st.session_state.tag_events)+1
        st.session_state.tag_events.append({
            "event_id": nid, "event_type": et, "player": player, "team": team,
            "minute": int(minute),
            "x": round(float(start[0]),2), "y": round(float(start[1]),2),
            "x2": round(float(end[0]),2) if end else np.nan,
            "y2": round(float(end[1]),2) if end else np.nan,
            "outcome": outcome, "tag": tag, "note": note,
            "start_marker": mk, "start_color": col, "start_edge": edge,
            "start_size": int(sz), "arrow_color": acol,
        })
        st.session_state.tag_start = None; st.session_state.tag_end = None
        st.session_state.tag_last_click = None
        st.session_state.tag_flash = f"✅ Saved {et} #{nid}"

    _init()

    ctrl_col, pitch_col = st.columns([0.85, 2.2], gap="large")
    with ctrl_col:
        st.markdown('<div class="tool-card">', unsafe_allow_html=True)
        t_theme = st.selectbox("Theme", list(THEMES.keys()), index=0, key="tg_theme")
        t_pitch = st.selectbox("Pitch shape", ["Rectangular","Square"], key="tg_shape")
        t_pm    = "rect" if t_pitch == "Rectangular" else "square"
        t_pw    = st.slider("Pitch width", 50.0, 80.0, 68.0, 1.0, key="tg_pw") if t_pm=="rect" else 100.0
        t_dw    = st.slider("Display size", 580, 900, 740, 20, key="tg_dw")
        t_thirds= st.checkbox("Show thirds", True, key="tg_thirds")
        st.markdown("---")
        auto_save = st.toggle("Auto-save", value=True, key="tg_auto")
        t_et    = st.selectbox("Event type", ["pass","carry","dribble","cross","shot","touch","defensive action","recovery"], key="tg_et")
        t_out   = st.selectbox("Outcome", ["successful","unsuccessful","key pass","assist","goal","ontarget","off target","blocked","touch"], key="tg_out")
        t_mk_l  = st.selectbox("Marker", MARKER_LABELS, index=1, key="tg_mk")
        t_mk    = MARKER_OPTIONS[t_mk_l]
        t_col   = st.color_picker("Dot color", _event_color(t_et), key="tg_col")
        t_edge  = st.color_picker("Edge color", "#FFFFFF", key="tg_edge")
        t_sz    = st.slider("Dot size", 4, 18, 8, key="tg_sz")
        t_ac    = st.color_picker("Arrow color", _event_color(t_et), key="tg_ac")
        t_player= st.text_input("Player", key="tg_player")
        t_team  = st.text_input("Team",   key="tg_team")
        t_min   = st.number_input("Minute", 0, 130, 0, 1, key="tg_min")
        t_tag   = st.selectbox("Action tag", ["","progressive","into final third","into box","key pass","under pressure","turnover","duel"], key="tg_tag")
        t_note  = st.text_area("Note", height=60, key="tg_note")

        needs_end = t_et in ["pass","carry","dribble","cross"]
        if st.session_state.tag_flash:
            st.success(st.session_state.tag_flash); st.session_state.tag_flash = ""

        if auto_save:
            if needs_end and st.session_state.tag_start:
                st.info(f"Start: {st.session_state.tag_start[0]:.1f},{st.session_state.tag_start[1]:.1f} — click END")
            elif needs_end: st.info("Click START point; next click = END + save")
            else:           st.info("Click anywhere to tag")
        else:
            bc1,bc2 = st.columns(2)
            with bc1:
                if st.button("Set START", disabled=st.session_state.tag_last_click is None):
                    st.session_state.tag_start = st.session_state.tag_last_click
                    if not needs_end: st.session_state.tag_end = None
                    st.rerun()
            with bc2:
                if st.button("Set END", disabled=(st.session_state.tag_last_click is None or not needs_end)):
                    st.session_state.tag_end = st.session_state.tag_last_click; st.rerun()
            if st.button("Save Event"):
                s = st.session_state.tag_start; e = st.session_state.tag_end
                if not s: st.warning("Set START first.")
                elif needs_end and not e: st.warning("Set END first.")
                else:
                    _save_event(t_et,t_out,t_player,t_team,t_min,t_tag,t_note,s,e,t_mk,t_col,t_edge,t_sz,t_ac)
                    st.rerun()

        bx1,bx2 = st.columns(2)
        with bx1:
            if st.button("Undo") and st.session_state.tag_events:
                st.session_state.tag_events.pop(); st.session_state.tag_start=None; st.rerun()
        with bx2:
            if st.button("Clear all"):
                for k in ["tag_events","tag_start","tag_end","tag_last_click","tag_last_processed"]:
                    st.session_state[k] = [] if k=="tag_events" else None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with pitch_col:
        img, iw, ih, y_max, pad = _make_pitch_img(t_theme, t_pm, t_pw, dw=t_dw,
                                                   thirds=t_thirds, cur_mk=t_mk, cur_col=t_col,
                                                   cur_edge=t_edge, cur_sz=t_sz)
        coords = streamlit_image_coordinates(img,
            key=f"tg_img_{t_theme}_{t_pm}_{t_pw}_{t_dw}_{st.session_state.tag_click_counter}")

        if coords and coords.get("x") is not None:
            try:
                x, y = _img_to_pitch(coords["x"], coords["y"], iw, ih, y_max, pad)
                if 0 <= x <= 100 and 0 <= y <= y_max:
                    raw = (int(coords["x"]), int(coords["y"]), st.session_state.tag_click_counter)
                    if st.session_state.tag_last_processed != raw:
                        st.session_state.tag_last_click = (x, y)
                        st.session_state.tag_last_processed = raw
                        if auto_save:
                            if needs_end:
                                if st.session_state.tag_start is None:
                                    st.session_state.tag_start = (x,y)
                                    st.session_state.tag_flash = f"Start: {x:.1f},{y:.1f}"
                                else:
                                    _save_event(t_et,t_out,t_player,t_team,t_min,t_tag,t_note,
                                                st.session_state.tag_start,(x,y),t_mk,t_col,t_edge,t_sz,t_ac)
                            else:
                                _save_event(t_et,t_out,t_player,t_team,t_min,t_tag,t_note,
                                            (x,y),None,t_mk,t_col,t_edge,t_sz,t_ac)
                        st.session_state.tag_click_counter += 1; st.rerun()
            except: pass

        evts_df = pd.DataFrame(st.session_state.tag_events) if st.session_state.tag_events else pd.DataFrame()
        if not evts_df.empty:
            with st.expander(f"Tagged events ({len(evts_df)})", expanded=False):
                st.dataframe(evts_df, use_container_width=True, hide_index=True)
            c1,c2 = st.columns(2)
            with c1:
                st.download_button("⬇ CSV", evts_df.to_csv(index=False).encode("utf-8-sig"),
                                   "tagged_events.csv", "text/csv", key="tg_csv")
            with c2:
                if st.button("Generate Map", key="tg_map"):
                    from matplotlib.lines import Line2D
                    theme = THEMES.get(t_theme, THEMES["The Athletic Dark"])
                    pitch = make_pitch(pitch_mode=t_pm, pitch_width=t_pw, theme=theme)
                    fig, ax = plt.subplots(figsize=(11,7))
                    fig.patch.set_facecolor(theme.get("bg","#0E1117"))
                    pitch.draw(ax=ax); ax.set_facecolor(theme.get("pitch","#1f5f3b"))
                    y_max_m = _tag_y_max(t_pm, t_pw)
                    if t_thirds:
                        tc = theme.get("lines","#334466")
                        for tx in (100/3,200/3): ax.plot([tx,tx],[0,y_max_m], color=tc, ls="--", lw=1.5, alpha=0.8)
                    for _,r in evts_df.iterrows():
                        try:
                            et=str(r.get("event_type","")).lower(); mk=r.get("start_marker","o")
                            sc=str(r.get("start_color",_event_color(et))); se=str(r.get("start_edge","#FFF"))
                            ss=int(float(r.get("start_size",9))); ac=str(r.get("arrow_color",sc))
                            x,y=float(r.get("x",0)),float(r.get("y",0))
                            x2,y2=r.get("x2"),r.get("y2")
                            if et in ["pass","carry","dribble","cross"] and pd.notna(x2) and pd.notna(y2):
                                pitch.arrows([x],[y],[float(x2)],[float(y2)],ax=ax,color=ac,width=2.4,headwidth=6,alpha=0.9,zorder=4)
                            if mk and str(mk).lower() not in {"none",""}:
                                pitch.scatter([x],[y],ax=ax,s=max(30,ss*ss*2),marker=mk,color=sc,edgecolors=se,linewidth=1.2,zorder=5)
                        except: pass
                    ax.set_title("Tagged Events", color=theme.get("text","white"), fontsize=16, weight="bold")
                    ax.set_xlim(-2,102); ax.set_ylim(-2,y_max_m+2)
                    st.session_state["tg_map_fig"] = fig
            if "tg_map_fig" in st.session_state:
                preview_and_download(st.session_state["tg_map_fig"], "tagged_events_map")
    st.stop()
