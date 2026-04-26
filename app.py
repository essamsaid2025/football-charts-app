import os
import io
import tempfile
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

from charts import (
    THEMES,
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
    defensive_regains_map,
    progressive_actions_chart,
    passing_direction_chart,
    carry_map,
    receive_map,
    zone_heatmap,
    player_comparison_dashboard,
    shot_spot_and_direction_map,
    chart_required_columns_note,
)

st.set_page_config(
    page_title="Football Scouting Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# UI STYLES
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
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
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
            color: white;
        }

        .app-subtitle {
            color: #cbd5e1;
            margin-top: 8px;
            font-size: 0.98rem;
        }

        .preview-shell {
            background: rgba(17,24,39,0.92);
            border: 1px solid #243041;
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

        section[data-testid="stSidebar"] {
            background: #0b1220;
        }

        .stButton > button, .stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">⚽ Football Scouting Studio</div>
        <div class="app-subtitle">
            Player scouting, match analysis, maps, dashboards, pizza charts, and exportable reports.
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
    "Touch Map (Scatter)": ["x", "y"],
    "Pass Map": ["outcome", "x", "y", "x2", "y2"],
    "Shot Map": ["outcome", "x", "y"],
    "Defensive Actions Map": ["x", "y"],
    "Progressive Actions": ["outcome"],
    "Passing Direction": ["x", "y", "x2", "y2"],
    "Carry Map": ["x", "y", "x2", "y2"],
    "Receive Map": ["x", "y"],
    "Zone Heatmap": ["x", "y"],
    "Shot Spot + Direction": ["x", "y", "x2", "y2"],
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
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Controls")

    with st.expander("📂 Output & File", expanded=True):
        mode = st.selectbox(
            "Choose output type",
            [
                "Match Charts",
                "Pizza Chart",
                "Shot Detail Card",
                "Defensive Actions Map",
                "Progressive Actions Chart",
                "Passing Direction Chart",
                "Carry Map",
                "Receive Map",
                "Zone Heatmap",
                "Player Comparison Dashboard",
                "Shot Spot + Direction Map",
            ],
        )
        uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

    with st.expander("🎨 Theme & Layout", expanded=True):
        theme_name = st.selectbox(
            "Theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index("Opta Analyst Light") if "Opta Analyst Light" in THEMES else 0,
        )
        attack_dir_ui = st.selectbox("Attack direction", ["Left → Right", "Right → Left"])
        attack_dir = "ltr" if attack_dir_ui == "Left → Right" else "rtl"
        pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular (recommended)", "Square (0-100)"])
        pitch_mode = "rect" if pitch_mode_ui.startswith("Rectangular") else "square"
        pitch_width = st.slider("Rect pitch width", min_value=50.0, max_value=80.0, value=64.0, step=1.0)
        flip_y = st.checkbox("Flip Y axis", value=False)
        vertical_pitch = st.checkbox("Show pitch vertical", value=False)

    with st.expander("🧩 Report Header", expanded=False):
        report_title = st.text_input("Title", value="Match Report")
        report_subtitle = st.text_input("Subtitle", value="")
        header_img = st.file_uploader(
            "Upload header image",
            type=["png", "jpg", "jpeg"],
            key="header_img_uploader",
        )
        header_img_side = st.selectbox("Image position", ["Left", "Right"], index=0)
        header_img_size = st.slider("Image size (% of figure width)", 5, 18, 10)
        header_img_width_frac = header_img_size / 100.0
        title_align = st.selectbox("Title align", ["Center", "Left", "Right"], index=0)
        subtitle_align = st.selectbox("Subtitle align", ["Center", "Left", "Right"], index=0)
        title_fontsize = st.slider("Title font size", 12, 28, 16)
        subtitle_fontsize = st.slider("Subtitle font size", 9, 20, 11)
        title_color = st.color_picker("Title color", "#FFFFFF")
        subtitle_color = st.color_picker("Subtitle color", "#A0A7B4")

    with st.expander("🤖 Advanced / xG", expanded=False):
        model_file = st.text_input("Model file path", value="xg_pipeline.joblib")
        model_exists = os.path.exists(model_file)
        xg_method_ui = st.radio("xG method", ["Zone", "Model"], index=1 if model_exists else 0)
        xg_method = "model" if xg_method_ui == "Model" else "zone"
        st.caption(f"Model exists: **{model_exists}**")

    with st.expander("📊 Charts Selection", expanded=False):
        all_charts = [
            "Outcome Bar",
            "Start Heatmap",
            "Touch Map (Scatter)",
            "Pass Map",
            "Shot Map",
            "Defensive Actions Map",
            "Progressive Actions",
            "Passing Direction",
            "Carry Map",
            "Receive Map",
            "Zone Heatmap",
            "Shot Spot + Direction",
        ]
        selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts[:10])
        st.markdown("**Required columns per selected chart**")
        for _ch in selected_charts:
            st.caption(f"{_ch}: {', '.join(CHART_REQUIREMENTS.get(_ch, []))}")

    with st.expander("🎯 Pass / Touch / Defensive Settings", expanded=False):
        pass_view = st.selectbox(
            "Pass map view",
            ["All passes", "Into Final Third", "Into Penalty Box", "Line-breaking", "Progressive passes"],
            index=0,
        )
        pass_scope = st.selectbox("Pass result scope", ["Attempts (all)", "Successful only", "Unsuccessful only"], index=0)
        pass_min_packing = st.slider("Min packing", 1, 3, 1)

        touch_dot_color = st.color_picker("Touch dots color", "#34D5FF")
        touch_dot_edge = st.color_picker("Touch edge color", "#0B0F14")
        touch_dot_size = st.slider("Touch dot size", 60, 520, 220)
        touch_alpha = st.slider("Touch alpha", 20, 100, 95) / 100.0

        def_map_title = st.text_input("Defensive map title", value="Ball Regains Map")
        def_show_zone_values = st.checkbox("Show defensive zone values", value=False)
        def_marker_size = st.slider("Defensive marker size", 60, 260, 110)
        def_zone_alpha = st.slider("Defensive zone alpha", 20, 100, 78) / 100.0

    with st.expander("🧭 Legends & Notes", expanded=False):
        show_legends = st.checkbox("Show legends", value=True)
        show_required_note = st.checkbox("Show required columns under every chart", value=True)

    with st.expander("🧱 Colors & Markers", expanded=False):
        col_pass_success = st.color_picker("Pass Successful", "#00FF6A")
        col_pass_unsuccess = st.color_picker("Pass Unsuccessful", "#FF4D4D")
        col_pass_key = st.color_picker("Key pass", "#00C2FF")
        col_pass_assist = st.color_picker("Assist", "#FFD400")

        col_shot_off = st.color_picker("Shot Off target", "#7A7A7A")
        col_shot_on = st.color_picker("Shot On target", "#111111")
        col_shot_goal = st.color_picker("Shot Goal", "#FF0000")
        col_shot_blocked = st.color_picker("Shot Blocked", "#AAAAAA")

        col_interception = st.color_picker("Interception", "#00C2FF")
        col_tackle = st.color_picker("Tackle", "#FF8A00")
        col_recovery = st.color_picker("Recovery", "#00FF6A")
        col_aerial = st.color_picker("Aerial Duel", "#FFD400")
        col_ground = st.color_picker("Ground Duel", "#FF4D4D")
        col_clearance = st.color_picker("Clearance", "#A78BFA")

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

        mk_shot_off = st.selectbox("Marker: Off target", marker_labels, index=marker_labels.index("X (x)"))
        mk_shot_on = st.selectbox("Marker: On target", marker_labels, index=marker_labels.index("Circle (o)"))
        mk_shot_goal = st.selectbox("Marker: Goal", marker_labels, index=marker_labels.index("Circle (o)"))
        mk_shot_blocked = st.selectbox("Marker: Blocked", marker_labels, index=marker_labels.index("Square (s)"))

        mk_pass_success = st.selectbox("Marker: Successful pass", marker_labels, index=marker_labels.index("Circle (o)"))
        mk_pass_unsuccess = st.selectbox("Marker: Unsuccessful pass", marker_labels, index=marker_labels.index("X (x)"))
        mk_pass_key = st.selectbox("Marker: Key pass", marker_labels, index=marker_labels.index("Diamond (D)"))
        mk_pass_assist = st.selectbox("Marker: Assist", marker_labels, index=marker_labels.index("Star (*)"))

        mk_interception = st.selectbox("Marker: Interception", marker_labels, index=marker_labels.index("Circle (o)"))
        mk_tackle = st.selectbox("Marker: Tackle", marker_labels, index=marker_labels.index("Square (s)"))
        mk_recovery = st.selectbox("Marker: Recovery", marker_labels, index=marker_labels.index("Diamond (D)"))
        mk_aerial = st.selectbox("Marker: Aerial Duel", marker_labels, index=marker_labels.index("Triangle up (^)"))
        mk_ground = st.selectbox("Marker: Ground Duel", marker_labels, index=marker_labels.index("X (x)"))
        mk_clearance = st.selectbox("Marker: Clearance", marker_labels, index=marker_labels.index("Star (*)"))

        touch_marker_label = st.selectbox("Marker: Touch", marker_labels, index=marker_labels.index("Circle (o)"))

    with st.expander("🍕 Pizza / Comparison", expanded=False):
        pizza_center_img = st.file_uploader(
            "Upload pizza center image",
            type=["png", "jpg", "jpeg"],
            key="pizza_center_uploader",
        )
        pizza_center_scale = st.slider("Center image size (Pizza)", 12, 32, 18) / 100.0
        compare_player_col_name = st.text_input("Comparison player column hint", value="player")

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
# SESSION STATE
# =========================================================
if "preview_images" not in st.session_state:
    st.session_state.preview_images = []
if "download_files" not in st.session_state:
    st.session_state.download_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# MAIN
# =========================================================
right_col = st.container()

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

        if mode == "Pizza Chart" or mode == "Player Comparison Dashboard":
            dfp = load_data(path)
            st.dataframe(dfp.head(20), use_container_width=True)
        else:
            dfp = None

        df_raw = load_data(path)
        if "outcome" not in [c.lower().strip() for c in df_raw.columns]:
            st.warning("Column `outcome` not found. Trying to derive it automatically.")
        df_raw = ensure_outcome_column(df_raw)
        df_raw = normalize_outcome_values(df_raw)

        cols_lower = _lower_cols(df_raw)
        if mode == "Match Charts":
            missing_by_chart = {}
            for ch in selected_charts:
                req = CHART_REQUIREMENTS.get(ch, [])
                miss = _missing_for_chart(cols_lower, req)
                if miss:
                    missing_by_chart[ch] = miss
            if missing_by_chart:
                st.error("❌ Missing required columns for selected charts:")
                for ch, miss in missing_by_chart.items():
                    st.write(f"**{ch}** missing → {', '.join(miss)}")

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

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="small-kpi"><div class="label">Mode</div><div class="value">{mode}</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="small-kpi"><div class="label">Rows</div><div class="value">{len(df2)}</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="small-kpi"><div class="label">Columns</div><div class="value">{len(df2.columns)}</div></div>', unsafe_allow_html=True)

        theme = THEMES.get(theme_name, list(THEMES.values())[0])

        # =========================
        # Pizza
        # =========================
        if mode == "Pizza Chart":
            cols_lower_map = {c.lower(): c for c in dfp.columns}
            player_col = cols_lower_map.get(compare_player_col_name.lower(), None) or cols_lower_map.get("player", None) or st.selectbox("Select player column", dfp.columns.tolist())
            players = sorted(dfp[player_col].dropna().astype(str).unique().tolist())
            selected_player = st.selectbox("Choose player", players)

            exclude = {player_col}
            for maybe in ["minutes", "team", "position", "pos", "league", "season", "age"]:
                if maybe in cols_lower_map:
                    exclude.add(cols_lower_map[maybe])

            metric_cols = [c for c in dfp.columns if c not in exclude]
            default_n = min(8, len(metric_cols))
            selected_metrics = st.multiselect("Choose metrics", metric_cols, default=metric_cols[:default_n])

            pizza_title = st.text_input("Pizza title", value=selected_player)
            pizza_subtitle = st.text_input("Pizza subtitle", value="Percentile vs peers")

            def pct_color(p):
                try:
                    p = float(p)
                except Exception:
                    p = 0.0
                if p >= 85:
                    return "#6D28D9"
                elif p >= 70:
                    return "#22A06B"
                elif p >= 50:
                    return "#A7A7A7"
                else:
                    return "#D64045"

            if st.button("Generate Pizza"):
                pizza_df = build_pizza_df(dfp, player_col, selected_player, selected_metrics)
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
                    theme_name=theme_name,
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Shot Detail Card
        # =========================
        elif mode == "Shot Detail Card":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.stop()

            shots_only["label"] = shots_only.apply(
                lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f} | ({_safe_float(r["x"]):.1f},{_safe_float(r["y"]):.1f})',
                axis=1,
            )
            selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
            shot_index = int(selected.split("|")[0].strip()) - 1
            card_title = st.text_input("Card title", value="Shot Detail")

            if st.button("Generate Shot Card"):
                fig, _ = shot_detail_card(
                    df2,
                    shot_index=shot_index,
                    title=card_title,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    theme_name=theme_name,
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Defensive Actions Map
        # =========================
        elif mode == "Defensive Actions Map":
            if st.button("Generate Defensive Map"):
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
                    vertical_pitch=vertical_pitch,
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Progressive Actions
        # =========================
        elif mode == "Progressive Actions Chart":
            if st.button("Generate Progressive Actions Chart"):
                fig = progressive_actions_chart(df2, theme_name=theme_name)
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Passing Direction
        # =========================
        elif mode == "Passing Direction Chart":
            if st.button("Generate Passing Direction Chart"):
                fig = passing_direction_chart(df2, theme_name=theme_name)
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Carry Map
        # =========================
        elif mode == "Carry Map":
            if st.button("Generate Carry Map"):
                fig = carry_map(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, vertical_pitch=vertical_pitch)
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Receive Map
        # =========================
        elif mode == "Receive Map":
            if st.button("Generate Receive Map"):
                fig = receive_map(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, vertical_pitch=vertical_pitch)
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Zone Heatmap
        # =========================
        elif mode == "Zone Heatmap":
            zone_event_type = st.selectbox("Zone event type", ["all", "pass", "shot", "touch", "defensive", "receive", "carry"])
            if st.button("Generate Zone Heatmap"):
                fig = zone_heatmap(
                    df2,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    theme_name=theme_name,
                    event_type=zone_event_type,
                    title=f"Zone Heatmap — {zone_event_type.title()}",
                    vertical_pitch=vertical_pitch,
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Player Comparison
        # =========================
        elif mode == "Player Comparison Dashboard":
            cols_lower_map = {c.lower(): c for c in dfp.columns}
            player_col = cols_lower_map.get(compare_player_col_name.lower(), None) or cols_lower_map.get("player", None) or st.selectbox("Select player column", dfp.columns.tolist())
            players = sorted(dfp[player_col].dropna().astype(str).unique().tolist())
            p1 = st.selectbox("Player 1", players, index=0)
            p2 = st.selectbox("Player 2", players, index=1 if len(players) > 1 else 0)

            exclude = {player_col}
            for maybe in ["minutes", "team", "position", "pos", "league", "season", "age"]:
                if maybe in cols_lower_map:
                    exclude.add(cols_lower_map[maybe])
            metric_cols = [c for c in dfp.columns if c not in exclude]
            chosen_metrics = st.multiselect("Comparison metrics", metric_cols, default=metric_cols[:min(8, len(metric_cols))])

            if st.button("Generate Comparison Dashboard"):
                fig = player_comparison_dashboard(
                    dfp,
                    player_col=player_col,
                    player_1=p1,
                    player_2=p2,
                    metrics=chosen_metrics,
                    theme_name=theme_name,
                    title=f"{p1} vs {p2}",
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Shot Spot + Direction
        # =========================
        elif mode == "Shot Spot + Direction Map":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.stop()

            plot_all = st.checkbox("Plot all shots", value=True)
            shot_index = None
            if not plot_all:
                shots_only["label"] = shots_only.apply(
                    lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f}',
                    axis=1,
                )
                selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
                shot_index = int(selected.split("|")[0].strip()) - 1

            if st.button("Generate Shot Spot + Direction Map"):
                fig = shot_spot_and_direction_map(
                    df2,
                    title="Shot Spot + Direction Map",
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    theme_name=theme_name,
                    plot_all=plot_all,
                    shot_index=shot_index,
                    vertical_pitch=vertical_pitch,
                )
                if show_required_note:
                    chart_required_columns_note(fig, CHART_REQUIREMENTS.get(mode.replace(" Chart", "").replace(" Map", ""), []), theme_name)
                st.pyplot(fig, use_container_width=True)

        # =========================
        # Match Report
        # =========================
        elif mode == "Match Charts":
            if st.button("Generate Report"):
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
                    pass_colors=pass_colors,
                    pass_markers=pass_markers,
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    def_colors=def_colors,
                    def_markers=def_markers,
                    charts_to_include=selected_charts,
                    touch_dot_color=touch_dot_color,
                    touch_dot_edge=touch_dot_edge,
                    touch_dot_size=touch_dot_size,
                    touch_alpha=touch_alpha,
                    touch_marker=touch_marker,
                    pass_view=pass_view,
                    pass_result_scope=pass_scope,
                    pass_min_packing=pass_min_packing,
                    vertical_pitch=vertical_pitch,
                    show_legends=show_legends,
                )

                st.success("Match report generated successfully.")
                for p in png_paths:
                    st.image(read_file_bytes(p), use_container_width=True)

                st.download_button(
                    "⬇️ Download report.pdf",
                    data=read_file_bytes(pdf_path),
                    file_name="report.pdf",
                    mime="application/pdf",
                )

    st.markdown("</div>", unsafe_allow_html=True)
