import os
import tempfile
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
    THEMES,
)

# =========================================================
# MUST be first Streamlit call
# =========================================================
st.set_page_config(page_title="Football Charts Generator", layout="wide")

st.markdown("## ⚽ Football Charts Generator")
st.caption("Upload CSV / Excel → Match Report / Pizza / Shot Card")


# -----------------------------
# Helpers
# -----------------------------
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
    """
    Light normalization in app (safe):
    - cleans BOM/NBSP/Â artifacts BEFORE lower()
    - maps common labels
    Note: charts.py will still do the strong normalization.
    """
    out = df.copy()
    if "outcome" not in out.columns:
        return out

    s = (
        out["outcome"]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)   # BOM
        .str.replace("\xa0", " ", regex=False)    # NBSP
        .str.replace("Â", " ", regex=False)       # cp1252 artifact
        .str.replace("â", " ", regex=False)       # if lower() happened elsewhere
        .str.strip()
        .str.lower()
    )

    mapping = {
        # shots
        "on target": "1ontarget",
        "ontarget": "1ontarget",
        "1 on target": "1ontarget",
        "shot on target": "1ontarget",
        "sot": "1ontarget",
        "saved": "1ontarget",

        "off target": "1offtarget",
        "offtarget": "1offtarget",
        "shot off target": "1offtarget",
        "miss": "1offtarget",
        "wide": "1offtarget",

        "goal": "goal",
        "scored": "goal",

        "blocked": "blocked",
        "block": "blocked",

        # passes
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
        "â assist": "assist",   # extra safety
        "a": "assist",

        # touch
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
            "percentile": 0 if pd.isna(pct) else round(float(pct), 1)
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


# -----------------------------
# Requirements shown in UI + pre-check
# -----------------------------
CHART_REQUIREMENTS = {
    "Outcome Bar": ["outcome"],
    "Start Heatmap": ["x", "y"],
    "Touch Map (Scatter)": ["x", "y", "(optional) outcome='touch'"],
    # ✅ IMPORTANT: Pass Map should NOT require x2/y2 (we show markers if missing)
    "Pass Map": ["outcome", "x", "y", "(optional) x2", "(optional) y2"],
    "Shot Map": ["outcome", "x", "y", "(optional) x2,y2", "(optional) xg"],
}

# -----------------------------
# Settings
# -----------------------------
with st.expander("🎛️ Settings", expanded=True):
    st.markdown("### 🧩 Report Header")

    report_title = st.text_input("Title", value="Match Report")
    report_subtitle = st.text_input("Subtitle", value="")

    header_img = st.file_uploader("Upload header image (Club logo / Player face) - PNG/JPG", type=["png", "jpg", "jpeg"])
    header_img_side = st.selectbox("Image position", ["Left", "Right"], index=0)

    header_img_size = st.slider("Image size (as % of figure width)", 5, 18, 10)
    header_img_width_frac = header_img_size / 100.0

    st.markdown("### Title / Subtitle style")
    title_align = st.selectbox("Title align", ["Center", "Left", "Right"], index=0)
    subtitle_align = st.selectbox("Subtitle align", ["Center", "Left", "Right"], index=0)

    title_fontsize = st.slider("Title font size", 12, 28, 16)
    subtitle_fontsize = st.slider("Subtitle font size", 9, 20, 11)

    title_color = st.color_picker("Title color", "#FFFFFF")
    subtitle_color = st.color_picker("Subtitle color", "#A0A7B4")

    st.markdown("---")

    st.markdown("### Theme")
    theme_name = st.selectbox(
        "Choose theme",
        list(THEMES.keys()),
        index=list(THEMES.keys()).index("The Athletic Dark") if "The Athletic Dark" in THEMES else 0,
    )

    attack_dir_ui = st.selectbox("Attack direction", ["Left → Right", "Right → Left"])
    attack_dir = "ltr" if attack_dir_ui == "Left → Right" else "rtl"

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

    st.markdown("### Report charts to include")
    all_charts = ["Outcome Bar", "Start Heatmap", "Touch Map (Scatter)", "Pass Map", "Shot Map"]
    selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts)

    if selected_charts:
        with st.expander("📌 Required columns for selected charts (pre-check)", expanded=False):
            for ch in selected_charts:
                st.write(f"**{ch}** → " + ", ".join(CHART_REQUIREMENTS.get(ch, [])))

    st.markdown("---")
    st.markdown("### Pass Map Filters (NEW)")

    # ✅ IMPORTANT: strings must include these keywords for charts.py filter:
    # "final third", "penalty box", "line", "progressive"
    pass_view = st.selectbox(
        "Pass map view",
        [
            "All passes",
            "Into Final Third",
            "Into Penalty Box",
            "Line-breaking (proxy)",
            "Progressive passes",
        ],
        index=0
    )

    pass_scope = st.selectbox(
        "Pass result scope",
        ["Attempts (all)", "Successful only", "Unsuccessful only"],
        index=0
    )
    pass_min_packing = st.slider("Min packing (proxy) for Line-breaking", 1, 3, 1)

    show_debug = st.checkbox("Show pass debug counts", value=True)

    st.markdown("---")
    st.markdown("### Pass colors")
    col_pass_success = st.color_picker("Successful", "#00FF6A")
    col_pass_unsuccess = st.color_picker("Unsuccessful", "#FF4D4D")
    col_pass_key = st.color_picker("Key pass", "#00C2FF")
    col_pass_assist = st.color_picker("Assist", "#FFD400")

    st.markdown("### Shot colors")
    col_shot_off = st.color_picker("Off target", "#FF8A00")
    col_shot_on = st.color_picker("On target", "#00C2FF")
    col_shot_goal = st.color_picker("Goal", "#00FF6A")
    col_shot_blocked = st.color_picker("Blocked", "#AAAAAA")

    st.markdown("### Outcome bar colors")
    bar_success = st.color_picker("Bar: successful", "#00FF6A")
    bar_unsuccess = st.color_picker("Bar: unsuccessful", "#FF4D4D")
    bar_key = st.color_picker("Bar: key pass", "#00C2FF")
    bar_assist = st.color_picker("Bar: assist", "#FFD400")
    bar_ont = st.color_picker("Bar: ontarget", "#00C2FF")
    bar_off = st.color_picker("Bar: off target", "#FF8A00")
    bar_goal = st.color_picker("Bar: goal", "#00FF6A")
    bar_blocked = st.color_picker("Bar: blocked", "#AAAAAA")

    st.markdown("---")
    st.markdown("### Event shapes (markers)")
    marker_options = {
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
    mk_shot_off = st.selectbox("Marker: Off target", marker_labels, index=marker_labels.index("Triangle up (^)"))
    mk_shot_on = st.selectbox("Marker: On target", marker_labels, index=marker_labels.index("Diamond (D)"))
    mk_shot_goal = st.selectbox("Marker: Goal", marker_labels, index=marker_labels.index("Star (*)"))
    mk_shot_blocked = st.selectbox("Marker: Blocked", marker_labels, index=marker_labels.index("Square (s)"))

    st.markdown("#### Pass start markers")
    mk_pass_success = st.selectbox("Marker: Successful pass", marker_labels, index=marker_labels.index("Circle (o)"))
    mk_pass_unsuccess = st.selectbox("Marker: Unsuccessful pass", marker_labels, index=marker_labels.index("X (x)"))
    mk_pass_key = st.selectbox("Marker: Key pass", marker_labels, index=marker_labels.index("Diamond (D)"))
    mk_pass_assist = st.selectbox("Marker: Assist", marker_labels, index=marker_labels.index("Star (*)"))

    st.markdown("#### Touch map marker")
    touch_marker_label = st.selectbox("Marker: Touch", marker_labels, index=marker_labels.index("Circle (o)"))
    touch_dot_color = st.color_picker("Touch dots color", "#34D5FF")
    touch_dot_edge = st.color_picker("Touch edge color", "#0B0F14")
    touch_dot_size = st.slider("Touch dot size", 60, 520, 220)
    touch_alpha = st.slider("Touch alpha", 20, 100, 95) / 100.0


pass_colors = {
    "successful": col_pass_success,
    "unsuccessful": col_pass_unsuccess,
    "key pass": col_pass_key,
    "assist": col_pass_assist
}
shot_colors = {
    "off target": col_shot_off,
    "ontarget": col_shot_on,
    "goal": col_shot_goal,
    "blocked": col_shot_blocked
}
bar_colors = {
    "successful": bar_success, "unsuccessful": bar_unsuccess, "key pass": bar_key, "assist": bar_assist,
    "ontarget": bar_ont, "off target": bar_off, "goal": bar_goal, "blocked": bar_blocked
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
touch_marker = marker_options[touch_marker_label]


# -----------------------------
# Header image load
# -----------------------------
img_obj = None
if header_img is not None:
    try:
        img_obj = Image.open(header_img).convert("RGBA")
    except Exception:
        img_obj = None


# -----------------------------
# Mode + Upload
# -----------------------------
mode = st.radio("Choose output type", ["Match Charts", "Pizza Chart", "Shot Detail Card"])
uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])


# =========================================================
# MAIN
# =========================================================
if uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        # ---------- Pizza ----------
        if mode == "Pizza Chart":
            dfp = load_data(path)
            st.success(f"Loaded ✅ rows: {len(dfp)}")
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

            if st.button("Generate Pizza"):
                pizza_df = build_pizza_df(dfp_filtered, player_col, selected_player, selected_metrics)
                base_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf"]
                slice_colors = [base_colors[i % len(base_colors)] for i in range(len(pizza_df))]
                fig = pizza_chart(pizza_df, title=pizza_title, subtitle=pizza_subtitle, slice_colors=slice_colors)

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "pizza.png")
                pdf_path = os.path.join(out_dir, "pizza.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.pyplot(fig)
                with open(pdf_path, "rb") as f2:
                    st.download_button("⬇️ Download pizza.pdf", f2, file_name="pizza.pdf")
                with open(png_path, "rb") as f2:
                    st.download_button("⬇️ Download pizza.png", f2, file_name="pizza.png")
            st.stop()

        # ---------- Match / Shot Card ----------
        df_raw = load_data(path)

        with st.expander("Raw file preview (first 25 rows)", expanded=False):
            st.write("Columns:", list(df_raw.columns))
            st.dataframe(df_raw.head(25), use_container_width=True)

        if "outcome" not in df_raw.columns:
            st.warning("Column `outcome` not found. Trying to derive it automatically.")
        df_raw = ensure_outcome_column(df_raw)
        df_raw = normalize_outcome_values(df_raw)

        # PRE-CHECK required columns for selected charts
        cols_lower = _lower_cols(df_raw)
        missing_by_chart = {}
        for ch in selected_charts:
            req = CHART_REQUIREMENTS.get(ch, [])
            miss = _missing_for_chart(cols_lower, req)
            if miss:
                missing_by_chart[ch] = miss

        if missing_by_chart and mode == "Match Charts":
            st.error("❌ Missing required columns for selected charts (fix file or unselect chart):")
            for ch, miss in missing_by_chart.items():
                st.write(f"**{ch}** missing → {', '.join(miss)}")
        else:
            st.success("✅ Required columns check passed (or not needed for this mode).")

        can_generate_report = (len(missing_by_chart) == 0)

        # Model load
        model_pipe = None
        if xg_method == "model":
            if model_exists:
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

        st.success(f"Prepared ✅ rows: {len(df2)}")
        if "xg_source" in df2.columns and len(df2):
            st.info(f"xG source used: **{df2['xg_source'].iloc[0]}**")

        # ✅ DEBUG: counts should be based on ALL passes (NOT only those with x2/y2)
        if show_debug:
            p_all = df2[df2["event_type"] == "pass"].copy()

            view = (pass_view or "All passes").lower().strip()
            scope = (pass_scope or "Attempts (all)").lower().strip()

            p_view = p_all.copy()
            if "final third" in view:
                p_view = p_view[p_view.get("into_final_third", False) == True]
            elif "penalty box" in view or "box" in view:
                p_view = p_view[p_view.get("into_penalty_box", False) == True]
            elif "line" in view:
                p_view = p_view[pd.to_numeric(p_view.get("packing_proxy", 0), errors="coerce").fillna(0).astype(int) >= int(pass_min_packing)]
            elif "progressive" in view:
                p_view = p_view[p_view.get("progressive_pass", False) == True]

            p_scope = p_view.copy()
            if "successful" in scope:
                p_scope = p_scope[p_scope.get("is_pass_successful", False) == True]
            elif "unsuccessful" in scope or "failed" in scope:
                p_scope = p_scope[p_scope.get("is_pass_unsuccessful", False) == True]
            else:
                if "is_pass_attempt" in p_scope.columns:
                    p_scope = p_scope[p_scope["is_pass_attempt"] == True]

            st.write("### 🧪 Pass Debug Counts")
            st.write({
                "passes_total_prepared": int(len(p_all)),
                "after_pass_view": int(len(p_view)),
                "after_scope": int(len(p_scope)),
                "into_box_true_total": int((p_all.get("into_penalty_box", False) == True).sum()) if len(p_all) else 0,
                "into_final_third_true_total": int((p_all.get("into_final_third", False) == True).sum()) if len(p_all) else 0,
                "progressive_true_total": int((p_all.get("progressive_pass", False) == True).sum()) if len(p_all) else 0,
            })

        with st.expander("Preview prepared data (first 25 rows)"):
            st.write("Prepared columns:", list(df2.columns))
            st.dataframe(df2.head(25), use_container_width=True)

        # ---------- Shot Detail Card ----------
        if mode == "Shot Detail Card":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.stop()

            shots_only["label"] = shots_only.apply(
                lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f} | ({_safe_float(r["x"]):.1f},{_safe_float(r["y"]):.1f})',
                axis=1
            )
            selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
            shot_index = int(selected.split("|")[0].strip()) - 1
            card_title = st.text_input("Card title", value="Shot Detail")

            if st.button("Generate Shot Card"):
                fig, _ = shot_detail_card(
                    df2,
                    shot_index=int(shot_index),
                    title=card_title,
                    pitch_mode=pitch_mode,
                    pitch_width=pitch_width,
                    shot_colors=shot_colors,
                    shot_markers=shot_markers,
                    theme_name=theme_name,
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "shot_card.png")
                pdf_path = os.path.join(out_dir, "shot_card.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)

                st.pyplot(fig)
                with open(png_path, "rb") as f2:
                    st.download_button("⬇️ Download shot_card.png", f2, file_name="shot_card.png")
                with open(pdf_path, "rb") as f2:
                    st.download_button("⬇️ Download shot_card.pdf", f2, file_name="shot_card.pdf")
            st.stop()

        # ---------- Report ----------
        generate_clicked = st.button("Generate Report", disabled=not can_generate_report)
        if not can_generate_report:
            st.info("Generate Report is disabled until required columns exist for all selected charts.")

        if generate_clicked:
            out_dir = os.path.join(tmp, "output")

            pdf_path, png_paths = build_report_from_prepared_df(
                df2,
                out_dir=out_dir,
                title=report_title,
                subtitle=report_subtitle,
                header_image=img_obj,
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
                bar_colors=bar_colors,
                charts_to_include=selected_charts,
                touch_dot_color=touch_dot_color,
                touch_dot_edge=touch_dot_edge,
                touch_dot_size=touch_dot_size,
                touch_alpha=touch_alpha,
                touch_marker=touch_marker,

                # ✅ pass filters sent to charts.py
                pass_view=pass_view,
                pass_result_scope=pass_scope,
                pass_min_packing=pass_min_packing,
            )

            st.subheader("Preview")
            for p in png_paths:
                st.image(p, use_container_width=True)

            st.subheader("Downloads")
            with open(pdf_path, "rb") as f2:
                st.download_button("⬇️ Download report.pdf", f2, file_name="report.pdf")
            for p in png_paths:
                name = os.path.basename(p)
                with open(p, "rb") as f2:
                    st.download_button(f"⬇️ Download {name}", f2, file_name=name)
