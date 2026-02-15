# app.py  (FULL UPDATED APP) — Header image + subtitle + existing pipeline
# =========================================================
import os
import tempfile
import joblib
import io

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages  # ✅ FIX

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
    THEMES,
)

st.set_page_config(page_title="Football Charts Generator", layout="wide")

# -----------------------------
# App title (simple)
# -----------------------------
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
    """
    Ensure df has 'outcome'. If missing, try to derive it from other common columns
    like 'event', 'result', 'shot_result', etc. Never crash; if can't derive, create
    an 'outcome' column filled with 'unknown'.
    """
    df = df_raw.copy()

    if "outcome" in df.columns:
        return df

    cols_lower = {c.lower().strip(): c for c in df.columns}

    # 1) Direct rename from common alternatives
    candidates = [
        "event", "event_type", "type",
        "result", "shot_result", "outcome_type",
        "shot_outcome", "final_outcome",
    ]
    for c in candidates:
        if c in cols_lower:
            df["outcome"] = df[cols_lower[c]]
            return df

    # 2) Build from boolean flags if available
    def _has(col): return col in cols_lower

    if _has("is_goal") or _has("goal"):
        base = pd.Series([np.nan] * len(df))
        if _has("is_goal"):
            base = np.where(
                pd.to_numeric(df[cols_lower["is_goal"]], errors="coerce").fillna(0).astype(int) == 1,
                "goal",
                base
            )
        if _has("goal"):
            base = np.where(
                pd.to_numeric(df[cols_lower["goal"]], errors="coerce").fillna(0).astype(int) == 1,
                "goal",
                base
            )

        df["outcome"] = base

        if _has("is_ontarget"):
            m = pd.to_numeric(df[cols_lower["is_ontarget"]], errors="coerce").fillna(0).astype(int) == 1
            df.loc[m & df["outcome"].isna(), "outcome"] = "1ontarget"
        if _has("is_offtarget"):
            m = pd.to_numeric(df[cols_lower["is_offtarget"]], errors="coerce").fillna(0).astype(int) == 1
            df.loc[m & df["outcome"].isna(), "outcome"] = "1offtarget"
        if _has("is_blocked"):
            m = pd.to_numeric(df[cols_lower["is_blocked"]], errors="coerce").fillna(0).astype(int) == 1
            df.loc[m & df["outcome"].isna(), "outcome"] = "blocked"

        df["outcome"] = df["outcome"].fillna("unknown")
        return df

    # 3) Fallback: create outcome with 'unknown'
    df["outcome"] = "unknown"
    return df


def normalize_outcome_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize outcome strings so charts expect consistent labels:
    goal / 1ontarget / 1offtarget / blocked / successful / unsuccessful / key pass / assist
    """
    out = df.copy()
    if "outcome" not in out.columns:
        return out

    s = out["outcome"].astype(str).str.strip().str.lower()

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
        "successfull": "successful",   # typo fix

        "unsuccessful": "unsuccessful",
        "unsuccess": "unsuccessful",
        "incomplete": "unsuccessful",
        "failed": "unsuccessful",
        "unsuccessfull": "unsuccessful",  # typo fix

        "key pass": "key pass",
        "keypass": "key pass",
        "kp": "key pass",

        "assist": "assist",
        "a": "assist",
    }

    out["outcome"] = s.map(lambda v: mapping.get(v, v))
    return out


def normalize_pass_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    If passes outcomes are sometimes labeled as accurate/inaccurate,
    convert them to successful/unsuccessful (and fix common typos).
    """
    out = df.copy()
    if "outcome" not in out.columns:
        return out

    s = out["outcome"].astype(str).str.strip().str.lower()

    mapping = {
        "accurate": "successful",
        "inaccurate": "unsuccessful",
        "accurate pass": "successful",
        "inaccurate pass": "unsuccessful",

        # extra typo variants
        "successfull": "successful",
        "unsuccessfull": "unsuccessful",
        "successful pass": "successful",
        "unsuccessful pass": "unsuccessful",
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


# -----------------------------
# Settings
# -----------------------------
with st.expander("🎛️ Settings", expanded=True):
    st.markdown("### 🧩 Report Header")
    report_title = st.text_input("Title", value="Match Report")
    report_subtitle = st.text_input("Subtitle (you control it)", value="")

    header_img = st.file_uploader(
        "Upload header image (Club logo / Player face) - PNG/JPG",
        type=["png", "jpg", "jpeg"]
    )
    header_img_side = st.selectbox("Image position", ["Left", "Right"], index=0)
    header_img_width = st.slider("Image width", 40, 180, 90)

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

    xg_method_ui = st.radio(
        "xG method",
        ["Zone", "Model"],
        index=1 if model_exists else 0
    )
    xg_method = "model" if xg_method_ui == "Model" else "zone"

    st.caption(f"Model exists: **{model_exists}**")

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
    "successful": bar_success,
    "unsuccessful": bar_unsuccess,
    "key pass": bar_key,
    "assist": bar_assist,
    "ontarget": bar_ont,
    "off target": bar_off,
    "goal": bar_goal,
    "blocked": bar_blocked
}

# -----------------------------
# Header Preview (UI only)
# -----------------------------
img_obj = None
if header_img is not None:
    try:
        img_obj = Image.open(header_img)
    except Exception:
        img_obj = None

if img_obj is not None:
    if header_img_side == "Left":
        c1, c2 = st.columns([1, 8], vertical_alignment="center")
        with c1:
            st.image(img_obj, width=header_img_width)
        with c2:
            st.markdown(f"## {report_title}")
            if report_subtitle.strip():
                st.caption(report_subtitle)
    else:
        c1, c2 = st.columns([8, 1], vertical_alignment="center")
        with c1:
            st.markdown(f"## {report_title}")
            if report_subtitle.strip():
                st.caption(report_subtitle)
        with c2:
            st.image(img_obj, width=header_img_width)
else:
    st.markdown(f"## {report_title}")
    if report_subtitle.strip():
        st.caption(report_subtitle)

# -----------------------------
# Mode + Upload
# -----------------------------
mode = st.radio("Choose output type", ["Match Charts", "Pizza Chart", "Shot Detail Card"])
uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])

# -----------------------------
# Main
# -----------------------------
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

                fig.savefig(png_path, dpi=220, bbox_inches="tight")
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")

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
            st.warning("Column `outcome` not found. Trying to derive it automatically (e.g., from `event`/`result`).")

        # ✅ FIX: create outcome if missing + normalize values + normalize pass accurate/inaccurate
        df_raw = ensure_outcome_column(df_raw)
        df_raw = normalize_outcome_values(df_raw)
        df_raw = normalize_pass_outcomes(df_raw)

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
                    theme_name=theme_name,
                )

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)
                png_path = os.path.join(out_dir, "shot_card.png")
                pdf_path = os.path.join(out_dir, "shot_card.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight")
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")

                st.pyplot(fig)
                with open(png_path, "rb") as f2:
                    st.download_button("⬇️ Download shot_card.png", f2, file_name="shot_card.png")
                with open(pdf_path, "rb") as f2:
                    st.download_button("⬇️ Download shot_card.pdf", f2, file_name="shot_card.pdf")
            st.stop()

        # ---------- Report ----------
        if st.button("Generate Report"):
            out_dir = os.path.join(tmp, "output")

            # Put subtitle inside title text for PDF (safe without changing charts.py)
            pdf_title = report_title.strip()
            if report_subtitle.strip():
                pdf_title = f"{pdf_title}\n{report_subtitle.strip()}"

            pdf_path, png_paths = build_report_from_prepared_df(
                df2,
                out_dir=out_dir,
                title=pdf_title,
                theme_name=theme_name,
                pitch_mode=pitch_mode,
                pitch_width=pitch_width,
                pass_colors=pass_colors,
                shot_colors=shot_colors,
                bar_colors=bar_colors,
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
