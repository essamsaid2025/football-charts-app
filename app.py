# app.py  (FULL UPDATED APP) — Zone + Model xG (default auto)
# =========================================
# Pipeline:
# - load_data
# - prepare_df_for_charts (clean + transforms + end-location + xG) ONCE
# - build_report_from_prepared_df for Match Charts
# - Shot Detail Card uses dropdown + NO shot type (handled in charts.py)
# =========================================

import os
import tempfile
import inspect
import joblib

import streamlit as st
import pandas as pd
import numpy as np

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
)

st.set_page_config(page_title="Football Charts Generator", layout="wide")

st.title("⚽ Football Charts Generator (Upload CSV / Excel)")
st.caption(
    "Match Charts: required outcome,x,y (passes need x2,y2) | "
    "Pizza: players table (per90) | "
    "Shot Card: shots need x,y (x2,y2 optional)"
)

# ----------------------------
# Settings (match charts + general)
# ----------------------------
with st.expander("🎛️ Settings", expanded=True):
    title_text = st.text_input("Title", value="Match Report")

    attack_dir_ui = st.selectbox("Attack direction", ["Left → Right", "Right → Left"])
    attack_dir = "ltr" if attack_dir_ui == "Left → Right" else "rtl"

    flip_y = st.checkbox("Flip Y axis (use this if your Y=0 is at the bottom)", value=False)

    pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular (recommended)", "Square (0-100)"])
    pitch_mode = "rect" if pitch_mode_ui.startswith("Rectangular") else "square"

    pitch_width = st.slider(
        "Rect pitch width (0-100 scale mapped to this)",
        min_value=50.0, max_value=80.0, value=64.0, step=1.0
    )

    st.markdown("### xG Settings (Zone + Model)")
    model_file = st.text_input("Model file path", value="xg_pipeline.joblib")
    model_exists = os.path.exists(model_file)

    # Default: Model if file exists, else Zone
    default_xg_ui = "Model" if model_exists else "Zone"
    xg_method_ui = st.radio(
        "xG method",
        ["Zone", "Model"],
        index=0 if default_xg_ui == "Zone" else 1,
        help="Model uses saved sklearn pipeline if available; otherwise falls back to Zone."
    )
    xg_method = "model" if xg_method_ui == "Model" else "zone"

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

mode = st.radio("Choose output type", ["Match Charts", "Pizza Chart", "Shot Detail Card"])
uploaded = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])


# ----------------------------
# Helpers for Pizza
# ----------------------------
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
        out.append(
            {
                "metric": m,
                "value": "" if pd.isna(val) else round(float(val), 2),
                "percentile": 0 if pd.isna(pct) else round(float(pct), 1),
            }
        )
    return pd.DataFrame(out)


# ----------------------------
# Main
# ----------------------------
if uploaded:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        # ============================
        # PIZZA MODE
        # ============================
        if mode == "Pizza Chart":
            try:
                dfp = load_data(path)  # IMPORTANT: no prepare_df_for_charts here
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()

            st.success(f"Loaded ✅ rows: {len(dfp)}")
            with st.expander("Preview data (first 25 rows)", expanded=False):
                st.dataframe(dfp.head(25), use_container_width=True)

            cols_lower = {c.lower(): c for c in dfp.columns}
            if "player" in cols_lower:
                player_col = cols_lower["player"]
            else:
                player_col = st.selectbox("Select player column", dfp.columns.tolist())

            players = sorted(dfp[player_col].dropna().astype(str).unique().tolist())
            if not players:
                st.error("No players found in the selected player column.")
                st.stop()

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
            if not metric_cols:
                st.error("No metric columns found (after excluding player/minutes/etc).")
                st.stop()

            default_n = min(8, len(metric_cols))
            selected_metrics = st.multiselect("Choose metrics", metric_cols, default=metric_cols[:default_n])

            pizza_title = st.text_input("Pizza title", value=selected_player)
            pizza_subtitle = st.text_input("Pizza subtitle", value="Percentile vs peers (per90)")

            if st.button("Generate Pizza"):
                if not selected_metrics:
                    st.error("Choose at least 1 metric.")
                    st.stop()

                try:
                    pizza_df = build_pizza_df(dfp_filtered, player_col, selected_player, selected_metrics)

                    base_colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#17becf"]
                    slice_colors = [base_colors[i % len(base_colors)] for i in range(len(pizza_df))]

                    fig = pizza_chart(
                        pizza_df,
                        title=pizza_title,
                        subtitle=pizza_subtitle,
                        slice_colors=slice_colors
                    )

                except Exception as e:
                    st.error(f"Pizza error: {e}")
                    st.stop()

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)

                png_path = os.path.join(out_dir, "pizza.png")
                pdf_path = os.path.join(out_dir, "pizza.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight")

                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")

                st.pyplot(fig)

                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download pizza.pdf", f, file_name="pizza.pdf")
                with open(png_path, "rb") as f:
                    st.download_button("⬇️ Download pizza.png", f, file_name="pizza.png")

            st.stop()

        # ============================
        # MATCH + SHOT CARD (prepared once) + xG method
        # ============================
        # Load model if requested and exists
        model_pipe = None
        if xg_method == "model":
            if os.path.exists(model_file):
                try:
                    model_pipe = joblib.load(model_file)
                except Exception as e:
                    st.warning(f"Could not load model file. Falling back to Zone. Reason: {e}")
                    model_pipe = None
            else:
                st.warning("Model file not found. Falling back to Zone.")
                model_pipe = None

        try:
            df_raw = load_data(path)

            # Call prepare_df_for_charts with extra args ONLY if supported
            sig = inspect.signature(prepare_df_for_charts)
            kwargs = dict(
                attack_direction=attack_dir,
                flip_y=flip_y,
                pitch_mode=pitch_mode,
                pitch_width=pitch_width,
            )
            if "xg_method" in sig.parameters:
                kwargs["xg_method"] = xg_method
            if "model_pipe" in sig.parameters:
                kwargs["model_pipe"] = model_pipe

            df2 = prepare_df_for_charts(df_raw, **kwargs)

        except Exception as e:
            st.error(str(e))
            st.stop()

        st.success(f"Loaded ✅ rows: {len(df2)}")

        # show which xG used
        if "xg_source" in df2.columns and len(df2):
            st.info(f"xG source used: **{df2['xg_source'].iloc[0]}**")

        with st.expander("Preview prepared data (first 25 rows)"):
            st.dataframe(df2.head(25), use_container_width=True)

        cols = st.columns(2)
        with cols[0]:
            st.write("Detected event types:")
            st.write(df2["event_type"].value_counts())
        with cols[1]:
            st.write("Detected outcomes (top 12):")
            st.write(df2["outcome"].value_counts().head(12))

        # ---------- SHOT DETAIL CARD MODE ----------
        if mode == "Shot Detail Card":
            shots_only = df2[df2["event_type"] == "shot"].copy().reset_index(drop=True)
            if shots_only.empty:
                st.error("No shots found in this file.")
                st.stop()

            st.write(f"Shots found: {len(shots_only)}")
            with st.expander("Shots table (first 50)", expanded=False):
                st.dataframe(shots_only.head(50), use_container_width=True)

            # Dropdown instead of raw index
            def _safe_float(v):
                try:
                    return float(v)
                except Exception:
                    return float("nan")

            shots_only["label"] = shots_only.apply(
                lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_safe_float(r.get("xg")):.2f} | ({_safe_float(r["x"]):.1f},{_safe_float(r["y"]):.1f})',
                axis=1
            )
            selected = st.selectbox("Select a shot", shots_only["label"].tolist(), index=0)
            shot_index = int(selected.split("|")[0].strip()) - 1

            card_title = st.text_input("Card title", value="Shot Detail")

            if st.button("Generate Shot Card"):
                try:
                    fig, _ = shot_detail_card(
                        df2,
                        shot_index=int(shot_index),
                        title=card_title,
                        pitch_mode=pitch_mode,
                        pitch_width=pitch_width,
                        shot_colors=shot_colors,
                        theme_name="Opta Dark",
                    )
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                out_dir = os.path.join(tmp, "output")
                os.makedirs(out_dir, exist_ok=True)

                png_path = os.path.join(out_dir, "shot_card.png")
                pdf_path = os.path.join(out_dir, "shot_card.pdf")

                fig.savefig(png_path, dpi=220, bbox_inches="tight")

                from matplotlib.backends.backend_pdf import PdfPages
                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")

                st.pyplot(fig)

                with open(png_path, "rb") as f:
                    st.download_button("⬇️ Download shot_card.png", f, file_name="shot_card.png")

                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download shot_card.pdf", f, file_name="shot_card.pdf")

            st.stop()

        # ---------- MATCH CHARTS MODE ----------
        if st.button("Generate Report"):
            out_dir = os.path.join(tmp, "output")
            pdf_path, png_paths = build_report_from_prepared_df(
                df2,
                out_dir=out_dir,
                title=title_text,
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
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Download report.pdf", f, file_name="report.pdf")

            for p in png_paths:
                name = os.path.basename(p)
                with open(p, "rb") as f:
                    st.download_button(f"⬇️ Download {name}", f, file_name=name)
