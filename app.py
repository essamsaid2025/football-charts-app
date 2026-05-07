import os
import tempfile
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw
from matplotlib.backends.backend_pdf import PdfPages

from charts import (
    load_data,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
    defensive_regains_map,
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
    try:
        from scouting_tools import (
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
        )
        recommendation_text = None
    except Exception:
        ROLE_TEMPLATES = None
        recommendation_text = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


st.set_page_config(
    page_title="Football Charts Generator",
    layout="wide",
    initial_sidebar_state="expanded",
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


        /* Dark sidebar fix */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #050b16 0%, #081426 100%) !important;
            border-right: 1px solid #243041 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {
            color: #f8fafc !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            background: rgba(15,23,42,0.72) !important;
            border-radius: 12px !important;
            padding: 6px 8px !important;
            margin-bottom: 6px !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(30,41,59,0.95) !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background: #0f172a !important;
            color: #f8fafc !important;
            border: 1px solid #334155 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SCOUTING MODULE
# =========================================================
def run_player_scouting_module():
    if ROLE_TEMPLATES is None:
        st.error("scouting_tools.py is missing. Upload scouting_tools.py beside app.py in the same GitHub repo.")
        st.stop()

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🧠 Player Scouting</div>
            <div class="app-subtitle">Role profiles • Position templates • Player comparison • Auto insights generator</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 📁 Scouting Data")
        uploaded_scouting = st.file_uploader(
            "Upload player scouting file",
            type=["csv", "xlsx", "xls"],
            key="scouting_file_uploader",
        )
        st.download_button(
            "⬇️ Download Scouting Template",
            data=make_template_csv(),
            file_name="scouting_template.csv",
            mime="text/csv",
            key="download_scouting_template",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Scouting Settings")
        min_minutes_global = st.number_input(
            "Default minimum minutes",
            min_value=0,
            value=300,
            step=50,
            key="scouting_min_minutes_global",
        )
        compare_by_position = st.checkbox(
            "Compare percentiles within same position",
            value=True,
            key="scouting_compare_by_position",
        )

    if uploaded_scouting is None:
        st.info("Upload a player scouting file to start. You can use the scouting template from the sidebar.")
        st.stop()

    try:
        df_raw = load_player_data(uploaded_scouting)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if df_raw.empty:
        st.error("The uploaded file is empty.")
        st.stop()

    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    cols = standard_columns(df_raw)

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🧩 Column Mapping")
        all_cols = [None] + df_raw.columns.tolist()
        player_col = st.selectbox("Player column", all_cols, index=all_cols.index(cols["player"]) if cols["player"] in all_cols else 0, key="scout_player_col")
        team_col = st.selectbox("Team column", all_cols, index=all_cols.index(cols["team"]) if cols["team"] in all_cols else 0, key="scout_team_col")
        position_col = st.selectbox("Position column", all_cols, index=all_cols.index(cols["position"]) if cols["position"] in all_cols else 0, key="scout_position_col")
        age_col = st.selectbox("Age column", all_cols, index=all_cols.index(cols["age"]) if cols["age"] in all_cols else 0, key="scout_age_col")
        minutes_col = st.selectbox("Minutes column", all_cols, index=all_cols.index(cols["minutes"]) if cols["minutes"] in all_cols else 0, key="scout_minutes_col")
        value_col = st.selectbox("Market value column", all_cols, index=all_cols.index(cols["market_value"]) if cols["market_value"] in all_cols else 0, key="scout_value_col")

    if not player_col:
        st.error("Select the player column first.")
        st.stop()

    base_exclude = [player_col, team_col, position_col, age_col, minutes_col, value_col]
    metric_cols = numeric_metrics(df_raw, base_exclude)
    df = coerce_numeric(df_raw, metric_cols + [c for c in [age_col, minutes_col, value_col] if c])

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        if position_col:
            positions = sorted(df[position_col].dropna().astype(str).unique().tolist())
            selected_positions = st.multiselect("Filter positions", positions, default=[], key="scout_filter_positions")
        else:
            selected_positions = []
    with f2:
        if team_col:
            teams = sorted(df[team_col].dropna().astype(str).unique().tolist())
            selected_teams = st.multiselect("Filter teams", teams, default=[], key="scout_filter_teams")
        else:
            selected_teams = []
    with f3:
        min_minutes = st.number_input("Min minutes", min_value=0, value=int(min_minutes_global), step=50, key="scout_min_minutes")
    with f4:
        max_age = st.number_input("Max age optional", min_value=0, value=0, step=1, help="0 means no age filter", key="scout_max_age")

    df_filtered = df.copy()
    if selected_positions and position_col:
        df_filtered = df_filtered[df_filtered[position_col].astype(str).isin(selected_positions)].copy()
    if selected_teams and team_col:
        df_filtered = df_filtered[df_filtered[team_col].astype(str).isin(selected_teams)].copy()
    if minutes_col:
        df_filtered = df_filtered[pd.to_numeric(df_filtered[minutes_col], errors="coerce").fillna(0) >= min_minutes].copy()
    if max_age > 0 and age_col:
        df_filtered = df_filtered[pd.to_numeric(df_filtered[age_col], errors="coerce") <= max_age].copy()

    if df_filtered.empty:
        st.warning("No players match current filters.")
        st.stop()

    role_options = list(ROLE_TEMPLATES.keys())
    role = st.selectbox(
        "🎯 Choose role / position template",
        role_options,
        index=role_options.index("Winger") if "Winger" in role_options else 0,
        key="scout_role_template",
    )
    matched_metrics, missing_template_metrics, metric_mapping = match_template_metrics(df_filtered, role)

    default_metrics = matched_metrics[:12] if matched_metrics else metric_cols[:8]
    custom_metrics = st.multiselect(
        "Metrics used for score/comparison",
        metric_cols,
        default=default_metrics,
        key="scout_custom_metrics",
    )

    if not custom_metrics:
        st.error("Choose at least one numeric metric.")
        st.stop()

    group_col = position_col if compare_by_position and position_col else None
    df_scored = add_percentiles_and_score(df_filtered, custom_metrics, group_col=group_col)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="small-kpi"><div class="label">Players</div><div class="value">{len(df_scored)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="small-kpi"><div class="label">Metrics</div><div class="value">{len(custom_metrics)}</div></div>', unsafe_allow_html=True)
    with k3:
        avg_score = df_scored["Scouting Score"].mean()
        st.markdown(f'<div class="small-kpi"><div class="label">Avg Score</div><div class="value">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
    with k4:
        top_name = df_scored.sort_values("Scouting Score", ascending=False).iloc[0][player_col]
        st.markdown(f'<div class="small-kpi"><div class="label">Top Player</div><div class="value" style="font-size:1rem;">{top_name}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧬 Role Profile", "⚔️ Player Comparison", "💡 Auto Insights", "⭐ Shortlist", "📋 Template Check"])

    with tab1:
        players = sorted(df_scored[player_col].dropna().astype(str).unique().tolist())
        selected_player = st.selectbox("Choose player", players, key="scouting_profile_player")
        prof = player_profile(df_scored, player_col, selected_player, custom_metrics, top_n=5)

        c1, c2 = st.columns([0.9, 1.4])
        with c1:
            score = prof.get("score")
            label = prof.get("label", "")
            st.markdown(f"""
            <div class="panel-card">
              <div class="panel-title">{selected_player}</div>
              <div class="panel-note">Role: {role}</div>
              <div style="font-size:2rem;font-weight:900;color:white;">{'NA' if pd.isna(score) else f'{score:.1f}/100'}</div>
              <div style="font-size:1.05rem;font-weight:800;color:white;">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            st.text_area("Auto profile text", profile_text(selected_player, prof), height=160, key="scout_profile_text")

        with c2:
            st.subheader("Strengths")
            strengths_df = pd.DataFrame(prof.get("strengths", []), columns=["Metric", "Percentile"])
            st.dataframe(strengths_df, use_container_width=True, hide_index=True)
            st.subheader("Weaknesses / Watch-outs")
            weaknesses_df = pd.DataFrame(prof.get("weaknesses", []), columns=["Metric", "Percentile"])
            st.dataframe(weaknesses_df, use_container_width=True, hide_index=True)

        st.subheader("Similar Players")
        sim = similar_players(df_scored, player_col, selected_player, custom_metrics, top_n=7)
        show_cols = [player_col]
        for c in [team_col, position_col, age_col, minutes_col, "Scouting Score", "Similarity Distance"]:
            if c and c in sim.columns and c not in show_cols:
                show_cols.append(c)
        st.dataframe(sim[show_cols], use_container_width=True, hide_index=True)

    with tab2:
        players = sorted(df_scored[player_col].dropna().astype(str).unique().tolist())
        c1, c2, c3 = st.columns([1, 1, 1.2])
        with c1:
            p1 = st.selectbox("Player A", players, index=0, key="scout_compare_p1")
        with c2:
            default_idx = 1 if len(players) > 1 else 0
            p2 = st.selectbox("Player B", players, index=default_idx, key="scout_compare_p2")
        with c3:
            chart_type = st.radio("Chart", ["Bar", "Radar"], horizontal=True, key="scout_compare_chart")

        compare_metrics = st.multiselect("Comparison metrics", custom_metrics, default=custom_metrics[:8], key="scout_compare_metrics")
        if p1 == p2:
            st.warning("Choose two different players.")
        elif compare_metrics:
            try:
                if chart_type == "Bar":
                    fig = comparison_chart(df_scored, player_col, p1, p2, compare_metrics, use_percentiles=True)
                else:
                    fig = radar_chart(df_scored, player_col, p1, p2, compare_metrics)
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))

            rows = df_scored[df_scored[player_col].astype(str).isin([str(p1), str(p2)])].copy()
            table_cols = [player_col]
            for c in [team_col, position_col, age_col, minutes_col, "Scouting Score"] + compare_metrics:
                if c and c in rows.columns and c not in table_cols:
                    table_cols.append(c)
            st.dataframe(rows[table_cols], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Dataset Insights")
        insights = auto_dataset_insights(df_scored, player_col, custom_metrics, team_col=team_col, age_col=age_col, minutes_col=minutes_col)
        if insights:
            for i, ins in enumerate(insights, start=1):
                st.markdown(f"**{i}.** {ins}")
        else:
            st.info("Not enough data to generate insights.")

        st.subheader("Top Ranked Players")
        top_cols = [player_col]
        for c in [team_col, position_col, age_col, minutes_col, value_col, "Scouting Score"]:
            if c and c in df_scored.columns and c not in top_cols:
                top_cols.append(c)
        st.dataframe(df_scored.sort_values("Scouting Score", ascending=False)[top_cols].head(25), use_container_width=True, hide_index=True)

    with tab4:
        st.subheader("Shortlist Builder")
        if "shortlist" not in st.session_state:
            st.session_state.shortlist = []

        players = sorted(df_scored[player_col].dropna().astype(str).unique().tolist())
        c1, c2, c3 = st.columns([1, 1, 1.3])
        with c1:
            add_player = st.selectbox("Player", players, key="shortlist_player")
        with c2:
            status = st.selectbox("Status", ["Watch", "Follow", "Recommend", "Reject"], key="shortlist_status")
        with c3:
            note = st.text_input("Scout note", value="", key="shortlist_note")

        if st.button("Add to shortlist", key="add_to_shortlist"):
            row = df_scored[df_scored[player_col].astype(str) == str(add_player)].iloc[0]
            item = {
                "Player": add_player,
                "Team": row.get(team_col, "") if team_col else "",
                "Position": row.get(position_col, "") if position_col else "",
                "Age": row.get(age_col, "") if age_col else "",
                "Minutes": row.get(minutes_col, "") if minutes_col else "",
                "Scouting Score": row.get("Scouting Score", ""),
                "Status": status,
                "Note": note,
            }
            st.session_state.shortlist.append(item)
            st.success("Added to shortlist.")

        shortlist_df = pd.DataFrame(st.session_state.shortlist)
        if not shortlist_df.empty:
            st.dataframe(shortlist_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Export shortlist CSV",
                data=shortlist_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="scouting_shortlist.csv",
                mime="text/csv",
                key="download_shortlist_csv",
            )
        else:
            st.info("No players in shortlist yet.")

    with tab5:
        st.subheader("Position Template Check")
        st.write("Matched metrics in your file:")
        if metric_mapping:
            st.dataframe(pd.DataFrame([{"Template Metric": k, "File Column": v} for k, v in metric_mapping.items()]), use_container_width=True, hide_index=True)
        else:
            st.warning("No template metrics matched your file columns. Rename columns or choose custom metrics manually.")

        st.write("Missing recommended metrics:")
        if missing_template_metrics:
            st.dataframe(pd.DataFrame({"Missing Metric": missing_template_metrics}), use_container_width=True, hide_index=True)
        else:
            st.success("All recommended template metrics were found.")

        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(50), use_container_width=True)



# =========================================================
# PLOTLY CLICK TAGGING TOOL
# =========================================================
def _tag_theme(theme_name: str) -> dict:
    return THEMES.get(theme_name, THEMES.get("The Athletic Dark", {}))


def _init_tag_state():
    defaults = {
        "tag_events": [],
        "tag_start": None,
        "tag_end": None,
        "tag_last_click": None,
        "tag_click_counter": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _tag_y_max(pitch_mode: str, pitch_width: float) -> float:
    return float(pitch_width if pitch_mode == "rect" else 100.0)


def _add_plotly_pitch_shapes(fig, theme: dict, y_max: float):
    line = theme.get("pitch_lines", "#E6E6E6")
    pitch_col = theme.get("pitch", "#1f5f3b")
    fig.update_layout(
        paper_bgcolor=theme.get("bg", "#0E1117"),
        plot_bgcolor=pitch_col,
        margin=dict(l=10, r=10, t=45, b=10),
        height=650,
        dragmode=False,
        clickmode="event+select",
        showlegend=False,
        font=dict(color=theme.get("text", "#FFFFFF")),
    )
    fig.update_xaxes(range=[-2, 102], showgrid=False, zeroline=False, visible=False, fixedrange=True)
    fig.update_yaxes(range=[-2, y_max + 2], showgrid=False, zeroline=False, visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)

    def add_line(x0, y0, x1, y1, width=2):
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=line, width=width), layer="below")

    def add_rect(x0, y0, x1, y1, width=2):
        fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=line, width=width), fillcolor="rgba(0,0,0,0)", layer="below")

    cy = y_max / 2.0
    add_rect(0, 0, 100, y_max, 2)
    add_line(50, 0, 50, y_max, 2)

    box_w = y_max * (40.32 / 68.0)
    six_w = y_max * (18.32 / 68.0)
    pen_depth = 16.5 / 105.0 * 100.0
    six_depth = 5.5 / 105.0 * 100.0
    add_rect(0, cy - box_w / 2, pen_depth, cy + box_w / 2, 2)
    add_rect(100 - pen_depth, cy - box_w / 2, 100, cy + box_w / 2, 2)
    add_rect(0, cy - six_w / 2, six_depth, cy + six_w / 2, 2)
    add_rect(100 - six_depth, cy - six_w / 2, 100, cy + six_w / 2, 2)

    rad_x = 9.15 / 105.0 * 100.0
    rad_y = 9.15 / 68.0 * y_max
    fig.add_shape(type="circle", x0=50-rad_x, y0=cy-rad_y, x1=50+rad_x, y1=cy+rad_y, line=dict(color=line, width=2), fillcolor="rgba(0,0,0,0)", layer="below")
    fig.add_trace(go.Scatter(x=[50, 11/105*100, 100-11/105*100], y=[cy, cy, cy], mode="markers", marker=dict(size=5, color=line), hoverinfo="skip"))

    goal_w = y_max * (7.32 / 68.0)
    add_line(0, cy-goal_w/2, -1.2, cy-goal_w/2, 2)
    add_line(0, cy+goal_w/2, -1.2, cy+goal_w/2, 2)
    add_line(100, cy-goal_w/2, 101.2, cy-goal_w/2, 2)
    add_line(100, cy+goal_w/2, 101.2, cy+goal_w/2, 2)


def _event_color(event_type: str) -> str:
    colors = {
        "pass": "#00C2FF",
        "carry": "#FF9300",
        "dribble": "#A78BFA",
        "cross": "#FFD400",
        "shot": "#00FF6A",
        "touch": "#FFFFFF",
        "defensive action": "#FF4D4D",
        "recovery": "#EF4444",
    }
    return colors.get(str(event_type).lower(), "#FFFFFF")



def _pitch_to_img(x, y, w, h, y_max):
    return int(round((float(x) / 100.0) * w)), int(round(h - (float(y) / float(y_max)) * h))


def _draw_arrow(draw, start, end, color, width=5):
    import math
    x1, y1 = start
    x2, y2 = end
    draw.line([start, end], fill=color, width=width)
    ang = math.atan2(y2 - y1, x2 - x1)
    size = 15
    a1 = ang + math.pi * 0.82
    a2 = ang - math.pi * 0.82
    p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
    p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
    draw.polygon([(x2, y2), p1, p2], fill=color)


def _draw_point(draw, x, y, w, h, y_max, fill, outline="#FFFFFF", label=None, r=9):
    px, py = _pitch_to_img(x, y, w, h, y_max)
    draw.ellipse([px-r, py-r, px+r, py+r], fill=fill, outline=outline, width=3)
    if label:
        draw.text((px + 10, py - 18), str(label), fill=outline)


def make_click_pitch_image(theme_name, pitch_mode, pitch_width, display_w=900):
    theme = _tag_theme(theme_name)
    y_max = _tag_y_max(pitch_mode, pitch_width)
    display_h = int(round(display_w * y_max / 100.0))
    bg = theme.get("pitch", "#1f5f3b")
    line = theme.get("pitch_lines", "#E6E6E6")
    img = Image.new("RGB", (display_w, display_h), bg)
    draw = ImageDraw.Draw(img)

    def P(x, y):
        return _pitch_to_img(x, y, display_w, display_h, y_max)

    lw = 3
    # outer pitch
    draw.rectangle([P(0, 0), P(100, y_max)], outline=line, width=lw)
    # halfway
    draw.line([P(50, 0), P(50, y_max)], fill=line, width=lw)
    # center circle / spot
    cx, cy = P(50, y_max / 2)
    rx = int(display_w * 8.7 / 100)
    ry = int(display_h * 8.7 / y_max)
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=line, width=lw)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=line)

    # boxes approximate on 100 x y_max pitch
    pen_len = 15.8
    six_len = 5.3
    pen_w = y_max * 40.3 / 68.0
    six_w = y_max * 18.3 / 68.0
    mid = y_max / 2
    # left / right penalty boxes
    draw.rectangle([P(0, mid - pen_w/2), P(pen_len, mid + pen_w/2)], outline=line, width=lw)
    draw.rectangle([P(100-pen_len, mid - pen_w/2), P(100, mid + pen_w/2)], outline=line, width=lw)
    draw.rectangle([P(0, mid - six_w/2), P(six_len, mid + six_w/2)], outline=line, width=lw)
    draw.rectangle([P(100-six_len, mid - six_w/2), P(100, mid + six_w/2)], outline=line, width=lw)
    # penalty spots
    for sx in (11, 89):
        px, py = P(sx, mid)
        draw.ellipse([px-3, py-3, px+3, py+3], fill=line)

    # saved events
    if st.session_state.get("tag_events"):
        d = pd.DataFrame(st.session_state.tag_events)
        for _, r in d.iterrows():
            try:
                et = str(r.get("event_type", "")).lower()
                col = _event_color(et)
                x, y = float(r.get("x", 0)), float(r.get("y", 0))
                x2, y2 = r.get("x2"), r.get("y2")
                if et in ["pass", "carry", "dribble", "cross"] and pd.notna(x2) and pd.notna(y2):
                    _draw_arrow(draw, P(x, y), P(float(x2), float(y2)), col, width=5)
                    _draw_point(draw, x, y, display_w, display_h, y_max, col, r=6)
                else:
                    _draw_point(draw, x, y, display_w, display_h, y_max, col, r=9)
            except Exception:
                pass

    # current selected points
    if st.session_state.get("tag_start"):
        sx, sy = st.session_state.tag_start
        _draw_point(draw, sx, sy, display_w, display_h, y_max, "#22C55E", label="START", r=10)
    if st.session_state.get("tag_end"):
        ex, ey = st.session_state.tag_end
        _draw_point(draw, ex, ey, display_w, display_h, y_max, "#EF4444", label="END", r=10)
    if st.session_state.get("tag_start") and st.session_state.get("tag_end"):
        sx, sy = st.session_state.tag_start
        ex, ey = st.session_state.tag_end
        _draw_arrow(draw, P(sx, sy), P(ex, ey), "#22C55E", width=4)
    if st.session_state.get("tag_last_click"):
        cx, cy = st.session_state.tag_last_click
        _draw_point(draw, cx, cy, display_w, display_h, y_max, "#FFFFFF", outline="#EF4444", label="CLICK", r=9)

    return img, display_w, display_h, y_max


def make_click_pitch(theme_name, pitch_mode, pitch_width):
    theme = _tag_theme(theme_name)
    y_max = _tag_y_max(pitch_mode, pitch_width)
    fig = go.Figure()
    _add_plotly_pitch_shapes(fig, theme, y_max)

    # saved events
    if st.session_state.tag_events:
        d = pd.DataFrame(st.session_state.tag_events)
        for _, r in d.iterrows():
            et = str(r.get("event_type", "")).lower()
            col = _event_color(et)
            x, y = float(r.get("x", 0)), float(r.get("y", 0))
            x2, y2 = r.get("x2"), r.get("y2")
            if et in ["pass", "carry", "dribble", "cross"] and pd.notna(x2) and pd.notna(y2):
                fig.add_trace(go.Scatter(x=[x, float(x2)], y=[y, float(y2)], mode="lines+markers", line=dict(color=col, width=4), marker=dict(size=8, color=col), hoverinfo="skip"))
                fig.add_annotation(x=float(x2), y=float(y2), ax=x, ay=y, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor=col)
            else:
                sym = "star" if et == "shot" else "square" if "defensive" in et else "circle"
                fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", marker=dict(size=15, symbol=sym, color=col, line=dict(color=theme.get("bg", "#000"), width=1.5)), hoverinfo="skip"))

    # current chosen points
    if st.session_state.tag_start:
        sx, sy = st.session_state.tag_start
        fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text", text=["START"], textposition="top center", marker=dict(size=17, color="#22C55E", line=dict(color="#FFFFFF", width=2)), hoverinfo="skip"))
    if st.session_state.tag_end:
        ex, ey = st.session_state.tag_end
        fig.add_trace(go.Scatter(x=[ex], y=[ey], mode="markers+text", text=["END"], textposition="top center", marker=dict(size=17, color="#EF4444", line=dict(color="#FFFFFF", width=2)), hoverinfo="skip"))
    if st.session_state.tag_start and st.session_state.tag_end:
        sx, sy = st.session_state.tag_start
        ex, ey = st.session_state.tag_end
        fig.add_trace(go.Scatter(x=[sx, ex], y=[sy, ey], mode="lines", line=dict(color="#22C55E", width=4, dash="dash"), hoverinfo="skip"))
    if st.session_state.tag_last_click:
        cx, cy = st.session_state.tag_last_click
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode="markers+text", text=["CLICK"], textposition="bottom center", marker=dict(size=15, color="#FFFFFF", line=dict(color="#EF4444", width=2)), hoverinfo="skip"))

    # IMPORTANT: visible-enough click grid. This is what captures the click.
    xs = np.arange(0, 100.01, 1.0)
    ys = np.arange(0, y_max + 0.01, 1.0)
    gx, gy = np.meshgrid(xs, ys)
    fig.add_trace(go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers", name="Click layer",
        marker=dict(size=9, color="rgba(255,255,255,0.001)", line=dict(width=0)),
        hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(title=dict(text="Click anywhere on the pitch", x=0.5, font=dict(color=theme.get("text", "#FFFFFF"), size=18)))
    return fig


def _events_dataframe():
    cols = ["event_id", "event_type", "player", "team", "minute", "x", "y", "x2", "y2", "outcome", "tag", "note"]
    if not st.session_state.tag_events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(st.session_state.tag_events).reindex(columns=cols)


def _save_tagged_map(df_events: pd.DataFrame, theme_name: str, pitch_mode: str, pitch_width: float, title: str):
    theme = _tag_theme(theme_name)
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme, vertical_pitch=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(theme.get("bg", "#0E1117"))
    pitch.draw(ax=ax)
    ax.set_facecolor(theme.get("pitch", "#1f5f3b"))
    y_max = _tag_y_max(pitch_mode, pitch_width)

    for _, r in df_events.iterrows():
        et = str(r.get("event_type", "")).lower()
        col = _event_color(et)
        x, y = float(r.get("x", 0)), float(r.get("y", 0))
        x2, y2 = r.get("x2"), r.get("y2")
        if et in ["pass", "carry", "dribble", "cross"] and pd.notna(x2) and pd.notna(y2):
            pitch.arrows([x], [y], [float(x2)], [float(y2)], ax=ax, color=col, width=2.4, headwidth=6, headlength=6, alpha=.9)
            pitch.scatter([x], [y], ax=ax, s=70, color=col, edgecolors="white", linewidth=1.2, zorder=4)
        else:
            marker = "*" if et == "shot" else "s" if "defensive" in et else "o"
            pitch.scatter([x], [y], ax=ax, s=140, marker=marker, color=col, edgecolors="white", linewidth=1.5, zorder=4)

    ax.set_title(title or "Tagged Events Map", color=theme.get("text", "white"), fontsize=18, weight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    return fig


def run_tagging_tool():
    _init_tag_state()
    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🖱️ Interactive Tagging Tool</div>
            <div class="app-subtitle">Click the pitch, set Start / End, save events, export CSV, or generate a themed map.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if streamlit_image_coordinates is None:
        st.error("Missing dependency for click tagging.")
        st.info("Add this to requirements.txt then reboot the app:")
        st.code("streamlit-image-coordinates")
        st.stop()

    control_col, pitch_col = st.columns([1.0, 1.65], gap="large")
    with control_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Tag Settings")
        tag_theme = st.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index("The Athletic Dark") if "The Athletic Dark" in THEMES else 0, key="tag_theme")
        tag_pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular", "Square"], index=0, key="tag_pitch_mode_ui")
        tag_pitch_mode = "rect" if tag_pitch_mode_ui == "Rectangular" else "square"
        tag_pitch_width = st.slider("Pitch width", 50.0, 80.0, 64.0, 1.0, key="tag_pitch_width") if tag_pitch_mode == "rect" else 100.0
        event_type = st.selectbox("Event type", ["pass", "carry", "dribble", "cross", "shot", "touch", "defensive action", "recovery"], key="tag_event_type")
        outcome = st.selectbox("Outcome", ["successful", "unsuccessful", "key pass", "assist", "goal", "ontarget", "off target", "blocked", "touch"], key="tag_outcome")
        player = st.text_input("Player", key="tag_player")
        team = st.text_input("Team", key="tag_team")
        minute = st.number_input("Minute", min_value=0, max_value=130, value=0, step=1, key="tag_minute")
        action_tag = st.selectbox("Action tag", ["", "progressive", "line breaking", "into final third", "into box", "key pass", "chance created", "under pressure", "turnover", "recovery", "duel", "dangerous action"], key="tag_action_tag")
        note = st.text_area("Note", height=70, key="tag_note")

        st.markdown("### Selected click")
        if st.session_state.tag_last_click:
            st.success(f"Clicked: x={st.session_state.tag_last_click[0]:.1f}, y={st.session_state.tag_last_click[1]:.1f}")
        else:
            st.info("Click on the pitch first.")

        c1, c2 = st.columns(2)
        needs_end = event_type in ["pass", "carry", "dribble", "cross"]
        with c1:
            if st.button("Use click as START", disabled=st.session_state.tag_last_click is None):
                st.session_state.tag_start = st.session_state.tag_last_click
                if not needs_end:
                    st.session_state.tag_end = None
                st.rerun()
        with c2:
            if st.button("Use click as END", disabled=(st.session_state.tag_last_click is None or not needs_end)):
                st.session_state.tag_end = st.session_state.tag_last_click
                st.rerun()

        if st.button("Save Event"):
            start = st.session_state.tag_start
            end = st.session_state.tag_end
            if start is None:
                st.warning("Choose START first.")
            elif needs_end and end is None:
                st.warning("This event needs END point too.")
            else:
                new_id = len(st.session_state.tag_events) + 1
                item = {
                    "event_id": new_id,
                    "event_type": event_type,
                    "player": player,
                    "team": team,
                    "minute": int(minute),
                    "x": round(float(start[0]), 2),
                    "y": round(float(start[1]), 2),
                    "x2": round(float(end[0]), 2) if end else np.nan,
                    "y2": round(float(end[1]), 2) if end else np.nan,
                    "outcome": outcome,
                    "tag": action_tag,
                    "note": note,
                }
                st.session_state.tag_events.append(item)
                st.session_state.tag_start = None
                st.session_state.tag_end = None
                st.session_state.tag_last_click = None
                st.success("Event saved.")
                st.rerun()

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Undo Last") and st.session_state.tag_events:
                st.session_state.tag_events.pop()
                st.rerun()
        with c4:
            if st.button("Clear All"):
                st.session_state.tag_events = []
                st.session_state.tag_start = None
                st.session_state.tag_end = None
                st.session_state.tag_last_click = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with pitch_col:
        st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
        pitch_img, img_w, img_h, y_max = make_click_pitch_image(tag_theme, tag_pitch_mode, tag_pitch_width, display_w=950)
        coords = streamlit_image_coordinates(
            pitch_img,
            key=f"tag_image_pitch_{tag_theme}_{tag_pitch_mode}_{tag_pitch_width}_{st.session_state.tag_click_counter}",
        )
        if coords and coords.get("x") is not None and coords.get("y") is not None:
            try:
                x = round((float(coords.get("x")) / float(img_w)) * 100.0, 2)
                y = round(y_max - ((float(coords.get("y")) / float(img_h)) * y_max), 2)
                if 0 <= x <= 100 and 0 <= y <= y_max:
                    new_click = (x, y)
                    if st.session_state.tag_last_click != new_click:
                        st.session_state.tag_last_click = new_click
                        st.session_state.tag_click_counter += 1
                        st.rerun()
            except Exception:
                pass

        df_events = _events_dataframe()
        st.subheader("Tagged Events")
        st.dataframe(df_events, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Tagged Events CSV", data=df_events.to_csv(index=False).encode("utf-8-sig"), file_name="tagged_events.csv", mime="text/csv", disabled=df_events.empty)

        if st.button("Generate Map From Tagged Events", disabled=df_events.empty):
            map_fig = _save_tagged_map(df_events, tag_theme, tag_pitch_mode, tag_pitch_width, "Tagged Events Map")
            png = io.BytesIO()
            map_fig.savefig(png, format="png", dpi=300, bbox_inches="tight", pad_inches=.25)
            png.seek(0)
            st.image(png.getvalue(), use_container_width=True)
            st.download_button("⬇️ Download Tagged Map PNG", data=png.getvalue(), file_name="tagged_events_map.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    app_section = st.radio(
        "Choose App Section",
        ["Charts Generator", "Player Scouting", "Tagging Tool"],
        index=0,
        key="app_section_selector",
    )

if app_section == "Player Scouting":
    run_player_scouting_module()
    st.stop()

if app_section == "Tagging Tool":
    run_tagging_tool()
    st.stop()

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">⚽ Football Charts Generator</div>
        <div class="app-subtitle">
            Upload CSV / Excel → Match Report / Pizza / Shot Card / Defensive Map
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


def _norm_col_name(c):
    return str(c).strip().lower()


def find_column_case_insensitive(df: pd.DataFrame, names: list[str]):
    lookup = {_norm_col_name(c): c for c in df.columns}
    for n in names:
        key = _norm_col_name(n)
        if key in lookup:
            return lookup[key]
    return None


def looks_like_pizza_file(df: pd.DataFrame) -> bool:
    player_col = find_column_case_insensitive(df, ["player", "player name", "name"])
    if player_col is None:
        return False
    core_missing = [c for c in ["outcome", "x", "y"] if find_column_case_insensitive(df, [c]) is None]
    numeric_cols = []
    for c in df.columns:
        if c == player_col:
            continue
        ser = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        if ser.notna().sum() >= 2:
            numeric_cols.append(c)
    return len(core_missing) >= 2 and len(numeric_cols) >= 2


def clean_metric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace("%", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.strip(),
        errors="coerce"
    )


def ensure_match_chart_columns_or_stop(df: pd.DataFrame):
    required = ["outcome", "x", "y"]
    missing = [c for c in required if find_column_case_insensitive(df, [c]) is None]
    if not missing:
        return

    if looks_like_pizza_file(df):
        st.error("This file looks like a Pizza Chart / player metrics file, not an event-data file.")
        st.info("Change **Choose output type** from the left panel to **Pizza Chart**. This file has Player + metrics, but it does not have outcome/x/y columns needed for Match Charts.")
    else:
        st.error(f"Missing columns: {missing}. Match Charts need at least: outcome, x, y.")
        st.info("For Pizza Chart use a player metrics file with a Player column and numeric metric columns. For Match Charts use event data with x/y locations and outcome.")
    with st.expander("Uploaded file columns", expanded=True):
        st.write(list(df.columns))
    st.stop()


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
        ["Match Charts", "Pizza Chart", "Shot Detail Card", "Defensive Actions Map"],
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
    ]
    selected_charts = st.multiselect("Choose charts", all_charts, default=all_charts)

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
    col_shot_off = st.color_picker("Off target", "#FF8A00")
    col_shot_on = st.color_picker("On target", "#00C2FF")
    col_shot_goal = st.color_picker("Goal", "#00FF6A")
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
    mk_shot_off = st.selectbox("Marker: Off target", marker_labels, index=marker_labels.index("Triangle up (^)"))
    mk_shot_on = st.selectbox("Marker: On target", marker_labels, index=marker_labels.index("Diamond (D)"))
    mk_shot_goal = st.selectbox("Marker: Goal", marker_labels, index=marker_labels.index("Star (*)"))
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

            cols_lower = {str(c).strip().lower(): c for c in dfp.columns}
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

            metric_cols = []
            for c in dfp_filtered.columns:
                if c in exclude:
                    continue
                converted = clean_metric_series(dfp_filtered[c])
                if converted.notna().sum() >= 2:
                    dfp_filtered[c] = converted
                    metric_cols.append(c)

            default_n = min(8, len(metric_cols))
            selected_metrics = st.multiselect("Choose metrics", metric_cols, default=metric_cols[:default_n])

            if not selected_metrics:
                st.warning("Choose at least one numeric metric to generate the Pizza Chart.")

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

            if st.button("Generate Pizza", key="generate_pizza", disabled=(not selected_metrics)):
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

        df_raw = load_data(path)

        with st.expander("Raw file preview (first 25 rows)", expanded=False):
            st.write("Columns:", list(df_raw.columns))
            st.dataframe(df_raw.head(25), use_container_width=True)

        ensure_match_chart_columns_or_stop(df_raw)

        if "outcome" not in df_raw.columns:
            st.warning("Column `outcome` not found. Trying to derive it automatically.")
        df_raw = ensure_outcome_column(df_raw)
        df_raw = normalize_outcome_values(df_raw)

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
