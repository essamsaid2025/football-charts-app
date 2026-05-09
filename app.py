import os
import tempfile
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


        /* FINAL FIX: force Streamlit sidebar dark + readable controls */
        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg,#050b16 0%,#081426 100%) !important;
            border-right:1px solid #25344a !important;
        }
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        section[data-testid="stSidebar"] [data-testid="stRadio"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{
            background: transparent !important;
            color:#f8fafc !important;
        }
        section[data-testid="stSidebar"] *,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div{
            color:#f8fafc !important;
            opacity:1 !important;
        }
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div{
            background:#0b1728 !important;
            color:#f8fafc !important;
            border-color:#334155 !important;
        }
        div[role="radiogroup"] label p,
        div[role="radiogroup"] label span{
            color:#f8fafc !important;
        }

        /* Tagging sidebar slider readability fix */
        section[data-testid="stSidebar"] [data-testid="stSlider"] *,
        section[data-testid="stSidebar"] [data-testid="stSlider"] label,
        section[data-testid="stSidebar"] [data-testid="stSlider"] p,
        section[data-testid="stSidebar"] [data-testid="stSlider"] span {
            color:#f8fafc !important;
            opacity:1 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-baseweb="slider"] {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
            background:#38bdf8 !important;
            border:2px solid #f8fafc !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] input {
            background:#0b1728 !important;
            color:#f8fafc !important;
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
        st.error("scouting_tools_v2.py is missing. Upload scouting_tools_v2.py beside app.py in the same GitHub repo.")
        st.stop()

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">🧠 Player Scouting</div>
            <div class="app-subtitle">Clean scoring • Role templates • Percentiles • Similar players • Shortlist • Data quality checks</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def _clean_txt(x):
        return str(x).strip().lower().replace("_", " ").replace("-", " ")

    def _infer_role_from_pos(pos):
        p = _clean_txt(pos)
        pos_map = {
            "gk": "Goalkeeper", "goalkeeper": "Goalkeeper",
            "cb": "Centre Back", "rcb": "Centre Back", "lcb": "Centre Back", "centre back": "Centre Back", "center back": "Centre Back",
            "rb": "Full Back / Wing Back", "lb": "Full Back / Wing Back", "rwb": "Full Back / Wing Back", "lwb": "Full Back / Wing Back", "full back": "Full Back / Wing Back",
            "dm": "Defensive Midfielder", "cdm": "Defensive Midfielder", "defensive midfielder": "Defensive Midfielder",
            "cm": "Central Midfielder", "mc": "Central Midfielder", "central midfielder": "Central Midfielder",
            "am": "Attacking Midfielder", "cam": "Attacking Midfielder", "attacking midfielder": "Attacking Midfielder",
            "rw": "Winger", "lw": "Winger", "winger": "Winger", "wide midfielder": "Winger",
            "st": "Striker", "cf": "Striker", "striker": "Striker", "centre forward": "Striker", "center forward": "Striker",
        }
        if p in pos_map:
            return pos_map[p]
        for k, v in pos_map.items():
            if k in p:
                return v
        return "Winger" if "Winger" in ROLE_TEMPLATES else list(ROLE_TEMPLATES.keys())[0]

    def _safe_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")

    def _pct_rank(s, higher=True):
        x = pd.to_numeric(s, errors="coerce")
        pct = x.rank(pct=True, method="average") * 100
        if not higher:
            pct = 100 - pct
        return pct.clip(0, 100)

    def _make_scored(df_in, metrics, lower_better, group_col=None, minutes_col=None, minutes_floor=900, reliability=True, weights=None):
        out = df_in.copy()
        pct_cols = []
        weights = weights or {}
        for m in metrics:
            if m not in out.columns:
                continue
            out[m] = _safe_numeric(out[m])
            pc = f"pct__{m}"
            higher = m not in lower_better
            if group_col and group_col in out.columns:
                out[pc] = out.groupby(group_col, dropna=False)[m].transform(lambda x: _pct_rank(x, higher))
            else:
                out[pc] = _pct_rank(out[m], higher)
            pct_cols.append(pc)
        if not pct_cols:
            out["Scouting Score"] = np.nan
            out["Reliability"] = np.nan
            out["Adjusted Score"] = np.nan
            return out
        w = np.array([float(weights.get(c.replace("pct__", ""), 1.0)) for c in pct_cols], dtype=float)
        w = np.where(np.isfinite(w), w, 1.0)
        mat = out[pct_cols].astype(float)
        out["Scouting Score"] = mat.mul(w, axis=1).sum(axis=1) / mat.notna().mul(w, axis=1).sum(axis=1).replace(0, np.nan)
        out["Scouting Score"] = out["Scouting Score"].round(1)
        if reliability and minutes_col and minutes_col in out.columns:
            mins = pd.to_numeric(out[minutes_col], errors="coerce").fillna(0)
            out["Reliability"] = (mins / float(max(minutes_floor, 1))).clip(0, 1).round(2)
            out["Adjusted Score"] = (out["Scouting Score"] * (0.65 + 0.35 * out["Reliability"])).round(1)
        else:
            out["Reliability"] = 1.0
            out["Adjusted Score"] = out["Scouting Score"]
        return out

    def _label(score):
        if pd.isna(score): return "Not enough data"
        if score >= 82: return "Elite / Priority"
        if score >= 70: return "Strong option"
        if score >= 58: return "Good watchlist"
        if score >= 45: return "Average / needs video"
        return "Risky"

    def _profile_from_row(df_scored, player_name, metrics, score_col="Adjusted Score", top_n=6):
        row = df_scored[df_scored[player_col].astype(str) == str(player_name)]
        if row.empty:
            return {}, pd.DataFrame(), pd.DataFrame()
        r = row.iloc[0]
        pairs = []
        for m in metrics:
            pc = f"pct__{m}"
            if pc in df_scored.columns and pd.notna(r.get(pc)):
                pairs.append({"Metric": m, "Raw Value": r.get(m, np.nan), "Percentile": round(float(r.get(pc)), 1), "Direction": "Lower is better" if m in lower_better else "Higher is better"})
        detail = pd.DataFrame(pairs).sort_values("Percentile", ascending=False)
        strengths = detail.head(top_n).copy()
        concerns = detail.tail(top_n).sort_values("Percentile").copy()
        info = {
            "score": r.get(score_col, np.nan),
            "raw_score": r.get("Scouting Score", np.nan),
            "reliability": r.get("Reliability", np.nan),
            "label": _label(r.get(score_col, np.nan)),
        }
        return info, strengths, concerns

    with st.sidebar:
        st.markdown("### 📁 Scouting Data")
        uploaded_scouting = st.file_uploader("Upload player scouting file", type=["csv", "xlsx", "xls"], key="scouting_file_uploader")
        st.download_button("⬇️ Download Scouting Template", data=make_template_csv(), file_name="scouting_template.csv", mime="text/csv", key="download_scouting_template")
        st.markdown("---")
        st.markdown("### ⚙️ Scouting Settings")
        min_minutes_global = st.number_input("Default minimum minutes", min_value=0, value=300, step=50, key="scouting_min_minutes_global")
        compare_scope = st.radio("Percentile comparison scope", ["Filtered players", "Same position only"], index=1, key="scouting_compare_scope")
        use_reliability = st.checkbox("Use minutes reliability adjustment", value=True, key="scouting_reliability")
        reliability_floor = st.number_input("Reliable sample = minutes", min_value=1, value=900, step=50, key="scouting_reliability_floor")

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
        selected_positions = st.multiselect("Filter positions", sorted(df[position_col].dropna().astype(str).unique().tolist()), default=[], key="scout_filter_positions") if position_col else []
    with f2:
        selected_teams = st.multiselect("Filter teams", sorted(df[team_col].dropna().astype(str).unique().tolist()), default=[], key="scout_filter_teams") if team_col else []
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
    auto_role = "Winger"
    if position_col and len(df_filtered[position_col].dropna()):
        auto_role = _infer_role_from_pos(df_filtered[position_col].dropna().astype(str).mode().iloc[0])
    role = st.selectbox("🎯 Choose role / position template", role_options, index=role_options.index(auto_role) if auto_role in role_options else 0, key="scout_role_template")
    matched_metrics, missing_template_metrics, metric_mapping = match_template_metrics(df_filtered, role)

    default_metrics = matched_metrics[:14] if matched_metrics else metric_cols[:10]
    custom_metrics = st.multiselect("Metrics used for score/comparison", metric_cols, default=default_metrics, key="scout_custom_metrics")
    if not custom_metrics:
        st.error("Choose at least one numeric metric.")
        st.stop()

    guessed_negative = [m for m in custom_metrics if any(w in _clean_txt(m) for w in ["conceded", "foul", "error", "mistake", "turnover", "dispossessed", "card", "lost"])]
    lower_better = st.multiselect("Lower is better metrics", custom_metrics, default=guessed_negative, help="Important: turnovers, fouls, errors, goals conceded etc should be lower-is-better.", key="scout_lower_better")

    with st.expander("Metric weights (optional)", expanded=False):
        st.caption("1.0 = normal. Raise important role metrics to 1.5/2.0. Set weak/noisy metrics to 0.5.")
        weights = {}
        weight_cols = st.columns(3)
        for i, m in enumerate(custom_metrics):
            with weight_cols[i % 3]:
                weights[m] = st.number_input(str(m), min_value=0.1, max_value=3.0, value=1.0, step=0.1, key=f"weight_{m}")

    group_col = position_col if compare_scope == "Same position only" and position_col else None
    df_scored = _make_scored(df_filtered, custom_metrics, lower_better, group_col=group_col, minutes_col=minutes_col, minutes_floor=reliability_floor, reliability=use_reliability, weights=weights)
    score_col = "Adjusted Score" if use_reliability else "Scouting Score"

    # Basic quality flags
    metric_coverage = df_scored[custom_metrics].notna().mean(axis=1).round(2)
    df_scored["Metric Coverage"] = metric_coverage
    if value_col and value_col in df_scored.columns:
        val = pd.to_numeric(df_scored[value_col], errors="coerce")
        df_scored["Value Score"] = (df_scored[score_col] / np.log10(val.clip(lower=1) + 10)).round(2)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="small-kpi"><div class="label">Players</div><div class="value">{len(df_scored)}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="small-kpi"><div class="label">Metrics</div><div class="value">{len(custom_metrics)}</div></div>', unsafe_allow_html=True)
    with k3:
        avg_score = df_scored[score_col].mean()
        st.markdown(f'<div class="small-kpi"><div class="label">Avg {score_col}</div><div class="value">{avg_score:.1f}</div></div>', unsafe_allow_html=True)
    with k4:
        top_name = df_scored.sort_values(score_col, ascending=False).iloc[0][player_col]
        st.markdown(f'<div class="small-kpi"><div class="label">Top Player</div><div class="value" style="font-size:1rem;">{top_name}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🧬 Player Profile", "⚔️ Compare", "🏆 Ranking", "💡 Insights", "⭐ Shortlist", "📋 Data Check"])

    players = sorted(df_scored[player_col].dropna().astype(str).unique().tolist())

    with tab1:
        selected_player = st.selectbox("Choose player", players, key="scouting_profile_player")
        info, strengths_df, concerns_df = _profile_from_row(df_scored, selected_player, custom_metrics, score_col=score_col, top_n=6)
        row = df_scored[df_scored[player_col].astype(str) == str(selected_player)].iloc[0]
        c1, c2 = st.columns([0.95, 1.45])
        with c1:
            score = info.get("score", np.nan)
            st.markdown(f"""
            <div class="panel-card">
              <div class="panel-title">{selected_player}</div>
              <div class="panel-note">Role: {role}</div>
              <div style="font-size:2rem;font-weight:900;color:white;">{'NA' if pd.isna(score) else f'{float(score):.1f}/100'}</div>
              <div style="font-size:1.05rem;font-weight:800;color:white;">{info.get('label','')}</div>
              <div class="panel-note">Raw score: {row.get('Scouting Score', np.nan)} | Reliability: {row.get('Reliability', np.nan)}</div>
            </div>
            """, unsafe_allow_html=True)
            rec = ""
            if recommendation_text is not None:
                try:
                    rec = recommendation_text(df_scored, player_col, selected_player, custom_metrics, role=role, team_col=team_col, position_col=position_col, age_col=age_col, minutes_col=minutes_col)
                except Exception:
                    rec = ""
            if not rec:
                top_txt = ", ".join([f"{r['Metric']} ({r['Percentile']:.0f}th)" for _, r in strengths_df.head(3).iterrows()])
                weak_txt = ", ".join([f"{r['Metric']} ({r['Percentile']:.0f}th)" for _, r in concerns_df.head(3).iterrows()])
                rec = f"{selected_player}: {info.get('label','')} with {score_col} {'NA' if pd.isna(score) else round(float(score),1)}. Strengths: {top_txt}. Watch-outs: {weak_txt}."
            st.text_area("Scout recommendation", rec, height=210, key="scout_recommendation_text")
        with c2:
            st.subheader("Strengths")
            st.dataframe(strengths_df, use_container_width=True, hide_index=True)
            st.subheader("Weaknesses / Watch-outs")
            st.dataframe(concerns_df, use_container_width=True, hide_index=True)

        st.subheader("Similar Players")
        try:
            sim = similar_players(df_scored, player_col, selected_player, custom_metrics, top_n=8)
            show_cols = [player_col]
            for c in [team_col, position_col, age_col, minutes_col, score_col, "Similarity Distance"]:
                if c and c in sim.columns and c not in show_cols:
                    show_cols.append(c)
            st.dataframe(sim[show_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.info(f"Similar players unavailable: {e}")

    with tab2:
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
                fig = comparison_chart(df_scored, player_col, p1, p2, compare_metrics, use_percentiles=True) if chart_type == "Bar" else radar_chart(df_scored, player_col, p1, p2, compare_metrics)
                st.pyplot(fig, use_container_width=True)
            except Exception as e:
                st.error(str(e))
            rows = df_scored[df_scored[player_col].astype(str).isin([str(p1), str(p2)])].copy()
            table_cols = [player_col]
            for c in [team_col, position_col, age_col, minutes_col, "Scouting Score", "Adjusted Score", "Reliability"] + compare_metrics:
                if c and c in rows.columns and c not in table_cols:
                    table_cols.append(c)
            st.dataframe(rows[table_cols], use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Top Ranked Players")
        top_cols = [player_col]
        for c in [team_col, position_col, age_col, minutes_col, value_col, "Scouting Score", "Adjusted Score", "Reliability", "Metric Coverage", "Value Score"]:
            if c and c in df_scored.columns and c not in top_cols:
                top_cols.append(c)
        st.dataframe(df_scored.sort_values(score_col, ascending=False)[top_cols].head(100), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export scored players CSV", data=df_scored.to_csv(index=False).encode("utf-8-sig"), file_name="scored_players.csv", mime="text/csv", key="download_scored_players")

    with tab4:
        st.subheader("Dataset Insights")
        insights = auto_dataset_insights(df_scored, player_col, custom_metrics, team_col=team_col, age_col=age_col, minutes_col=minutes_col)
        if value_col and "Value Score" in df_scored.columns:
            best_value = df_scored.sort_values("Value Score", ascending=False).head(5)
            names = ", ".join([f"{r[player_col]} ({r['Value Score']})" for _, r in best_value.iterrows() if pd.notna(r.get("Value Score"))])
            if names:
                insights.insert(1, f"Best value targets by score/value ratio: {names}.")
        if insights:
            for i, ins in enumerate(insights, start=1):
                st.markdown(f"**{i}.** {ins}")
        else:
            st.info("Not enough data to generate insights.")

    with tab5:
        st.subheader("Shortlist Builder")
        if "shortlist" not in st.session_state:
            st.session_state.shortlist = []
        c1, c2, c3 = st.columns([1, 1, 1.3])
        with c1:
            add_player = st.selectbox("Player", players, key="shortlist_player")
        with c2:
            status = st.selectbox("Status", ["Watch", "Follow", "Recommend", "Reject"], key="shortlist_status")
        with c3:
            note = st.text_input("Scout note", value="", key="shortlist_note")
        if st.button("Add to shortlist", key="add_to_shortlist"):
            row = df_scored[df_scored[player_col].astype(str) == str(add_player)].iloc[0]
            item = {"Player": add_player, "Score": row.get(score_col, ""), "Status": status, "Note": note}
            for c, label in [(team_col, "Team"), (position_col, "Position"), (age_col, "Age"), (minutes_col, "Minutes")]:
                item[label] = row.get(c, "") if c else ""
            st.session_state.shortlist.append(item)
            st.success("Added to shortlist.")
        shortlist_df = pd.DataFrame(st.session_state.shortlist)
        if not shortlist_df.empty:
            st.dataframe(shortlist_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Export shortlist CSV", data=shortlist_df.to_csv(index=False).encode("utf-8-sig"), file_name="scouting_shortlist.csv", mime="text/csv", key="download_shortlist_csv")
        else:
            st.info("No players in shortlist yet.")

    with tab6:
        st.subheader("Template Match Check")
        if metric_mapping:
            st.dataframe(pd.DataFrame([{"Template Metric": k, "File Column": v} for k, v in metric_mapping.items()]), use_container_width=True, hide_index=True)
        else:
            st.warning("No template metrics matched your file columns. Rename columns or choose custom metrics manually.")
        if missing_template_metrics:
            st.write("Missing recommended metrics:")
            st.dataframe(pd.DataFrame({"Missing Metric": missing_template_metrics}), use_container_width=True, hide_index=True)
        else:
            st.success("All recommended template metrics were found.")
        st.subheader("Metric Data Quality")
        quality_rows = []
        for m in custom_metrics:
            s = pd.to_numeric(df_filtered[m], errors="coerce") if m in df_filtered.columns else pd.Series(dtype=float)
            quality_rows.append({"Metric": m, "Valid Values": int(s.notna().sum()), "Missing": int(s.isna().sum()), "Mean": round(float(s.mean()), 2) if s.notna().any() else np.nan, "Lower Better": m in lower_better})
        st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
        st.subheader("Raw Data Preview")
        st.dataframe(df_raw.head(50), use_container_width=True)


# =========================================================
# PLOTLY CLICK TAGGING TOOL
# =========================================================
# Tagging-only themes. This does not change the Charts Generator logic.
TAGGING_THEMES = dict(THEMES)
TAGGING_THEMES["Opta Analyst Light"] = {
    "bg": "#ECECEC",
    "panel": "#F5F5F5",
    "panel_2": "#E9E9E9",
    "pitch": "#ECECEC",
    "pitch_stripe": None,
    "text": "#201C2B",
    "muted": "#7A7584",
    "lines": "#A7A7A7",
    "goal": "#8F8F8F",
    "pitch_lines": "#9F9F9F",
    "accent": "#6D28D9",
    "accent_2": "#8B5CF6",
    "danger": "#D64045",
    "warning": "#B0B0B0",
    "success": "#22A06B",
    "legend_bg": "#F5F5F5",
    "legend_border": "#B8B8B8",
    "legend_text": "#201C2B",
}

def _tag_theme(theme_name: str) -> dict:
    return TAGGING_THEMES.get(theme_name, TAGGING_THEMES.get("The Athletic Dark", {}))



def _init_tag_state():
    defaults = {
        "tag_events": [],
        "tag_start": None,
        "tag_end": None,
        "tag_last_click": None,
        "tag_click_counter": 0,
        "tag_last_processed_click": None,
        "tag_flash": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _tag_y_max(pitch_mode: str, pitch_width: float) -> float:
    return float(pitch_width if pitch_mode == "rect" else 100.0)


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


def _pitch_to_img(x, y, w, h, y_max, pad=18):
    inner_w = max(1, w - 2 * pad)
    inner_h = max(1, h - 2 * pad)
    px = pad + int(round((float(x) / 100.0) * inner_w))
    py = pad + int(round(inner_h - (float(y) / float(y_max)) * inner_h))
    return px, py


def _img_to_pitch(px, py, w, h, y_max, pad=18):
    inner_w = max(1, w - 2 * pad)
    inner_h = max(1, h - 2 * pad)
    x = ((float(px) - pad) / inner_w) * 100.0
    y = y_max - (((float(py) - pad) / inner_h) * y_max)
    return round(x, 2), round(y, 2)


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


def _draw_point(draw, x, y, w, h, y_max, fill, outline="#FFFFFF", label=None, r=9, pad=18, marker="o"):
    px, py = _pitch_to_img(x, y, w, h, y_max, pad=pad)
    mk = str(marker or "o").lower()

    if mk in ["s", "square"]:
        draw.rectangle([px-r, py-r, px+r, py+r], fill=fill, outline=outline, width=3)
    elif mk in ["d", "diamond"]:
        draw.polygon([(px, py-r), (px+r, py), (px, py+r), (px-r, py)], fill=fill, outline=outline)
        draw.line([(px, py-r), (px+r, py), (px, py+r), (px-r, py), (px, py-r)], fill=outline, width=3)
    elif mk in ["^", "triangle up"]:
        draw.polygon([(px, py-r), (px+r, py+r), (px-r, py+r)], fill=fill, outline=outline)
        draw.line([(px, py-r), (px+r, py+r), (px-r, py+r), (px, py-r)], fill=outline, width=3)
    elif mk in ["v", "triangle down"]:
        draw.polygon([(px-r, py-r), (px+r, py-r), (px, py+r)], fill=fill, outline=outline)
        draw.line([(px-r, py-r), (px+r, py-r), (px, py+r), (px-r, py-r)], fill=outline, width=3)
    elif mk in ["*", "star"]:
        draw.line([px-r, py, px+r, py], fill=outline, width=3)
        draw.line([px, py-r, px, py+r], fill=outline, width=3)
        draw.line([px-r, py-r, px+r, py+r], fill=outline, width=3)
        draw.line([px-r, py+r, px+r, py-r], fill=outline, width=3)
        draw.ellipse([px-r//2, py-r//2, px+r//2, py+r//2], fill=fill, outline=outline, width=2)
    elif mk in ["x"]:
        draw.line([px-r, py-r, px+r, py+r], fill=fill, width=4)
        draw.line([px-r, py+r, px+r, py-r], fill=fill, width=4)
    elif mk in ["+", "p"]:
        draw.line([px-r, py, px+r, py], fill=fill, width=4)
        draw.line([px, py-r, px, py+r], fill=fill, width=4)
    else:
        draw.ellipse([px-r, py-r, px+r, py+r], fill=fill, outline=outline, width=3)

    if label:
        draw.text((px + 10, py - 18), str(label), fill=outline)


def make_click_pitch_image(theme_name, pitch_mode, pitch_width, display_w=760, start_marker="o", start_color=None, start_edge="#FFFFFF", start_size=7):
    theme = _tag_theme(theme_name)
    y_max = _tag_y_max(pitch_mode, pitch_width)
    pad = 22
    display_h = int(round((display_w - 2 * pad) * y_max / 100.0)) + 2 * pad
    bg = theme.get("pitch", "#1f5f3b")
    line = theme.get("pitch_lines", "#E6E6E6")
    img = Image.new("RGB", (display_w, display_h), bg)
    draw = ImageDraw.Draw(img)

    def P(x, y):
        return _pitch_to_img(x, y, display_w, display_h, y_max, pad=pad)

    def rect(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    lw = 3
    draw.rectangle(rect(P(0, 0), P(100, y_max)), outline=line, width=lw)
    draw.line([P(50, 0), P(50, y_max)], fill=line, width=lw)
    cx, cy = P(50, y_max / 2)
    rx = int((display_w - 2 * pad) * 9.15 / 105.0)
    ry = int((display_h - 2 * pad) * 9.15 / 68.0)
    draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], outline=line, width=lw)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=line)

    pen_len = 16.5 / 105.0 * 100.0
    six_len = 5.5 / 105.0 * 100.0
    pen_w = y_max * 40.32 / 68.0
    six_w = y_max * 18.32 / 68.0
    mid = y_max / 2
    draw.rectangle(rect(P(0, mid - pen_w/2), P(pen_len, mid + pen_w/2)), outline=line, width=lw)
    draw.rectangle(rect(P(100-pen_len, mid - pen_w/2), P(100, mid + pen_w/2)), outline=line, width=lw)
    draw.rectangle(rect(P(0, mid - six_w/2), P(six_len, mid + six_w/2)), outline=line, width=lw)
    draw.rectangle(rect(P(100-six_len, mid - six_w/2), P(100, mid + six_w/2)), outline=line, width=lw)
    for sx in (11 / 105.0 * 100.0, 100 - 11 / 105.0 * 100.0):
        px, py = P(sx, mid)
        draw.ellipse([px-3, py-3, px+3, py+3], fill=line)

    # goals
    goal_w = y_max * (7.32 / 68.0)
    draw.line([P(0, mid-goal_w/2), (P(0, mid-goal_w/2)[0]-10, P(0, mid-goal_w/2)[1])], fill=line, width=lw)
    draw.line([P(0, mid+goal_w/2), (P(0, mid+goal_w/2)[0]-10, P(0, mid+goal_w/2)[1])], fill=line, width=lw)
    draw.line([P(100, mid-goal_w/2), (P(100, mid-goal_w/2)[0]+10, P(100, mid-goal_w/2)[1])], fill=line, width=lw)
    draw.line([P(100, mid+goal_w/2), (P(100, mid+goal_w/2)[0]+10, P(100, mid+goal_w/2)[1])], fill=line, width=lw)

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
                    _draw_point(draw, x, y, display_w, display_h, y_max, start_color or col, outline=start_edge, r=int(start_size), pad=pad, marker=start_marker)
                else:
                    _draw_point(draw, x, y, display_w, display_h, y_max, col, r=9, pad=pad)
            except Exception:
                pass

    if st.session_state.get("tag_start"):
        sx, sy = st.session_state.tag_start
        _draw_point(draw, sx, sy, display_w, display_h, y_max, start_color or "#22C55E", outline=start_edge, label="START", r=int(start_size)+3, pad=pad, marker=start_marker)
    if st.session_state.get("tag_last_click"):
        cx, cy = st.session_state.tag_last_click
        _draw_point(draw, cx, cy, display_w, display_h, y_max, "#FFFFFF", outline="#EF4444", label="LAST", r=8, pad=pad)

    return img, display_w, display_h, y_max, pad


def _events_dataframe():
    cols = ["event_id", "event_type", "player", "team", "minute", "x", "y", "x2", "y2", "outcome", "tag", "note"]
    if not st.session_state.tag_events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(st.session_state.tag_events).reindex(columns=cols)


def _save_tagged_event(event_type, outcome, player, team, minute, action_tag, note, start, end=None):
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
    st.session_state.tag_flash = f"Saved {event_type} #{new_id}"


def _save_tagged_map(df_events: pd.DataFrame, theme_name: str, pitch_mode: str, pitch_width: float, title: str, start_marker="o", start_color=None, start_edge="#FFFFFF", start_size=70):
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
            pitch.scatter([x], [y], ax=ax, s=start_size, marker=start_marker or "o", color=start_color or col, edgecolors=start_edge, linewidth=1.2, zorder=4)
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
            <div class="app-title">🖱️ Interactive Tagging Tool — V5 CONFIRMED</div>
            <div class="app-subtitle">V5 CONFIRMED: Opta Analyst Light + start point shape/color/size controls.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if streamlit_image_coordinates is None:
        st.error("Missing dependency for click tagging.")
        st.info("Add this to requirements.txt then reboot the app:")
        st.code("streamlit-image-coordinates")
        st.stop()

    control_col, pitch_col = st.columns([0.85, 2.15], gap="large")
    with control_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.subheader("Tag Settings")
        tag_theme = st.selectbox("Theme", list(TAGGING_THEMES.keys()), index=list(TAGGING_THEMES.keys()).index("Opta Analyst Light") if "Opta Analyst Light" in TAGGING_THEMES else 0, key="tag_theme")
        tag_pitch_mode_ui = st.selectbox("Pitch shape", ["Rectangular", "Square"], index=0, key="tag_pitch_mode_ui")
        tag_pitch_mode = "rect" if tag_pitch_mode_ui == "Rectangular" else "square"
        tag_pitch_width = st.slider("Pitch width", 50.0, 80.0, 68.0, 1.0, key="tag_pitch_width") if tag_pitch_mode == "rect" else 100.0
        display_w = st.slider("Pitch display size", 620, 920, 760, 20, key="tag_display_w")

        st.markdown("### 🎯 Arrow start point")
        st.caption("✅ V5 CONFIRMED — لو شايف السطر ده يبقى التعديل شغال 100%")
        tag_marker_options = {
            "Circle": "o",
            "Square": "s",
            "Diamond": "D",
            "Triangle Up": "^",
            "Triangle Down": "v",
            "Star": "*",
            "X": "x",
            "Plus": "+",
            "Pentagon": "p",
            "Hexagon": "h",
        }
        tag_start_marker_label = st.selectbox("Start point shape", list(tag_marker_options.keys()), index=0, key="tag_start_marker_label")
        tag_start_marker = tag_marker_options[tag_start_marker_label]
        tag_start_color = st.color_picker("Start point color", "#6D28D9", key="tag_start_color")
        tag_start_edge = st.color_picker("Start point edge color", "#FFFFFF", key="tag_start_edge")
        tag_start_size = st.slider("Start point size", 5, 22, 9, 1, key="tag_start_size")

        auto_save = st.toggle("Auto-save clicks", value=True, key="tag_auto_save")
        event_type = st.selectbox("Event type", ["pass", "carry", "dribble", "cross", "shot", "touch", "defensive action", "recovery"], key="tag_event_type")
        outcome = st.selectbox("Outcome", ["successful", "unsuccessful", "key pass", "assist", "goal", "ontarget", "off target", "blocked", "touch"], key="tag_outcome")
        player = st.text_input("Player", key="tag_player")
        team = st.text_input("Team", key="tag_team")
        minute = st.number_input("Minute", min_value=0, max_value=130, value=0, step=1, key="tag_minute")
        action_tag = st.selectbox("Action tag", ["", "progressive", "line breaking", "into final third", "into box", "key pass", "chance created", "under pressure", "turnover", "recovery", "duel", "dangerous action"], key="tag_action_tag")
        note = st.text_area("Note", height=70, key="tag_note")
        needs_end = event_type in ["pass", "carry", "dribble", "cross"]
        if st.session_state.tag_flash:
            st.success(st.session_state.tag_flash)
            st.session_state.tag_flash = ""
        if auto_save:
            if needs_end and st.session_state.tag_start:
                st.info(f"Start saved at x={st.session_state.tag_start[0]:.1f}, y={st.session_state.tag_start[1]:.1f}. Click END now.")
            elif needs_end:
                st.info("Click START point. Next click will be END and save automatically.")
            else:
                st.info("Click any point to save this event immediately.")
        else:
            st.info("Manual mode: click a point, then use START/END/Save buttons.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Set START", disabled=st.session_state.tag_last_click is None):
                    st.session_state.tag_start = st.session_state.tag_last_click
                    if not needs_end:
                        st.session_state.tag_end = None
                    st.rerun()
            with c2:
                if st.button("Set END", disabled=(st.session_state.tag_last_click is None or not needs_end)):
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
                    _save_tagged_event(event_type, outcome, player, team, minute, action_tag, note, start, end)
                    st.rerun()
        c3, c4 = st.columns(2)
        with c3:
            if st.button("Undo Last") and st.session_state.tag_events:
                st.session_state.tag_events.pop()
                st.session_state.tag_start = None
                st.rerun()
        with c4:
            if st.button("Clear All"):
                st.session_state.tag_events = []
                st.session_state.tag_start = None
                st.session_state.tag_end = None
                st.session_state.tag_last_click = None
                st.session_state.tag_last_processed_click = None
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with pitch_col:
        st.markdown('<div class="preview-shell">', unsafe_allow_html=True)
        pitch_img, img_w, img_h, y_max, pad = make_click_pitch_image(tag_theme, tag_pitch_mode, tag_pitch_width, display_w=display_w, start_marker=tag_start_marker, start_color=tag_start_color, start_edge=tag_start_edge, start_size=tag_start_size)
        coords = streamlit_image_coordinates(
            pitch_img,
            key=f"tag_image_pitch_{tag_theme}_{tag_pitch_mode}_{tag_pitch_width}_{display_w}_{st.session_state.tag_click_counter}",
        )
        if coords and coords.get("x") is not None and coords.get("y") is not None:
            try:
                x, y = _img_to_pitch(coords.get("x"), coords.get("y"), img_w, img_h, y_max, pad=pad)
                if 0 <= x <= 100 and 0 <= y <= y_max:
                    new_click = (x, y)
                    raw_click = (int(coords.get("x")), int(coords.get("y")), st.session_state.tag_click_counter)
                    if st.session_state.tag_last_processed_click != raw_click:
                        st.session_state.tag_last_click = new_click
                        st.session_state.tag_last_processed_click = raw_click
                        if auto_save:
                            if needs_end:
                                if st.session_state.tag_start is None:
                                    st.session_state.tag_start = new_click
                                    st.session_state.tag_flash = f"Start set: x={x:.1f}, y={y:.1f}"
                                else:
                                    _save_tagged_event(event_type, outcome, player, team, minute, action_tag, note, st.session_state.tag_start, new_click)
                            else:
                                _save_tagged_event(event_type, outcome, player, team, minute, action_tag, note, new_click, None)
                        st.session_state.tag_click_counter += 1
                        st.rerun()
            except Exception:
                pass

        df_events = _events_dataframe()
        st.subheader("Tagged Events")
        st.dataframe(df_events, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download Tagged Events CSV", data=df_events.to_csv(index=False).encode("utf-8-sig"), file_name="tagged_events.csv", mime="text/csv", disabled=df_events.empty)
        if st.button("Generate Map From Tagged Events", disabled=df_events.empty):
            map_fig = _save_tagged_map(df_events, tag_theme, tag_pitch_mode, tag_pitch_width, "Tagged Events Map", start_marker=tag_start_marker, start_color=tag_start_color, start_edge=tag_start_edge, start_size=tag_start_size * 12)
            png = io.BytesIO()
            map_fig.savefig(png, format="png", dpi=300, bbox_inches="tight", pad_inches=.25)
            png.seek(0)
            st.image(png.getvalue(), use_container_width=True)
            st.download_button("⬇️ Download Tagged Map PNG", data=png.getvalue(), file_name="tagged_events_map.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)


with st.sidebar:
    st.success("APP FILE VERSION: TAGGING V5 CONFIRMED")
    st.caption("لو مش شايف الرسالة دي يبقى Streamlit مش بيقرأ app.py الجديد")
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
# PLAYER METRIC CHARTS - restored old chart options
# =========================================================
PLAYER_METRIC_MODES = {"Scatter Plot", "Bar Chart", "Percentile Bar", "MPL Pizza", "Athletic Pizza", "Old Simple Pizza"}


def _metric_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in d.columns:
        s = clean_metric_series(d[c])
        if s.notna().sum() > 0:
            d[c] = s
    return d


def _default_player_col(df: pd.DataFrame):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for k in ["player", "player name", "name"]:
        if k in lower:
            return lower[k]
    return df.columns[0]


def _metric_columns(df: pd.DataFrame, exclude_cols=None, min_valid=2):
    exclude_cols = set(exclude_cols or [])
    out = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        s = clean_metric_series(df[c])
        if s.notna().sum() >= min_valid:
            out.append(c)
    return out


def _build_metric_table(df: pd.DataFrame, player_col: str, player_name: str, metrics: list[str], value_mode: str = "Percentile") -> pd.DataFrame:
    d = df.copy()
    row = d[d[player_col].astype(str) == str(player_name)]
    if row.empty:
        raise ValueError("Player not found.")
    row = row.iloc[0]
    rows = []
    for m in metrics:
        series = clean_metric_series(d[m])
        val = pd.to_numeric(row[m], errors="coerce")
        pct = _percentile_rank(series, val)
        pct = 0 if pd.isna(pct) else float(pct)
        val_num = 0 if pd.isna(val) else float(val)
        rows.append({
            "metric": str(m),
            "value": round(val_num, 1),
            "percentile": round(pct, 1),
            "plot_value": round(pct if value_mode == "Percentile" else val_num, 1),
        })
    return pd.DataFrame(rows)


def _save_fig_files(fig, base_name: str, tmp_dir: str):
    out_dir = os.path.join(tmp_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{base_name}.png")
    pdf_path = os.path.join(out_dir, f"{base_name}.pdf")
    fig.savefig(png_path, dpi=350, bbox_inches="tight", pad_inches=0.25)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
    return [(f"{base_name}.png", read_file_bytes(png_path), "image/png"), (f"{base_name}.pdf", read_file_bytes(pdf_path), "application/pdf")]


def _scatter_chart(df: pd.DataFrame, x_metric: str, y_metric: str, label_col: str, highlight_value: str, title: str, bg: str, text: str, color: str, highlight: str):
    d = df.copy()
    d[x_metric] = clean_metric_series(d[x_metric])
    d[y_metric] = clean_metric_series(d[y_metric])
    d = d.dropna(subset=[x_metric, y_metric])
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.scatter(d[x_metric], d[y_metric], s=80, c=color, alpha=.65, edgecolors="white", linewidths=.7)
    if highlight_value is not None:
        h = d[d[label_col].astype(str) == str(highlight_value)]
        if not h.empty:
            ax.scatter(h[x_metric], h[y_metric], s=220, c=highlight, edgecolors="white", linewidths=1.6, zorder=5)
            for _, r in h.iterrows():
                ax.text(float(r[x_metric]), float(r[y_metric]), str(r[label_col]), color=text, fontsize=10, weight="bold", ha="left", va="bottom")
    ax.set_title(title, color=text, fontsize=16, weight="bold")
    ax.set_xlabel(x_metric, color=text); ax.set_ylabel(y_metric, color=text)
    ax.tick_params(colors=text)
    for sp in ax.spines.values(): sp.set_color(text)
    ax.grid(alpha=.18)
    return fig


def _bar_chart(table: pd.DataFrame, title: str, value_col: str, bg: str, text: str, color: str, horizontal=True):
    d = table.copy().iloc[::-1] if horizontal else table.copy()
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(d) * .45)))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    if horizontal:
        ax.barh(d["metric"], d[value_col], color=color, alpha=.9)
        ax.set_xlim(0, max(100, float(pd.to_numeric(d[value_col], errors="coerce").max() or 100)))
        for i, v in enumerate(d[value_col]): ax.text(float(v)+1, i, f"{float(v):.1f}", color=text, va="center", fontsize=9, weight="bold")
    else:
        ax.bar(d["metric"], d[value_col], color=color, alpha=.9)
        ax.tick_params(axis="x", rotation=30)
    ax.set_title(title, color=text, fontsize=16, weight="bold")
    ax.tick_params(colors=text)
    for sp in ax.spines.values(): sp.set_color(text)
    ax.grid(axis="x" if horizontal else "y", alpha=.18)
    return fig


def _percentile_bar(table: pd.DataFrame, title: str, bg: str, text: str, good: str, mid: str, low: str):
    d = table.copy().iloc[::-1]
    vals = pd.to_numeric(d["percentile"], errors="coerce").fillna(0)
    colors = [good if v >= 70 else mid if v >= 50 else low for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(d) * .45)))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.barh(d["metric"], vals, color=colors, alpha=.95)
    ax.set_xlim(0, 100)
    for i, v in enumerate(vals): ax.text(float(v)+1, i, f"{float(v):.1f}", color=text, va="center", fontsize=9, weight="bold")
    ax.axvline(50, color=text, alpha=.25, lw=1)
    ax.axvline(70, color=text, alpha=.25, lw=1)
    ax.set_title(title, color=text, fontsize=16, weight="bold")
    ax.tick_params(colors=text)
    for sp in ax.spines.values(): sp.set_color(text)
    ax.grid(axis="x", alpha=.18)
    return fig


def _category_colors(categories, attacking_color, possession_color, defending_color):
    cmap = {"Attacking": attacking_color, "Possession": possession_color, "Defending": defending_color}
    return [cmap.get(c, attacking_color) for c in categories]


def _old_scale_colors(percentiles):
    return [
        "#1A78CF" if float(p) >= 85 else
        "#2ECC71" if float(p) >= 70 else
        "#FF9300" if float(p) >= 50 else
        "#D70232"
        for p in percentiles
    ]


def _render_player_metric_mode(mode, dfp, tmp_dir, header_img_obj, pizza_img_obj, default_title, default_subtitle):
    st.session_state.data_info = {"rows": len(dfp), "cols": len(dfp.columns), "mode": mode}
    k1, k2, k3 = st.columns(3)
    with k1: st.markdown(f'<div class="small-kpi"><div class="label">Mode</div><div class="value">{mode}</div></div>', unsafe_allow_html=True)
    with k2: st.markdown(f'<div class="small-kpi"><div class="label">Rows</div><div class="value">{len(dfp)}</div></div>', unsafe_allow_html=True)
    with k3: st.markdown(f'<div class="small-kpi"><div class="label">Columns</div><div class="value">{len(dfp.columns)}</div></div>', unsafe_allow_html=True)
    with st.expander("Preview data (first 25 rows)", expanded=False):
        st.write("Columns:", list(dfp.columns)); st.dataframe(dfp.head(25), use_container_width=True)

    dfm = _metric_clean_df(dfp)
    player_col = st.selectbox("Player column", dfm.columns.tolist(), index=dfm.columns.tolist().index(_default_player_col(dfm)) if _default_player_col(dfm) in dfm.columns else 0, key=f"pm_player_col_{mode}")
    meta_cols = {player_col}
    for maybe in ["team", "squad", "position", "pos", "age", "minutes", "mins", "league", "season", "id"]:
        for c in dfm.columns:
            if str(c).strip().lower() == maybe:
                meta_cols.add(c)
    metric_cols = _metric_columns(dfm, meta_cols, min_valid=1)
    if not metric_cols:
        st.error("No numeric metric columns found."); st.stop()
    players = sorted(dfm[player_col].dropna().astype(str).unique().tolist())
    selected_player = st.selectbox("Choose player", players, key=f"pm_player_{mode}")

    if mode == "Scatter Plot":
        x_metric = st.selectbox("X-axis metric", metric_cols, index=0)
        y_metric = st.selectbox("Y-axis metric", metric_cols, index=min(1, len(metric_cols)-1))
        show_highlight = st.checkbox("Highlight selected player", value=True)
        if st.button("Generate Scatter Plot", key="gen_scatter"):
            fig = _scatter_chart(dfm, x_metric, y_metric, player_col, selected_player if show_highlight else None, default_title or f"{x_metric} vs {y_metric}", "#0E1117", "#FFFFFF", bar_success if 'bar_success' in globals() else "#38BDF8", "#FF9300")
            st.session_state.preview_images = [fig_to_png_bytes(fig)]
            st.session_state.download_files = _save_fig_files(fig, "scatter_plot", tmp_dir)
            st.session_state.messages = [("success", "Scatter Plot generated successfully.")]
    else:
        default = metric_cols[:min(12, len(metric_cols))]
        selected_metrics = st.multiselect("Choose metrics", metric_cols, default=default, key=f"pm_metrics_{mode}")
        value_mode = st.radio("Use values or percentiles?", ["Percentile", "Raw Value"], horizontal=True, key=f"pm_value_{mode}") if mode == "Bar Chart" else "Percentile"
        categories = []
        if mode in ["MPL Pizza", "Athletic Pizza"] and selected_metrics:
            st.markdown("Metric categories")
            for m in selected_metrics:
                categories.append(st.selectbox(str(m), ["Attacking", "Possession", "Defending"], key=f"cat_{mode}_{m}"))
        horizontal = st.checkbox("Horizontal bars", value=True, key="bar_horizontal") if mode == "Bar Chart" else True
        if st.button(f"Generate {mode}", disabled=not selected_metrics, key=f"gen_{mode}"):
            table = _build_metric_table(dfm, player_col, selected_player, selected_metrics, value_mode=value_mode)
            center_img = pizza_img_obj if pizza_img_obj is not None else header_img_obj
            if mode == "Bar Chart":
                fig = _bar_chart(table, default_title or selected_player, "plot_value", "#0E1117", "#FFFFFF", "#38BDF8", horizontal=horizontal)
                base = "bar_chart"
            elif mode == "Percentile Bar":
                fig = _percentile_bar(table, default_title or f"{selected_player} Percentile Bar", "#0E1117", "#FFFFFF", "#2ECC71", "#FF9300", "#D70232")
                base = "percentile_bar"
            elif mode in ["MPL Pizza", "Athletic Pizza"]:
                colors = _category_colors(categories, "#1A78CF", "#FF9300", "#D70232") if categories else _old_scale_colors(table["percentile"])
                fig = pizza_chart(table[["metric", "value", "percentile"]], title=default_title or selected_player, subtitle=default_subtitle or "Percentile vs peers", slice_colors=colors, show_values_legend=False, center_image=center_img, center_img_scale=pizza_center_scale, footer_text="")
                base = "mpl_pizza" if mode == "MPL Pizza" else "athletic_pizza"
            else:
                colors = _old_scale_colors(table["percentile"])
                fig = pizza_chart(table[["metric", "value", "percentile"]], title=default_title or selected_player, subtitle=default_subtitle or "Percentile vs peers", slice_colors=colors, show_values_legend=False, center_image=center_img, center_img_scale=pizza_center_scale, footer_text="")
                base = "old_simple_pizza"
            st.session_state.preview_images = [fig_to_png_bytes(fig)]
            st.session_state.download_files = _save_fig_files(fig, base, tmp_dir)
            st.session_state.messages = [("success", f"{mode} generated successfully.")]

    if st.session_state.messages:
        for level, msg in st.session_state.messages:
            getattr(st, level)(msg)
    if st.session_state.preview_images:
        st.subheader("Preview")
        for img in st.session_state.preview_images:
            st.image(img, use_container_width=True)
        st.subheader("Downloads")
        for fname, fbytes, mime in st.session_state.download_files:
            st.download_button(f"⬇️ Download {fname}", data=fbytes, file_name=fname, mime=mime, key=f"dl_{mode}_{fname}")
    else:
        st.markdown("""<div class="preview-placeholder">Choose settings, then click Generate.</div>""", unsafe_allow_html=True)


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
            "Scatter Plot",
            "Bar Chart",
            "Percentile Bar",
            "MPL Pizza",
            "Athletic Pizza",
            "Old Simple Pizza",
            "Pizza Chart",
            "Shot Detail Card",
            "Defensive Actions Map",
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

        if mode in PLAYER_METRIC_MODES:
            dfp = load_data(path)
            _render_player_metric_mode(mode, dfp, tmp, header_img_obj, pizza_img_obj, report_title, report_subtitle)
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

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
