"""
Football Analysis Suite  v4  — Professional Analyst Platform
Upgrades:
- Universal layout engine (title, subtitle, competition, player, logo, footer)
- Attacking direction arrows on ALL pitch charts
- Advanced legend controls
- 8 professional themes (Opta, Athletic, StatsBomb, FotMob, Wyscout, SofaScore…)
- The Athletic shot map (reference image 2 replica)
- Opta Analyst pass map (reference image 1 replica)
- Stat blocks configurable per chart
- Full vertical pitch with correct aspect ratio
- PNG + PDF exports preserve all overlays
"""

import os, io, math, sys, importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from PIL import Image

# ── module loader ─────────────────────────────────────────────────────────────
def _load_local(alias, *filenames):
    if alias in sys.modules:
        return sys.modules[alias]
    search = [os.path.dirname(os.path.abspath(__file__)), "/home/claude", "/mnt/project"]
    for fn in filenames:
        for base in search:
            p = os.path.join(base, fn)
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location(alias, p)
                mod  = importlib.util.module_from_spec(spec)
                sys.modules[alias] = mod
                spec.loader.exec_module(mod)
                return mod
    return None

_load_local("charts",          "charts.py",          "charts__8_.py")
_load_local("charts_extra",    "charts_extra.py",    "charts_extra__1_.py")
_load_local("charts_pro",      "charts_pro.py")
_load_local("scouting_tools_v2","scouting_tools_v2.py","scouting_tools_v2__6_.py")

# ── core imports ──────────────────────────────────────────────────────────────
from charts import (
    load_data, prepare_df_for_charts,
    pizza_chart, mpl_pizza_dark, athletic_pizza, shot_detail_card,
    defensive_regains_map, outcome_bar, start_location_heatmap,
    touch_map, pass_map, shot_map, defensive_actions_map,
    THEMES, make_pitch,
)
from charts_extra import (
    register_extra_themes, overlay_image_on_fig,
    goal_location_map, goal_mouth_map, goal_shot_report_map,
    vertical_event_map, progressive_carries_map,
    pressure_map, xg_timeline, passing_network,
    draw_tagging_pitch,
)
register_extra_themes()

from charts_pro import (
    PRO_THEMES, ALL_PRO_THEME_NAMES,
    default_layout_cfg, stat_block,
    draw_logo, draw_title_block, draw_footer,
    draw_attack_direction, draw_opta_attack_arrows,
    draw_stat_blocks_bottom, draw_stat_blocks_right,
    draw_custom_legend, layout_controls_ui, image_overlay_controls_pro,
    athletic_shot_map_pro, opta_pass_map_pro, pro_pass_map, pro_shot_map,
)

try:
    from scouting_tools_v2 import (
        ROLE_TEMPLATES, load_player_data, standard_columns,
        numeric_metrics, match_template_metrics, add_percentiles_and_score,
        player_profile, similar_players, comparison_chart, radar_chart,
        make_template_csv, recommendation_text, NEGATIVE_METRIC_WORDS,
    )
except Exception:
    ROLE_TEMPLATES = None; recommendation_text = None
    NEGATIVE_METRIC_WORDS = ["conceded","fouls","errors","turnovers","dispossessed"]

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_SIC = True
except Exception:
    streamlit_image_coordinates = None; HAS_SIC = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="⚽ Football Analysis Suite v4",
                   layout="wide", initial_sidebar_state="expanded")

# Merge all themes
for k, v in PRO_THEMES.items():
    if k not in THEMES:
        THEMES[k] = v

ALL_THEMES = list(THEMES.keys())

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--bg:#07111f;--s1:#0d1f35;--s2:#112240;--bdr:#1e3a5f;
      --tx:#e8f0fe;--mu:#6b8cae;--ac:#00d4ff;--gr:#00e676;--re:#ff4060;--go:#ffd060;}
.stApp{background:var(--bg);color:var(--tx);font-family:'Inter',-apple-system,sans-serif;}
.block-container{padding:1rem 1.5rem 2rem;max-width:100%;}
section[data-testid="stSidebar"]{background:#050e1c!important;border-right:1px solid var(--bdr)!important;}
section[data-testid="stSidebar"] *{color:var(--tx)!important;}
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[data-baseweb="select"]>div,
section[data-testid="stSidebar"] textarea{background:#0a1929!important;border-color:var(--bdr)!important;color:var(--tx)!important;}
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  background:rgba(13,31,53,.7)!important;border:1px solid var(--bdr)!important;
  border-radius:8px!important;padding:6px 10px!important;margin-bottom:5px!important;}
.hdr{display:flex;align-items:center;gap:14px;padding:16px 20px;
  background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(124,58,237,.06));
  border:1px solid rgba(0,212,255,.15);border-radius:16px;margin-bottom:18px;}
.hdr .ic{font-size:2rem;}.hdr .ti{font-size:1.5rem;font-weight:800;line-height:1.1;}
.hdr .su{font-size:.87rem;color:var(--mu);margin-top:3px;}
.card{background:var(--s1);border:1px solid var(--bdr);border-radius:14px;padding:16px;margin-bottom:12px;}
.ctitle{font-size:.85rem;font-weight:700;color:var(--ac);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;}
.krow{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px;}
.kc{background:var(--s2);border:1px solid var(--bdr);border-radius:10px;padding:9px 13px;text-align:center;flex:1;min-width:80px;}
.kc .kv{font-size:1.1rem;font-weight:800;}.kc .kl{font-size:.73rem;color:var(--mu);margin-top:2px;}
.empty{border:1.5px dashed var(--bdr);background:rgba(13,31,53,.5);border-radius:14px;
  padding:36px 20px;text-align:center;color:var(--mu);}
.empty .ei{font-size:2.2rem;margin-bottom:8px;}
.empty .et{font-size:1rem;font-weight:700;color:var(--tx);margin-bottom:5px;}
.divd{height:1px;background:linear-gradient(90deg,transparent,var(--bdr),transparent);margin:14px 0;}
.stButton>button{border-radius:10px;border:1px solid var(--bdr);
  background:linear-gradient(135deg,#0ea5e9,#2563eb);color:white;font-weight:700;
  padding:.55rem 1rem;width:100%;}
.stDownloadButton>button{border-radius:10px;border:1px solid var(--bdr);
  background:var(--s2);color:var(--tx);font-weight:600;width:100%;}
div[data-testid="stFileUploader"] section{background:var(--s2);
  border:1.5px dashed var(--bdr);border-radius:12px;}
.stExpander{background:var(--s2)!important;border:1px solid var(--bdr)!important;border-radius:12px!important;}
.req-badge{display:inline-block;background:rgba(0,212,255,.12);border:1px solid rgba(0,212,255,.25);
  color:var(--ac);border-radius:6px;padding:2px 8px;font-size:.75rem;margin:2px;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MARKER_OPTS = {
    "None": None, "Circle": "o", "Star": "*",
    "Triangle ▲": "^", "Triangle ▼": "v",
    "Square": "s", "Diamond": "D", "Plus": "+", "X": "x",
    "Pentagon": "p", "Hexagon": "h",
}
MKL = list(MARKER_OPTS.keys())

LEGEND_POSITIONS = [
    "lower center","upper center","lower left","upper left",
    "lower right","upper right","center left","center right","best"
]

def _bytes(fig, dpi=240):
    b = io.BytesIO()
    fig.canvas.draw()
    extras = list(fig.legends)
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            extras.append(leg)
    fig.savefig(
        b, format="png", dpi=dpi, bbox_inches="tight", pad_inches=.18,
        bbox_extra_artists=extras or None,
    )
    b.seek(0); return b.read()

def _pdf(fig):
    b = io.BytesIO()
    with PdfPages(b) as p:
        fig.canvas.draw()
        extras = list(fig.legends)
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                extras.append(leg)
        p.savefig(fig, bbox_inches="tight", pad_inches=.18,
                  bbox_extra_artists=extras or None)
    b.seek(0); return b.read()

def _clean(s):
    return pd.to_numeric(
        s.astype(str).str.replace("%","",regex=False)
                     .str.replace(",","",regex=False).str.strip(),
        errors="coerce")

def _pct(s, hi=True):
    x = pd.to_numeric(s, errors="coerce")
    p = x.rank(pct=True, method="average") * 100
    return (100 - p if not hi else p).clip(0, 100)

def hdr(icon, title, sub=""):
    s = f'<div class="su">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="hdr"><div class="ic">{icon}</div>'
        f'<div><div class="ti">{title}</div>{s}</div></div>',
        unsafe_allow_html=True)

def empty(icon="📊", title="Configure and generate", msg=""):
    st.markdown(
        f'<div class="empty"><div class="ei">{icon}</div>'
        f'<div class="et">{title}</div><div>{msg}</div></div>',
        unsafe_allow_html=True)

def req_badge(cols):
    badges = "".join([f'<span class="req-badge">{c}</span>' for c in cols])
    st.markdown(
        f"<div style='margin-bottom:8px'>"
        f"<b style='color:#6b8cae;font-size:.78rem'>Requires:</b> {badges}</div>",
        unsafe_allow_html=True)

def dl_row(fig, name):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ PNG", _bytes(fig), f"{name}.png",
                           "image/png", key=f"p_{name}_{id(fig)}")
    with c2:
        st.download_button("⬇ PDF", _pdf(fig), f"{name}.pdf",
                           "application/pdf", key=f"f_{name}_{id(fig)}")
    plt.close(fig)

def preview(fig, name):
    st.image(_bytes(fig, dpi=180), use_container_width=True)
    dl_row(fig, name)

def load_img(f):
    if f is None: return None
    try: return Image.open(f).convert("RGBA")
    except: return None

def load_ev(f):
    n = getattr(f, "name", "x"); ext = n.lower().rsplit(".", 1)[-1]
    if ext == "csv":
        for enc in ["utf-8","utf-8-sig","cp1256","latin1"]:
            try: f.seek(0); return pd.read_csv(f, encoding=enc)
            except: pass
    return pd.read_excel(f)

def ensure_outcome(df):
    if "outcome" in df.columns: return df
    for c in df.columns:
        if c.lower().strip() in ["event","result","type","event_type"]:
            df = df.copy(); df["outcome"] = df[c]; return df
    df = df.copy(); df["outcome"] = "unknown"; return df

def nofile():
    empty("📂", "No file uploaded", "Upload an event-data CSV/Excel from the sidebar.")

# ─────────────────────────────────────────────────────────────────────────────
# COLOR/MARKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def shot_cm(px):
    c1, c2 = st.columns(2)
    with c1:
        co  = st.color_picker("Off target",  "#FF8A00", key=f"{px}_co")
        con = st.color_picker("On target",   "#00C2FF", key=f"{px}_con")
    with c2:
        cg  = st.color_picker("Goal",        "#00FF6A", key=f"{px}_cg")
        cb  = st.color_picker("Blocked",     "#AAAAAA", key=f"{px}_cb")
    c3, c4 = st.columns(2)
    with c3:
        mo  = st.selectbox("Marker off-target", MKL, index=MKL.index("Triangle ▲"), key=f"{px}_mo")
        mon = st.selectbox("Marker on-target",  MKL, index=MKL.index("Diamond"),    key=f"{px}_mon")
    with c4:
        mg  = st.selectbox("Marker goal",    MKL, index=MKL.index("Star"),   key=f"{px}_mg")
        mb  = st.selectbox("Marker blocked", MKL, index=MKL.index("Square"), key=f"{px}_mb")
    colors  = {"off target": co, "ontarget": con, "goal": cg, "blocked": cb}
    markers = {"off target": MARKER_OPTS[mo], "ontarget": MARKER_OPTS[mon],
               "goal": MARKER_OPTS[mg], "blocked": MARKER_OPTS[mb]}
    return colors, markers

def pass_cm(px):
    c1, c2 = st.columns(2)
    with c1:
        cs = st.color_picker("Successful",   "#00FF6A", key=f"{px}_cs")
        cu = st.color_picker("Unsuccessful", "#FF4D4D", key=f"{px}_cu")
    with c2:
        ck = st.color_picker("Key pass", "#00C2FF", key=f"{px}_ck")
        ca = st.color_picker("Assist",   "#FFD400", key=f"{px}_ca")
    c3, c4 = st.columns(2)
    with c3:
        ms = st.selectbox("Marker succ",   MKL, index=MKL.index("Circle"),  key=f"{px}_ms")
        mu = st.selectbox("Marker unsucc", MKL, index=MKL.index("X"),       key=f"{px}_mu")
    with c4:
        mk = st.selectbox("Marker key",    MKL, index=MKL.index("Diamond"), key=f"{px}_mk")
        ma = st.selectbox("Marker assist", MKL, index=MKL.index("Star"),    key=f"{px}_ma")
    colors  = {"successful": cs, "unsuccessful": cu, "key pass": ck, "assist": ca}
    markers = {"successful": MARKER_OPTS[ms], "unsuccessful": MARKER_OPTS[mu],
               "key pass": MARKER_OPTS[mk], "assist": MARKER_OPTS[ma]}
    return colors, markers

def def_cm(px):
    defs = {
        "interception": ("#00C2FF","Circle"), "tackle": ("#FF8A00","Square"),
        "recovery": ("#00FF6A","Diamond"), "aerial_duel": ("#FFD400","Triangle ▲"),
        "ground_duel": ("#FF4D4D","X"), "clearance": ("#A78BFA","Star"),
    }
    cols_o = {}; mkrs = {}
    c1, c2 = st.columns(2)
    for i, (act, (dc, dm)) in enumerate(defs.items()):
        lbl = act.replace("_"," ").title()
        with (c1 if i % 2 == 0 else c2):
            cols_o[act] = st.color_picker(lbl, dc, key=f"{px}_dc_{act}")
            ml = st.selectbox(f"{lbl} marker", MKL, index=MKL.index(dm), key=f"{px}_dm_{act}")
            mkrs[act] = MARKER_OPTS[ml]
    return cols_o, mkrs

# ─────────────────────────────────────────────────────────────────────────────
# LEGEND CONTROLS (full)
# ─────────────────────────────────────────────────────────────────────────────
def legend_controls_full(prefix, all_items, default_active=None):
    if default_active is None:
        default_active = list(all_items)
    with st.expander("🎨 Legend Controls", expanded=False):
        st.markdown("**Show/hide items**")
        active = []
        n_cols = min(3, max(1, len(all_items)))
        cols = st.columns(n_cols)
        for i, item in enumerate(all_items):
            with cols[i % n_cols]:
                if st.checkbox(item, value=(item in default_active),
                               key=f"{prefix}_leg_{i}"):
                    active.append(item)
        st.markdown("**Legend style**")
        c1, c2, c3 = st.columns(3)
        with c1:
            pos = st.selectbox("Position", LEGEND_POSITIONS,
                               index=LEGEND_POSITIONS.index("lower center"),
                               key=f"{prefix}_leg_pos")
        with c2:
            fs = st.slider("Font size", 7, 16, 9, key=f"{prefix}_leg_fs")
        with c3:
            title_txt = st.text_input("Legend title", "", key=f"{prefix}_leg_title")
        frame = st.checkbox("Show frame", False, key=f"{prefix}_leg_frame")
    return dict(active=active, pos=pos, fontsize=fs,
                title=title_txt or None, frame=frame)

def apply_legend_style(ax, legend_cfg, theme):
    handles, labels = ax.get_legend_handles_labels()
    active = legend_cfg.get("active", labels)
    filtered = [(h, l) for h, l in zip(handles, labels) if l in active]
    if not filtered:
        if ax.get_legend(): ax.get_legend().remove()
        return
    hs, ls = zip(*filtered)
    bbox = (0.5, -0.02) if "center" in legend_cfg.get("pos","") else None
    leg = ax.legend(hs, ls,
        loc=legend_cfg.get("pos","lower center"),
        fontsize=legend_cfg.get("fontsize", 9),
        title=legend_cfg.get("title"),
        frameon=legend_cfg.get("frame", False),
        bbox_to_anchor=bbox)
    for t in leg.get_texts():
        t.set_color(theme.get("text", "white"))
    if leg.get_title():
        leg.get_title().set_color(theme.get("muted","#888888"))

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE OVERLAY CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
def img_overlay_controls(prefix, label="Overlay Image",
                          default_x=0.02, default_y=0.88,
                          default_w=0.10, default_h=0.10):
    with st.expander(f"🖼️ {label}", expanded=False):
        f = st.file_uploader(label, type=["png","jpg","jpeg"],
                             key=f"{prefix}_img_f", label_visibility="collapsed")
        img = load_img(f)
        if img:
            col_x, col_y = st.columns(2)
            with col_x: x = st.slider("X %", 0, 95, int(default_x*100), key=f"{prefix}_ix") / 100
            with col_y: y = st.slider("Y %", 0, 95, int(default_y*100), key=f"{prefix}_iy") / 100
            col_w, col_h = st.columns(2)
            with col_w: w = st.slider("W %", 4, 30, int(default_w*100), key=f"{prefix}_iw") / 100
            with col_h: h = st.slider("H %", 4, 30, int(default_h*100), key=f"{prefix}_ih") / 100
            circle = st.checkbox("Circle crop", False, key=f"{prefix}_ic")
            bdr_lw = st.slider("Border", 0.0, 5.0, 0.0, 0.5, key=f"{prefix}_ib")
        else:
            x, y, w, h, circle, bdr_lw = default_x, default_y, default_w, default_h, False, 0.0
    return dict(img=img, x=x, y=y, w=w, h=h, circle=circle, bdr_lw=bdr_lw)

def apply_overlay(fig, ov):
    if ov.get("img"):
        try:
            overlay_image_on_fig(fig, ov["img"], x=ov["x"], y=ov["y"],
                                 w=ov["w"], h=ov["h"],
                                 circle_crop=ov["circle"],
                                 border_lw=ov.get("bdr_lw", 0.0))
        except Exception:
            pass

# ─────────────────────────────────────────────────────────────────────────────
# STAT BLOCKS UI
# ─────────────────────────────────────────────────────────────────────────────
def stat_blocks_ui(prefix, n_max=6):
    blocks = []
    with st.expander("📊 Stat Blocks", expanded=False):
        n = st.number_input("Number of stat blocks", 0, n_max, 0, key=f"{prefix}_sbn")
        accent = THEMES.get("The Athletic Dark", {}).get("accent","#C8102E")
        for i in range(int(n)):
            c1, c2, c3 = st.columns(3)
            with c1: val = st.text_input(f"Value {i+1}", "", key=f"{prefix}_sbv{i}")
            with c2: lbl = st.text_input(f"Label {i+1}", "", key=f"{prefix}_sbl{i}")
            with c3: col = st.color_picker(f"Color {i+1}", accent, key=f"{prefix}_sbc{i}")
            if val or lbl:
                blocks.append(stat_block(val, lbl, color=col))
    return blocks

# ─────────────────────────────────────────────────────────────────────────────
# EVENT DATA SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def ev_sidebar(prefix):
    with st.sidebar:
        st.markdown("### 📂 Event Data")
        f = st.file_uploader("CSV / Excel", type=["csv","xlsx","xls"],
                             key=f"{prefix}_f")
        st.markdown('<div class="divd"></div>', unsafe_allow_html=True)
        st.markdown("### 🎨 Theme & Pitch")
        tn = st.selectbox("Theme", ALL_THEMES,
                          index=ALL_THEMES.index("The Athletic Dark")
                          if "The Athletic Dark" in ALL_THEMES else 0,
                          key=f"{prefix}_tn")
        adir = st.selectbox("Attack direction",
                            ["Left → Right","Right → Left"], key=f"{prefix}_ad")
        fy   = st.checkbox("Flip Y axis", False, key=f"{prefix}_fy")
        ps   = st.selectbox("Pitch shape", ["Rectangular","Square"],
                            key=f"{prefix}_ps")
        pm   = "rect" if ps == "Rectangular" else "square"
        pw   = st.slider("Pitch width", 50.0, 80.0, 68.0, 1.0,
                         key=f"{prefix}_pw") if pm == "rect" else 100.0

    S = dict(tn=tn, ad="ltr" if "Left" in adir else "rtl",
             fy=fy, pm=pm, pw=pw)
    if f is None:
        return None, S
    try:
        raw = load_ev(f)
        raw = ensure_outcome(raw)
        df  = prepare_df_for_charts(raw, attack_direction=S["ad"],
                                    flip_y=S["fy"], pitch_mode=S["pm"],
                                    pitch_width=S["pw"], xg_method="zone")
        return df, S
    except Exception as e:
        st.error(f"Load error: {e}"); return None, S

# ─────────────────────────────────────────────────────────────────────────────
# VERTICAL PITCH HELPER
# ─────────────────────────────────────────────────────────────────────────────
def make_full_vertical_pitch(pitch_mode, pitch_width, theme):
    from mplsoccer import VerticalPitch
    pc    = theme.get("pitch","#1f5f3b")
    lc    = theme.get("pitch_lines","#E6E6E6")
    stripe = theme.get("pitch_stripe")
    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    fig_w = 8.0
    fig_h = fig_w * (100.0 / max(y_max, 1.0)) * 0.95
    vp = VerticalPitch(
        pitch_type="custom", pitch_length=100, pitch_width=y_max,
        pitch_color=pc, line_color=lc, line_zorder=2,
        stripe=bool(stripe), stripe_color=stripe or pc,
    )
    fig, ax = vp.draw(figsize=(fig_w, fig_h), constrained_layout=True)
    fig.patch.set_facecolor(theme["bg"])
    return fig, ax, vp

# ─────────────────────────────────────────────────────────────────────────────
# MOMENTUM CHART
# ─────────────────────────────────────────────────────────────────────────────
def momentum_chart(df, team_a="Home", team_b="Away", team_col=None,
                   minute_col=None, title="Match Momentum",
                   color_a="#00C2FF", color_b="#FF4060",
                   theme_name="The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme["bg"]; text_col = theme["text"]; muted = theme["muted"]
    d = df.copy()
    min_c = minute_col
    if not min_c:
        for cand in ["minute","min","time","match_time"]:
            if cand in d.columns: min_c = cand; break
    if not min_c:
        d["_min"] = np.arange(len(d)); min_c = "_min"
    d[min_c] = pd.to_numeric(d[min_c], errors="coerce").fillna(0)
    d = d.sort_values(min_c)
    bins = np.arange(0, 95, 5)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    if team_col and team_col in d.columns:
        for tname, color, sign in [(team_a, color_a, 1), (team_b, color_b, -1)]:
            td = d[d[team_col].astype(str) == str(tname)]
            counts, _ = np.histogram(td[min_c], bins=bins)
            xs = (bins[:-1] + bins[1:]) / 2
            ax.bar(xs, counts * sign, width=4.5, color=color, alpha=0.75, label=tname)
    else:
        counts, _ = np.histogram(d[min_c], bins=bins)
        xs = (bins[:-1] + bins[1:]) / 2
        ax.bar(xs, counts, width=4.5, color=color_a, alpha=0.75, label=team_a)
    ax.axhline(0, color=muted, lw=1.5)
    ax.axvline(45, color=muted, ls="--", lw=1, alpha=0.5)
    ax.set_title(title, color=text_col, fontsize=16, weight="bold")
    ax.set_xlabel("Minute", color=muted); ax.set_ylabel("Actions", color=muted)
    ax.tick_params(colors=muted)
    for sp in ax.spines.values(): sp.set_color(theme.get("lines","#2A3240"))
    leg = ax.legend(frameon=False)
    for t in leg.get_texts(): t.set_color(text_col)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SHOT ZONE MAP
# ─────────────────────────────────────────────────────────────────────────────
def shot_zone_map(df, theme_name="The Athletic Dark", pitch_mode="rect",
                  pitch_width=68.0, title="Shot Zones"):
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme.get("bg","#0E1117"); text_col = theme.get("text","white")
    lc = theme.get("pitch_lines","white"); pc = theme.get("pitch","#1f5f3b")
    shots = df[df["event_type"]=="shot"].copy() if "event_type" in df.columns else df.copy()
    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    zones = {
        "6-yard box":    (94, y_max*.5-y_max*.054, 100, y_max*.5+y_max*.054),
        "Penalty spot":  (88, y_max*.5-y_max*.135, 94,  y_max*.5+y_max*.135),
        "Central zone":  (78, y_max*.5-y_max*.20,  88,  y_max*.5+y_max*.20),
        "Left channel":  (78, y_max*.5-y_max*.40,  100, y_max*.5-y_max*.20),
        "Right channel": (78, y_max*.5+y_max*.20,  100, y_max*.5+y_max*.40),
        "Long range L":  (55, y_max*.5-y_max*.50,  78,  y_max*.5-y_max*.15),
        "Long range C":  (55, y_max*.5-y_max*.15,  78,  y_max*.5+y_max*.15),
        "Long range R":  (55, y_max*.5+y_max*.15,  78,  y_max*.5+y_max*.50),
    }
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(bg); ax.set_facecolor(pc)
    ax.set_xlim(48, 104); ax.set_ylim(-2, y_max+2); ax.axis("off")
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    pitch.draw(ax=ax); ax.set_facecolor(pc)
    cmap = LinearSegmentedColormap.from_list("shots",
           ["#0B2A4A","#1A78CF","#FF9300","#C8102E"])
    for zname, (x0,y0,x1,y1) in zones.items():
        mask = (shots["x"]>=x0)&(shots["x"]<x1)&(shots["y"]>=y0)&(shots["y"]<y1)
        n = mask.sum()
        xg_sum = pd.to_numeric(shots.loc[mask,"xg"], errors="coerce").fillna(0).sum()
        goals_n = (shots.loc[mask,"outcome"].astype(str).str.lower()=="goal").sum()
        alpha = min(0.85, 0.1 + n/max(1,len(shots))*3)
        rect = Rectangle((x0,y0), x1-x0, y1-y0,
                         facecolor=cmap(min(1.0,n/max(1,len(shots))*4)),
                         edgecolor=lc, linewidth=1.5, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        cx, cy = (x0+x1)/2, (y0+y1)/2
        ax.text(cx, cy+1.5, str(n), ha="center", va="center",
                fontsize=13, weight="900", color="white", zorder=5)
        ax.text(cx, cy-1.5, f"xG {xg_sum:.2f}", ha="center", va="center",
                fontsize=9, color="#DDDDDD", zorder=5)
        if goals_n:
            ax.text(cx+4, cy+1.5, f"⚽{goals_n}", ha="center", va="center",
                    fontsize=9, color="#00FF6A", zorder=5)
    ax.set_title(title, color=text_col, fontsize=18, weight="bold")
    ax.set_xlim(48, 104); ax.set_ylim(-2, y_max+2)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Football Analysis Suite v4")
    st.markdown('<div class="divd"></div>', unsafe_allow_html=True)
    section = st.radio("Navigate", [
        "🏠 Home",
        "⚔️ Attacking Charts",
        "🎯 Pro Shot Map",
        "📨 Pro Pass Map",
        "🛡️ Defensive Charts",
        "🔄 Distribution Charts",
        "🎯 Specialist Charts",
        "🍕 Radars & Pizza",
        "🧠 Player Scouting",
        "🖱️ Tagging Tool",
    ], key="nav")

# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if section == "🏠 Home":
    hdr("⚽","Football Analysis Suite v4",
        "Professional charts · Opta · The Athletic · StatsBomb · Wyscout · FotMob")

    tiles = [
        ("🎯","Pro Shot Map","The Athletic replica — xG size, body parts, avg distance","x, y, outcome, xg"),
        ("📨","Pro Pass Map","Opta Analyst replica — arrows, accuracy%, attack direction","x, y, x2, y2, outcome"),
        ("⚔️","Attacking Charts","Shot maps, goal location, shot zones, xG timeline","x, y, outcome"),
        ("🛡️","Defensive Charts","Action maps, pressure, momentum","x, y + def cols"),
        ("🔄","Distribution Charts","Pass maps, carries, passing network","x, y, x2, y2"),
        ("🎯","Specialist Charts","Goal mouth, vertical maps, shot card","x, y, z/height"),
        ("🍕","Radars & Pizza","MPL pizza, Athletic style, percentile bars","Player + metrics"),
        ("🧠","Player Scouting","Role templates, scoring, scout reports","Player + metrics"),
        ("🖱️","Tagging Tool","Click-to-tag events on interactive pitch","No file needed"),
    ]
    cols = st.columns(3)
    for i, (ic, ti, de, req) in enumerate(tiles):
        with cols[i % 3]:
            st.markdown(f"""<div class="card" style="min-height:140px;">
            <div style="font-size:1.5rem">{ic}</div>
            <div style="font-weight:800;font-size:.98rem;color:var(--tx);margin:6px 0 4px">{ti}</div>
            <div style="font-size:.82rem;color:var(--mu);margin-bottom:6px">{de}</div>
            <div style="font-size:.75rem;color:var(--ac)">Requires: {req}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divd"></div>', unsafe_allow_html=True)
    st.markdown("""
**v4 Highlights**
- 🎯 **The Athletic Shot Map** — exact replica with xG sizing, body-part counts, average distance, p90 stats
- 📨 **Opta Analyst Pass Map** — exact replica with attacking direction chevrons, accuracy %, player photo
- 🏗️ **Universal Layout Engine** — title / subtitle / competition / player / match / logo / footer on every chart
- ⬆ **Attacking direction** arrows on all pitch charts — position, color, label, size all configurable
- 📖 **Advanced legend** — show/hide items, reorder, marker scale, font size, title, columns, frame
- 📊 **Configurable stat blocks** — add unlimited value+label stats around any chart
- 🎨 **8 pro themes** — Opta Light/Dark, The Athletic Light/Dark, StatsBomb, FotMob, Wyscout, SofaScore
- 📐 **Vertical pitch fixed** — correct aspect ratio, no cropping
- 💾 **PNG + PDF exports** preserve logos, images, legends, stat blocks
""")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🎯  PRO SHOT MAP  (The Athletic replica)
# ─────────────────────────────────────────────────────────────────────────────
if section == "🎯 Pro Shot Map":
    df, S = ev_sidebar("psm")
    hdr("🎯","Pro Shot Map","The Athletic replica — xG size · body parts · average distance · p90 stats")
    req_badge(["x","y","outcome","xg"])

    if df is None:
        nofile()
        st.stop()

    L, R = st.columns([1, 2.4])
    with L:
        st.markdown('<div class="card"><div class="ctitle">The Athletic Shot Map</div>', unsafe_allow_html=True)
        cfg = default_layout_cfg()
        cfg["theme_name"] = st.selectbox("Theme", ALL_THEMES,
            index=ALL_THEMES.index("The Athletic Light") if "The Athletic Light" in ALL_THEMES else 0,
            key="psm_theme_main")
        theme = THEMES.get(cfg["theme_name"], {})

        cfg["player_name"] = st.text_input("Player name", "", key="psm_pn")
        cfg["subtitle"]    = st.text_input("Subtitle (e.g. All non-penalty shots)", "", key="psm_sub")
        cfg["competition"] = st.text_input("Competition", "", key="psm_cp")
        cfg["season"]      = st.text_input("Season", "", key="psm_sea")
        cfg["analyst_credit"] = st.text_input("Credit (bottom-right)", "The Athletic", key="psm_cr")
        cfg["footer_note"] = st.text_input("Footer note", "", key="psm_fn")

        st.markdown("**Shot colours**")
        goal_col   = st.color_picker("Goal fill",    "#C8102E", key="psm_gc")
        nogoal_col = st.color_picker("No-Goal fill", "#FFFFFF", key="psm_ngc")
        goal_edge  = st.color_picker("Goal edge",    "#C8102E", key="psm_ge")
        nogoal_edge= st.color_picker("No-Goal edge", "#555555", key="psm_nge")

        st.markdown("**Dot size range (by xG)**")
        c1, c2 = st.columns(2)
        with c1: min_sz = st.slider("Min size", 10, 200, 30, key="psm_mins")
        with c2: max_sz = st.slider("Max size", 100, 1500, 700, key="psm_maxs")

        st.markdown("**Body part column (optional)**")
        bp_opts = [None] + list(df.columns)
        bp_col = st.selectbox("Body part column", bp_opts, key="psm_bp")

        st.markdown("**Stats overrides** (leave blank = auto)")
        c1, c2, c3 = st.columns(3)
        with c1:
            tot_ov = st.text_input("Total shots", "", key="psm_tot")
            rf_ov  = st.text_input("Right foot",  "", key="psm_rf")
        with c2:
            lf_ov  = st.text_input("Left foot",   "", key="psm_lf")
            hd_ov  = st.text_input("Head",        "", key="psm_hd")
        with c3:
            oth_ov = st.text_input("Other",       "", key="psm_oth")
            min_pl = st.number_input("Minutes played (for p90)", 0, 5000, 0, key="psm_mp")

        logo_ov   = img_overlay_controls("psm_logo","Logo image",
                                         default_x=0.78, default_y=0.89,
                                         default_w=0.14, default_h=0.10)
        player_ov = img_overlay_controls("psm_plyr","Player image",
                                         default_x=0.02, default_y=0.88,
                                         default_w=0.10, default_h=0.10)
        cfg["logo_img"]    = logo_ov["img"]
        cfg["logo_x"]      = logo_ov["x"]; cfg["logo_y"] = logo_ov["y"]
        cfg["logo_w"]      = logo_ov["w"]; cfg["logo_h"] = logo_ov["h"]
        cfg["logo_circle"] = logo_ov["circle"]
        cfg["player_img"]  = player_ov["img"]
        cfg["player_x"]    = player_ov["x"]; cfg["player_y"] = player_ov["y"]
        cfg["player_w"]    = player_ov["w"]; cfg["player_h"] = player_ov["h"]

        gen = st.button("Generate", key="psm_gen")
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        if gen:
            try:
                fig = athletic_shot_map_pro(
                    df, cfg=cfg,
                    goal_color=goal_col, no_goal_color=nogoal_col,
                    goal_edge=goal_edge, no_goal_edge=nogoal_edge,
                    min_dot_size=min_sz, max_dot_size=max_sz,
                    pitch_mode=S["pm"], pitch_width=S["pw"],
                    body_part_col=bp_col,
                    total_shots_override=tot_ov or None,
                    right_foot_override=rf_ov or None,
                    left_foot_override=lf_ov or None,
                    head_override=hd_ov or None,
                    other_override=oth_ov or None,
                    minutes_played=float(min_pl) if min_pl else None,
                )
                st.session_state["psm_fig"] = fig
            except Exception as e:
                st.error(f"Error: {e}")
        if "psm_fig" in st.session_state:
            preview(st.session_state["psm_fig"], "athletic_shot_map")
        else:
            empty("🎯","Configure and generate","Replica of The Athletic shot map style")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 📨  PRO PASS MAP  (Opta Analyst replica)
# ─────────────────────────────────────────────────────────────────────────────
if section == "📨 Pro Pass Map":
    df, S = ev_sidebar("ppm")
    hdr("📨","Pro Pass Map","Opta Analyst replica — arrows · accuracy % · attacking direction")
    req_badge(["x","y","x2","y2","outcome"])

    if df is None:
        nofile()
        st.stop()

    L, R = st.columns([1, 2.4])
    with L:
        st.markdown('<div class="card"><div class="ctitle">Opta Analyst Pass Map</div>', unsafe_allow_html=True)
        cfg = default_layout_cfg()
        cfg["theme_name"] = st.selectbox("Theme", ALL_THEMES,
            index=ALL_THEMES.index("Opta Analyst Light") if "Opta Analyst Light" in ALL_THEMES else 0,
            key="pro_pass_theme")

        cfg["player_name"] = st.text_input("Player name", "", key="ppm_chart_pn")
        cfg["competition"] = st.text_input("Competition", "", key="ppm_chart_cp")
        cfg["match_info"]  = st.text_input("Match info (e.g. Iraq 1-4 Norway)", "", key="ppm_chart_mi")
        cfg["season"]      = st.text_input("Date / Season", "", key="ppm_chart_sea")
        cfg["footer_note"] = st.text_input("Footer note", "", key="ppm_chart_fn")

        # Attack direction
        cfg["show_attack_dir"]     = st.checkbox("Show attack direction", True, key="ppm_chart_sad")
        cfg["attack_dir"]          = st.selectbox("Direction", ["ltr","rtl"], key="ppm_attack_dir_main")
        cfg["attack_dir_label"]    = st.text_input("Direction label", "Attacking Direction", key="ppm_chart_adl")
        cfg["attack_dir_color"]    = st.color_picker("Direction color", "#777777", key="ppm_chart_adc")

        st.markdown("**Pass colours**")
        succ_col  = st.color_picker("Successful",   "#C8102E", key="ppm_chart_sc")
        unsucc_col= st.color_picker("Unsuccessful", "#AAAAAA", key="ppm_chart_uc")

        st.markdown("**Arrow style**")
        c1, c2 = st.columns(2)
        with c1: arrow_alpha = st.slider("Opacity", 0.3, 1.0, 0.82, key="ppm_chart_alpha")
        with c2: arrow_w     = st.slider("Width",   0.5, 4.0, 1.8,  key="ppm_chart_aw")

        st.markdown("**Stat overrides** (leave blank = auto)")
        c1, c2, c3 = st.columns(3)
        with c1: succ_ov = st.text_input("Successful",   "", key="ppm_chart_sov")
        with c2: un_ov   = st.text_input("Unsuccessful", "", key="ppm_chart_uov")
        with c3: acc_ov  = st.text_input("Accuracy %",   "", key="ppm_chart_aov")

        logo_ov   = img_overlay_controls("ppm_logo","Logo image (top-right)",
                                         default_x=0.755, default_y=0.568,
                                         default_w=0.13, default_h=0.055)
        player_ov = img_overlay_controls("ppm_plyr","Player/background photo",
                                         default_x=0.03, default_y=0.895,
                                         default_w=0.10, default_h=0.09)
        cfg["logo_img"]   = logo_ov["img"]
        cfg["logo_x"]     = logo_ov["x"]; cfg["logo_y"] = logo_ov["y"]
        cfg["logo_w"]     = logo_ov["w"]; cfg["logo_h"] = logo_ov["h"]
        cfg["logo_circle"]= logo_ov["circle"]
        cfg["player_img"] = player_ov["img"]
        cfg["player_x"]   = player_ov["x"]; cfg["player_y"] = player_ov["y"]
        cfg["player_w"]   = player_ov["w"]; cfg["player_h"] = player_ov["h"]

        gen = st.button("Generate", key="ppm_chart_gen")
        st.markdown('</div>', unsafe_allow_html=True)

    with R:
        if gen:
            try:
                fig = opta_pass_map_pro(
                    df, cfg=cfg,
                    successful_color=succ_col,
                    unsuccessful_color=unsucc_col,
                    arrow_alpha=arrow_alpha,
                    arrow_width=arrow_w,
                    pitch_mode=S["pm"], pitch_width=S["pw"],
                    successful_override=succ_ov or None,
                    unsuccessful_override=un_ov  or None,
                    accuracy_override=acc_ov    or None,
                )
                st.session_state["ppm_fig"] = fig
            except Exception as e:
                st.error(f"Error: {e}")
        if "ppm_fig" in st.session_state:
            preview(st.session_state["ppm_fig"], "opta_pass_map")
        else:
            empty("📨","Configure and generate","Replica of the Opta Analyst pass map style")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ⚔️  ATTACKING CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "⚔️ Attacking Charts":
    df, S = ev_sidebar("atk")
    hdr("⚔️","Attacking Charts",
        "Shot map · Goal location · Shot zones · Touch map · xG Timeline")

    tabs = st.tabs(["Shot Map","Opta Goal Location","Shot Zones",
                    "Touch Map","Start Heatmap","xG Timeline"])

    # ── Shot Map ─────────────────────────────────────────────────────────────
    with tabs[0]:
        req_badge(["x","y","outcome"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Shot Map</div>', unsafe_allow_html=True)
                t = st.text_input("Title","Shot Map",key="sm_t")
                show_xg  = st.checkbox("Show xG labels", True, key="sm_xg")
                size_xg  = st.checkbox("Size dots by xG", False, key="sm_sxg")
                sc, sm_m = shot_cm("sm")
                leg_cfg  = legend_controls_full("sm",["off target","ontarget","goal","blocked"])

                # Layout controls
                cfg_sm = default_layout_cfg()
                cfg_sm["theme_name"] = S["tn"]
                cfg_sm["show_attack_dir"] = st.checkbox("Show attack direction", True, key="sm_sad")
                cfg_sm["attack_dir"]      = S["ad"]
                cfg_sm["attack_dir_color"]= st.color_picker("Arrow color","#888888",key="sm_adc")

                blocks = stat_blocks_ui("sm")
                ov = img_overlay_controls("sm_ov")
                gen = st.button("Generate", key="sm_gen")
                st.markdown('</div>', unsafe_allow_html=True)
            with R:
                if gen:
                    theme = THEMES[S["tn"]]
                    sc_f  = {k:v for k,v in sc.items()   if k in leg_cfg["active"]}
                    sm_f  = {k:v for k,v in sm_m.items() if k in leg_cfg["active"]}
                    fig = shot_map(df, shot_colors=sc_f, shot_markers=sm_f,
                                   pitch_mode=S["pm"], pitch_width=S["pw"],
                                   show_xg=show_xg, theme_name=S["tn"])
                    ax = fig.axes[0]
                    ax.set_title(t, color=theme["text"], fontsize=16, weight="bold")
                    draw_attack_direction(ax, cfg_sm, theme, S["pm"], S["pw"])
                    apply_legend_style(ax, leg_cfg, theme)
                    if blocks:
                        draw_stat_blocks_bottom(fig, blocks, theme, y=0.02)
                    apply_overlay(fig, ov)
                    st.session_state["atk_sm"] = fig
                if "atk_sm" in st.session_state:
                    preview(st.session_state["atk_sm"], "shot_map")
                else:
                    empty("🎯","Configure and generate")

    # ── Opta Goal Location ───────────────────────────────────────────────────
    with tabs[1]:
        req_badge(["x","y","outcome"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Opta Goal Location Map</div>', unsafe_allow_html=True)
                gl_title = st.text_input("Player/Title","Player Name",key="gl_t")
                gl_sub   = st.text_input("Subtitle","Club | League 2024-25",key="gl_su")
                gl_tn    = st.selectbox("Theme", ALL_THEMES,
                    index=ALL_THEMES.index("Opta Analyst Light") if "Opta Analyst Light" in ALL_THEMES else 0,
                    key="gl_tn")
                gl_color = st.color_picker("Goal dot","#C8102E",key="gl_gc")
                gl_edge  = st.color_picker("Goal edge","#8B0000",key="gl_ge")
                gl_ds    = st.slider("Dot size",60,400,160,key="gl_ds")
                st.markdown("**Stats panel** (value → label)")
                n_stats = st.number_input("Rows",1,6,3,key="gl_ns")
                stats = []
                for i in range(int(n_stats)):
                    c1,c2 = st.columns(2)
                    with c1: v = st.text_input(f"Value {i+1}","",key=f"gl_sv{i}")
                    with c2: lb= st.text_input(f"Label {i+1}","",key=f"gl_sl{i}")
                    if v or lb: stats.append((v,lb))
                player_ov = img_overlay_controls("gl_plyr","Player image",0.01,0.89,0.07,0.09)
                logo_ov   = img_overlay_controls("gl_logo","Logo image",0.87,0.89,0.10,0.09)
                gen = st.button("Generate",key="gl_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig = _opta_goal_combo(df,gl_title,gl_sub,
                        player_ov.get("img"),logo_ov.get("img"),
                        gl_color,gl_edge,gl_ds,gl_tn,S["pm"],S["pw"],stats or None)
                    st.session_state["atk_gl"] = fig
                if "atk_gl" in st.session_state:
                    preview(st.session_state["atk_gl"],"opta_goal_location")
                else:
                    empty("🥅","Configure and generate")

    # ── Shot Zones ───────────────────────────────────────────────────────────
    with tabs[2]:
        req_badge(["x","y","outcome","xg"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Shot Zone Map</div>', unsafe_allow_html=True)
                sz_t = st.text_input("Title","Shot Zones",key="sz_t")
                ov   = img_overlay_controls("sz_ov")
                gen  = st.button("Generate",key="sz_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig = shot_zone_map(df,theme_name=S["tn"],pitch_mode=S["pm"],
                                        pitch_width=S["pw"],title=sz_t)
                    apply_overlay(fig,ov)
                    st.session_state["atk_sz"] = fig
                if "atk_sz" in st.session_state:
                    preview(st.session_state["atk_sz"],"shot_zones")
                else:
                    empty("🗺️","Configure and generate")

    # ── Touch Map ────────────────────────────────────────────────────────────
    with tabs[3]:
        req_badge(["x","y"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Touch Map</div>', unsafe_allow_html=True)
                tm_t = st.text_input("Title","Touch Map",key="tm_t")
                tm_c = st.color_picker("Dot color","#34D5FF",key="tm_c")
                tm_e = st.color_picker("Edge color","#0B0F14",key="tm_e")
                tm_s = st.slider("Dot size",60,500,220,key="tm_s")
                tm_a = st.slider("Opacity",20,100,90,key="tm_a")/100
                tm_ml= st.selectbox("Marker",MKL,index=1,key="tm_ml")
                tm_v = st.checkbox("Vertical pitch",False,key="tm_v")

                cfg_tm = default_layout_cfg()
                cfg_tm["theme_name"] = S["tn"]
                cfg_tm["show_attack_dir"] = st.checkbox("Attack arrow",True,key="tm_sad")
                cfg_tm["attack_dir"] = S["ad"]

                leg_cfg = legend_controls_full("tm",["Touches"])
                ov  = img_overlay_controls("tm_ov")
                gen = st.button("Generate",key="tm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme = THEMES[S["tn"]]
                    if tm_v:
                        fig,ax,vp = make_full_vertical_pitch(S["pm"],S["pw"],theme)
                        d2 = df.dropna(subset=["x","y"])
                        vp.scatter(d2["x"],d2["y"],ax=ax,s=tm_s,
                                   marker=MARKER_OPTS[tm_ml],color=tm_c,
                                   edgecolors=tm_e,linewidth=2,alpha=tm_a,zorder=5)
                        ax.set_title(tm_t,color=theme["text"],fontsize=16,weight="bold")
                    else:
                        fig = touch_map(df,pitch_mode=S["pm"],pitch_width=S["pw"],
                                        theme_name=S["tn"],dot_color=tm_c,edge_color=tm_e,
                                        dot_size=tm_s,alpha=tm_a,marker=MARKER_OPTS[tm_ml])
                        ax = fig.axes[0]
                        ax.set_title(tm_t,color=theme["text"],fontsize=16,weight="bold")
                        draw_attack_direction(ax,cfg_tm,theme,S["pm"],S["pw"],tm_v)
                    apply_overlay(fig,ov)
                    st.session_state["atk_tm"] = fig
                if "atk_tm" in st.session_state:
                    preview(st.session_state["atk_tm"],"touch_map")
                else:
                    empty("👟","Configure and generate")

    # ── Start Heatmap ────────────────────────────────────────────────────────
    with tabs[4]:
        req_badge(["x","y"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Start Heatmap</div>', unsafe_allow_html=True)
                ht_t = st.text_input("Title","Start Location Heatmap",key="ht_t")
                ht_v = st.checkbox("Vertical pitch",False,key="ht_v")
                cfg_ht = default_layout_cfg(); cfg_ht["theme_name"]=S["tn"]
                cfg_ht["show_attack_dir"] = st.checkbox("Attack arrow",True,key="ht_sad")
                cfg_ht["attack_dir"] = S["ad"]
                ov   = img_overlay_controls("ht_ov")
                gen  = st.button("Generate",key="ht_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme = THEMES[S["tn"]]
                    if ht_v:
                        fig,ax,vp = make_full_vertical_pitch(S["pm"],S["pw"],theme)
                        d2 = df.dropna(subset=["x","y"])
                        try:
                            vp.kdeplot(d2["x"],d2["y"],ax=ax,fill=True,levels=50,cmap="Reds",alpha=0.72)
                        except:
                            vp.scatter(d2["x"],d2["y"],ax=ax,s=30,alpha=0.5,color="#FF4060")
                        ax.set_title(ht_t,color=theme["text"],fontsize=16,weight="bold")
                    else:
                        fig = start_location_heatmap(df,pitch_mode=S["pm"],pitch_width=S["pw"],theme_name=S["tn"])
                        ax = fig.axes[0]
                        ax.set_title(ht_t,color=theme["text"],fontsize=16,weight="bold")
                        draw_attack_direction(ax,cfg_ht,theme,S["pm"],S["pw"],ht_v)
                    apply_overlay(fig,ov)
                    st.session_state["atk_ht"] = fig
                if "atk_ht" in st.session_state:
                    preview(st.session_state["atk_ht"],"start_heatmap")
                else:
                    empty("🌡️","Configure and generate")

    # ── xG Timeline ──────────────────────────────────────────────────────────
    with tabs[5]:
        req_badge(["xg","minute (optional)","team (optional)"])
        if df is None: nofile()
        else:
            L, R = st.columns([1, 2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">xG Timeline</div>', unsafe_allow_html=True)
                xt_t  = st.text_input("Title","xG Timeline",key="xt_t")
                xt_a  = st.text_input("Team A","Home",key="xt_a")
                xt_b  = st.text_input("Team B","Away",key="xt_b")
                tcol_opts = [None]+list(df.columns)
                xt_tc = st.selectbox("Team column",tcol_opts,key="xt_tc")
                mcol_opts = [None]+list(df.columns)
                xt_mc = st.selectbox("Minute column",mcol_opts,key="xt_mc")
                xt_ca = st.color_picker("Team A color","#00C2FF",key="xt_ca")
                xt_cb = st.color_picker("Team B color","#FF4060",key="xt_cb")
                ov    = img_overlay_controls("xt_ov",default_x=0.85,default_y=0.88)
                gen   = st.button("Generate",key="xt_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig = xg_timeline(df,team_a=xt_a,team_b=xt_b,
                                      team_col=xt_tc,minute_col=xt_mc,
                                      title=xt_t,color_a=xt_ca,color_b=xt_cb,
                                      theme_name=S["tn"])
                    apply_overlay(fig,ov)
                    st.session_state["atk_xt"] = fig
                if "atk_xt" in st.session_state:
                    preview(st.session_state["atk_xt"],"xg_timeline")
                else:
                    empty("📈","Configure and generate")
    st.stop()

# ── helper used by Attacking tab 1 ───────────────────────────────────────────
def _opta_goal_combo(df,title,subtitle,player_img,logo_img,
                     goal_color,goal_edge,dot_size,theme_name,pitch_mode,pitch_width,stat_labels):
    from matplotlib.patches import FancyBboxPatch
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg=theme.get("bg","#F3F3F4"); text_col=theme.get("text","#151326")
    muted_col=theme.get("muted","#77727F"); lc=theme.get("pitch_lines","#9B9B9B")
    goals = df[df["outcome"].astype(str).str.lower()=="goal"].copy()
    y_max = float(pitch_width if pitch_mode=="rect" else 100.0); mid=y_max/2.0
    fig = plt.figure(figsize=(14,10)); fig.patch.set_facecolor(bg)
    ax_hdr=fig.add_axes([0.0,0.88,1.0,0.12]); ax_hdr.set_facecolor(bg); ax_hdr.axis("off")
    fig.text(0.06,0.965,title,fontsize=28,weight="900",color=text_col,va="top")
    fig.text(0.06,0.918,subtitle,fontsize=13,color=muted_col,va="top")
    ax_pitch=fig.add_axes([0.01,0.04,0.60,0.81]); ax_pitch.set_facecolor(bg)
    ax_pitch.set_xlim(48,104); ax_pitch.set_ylim(-2,y_max+2); ax_pitch.axis("off")
    from charts_pro import _draw_half_pitch_lines
    _draw_half_pitch_lines(ax_pitch,y_max,lc,lw=2.0,show_full=False)
    if not goals.empty:
        ax_pitch.scatter(goals["x"],goals["y"],s=dot_size,
                         color=goal_color,edgecolors=goal_edge,
                         linewidth=1.2,zorder=6,alpha=0.92)
    ax_stats=fig.add_axes([0.63,0.04,0.35,0.81])
    ax_stats.set_facecolor(bg); ax_stats.axis("off")
    ax_stats.set_xlim(0,1); ax_stats.set_ylim(0,1)
    if stat_labels:
        y_pos=0.94
        for i,(val,label) in enumerate(stat_labels):
            circ=plt.Circle((0.12,y_pos-0.04),0.065,color=goal_color,
                            transform=ax_stats.transAxes,zorder=4)
            ax_stats.add_patch(circ)
            ax_stats.text(0.12,y_pos-0.04,str(val),fontsize=15,weight="900",
                          color="white",ha="center",va="center",
                          transform=ax_stats.transAxes,zorder=5)
            ax_stats.text(0.25,y_pos-0.04,str(label),fontsize=16,weight="700",
                          color=text_col,va="center",transform=ax_stats.transAxes)
            if i<len(stat_labels)-1:
                sep_y=y_pos-0.12
                ax_stats.plot([0.02,0.98],[sep_y,sep_y],
                              color=theme.get("lines","#CCCCCC"),lw=1.0,
                              transform=ax_stats.transAxes)
            y_pos-=0.175
    if player_img:
        draw_logo(fig,player_img,0.01,0.89,0.07,0.09,circle_crop=True)
    if logo_img:
        draw_logo(fig,logo_img,0.87,0.89,0.10,0.09)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# 🛡️  DEFENSIVE CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🛡️ Defensive Charts":
    df, S = ev_sidebar("def")
    hdr("🛡️","Defensive Charts",
        "Actions map · Regains heatmap · Pressure map · Momentum · Outcome bar")

    tabs = st.tabs(["Defensive Actions","Ball Regains","Pressure Map",
                    "Momentum Chart","Outcome Distribution"])

    with tabs[0]:
        req_badge(["x","y","outcome","interception/tackle/..."])
        if df is None: nofile()
        else:
            L, R = st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Defensive Actions Map</div>',unsafe_allow_html=True)
                t = st.text_input("Title","Defensive Actions",key="da_t")
                dc,dm = def_cm("da")
                all_acts=["interception","tackle","recovery","aerial_duel","ground_duel","clearance"]
                leg_cfg = legend_controls_full("da",all_acts)
                cfg_da = default_layout_cfg(); cfg_da["theme_name"]=S["tn"]
                cfg_da["show_attack_dir"]=st.checkbox("Attack arrow",True,key="da_sad")
                cfg_da["attack_dir"]=S["ad"]
                dv=st.checkbox("Vertical",False,key="da_v")
                blocks=stat_blocks_ui("da")
                ov=img_overlay_controls("da_ov")
                gen=st.button("Generate",key="da_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme=THEMES[S["tn"]]
                    dc_f={k:v for k,v in dc.items() if k in leg_cfg["active"]}
                    dm_f={k:v for k,v in dm.items() if k in leg_cfg["active"]}
                    fig=defensive_actions_map(df,def_colors=dc_f,def_markers=dm_f,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],
                                              theme_name=S["tn"])
                    ax=fig.axes[0]
                    ax.set_title(t,color=theme["text"],fontsize=16,weight="bold")
                    draw_attack_direction(ax,cfg_da,theme,S["pm"],S["pw"],dv)
                    apply_legend_style(ax,leg_cfg,theme)
                    if blocks: draw_stat_blocks_bottom(fig,blocks,theme,y=0.02)
                    apply_overlay(fig,ov)
                    st.session_state["def_da"]=fig
                if "def_da" in st.session_state: preview(st.session_state["def_da"],"defensive_actions")
                else: empty("🛡️","Configure and generate")

    with tabs[1]:
        req_badge(["x","y","interception/tackle/..."])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Ball Regains Heatmap</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Ball Regains",key="br_t")
                bz=st.checkbox("Show zone counts",True,key="br_z")
                bms=st.slider("Marker size",60,260,110,key="br_ms")
                dc,dm=def_cm("br")
                all_acts=["interception","tackle","recovery","aerial_duel","ground_duel","clearance"]
                leg_cfg=legend_controls_full("br",all_acts)
                ov=img_overlay_controls("br_ov")
                gen=st.button("Generate",key="br_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    dc_f={k:v for k,v in dc.items() if k in leg_cfg["active"]}
                    dm_f={k:v for k,v in dm.items() if k in leg_cfg["active"]}
                    fig=defensive_regains_map(df,title=t,def_colors=dc_f,def_markers=dm_f,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],
                                              theme_name=S["tn"],marker_size=bms,
                                              show_zone_values=bz)
                    apply_overlay(fig,ov)
                    st.session_state["def_br"]=fig
                if "def_br" in st.session_state: preview(st.session_state["def_br"],"ball_regains")
                else: empty("🗺️","Configure and generate")

    with tabs[2]:
        req_badge(["x","y","pressure col (optional)"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Pressure Map</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Pressure Map",key="pr_t")
                pc_opts=[None]+list(df.columns)
                pc=st.selectbox("Pressure column",pc_opts,key="pr_pc")
                cfg_pr=default_layout_cfg(); cfg_pr["theme_name"]=S["tn"]
                cfg_pr["show_attack_dir"]=st.checkbox("Attack arrow",True,key="pr_sad")
                cfg_pr["attack_dir"]=S["ad"]
                pv=st.checkbox("Vertical",False,key="pr_v")
                ov=img_overlay_controls("pr_ov")
                gen=st.button("Generate",key="pr_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme=THEMES[S["tn"]]
                    fig=pressure_map(df,title=t,theme_name=S["tn"],
                                     pitch_mode=S["pm"],pitch_width=S["pw"],
                                     pressure_col=pc or "pressure")
                    ax=fig.axes[0]
                    draw_attack_direction(ax,cfg_pr,theme,S["pm"],S["pw"],pv)
                    apply_overlay(fig,ov)
                    st.session_state["def_pr"]=fig
                if "def_pr" in st.session_state: preview(st.session_state["def_pr"],"pressure_map")
                else: empty("⚡","Configure and generate")

    with tabs[3]:
        req_badge(["minute col (optional)","team col (optional)"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Momentum Chart</div>',unsafe_allow_html=True)
                mm_t=st.text_input("Title","Match Momentum",key="mm_t")
                mm_a=st.text_input("Team A","Home",key="mm_a")
                mm_b=st.text_input("Team B","Away",key="mm_b")
                tc_opts=[None]+list(df.columns)
                mm_tc=st.selectbox("Team column",tc_opts,key="mm_tc")
                mc_opts=[None]+list(df.columns)
                mm_mc=st.selectbox("Minute column",mc_opts,key="mm_mc")
                mm_ca=st.color_picker("Team A","#00C2FF",key="mm_ca")
                mm_cb=st.color_picker("Team B","#FF4060",key="mm_cb")
                ov=img_overlay_controls("mm_ov")
                gen=st.button("Generate",key="mm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=momentum_chart(df,team_a=mm_a,team_b=mm_b,
                                        team_col=mm_tc,minute_col=mm_mc,
                                        title=mm_t,color_a=mm_ca,color_b=mm_cb,
                                        theme_name=S["tn"])
                    apply_overlay(fig,ov)
                    st.session_state["def_mm"]=fig
                if "def_mm" in st.session_state: preview(st.session_state["def_mm"],"momentum")
                else: empty("📊","Configure and generate")

    with tabs[4]:
        req_badge(["outcome"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Outcome Distribution</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Outcome Distribution",key="ob_t")
                all_outc=["successful","unsuccessful","key pass","assist","goal","ontarget","off target","blocked"]
                leg_cfg=legend_controls_full("ob",all_outc)
                bar_colors={
                    "successful":  st.color_picker("Successful","#00FF6A",key="ob_cs"),
                    "unsuccessful":st.color_picker("Unsuccessful","#FF4D4D",key="ob_cu"),
                    "key pass":    st.color_picker("Key pass","#00C2FF",key="ob_ck"),
                    "assist":      st.color_picker("Assist","#FFD400",key="ob_ca"),
                    "goal":        st.color_picker("Goal","#00FF6A",key="ob_cg"),
                    "ontarget":    st.color_picker("On target","#00C2FF",key="ob_co"),
                    "off target":  st.color_picker("Off target","#FF8A00",key="ob_cf"),
                    "blocked":     st.color_picker("Blocked","#AAAAAA",key="ob_cb"),
                }
                ov=img_overlay_controls("ob_ov")
                gen=st.button("Generate",key="ob_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    al=leg_cfg["active"]
                    df_f=df[df["outcome"].isin(al)].copy() if al else df.copy()
                    fig=outcome_bar(df_f,bar_colors={k:v for k,v in bar_colors.items() if k in al},
                                    theme_name=S["tn"])
                    fig.axes[0].set_title(t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["def_ob"]=fig
                if "def_ob" in st.session_state: preview(st.session_state["def_ob"],"outcome_distribution")
                else: empty("📊","Configure and generate")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🔄  DISTRIBUTION CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🔄 Distribution Charts":
    df, S = ev_sidebar("dist")
    hdr("🔄","Distribution Charts","Pass map · Progressive carries · Passing network")

    tabs = st.tabs(["Pass Map","Progressive Carries","Passing Network"])

    with tabs[0]:
        req_badge(["x","y","x2","y2","outcome"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Pass Map</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Pass Map",key="pm_t")
                pv=st.selectbox("Pass view",["All passes","Into Final Third","Into Penalty Box","Line-breaking","Progressive passes"],key="pm_v")
                ps=st.selectbox("Result scope",["Attempts (all)","Successful only","Unsuccessful only"],key="pm_s")
                ppk=st.slider("Min packing",1,5,1,key="pm_pk")
                pc,pm2=pass_cm("pm")
                all_pass=["successful","unsuccessful","key pass","assist"]
                leg_cfg=legend_controls_full("pm",all_pass)
                cfg_pm=default_layout_cfg(); cfg_pm["theme_name"]=S["tn"]
                cfg_pm["show_attack_dir"]=st.checkbox("Attack arrow",True,key="pm_sad")
                cfg_pm["attack_dir"]=S["ad"]
                cfg_pm["attack_dir_color"]=st.color_picker("Arrow color","#888888",key="pm_adc")
                dv=st.checkbox("Vertical",False,key="pm_v2")
                blocks=stat_blocks_ui("pm")
                ov=img_overlay_controls("pm_ov")
                gen=st.button("Generate",key="pm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme=THEMES[S["tn"]]
                    al=leg_cfg["active"]
                    pc_f={k:v for k,v in pc.items() if k in al}
                    pm_f={k:v for k,v in pm2.items() if k in al}
                    fig=pass_map(df,pass_colors=pc_f,pass_markers=pm_f,
                                  pitch_mode=S["pm"],pitch_width=S["pw"],
                                  theme_name=S["tn"],pass_view=pv,
                                  result_scope=ps,min_packing=ppk)
                    ax=fig.axes[0]
                    ax.set_title(t,color=theme["text"],fontsize=16,weight="bold")
                    draw_attack_direction(ax,cfg_pm,theme,S["pm"],S["pw"],dv)
                    apply_legend_style(ax,leg_cfg,theme)
                    if blocks: draw_stat_blocks_bottom(fig,blocks,theme,y=0.02)
                    apply_overlay(fig,ov)
                    st.session_state["dist_pm"]=fig
                if "dist_pm" in st.session_state: preview(st.session_state["dist_pm"],"pass_map")
                else: empty("➡️","Configure and generate")

    with tabs[1]:
        req_badge(["x","y","x2","y2"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Progressive Carries</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Progressive Carries",key="pc_t")
                cc=st.color_picker("Carry color","#FF9300",key="pc_c")
                md=st.slider("Min distance",1.0,20.0,5.0,key="pc_md")
                cfg_pc=default_layout_cfg(); cfg_pc["theme_name"]=S["tn"]
                cfg_pc["show_attack_dir"]=st.checkbox("Attack arrow",True,key="pc_sad")
                cfg_pc["attack_dir"]=S["ad"]
                dv=st.checkbox("Vertical",False,key="pc_v")
                ov=img_overlay_controls("pc_ov")
                gen=st.button("Generate",key="pc_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme=THEMES[S["tn"]]
                    fig=progressive_carries_map(df,title=t,theme_name=S["tn"],
                                                 pitch_mode=S["pm"],pitch_width=S["pw"],
                                                 carry_color=cc,min_distance=md,
                                                 vertical_pitch=dv)
                    ax=fig.axes[0]
                    draw_attack_direction(ax,cfg_pc,theme,S["pm"],S["pw"],dv)
                    apply_overlay(fig,ov)
                    st.session_state["dist_pc"]=fig
                if "dist_pc" in st.session_state: preview(st.session_state["dist_pc"],"progressive_carries")
                else: empty("🏃","Configure and generate")

    with tabs[2]:
        req_badge(["x","y","player col","recipient col (optional)"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Passing Network</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Passing Network",key="pn_t")
                all_cols=[None]+list(df.columns)
                plcol=st.selectbox("Player column",all_cols,key="pn_pl")
                rccol=st.selectbox("Recipient column",all_cols,key="pn_rc")
                nc=st.color_picker("Node","#00C2FF",key="pn_nc")
                ec=st.color_picker("Edge","#AAAAAA",key="pn_ec")
                mp=st.slider("Min passes",1,10,3,key="pn_mp")
                ov=img_overlay_controls("pn_ov")
                gen=st.button("Generate",key="pn_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen and plcol:
                    fig=passing_network(df,player_col=plcol,
                                         recipient_col=rccol or "recipient",
                                         title=t,theme_name=S["tn"],
                                         pitch_mode=S["pm"],pitch_width=S["pw"],
                                         node_color=nc,edge_color=ec,min_passes=mp)
                    apply_overlay(fig,ov)
                    st.session_state["dist_pn"]=fig
                if "dist_pn" in st.session_state: preview(st.session_state["dist_pn"],"passing_network")
                else: empty("🕸️","Configure and generate")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🎯  SPECIALIST CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🎯 Specialist Charts":
    df, S = ev_sidebar("spec")
    hdr("🎯","Specialist Charts",
        "CannonStats goal mouth · Combined shot report · Vertical maps · Shot card")

    tabs = st.tabs(["Goal Mouth Map","Shot Report","Vertical Map","Shot Detail Card"])

    with tabs[0]:
        req_badge(["outcome","y","z/height (optional)","xg (optional)"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Goal Mouth Map</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Team",key="gm_t")
                sub=st.text_input("Subtitle","Shots on Target Map",key="gm_s")
                gtn=st.selectbox("Theme",ALL_THEMES,
                    index=ALL_THEMES.index("Opta Analyst Light") if "Opta Analyst Light" in ALL_THEMES else 0,
                    key="gm_tn")
                gc=st.color_picker("Goal","#7A2232",key="gm_gc")
                sc=st.color_picker("Save","#FFFFFF",key="gm_sc")
                ge=st.color_picker("Goal edge","#3D0A18",key="gm_ge")
                se=st.color_picker("Save edge","#1A2E5A",key="gm_se")
                sxg=st.checkbox("Size by xG",True,key="gm_sxg")
                all_leg=["Save","Goal","Post-Shot xG Value"]
                leg_cfg_gm=legend_controls_full("gm",all_leg)
                st.markdown("**Stats row**")
                n_st=st.number_input("Stat columns",1,5,3,key="gm_ns")
                stats_row={}
                for i in range(int(n_st)):
                    c1,c2=st.columns(2)
                    with c1: k=st.text_input(f"Label {i+1}","",key=f"gm_sk{i}")
                    with c2: v=st.text_input(f"Value {i+1}","",key=f"gm_sv{i}")
                    if k: stats_row[k]=v
                fl=st.text_input("Footer left","Excluding Own Goals",key="gm_fl")
                fr=st.text_input("Footer right","",key="gm_fr")
                logo_ov=img_overlay_controls("gm_logo","Logo",0.84,0.88,0.12,0.10)
                gen=st.button("Generate",key="gm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=goal_mouth_map(df,title=t,subtitle=sub,
                                        stats_row=stats_row or None,theme_name=gtn,
                                        goal_color=gc,save_color=sc,goal_edge=ge,save_edge=se,
                                        size_by_xg=sxg,pitch_mode=S["pm"],pitch_width=S["pw"],
                                        logo_img=logo_ov["img"],
                                        logo_x=logo_ov["x"],logo_y=logo_ov["y"],
                                        logo_w=logo_ov["w"],logo_h=logo_ov["h"],
                                        footer_left=fl,footer_right=fr,
                                        active_legend_items=leg_cfg_gm["active"])
                    st.session_state["spec_gm"]=fig
                if "spec_gm" in st.session_state: preview(st.session_state["spec_gm"],"goal_mouth_map")
                else: empty("🥅","Configure and generate")

    with tabs[1]:
        req_badge(["x","y","outcome","z/height (optional)","xg (optional)"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Combined Shot Report</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Shot Report",key="gsr_t")
                sub=st.text_input("Subtitle","Goals and shots on target",key="gsr_s")
                gtn=st.selectbox("Theme",ALL_THEMES,
                    index=ALL_THEMES.index("Opta Analyst Light") if "Opta Analyst Light" in ALL_THEMES else 0,
                    key="gsr_tn")
                gc=st.color_picker("Goal","#D94D61",key="gsr_gc")
                sc=st.color_picker("Save","#FFFFFF",key="gsr_sc")
                ge=st.color_picker("Goal edge","#B73C4F",key="gsr_ge")
                se=st.color_picker("Save edge","#11113A",key="gsr_se")
                ds=st.slider("Dot size",60,350,130,key="gsr_ds")
                all_gsr=["Pitch goals","Goal","Save","xG size"]
                leg_cfg_gsr=legend_controls_full("gsr",all_gsr)
                logo_ov=img_overlay_controls("gsr_logo","Logo",0.84,0.88,0.12,0.10)
                gen=st.button("Generate",key="gsr_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=goal_shot_report_map(df,title=t,subtitle=sub,theme_name=gtn,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],
                                              goal_color=gc,save_color=sc,
                                              goal_edge=ge,save_edge=se,dot_size=ds,
                                              logo_img=logo_ov["img"],
                                              logo_x=logo_ov["x"],logo_y=logo_ov["y"],
                                              logo_w=logo_ov["w"],logo_h=logo_ov["h"],
                                              active_legend_items=leg_cfg_gsr["active"])
                    st.session_state["spec_gsr"]=fig
                if "spec_gsr" in st.session_state: preview(st.session_state["spec_gsr"],"shot_report")
                else: empty("🎯","Configure and generate")

    with tabs[2]:
        req_badge(["x","y","outcome"])
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Vertical Pitch Map</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Vertical Map",key="vp_t")
                et=st.selectbox("Event type",["pass","shot","touch","all"],key="vp_et")
                dta=st.checkbox("Show arrows",True,key="vp_arr")
                pc2,pm3=pass_cm("vp_pc")
                sc2,sm2=shot_cm("vp_sc")
                dc_col=st.color_picker("Dot color","#00C2FF",key="vp_dc")
                ds=st.slider("Dot size",30,300,80,key="vp_ds")
                all_et=["successful","unsuccessful","key pass","assist",
                        "off target","ontarget","goal","blocked"]
                leg_cfg=legend_controls_full("vp",all_et)
                ov=img_overlay_controls("vp_ov")
                gen=st.button("Generate",key="vp_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    theme=THEMES[S["tn"]]
                    fig,ax,vp2=make_full_vertical_pitch(S["pm"],S["pw"],theme)
                    et_lower=et.lower()
                    if et_lower!="all" and "event_type" in df.columns:
                        d2=df[df["event_type"].astype(str).str.lower()==et_lower].copy()
                    else:
                        d2=df.copy()
                    d2=d2.dropna(subset=["x","y"])
                    from charts import PASS_ORDER,SHOT_ORDER
                    active=leg_cfg["active"]
                    if et_lower=="pass" and "outcome" in d2.columns:
                        for outc in PASS_ORDER:
                            if outc not in active: continue
                            sub=d2[d2["outcome"]==outc]
                            if sub.empty: continue
                            col=pc2.get(outc,"#AAAAAA"); mk=pm3.get(outc,"o") or "o"
                            if dta and "x2" in sub.columns:
                                valid=sub.dropna(subset=["x2","y2"])
                                if not valid.empty:
                                    vp2.arrows(valid["x"],valid["y"],valid["x2"],valid["y2"],
                                               ax=ax,color=col,width=1.8,alpha=0.85)
                            vp2.scatter(sub["x"],sub["y"],ax=ax,s=60,marker=mk,
                                        color=col,edgecolors="white",linewidth=0.8,zorder=5)
                    elif et_lower=="shot" and "outcome" in d2.columns:
                        for outc in SHOT_ORDER:
                            if outc not in active: continue
                            sub=d2[d2["outcome"]==outc]
                            if sub.empty: continue
                            col=sc2.get(outc,"#AAAAAA"); mk=sm2.get(outc,"o") or "o"
                            vp2.scatter(sub["x"],sub["y"],ax=ax,s=120,marker=mk,
                                        color=col,edgecolors="white",linewidth=1.2,zorder=5)
                    else:
                        vp2.scatter(d2["x"],d2["y"],ax=ax,s=ds,color=dc_col,
                                    edgecolors="white",linewidth=0.8,alpha=0.9,zorder=5)
                    ax.set_title(t,color=theme["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["spec_vp"]=fig
                if "spec_vp" in st.session_state: preview(st.session_state["spec_vp"],"vertical_map")
                else: empty("📍","Configure and generate")

    with tabs[3]:
        req_badge(["x","y","outcome","event_type=shot"])
        if df is None: nofile()
        else:
            shots=df[df["event_type"]=="shot"].copy().reset_index(drop=True) if "event_type" in df.columns else df.copy().reset_index(drop=True)
            if shots.empty: st.warning("No shots found.")
            else:
                def _sf(v):
                    try: return float(v)
                    except: return float("nan")
                shots["label"]=shots.apply(
                    lambda r:f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_sf(r.get("xg","0")):.2f}',axis=1)
                L,R=st.columns([1,2.2])
                with L:
                    st.markdown('<div class="card"><div class="ctitle">Shot Detail Card</div>',unsafe_allow_html=True)
                    sel=st.selectbox("Select shot",shots["label"].tolist(),key="sc_sel")
                    sct=st.text_input("Title","Shot Detail",key="sc_t")
                    sc3,sm3=shot_cm("sc3")
                    leg_cfg_sc=legend_controls_full("sc3",["off target","ontarget","goal","blocked"])
                    ov=img_overlay_controls("sc3_ov")
                    gen=st.button("Generate",key="sc3_gen")
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    if gen:
                        idx=int(sel.split("|")[0].strip())-1
                        al3=leg_cfg_sc["active"]
                        sc_f={k:v for k,v in sc3.items() if k in al3}
                        sm_f={k:v for k,v in sm3.items() if k in al3}
                        fig,_=shot_detail_card(df,shot_index=idx,title=sct,
                                               pitch_mode=S["pm"],pitch_width=S["pw"],
                                               shot_colors=sc_f,shot_markers=sm_f,
                                               theme_name=S["tn"])
                        apply_overlay(fig,ov)
                        st.session_state["spec_sc"]=fig
                    if "spec_sc" in st.session_state: preview(st.session_state["spec_sc"],"shot_detail_card")
                    else: empty("🃏","Select a shot and generate")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🍕  RADARS & PIZZA
# ─────────────────────────────────────────────────────────────────────────────
if section == "🍕 Radars & Pizza":
    hdr("🍕","Radars & Pizza","MPL pizza · Athletic style · Percentile bar · Scatter plot")
    with st.sidebar:
        st.markdown("### 📂 Player Metrics File")
        pf=st.file_uploader("CSV / Excel",type=["csv","xlsx","xls"],key="piz_f")

    def _load_pf(f):
        if f is None: return None
        n=getattr(f,"name","x"); ext=n.lower().rsplit(".",1)[-1]
        if ext=="csv":
            for enc in ["utf-8","utf-8-sig","cp1256","latin1"]:
                try: f.seek(0); return pd.read_csv(f,encoding=enc)
                except: pass
        return pd.read_excel(f)

    def _dpc(df_):
        lw={str(c).strip().lower():c for c in df_.columns}
        for k in ["player","player name","name"]:
            if k in lw: return lw[k]
        return df_.columns[0]

    def _mc(df_,excl,mv=1):
        return [c for c in df_.columns if c not in excl and _clean(df_[c]).notna().sum()>=mv]

    def _tbl(df_,pcol,player,metrics):
        d=df_.copy()
        r=d[d[pcol].astype(str)==str(player)]
        if r.empty: raise ValueError("Player not found")
        r=r.iloc[0]; rows=[]
        for m in metrics:
            s=_clean(d[m]); v=pd.to_numeric(r[m],errors="coerce")
            pct=float((s<float(v)).mean()*100) if s.notna().any() and not pd.isna(v) else 0.0
            rows.append({"metric":m,"value":round(float(v),2) if not pd.isna(v) else 0,
                         "percentile":round(pct,1),"plot_value":round(pct,1)})
        return pd.DataFrame(rows)

    dfp=_load_pf(pf)
    tabs=st.tabs(["MPL Pizza (Dark)","Athletic Pizza (Light)","Percentile Bar","Scatter Plot"])

    for tab_idx,(tab,ck) in enumerate(zip(tabs,["mpl","ath","pb","sc"])):
        with tab:
            if dfp is None: empty("📂","Upload a player metrics file"); continue
            pcol=_dpc(dfp); mc_=_mc(dfp,{pcol},mv=1)
            if not mc_: st.error("No numeric columns."); continue
            players=sorted(dfp[pcol].dropna().astype(str).unique().tolist())

            if ck in ["mpl","ath","pb"]:
                L,R=st.columns([1,2])
                with L:
                    st.markdown(f'<div class="card"><div class="ctitle">{tab}</div>',unsafe_allow_html=True)
                    sp=st.selectbox("Player",players,key=f"{ck}_p")
                    ct=st.text_input("Title",sp,key=f"{ck}_t")
                    cs=st.text_input("Subtitle","Percentile vs peers",key=f"{ck}_s")
                    sm_m=st.multiselect("Metrics",mc_,default=mc_[:min(12,len(mc_))],key=f"{ck}_m")
                    cats=[]
                    if ck in ["mpl","ath"] and sm_m:
                        st.markdown("**Categories**")
                        for m in sm_m:
                            cats.append(st.selectbox(m,["Attacking","Possession","Defending"],key=f"{ck}_cat_{m}"))
                    if ck=="mpl":
                        atk_c=st.color_picker("Attacking","#1A78CF",key="mpl_ak")
                        pos_c=st.color_picker("Possession","#FF9300",key="mpl_po")
                        def_c=st.color_picker("Defending","#D70232",key="mpl_df")
                        bg_c =st.color_picker("Background","#222222",key="mpl_bg")
                        ci_f =st.file_uploader("Center image",type=["png","jpg","jpeg"],key="mpl_ci")
                        ci   =load_img(ci_f)
                        cs2  =st.slider("Center scale",8,28,16,key="mpl_cs")/100
                    elif ck=="ath":
                        atk_c=st.color_picker("Attacking","#4B78B9",key="ath_ak")
                        pos_c=st.color_picker("Possession","#F0C987",key="ath_po")
                        def_c=st.color_picker("Defending","#9E374B",key="ath_df")
                        ci_f =st.file_uploader("Center image",type=["png","jpg","jpeg"],key="ath_ci")
                        ci   =load_img(ci_f)
                        cs2  =st.slider("Center scale",8,28,14,key="ath_cs")/100
                    elif ck=="pb":
                        gc_=st.color_picker("Good ≥70","#00e676",key="pb_g")
                        mc2_=st.color_picker("Mid 50-70","#ffd060",key="pb_mid_col")
                        lc_=st.color_picker("Low <50","#ff4060",key="pb_l")
                    ov=img_overlay_controls(f"{ck}_ov","Extra overlay")
                    gen=st.button("Generate",key=f"{ck}_gen",disabled=not sm_m)
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    fk=f"piz_{ck}_fig"
                    if gen and sm_m:
                        try:
                            tbl=_tbl(dfp,pcol,sp,sm_m)
                            if ck=="mpl":
                                fig=mpl_pizza_dark(tbl,title=ct,subtitle=cs,
                                                    categories=cats or None,
                                                    attacking_color=atk_c,
                                                    possession_color=pos_c,
                                                    defending_color=def_c,
                                                    center_image=ci,
                                                    center_img_scale=cs2,
                                                    bg_color=bg_c)
                            elif ck=="ath":
                                fig=athletic_pizza(tbl,title=ct,subtitle=cs,
                                                    categories=cats or None,
                                                    attacking_color=atk_c,
                                                    possession_color=pos_c,
                                                    defending_color=def_c)
                                if ci:
                                    overlay_image_on_fig(fig,ci,x=0.42,y=0.42,
                                                         w=cs2,h=cs2,circle_crop=True)
                            else:
                                d2=tbl.sort_values("percentile",ascending=True)
                                vals=d2["percentile"].tolist()
                                clrs=[gc_ if v>=70 else mc2_ if v>=50 else lc_ for v in vals]
                                fig,ax=plt.subplots(figsize=(10,max(4.5,len(d2)*.45)))
                                fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                                ax.barh(d2["metric"],vals,color=clrs,alpha=.92)
                                ax.set_xlim(0,100)
                                for i,v in enumerate(vals):
                                    ax.text(min(97,float(v)+1.2),i,f"{v:.0f}",
                                            va="center",color="#e8f0fe",fontsize=9,weight="bold")
                                ax.axvline(50,color="#1e3a5f",lw=1.5,ls="--")
                                ax.axvline(70,color="#1e3a5f",lw=1.5,ls="--")
                                ax.set_title(ct,color="#e8f0fe",fontsize=16,weight="bold")
                                ax.set_xlabel("Percentile",color="#6b8cae")
                                ax.tick_params(colors="#6b8cae")
                                for sp2 in ax.spines.values(): sp2.set_color("#1e3a5f")
                            apply_overlay(fig,ov)
                            st.session_state[fk]=fig
                        except Exception as e: st.error(str(e))
                    if fk in st.session_state: preview(st.session_state[fk],ck)
                    else: empty("🍕","Configure and generate")
            else:  # scatter
                L,R=st.columns([1,2])
                with L:
                    st.markdown('<div class="card"><div class="ctitle">Scatter Plot</div>',unsafe_allow_html=True)
                    hi_p=st.selectbox("Highlight player",["(none)"]+players,key="sc_hp")
                    sx=st.selectbox("X metric",mc_,index=0,key="sc_x")
                    sy=st.selectbox("Y metric",mc_,index=min(1,len(mc_)-1),key="sc_y")
                    sct2=st.text_input("Title","Scatter Plot",key="sc_t2")
                    sdc=st.color_picker("Dot","#00d4ff",key="sc_dc")
                    shc=st.color_picker("Highlight","#ffd060",key="sc_hc")
                    ov=img_overlay_controls("sc_ov")
                    gen=st.button("Generate",key="sc_gen")
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    if gen:
                        d2=dfp.copy(); d2[sx]=_clean(d2[sx]); d2[sy]=_clean(d2[sy])
                        d2=d2.dropna(subset=[sx,sy])
                        fig,ax=plt.subplots(figsize=(9,6))
                        fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                        ax.scatter(d2[sx],d2[sy],s=75,color=sdc,alpha=.72,
                                   edgecolors="white",lw=.6)
                        if hi_p!="(none)":
                            h=d2[d2[pcol].astype(str)==str(hi_p)]
                            if not h.empty:
                                ax.scatter(h[sx],h[sy],s=200,color=shc,
                                           edgecolors="white",lw=1.5,zorder=5)
                                for _,rr in h.iterrows():
                                    ax.text(float(rr[sx]),float(rr[sy]),
                                            f"  {rr[pcol]}",color="#e8f0fe",
                                            fontsize=10,weight="bold")
                        ax.axvline(d2[sx].median(),color="#1e3a5f",lw=1.5,ls="--")
                        ax.axhline(d2[sy].median(),color="#1e3a5f",lw=1.5,ls="--")
                        ax.set_title(sct2,color="#e8f0fe",fontsize=16,weight="bold")
                        ax.set_xlabel(sx,color="#6b8cae"); ax.set_ylabel(sy,color="#6b8cae")
                        ax.tick_params(colors="#6b8cae")
                        for sp3 in ax.spines.values(): sp3.set_color("#1e3a5f")
                        apply_overlay(fig,ov)
                        st.session_state["piz_sc_fig"]=fig
                    if "piz_sc_fig" in st.session_state: preview(st.session_state["piz_sc_fig"],"scatter")
                    else: empty("📈","Configure and generate")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🧠  PLAYER SCOUTING
# ─────────────────────────────────────────────────────────────────────────────
if section == "🧠 Player Scouting":
    if ROLE_TEMPLATES is None:
        st.error("scouting_tools_v2 not found."); st.stop()
    hdr("🧠","Player Scouting",
        "Role templates · percentile scoring · shortlists · recommendations")

    def _cl(x): return str(x).strip().lower().replace("_"," ").replace("-"," ")
    def _sn(s): return pd.to_numeric(s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False),errors="coerce")

    def _ir(pos):
        p=_cl(pos)
        mapping={"gk":"Goalkeeper","goalkeeper":"Goalkeeper","cb":"Centre Back",
                 "rb":"Full Back / Wing Back","lb":"Full Back / Wing Back",
                 "rwb":"Full Back / Wing Back","lwb":"Full Back / Wing Back",
                 "dm":"Defensive Midfielder","cdm":"Defensive Midfielder",
                 "cm":"Central Midfielder","mc":"Central Midfielder",
                 "am":"Attacking Midfielder","cam":"Attacking Midfielder",
                 "rw":"Winger","lw":"Winger","st":"Striker","cf":"Striker"}
        for k,v in mapping.items():
            if k in p: return v
        return list(ROLE_TEMPLATES.keys())[0]

    def _ms(df_in,metrics,lb,gc=None,mc_arg=None,mf=900,ws=None):
        out=df_in.copy(); pcs=[]; ws=ws or {}
        for m in metrics:
            if m not in out.columns: continue
            out[m]=_sn(out[m]); pc=f"pct__{m}"; hi=m not in lb
            if gc and gc in out.columns:
                out[pc]=out.groupby(gc,dropna=False)[m].transform(lambda x:_pct(x,hi))
            else:
                out[pc]=_pct(out[m],hi)
            pcs.append(pc)
        if not pcs:
            out["Scouting Score"]=np.nan; out["Adjusted Score"]=np.nan; return out
        w=np.array([float(ws.get(c.replace("pct__",""),1.0)) for c in pcs],dtype=float)
        w=np.where(np.isfinite(w),w,1.0)
        mat=out[pcs].astype(float)
        out["Scouting Score"]=(mat.mul(w,axis=1).sum(axis=1)/mat.notna().mul(w,axis=1).sum(axis=1).replace(0,np.nan)).round(1)
        if mc_arg and mc_arg in out.columns:
            mins=pd.to_numeric(out[mc_arg],errors="coerce").fillna(0)
            out["Reliability"]=(mins/float(max(mf,1))).clip(0,1).round(2)
            out["Adjusted Score"]=(out["Scouting Score"]*(0.65+0.35*out["Reliability"])).round(1)
        else:
            out["Reliability"]=1.0; out["Adjusted Score"]=out["Scouting Score"]
        return out

    def _match_tm(metrics_list,template_metrics):
        return [m for m in metrics_list if any(t.lower() in m.lower() or m.lower() in t.lower() for t in template_metrics)]

    with st.sidebar:
        st.markdown("### 📂 Scouting File")
        sf=st.file_uploader("CSV / Excel",type=["csv","xlsx","xls"],key="sc_f")
        if ROLE_TEMPLATES:
            st.download_button("⬇ Template CSV",data=make_template_csv(),
                               file_name="scouting_template.csv",mime="text/csv",key="sc_tmpl")
        st.markdown('<div class="divd"></div>',unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        mmg=st.number_input("Default min minutes",0,value=300,step=50,key="sc_mmg")
        scp=st.radio("Percentile scope",["All filtered","Same position"],index=1,key="sc_scp")
        ur=st.checkbox("Reliability adjustment",True,key="sc_ur")
        rf=st.number_input("'Reliable' ≥ minutes",1,value=900,step=50,key="sc_rf")

    if sf is None: st.info("Upload a scouting file to begin."); st.stop()
    try: dfr=load_player_data(sf)
    except Exception as e: st.error(f"Load error: {e}"); st.stop()
    dfr.columns=[str(c).strip() for c in dfr.columns]
    cs_=standard_columns(dfr); ac=[None]+dfr.columns.tolist()

    with st.sidebar:
        st.markdown("### 🧩 Columns")
        pcol=st.selectbox("Player",ac,index=ac.index(cs_["player"]) if cs_["player"] in ac else 0,key="sc_pc")
        tcol=st.selectbox("Team",ac,index=ac.index(cs_["team"]) if cs_["team"] in ac else 0,key="sc_tc")
        poscol=st.selectbox("Position",ac,index=ac.index(cs_["position"]) if cs_["position"] in ac else 0,key="sc_pos")
        agcol=st.selectbox("Age",ac,index=ac.index(cs_["age"]) if cs_["age"] in ac else 0,key="sc_ag")
        mincol=st.selectbox("Minutes",ac,index=ac.index(cs_["minutes"]) if cs_["minutes"] in ac else 0,key="sc_min")

    if not pcol: st.error("Choose a player column."); st.stop()
    metrics_=numeric_metrics(dfr,[pcol,tcol,poscol,agcol,mincol])
    roles=list(ROLE_TEMPLATES.keys())
    default_role=(_ir(dfr[poscol].dropna().iloc[0])
                  if poscol and poscol in dfr.columns and dfr[poscol].notna().any()
                  else roles[0])
    role=st.selectbox("Role template",roles,index=roles.index(default_role),key="sc_role")
    template_metrics_=[m for group in ROLE_TEMPLATES[role].values() for m in group]
    matched_=_match_tm(metrics_,template_metrics_)
    selected_=st.multiselect("Metrics",metrics_,default=matched_[:12] if matched_ else metrics_[:12],key="sc_metrics")
    lower_better=st.multiselect("Lower is better",selected_,
                                 default=[m for m in selected_ if any(w in m.lower() for w in NEGATIVE_METRIC_WORDS)],
                                 key="sc_lb")
    if not selected_: st.info("Choose at least one metric."); st.stop()
    gc_=poscol if scp=="Same position" and poscol else None
    scored=_ms(dfr,selected_,lower_better,gc=gc_,mc_arg=mincol if ur else None,mf=rf)
    if mincol and mincol in scored.columns:
        scored=scored[pd.to_numeric(scored[mincol],errors="coerce").fillna(0)>=float(mmg)].copy()
    scored=scored.sort_values("Adjusted Score",ascending=False)
    c1,c2,c3=st.columns(3)
    c1.metric("Players",len(scored)); c2.metric("Metrics",len(selected_))
    c3.metric("Top score","-" if scored.empty else scored["Adjusted Score"].iloc[0])
    show_cols=[c for c in [pcol,tcol,poscol,agcol,mincol,"Scouting Score","Reliability","Adjusted Score"]
               if c and c in scored.columns]
    st.dataframe(scored[show_cols].head(80),use_container_width=True)
    st.download_button("⬇ Scouting CSV",
                       scored.to_csv(index=False).encode("utf-8-sig"),
                       "scouting_results.csv","text/csv",key="sc_results")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 🖱️  TAGGING TOOL
# ─────────────────────────────────────────────────────────────────────────────
if section == "🖱️ Tagging Tool":
    hdr("🖱️","Tagging Tool",
        "Click the pitch to tag events, passes, carries, shots and more")

    for key,default in [("tag_events",[]),("tag_start",None),("tag_last_click",None)]:
        if key not in st.session_state:
            st.session_state[key]=default

    if not HAS_SIC:
        st.warning("Install **streamlit-image-coordinates** to enable click-to-tag. "
                   "Use Manual Entry below in the meantime.")

    L,R=st.columns([1,2.2])
    with L:
        st.markdown('<div class="card"><div class="ctitle">Tag Settings</div>',unsafe_allow_html=True)
        tn=st.selectbox("Theme",ALL_THEMES,
                        index=ALL_THEMES.index("The Athletic Dark") if "The Athletic Dark" in ALL_THEMES else 0,
                        key="tag_tn")
        ps_tag=st.selectbox("Pitch shape",["Rectangular","Square"],key="tag_ps")
        pm_tag="rect" if ps_tag=="Rectangular" else "square"
        pw_tag=st.slider("Pitch width",50.0,80.0,68.0,1.0,key="tag_pw") if pm_tag=="rect" else 100.0
        display_w=st.slider("Display width",600,1100,900,10,key="tag_dw")
        show_thirds=st.checkbox("Show thirds",True,key="tag_thirds")
        event_type=st.selectbox("Event type",
                                 ["pass","carry","dribble","cross","shot","touch","defensive action","recovery"],
                                 key="tag_et")
        outcome=st.selectbox("Outcome",
                              ["successful","unsuccessful","key pass","assist",
                               "goal","ontarget","off target","blocked","touch"],
                              key="tag_out")
        two_click=event_type in ["pass","carry","dribble","cross"]
        if two_click:
            st.info(f"**{event_type.title()}**: click START then END point.")
        player=st.text_input("Player","",key="tag_player")
        team  =st.text_input("Team","",key="tag_team")
        minute=st.number_input("Minute",0,130,0,key="tag_min")
        xg_val=st.number_input("xG",0.0,1.0,0.0,0.01,key="tag_xg") if event_type=="shot" else 0.0
        col_ev =st.color_picker("Marker color","#22C55E",key="tag_col")
        edge_ev=st.color_picker("Marker edge","#FFFFFF",key="tag_edge")
        marker_name=st.selectbox("Marker",MKL,index=1,key="tag_marker")
        marker_ev=MARKER_OPTS[marker_name]
        size_ev=st.slider("Marker size",4,22,9,key="tag_size")
        c1,c2,c3=st.columns(3)
        with c1:
            if st.button("↩ Undo",key="tag_undo") and st.session_state["tag_events"]:
                st.session_state["tag_events"].pop()
                st.session_state["tag_start"]=None
                st.rerun()
        with c2:
            if st.button("🗑 Clear",key="tag_clear"):
                st.session_state["tag_events"]=[]; st.session_state["tag_start"]=None; st.rerun()
        with c3:
            if two_click and st.session_state["tag_start"]:
                if st.button("✖ Cancel",key="tag_cancel"):
                    st.session_state["tag_start"]=None; st.rerun()
        if two_click:
            if st.session_state["tag_start"]:
                sp=st.session_state["tag_start"]
                st.success(f"Start: ({sp[0]:.1f},{sp[1]:.1f}) — click END")
            else:
                st.info("Click pitch to set START point")
        st.markdown('</div>',unsafe_allow_html=True)

        with st.expander("✏️ Manual Event Entry"):
            me_x=st.number_input("X",0.0,100.0,50.0,key="me_x")
            me_y=st.number_input("Y",0.0,float(pw_tag),float(pw_tag)/2,key="me_y")
            me_x2=st.number_input("X2",0.0,100.0,60.0,key="me_x2") if two_click else None
            me_y2=st.number_input("Y2",0.0,float(pw_tag),float(pw_tag)/2,key="me_y2") if two_click else None
            if st.button("Add Event",key="me_add"):
                ev={"event_type":event_type,"outcome":outcome,
                    "x":me_x,"y":me_y,"player":player,"team":team,
                    "minute":minute,"xg":xg_val if event_type=="shot" else 0,
                    "start_color":col_ev,"start_edge":edge_ev,
                    "start_marker":marker_ev,"start_size":size_ev,
                    "arrow_color":col_ev}
                if two_click and me_x2 is not None:
                    ev["x2"]=me_x2; ev["y2"]=me_y2
                st.session_state["tag_events"].append(ev)
                st.rerun()

    with R:
        try:
            img_pil,img_w,img_h,y_max_tag,pad_tag=draw_tagging_pitch(
                tn,pm_tag,pw_tag,display_w,show_thirds,
                st.session_state["tag_events"],
                start_point=st.session_state["tag_start"],
                current_marker=marker_ev,current_color=col_ev,
                current_edge=edge_ev,current_size=size_ev)
        except Exception as e:
            st.error(f"Pitch render error: {e}"); st.stop()

        if HAS_SIC:
            click=streamlit_image_coordinates(img_pil,key="tag_pitch")
            if click and "x" in click and "y" in click:
                key_c=(int(click["x"]),int(click["y"]))
                if key_c!=st.session_state["tag_last_click"]:
                    st.session_state["tag_last_click"]=key_c
                    inner_w=img_w-2*pad_tag; inner_h=img_h-2*pad_tag
                    px=min(max(int(click["x"])-pad_tag,0),inner_w)
                    py=min(max(int(click["y"])-pad_tag,0),inner_h)
                    x_pitch=round((px/inner_w)*100.0,2)
                    y_pitch=round((1.0-py/inner_h)*float(y_max_tag),2)
                    if two_click and st.session_state["tag_start"] is None:
                        st.session_state["tag_start"]=(x_pitch,y_pitch)
                    else:
                        ev={"event_type":event_type,"outcome":outcome,
                            "x":x_pitch,"y":y_pitch,"player":player,"team":team,
                            "minute":minute,"xg":xg_val if event_type=="shot" else 0,
                            "start_color":col_ev,"start_edge":edge_ev,
                            "start_marker":marker_ev,"start_size":size_ev,
                            "arrow_color":col_ev}
                        if two_click and st.session_state["tag_start"] is not None:
                            sx,sy=st.session_state["tag_start"]
                            ev["x"]=sx; ev["y"]=sy; ev["x2"]=x_pitch; ev["y2"]=y_pitch
                            st.session_state["tag_start"]=None
                        st.session_state["tag_events"].append(ev)
                    st.rerun()
        else:
            buf=io.BytesIO(); img_pil.save(buf,format="PNG")
            st.image(buf.getvalue(),use_container_width=True)
            st.info("Install streamlit-image-coordinates to enable click-to-tag.")

        events_df=pd.DataFrame(st.session_state["tag_events"])
        if not events_df.empty:
            display_cols=[c for c in ["event_type","outcome","x","y","x2","y2","player","team","minute","xg"]
                          if c in events_df.columns]
            st.dataframe(events_df[display_cols],use_container_width=True)
            st.download_button("⬇ Tagged Events CSV",
                               events_df.to_csv(index=False).encode("utf-8-sig"),
                               "tagged_events.csv","text/csv",key="tag_csv")
            if st.button("📊 Generate Pitch Chart from Tags",key="tag_chart"):
                try:
                    ec=events_df.copy()
                    for c in ["x","y","x2","y2"]:
                        if c in ec.columns: ec[c]=pd.to_numeric(ec[c],errors="coerce")
                    ET_MAP={"defensive action":"defensive","recovery":"defensive",
                            "carry":"carry","dribble":"carry","cross":"pass"}
                    if "event_type" in ec.columns:
                        ec["event_type"]=(ec["event_type"].astype(str).str.strip().str.lower()
                                          .map(lambda v:ET_MAP.get(v,v)))
                    prepared=prepare_df_for_charts(ensure_outcome(ec),
                                                   pitch_mode=pm_tag,pitch_width=pw_tag)
                    et_counts=prepared.get("event_type",pd.Series(dtype=str)).value_counts()
                    has_xy2=(prepared["x2"].notna()&prepared["y2"].notna()).any()
                    if "shot" in et_counts.index and et_counts["shot"]>0:
                        fig=shot_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn)
                        fig.axes[0].set_title("Tagged Shots",color=THEMES[tn]["text"],fontsize=14,weight="bold")
                    elif "pass" in et_counts.index and et_counts["pass"]>0:
                        fig=pass_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn)
                        fig.axes[0].set_title("Tagged Passes",color=THEMES[tn]["text"],fontsize=14,weight="bold")
                    elif "defensive" in et_counts.index:
                        fig=defensive_actions_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn)
                        fig.axes[0].set_title("Tagged Defensive Actions",color=THEMES[tn]["text"],fontsize=14,weight="bold")
                    elif "carry" in et_counts.index and has_xy2:
                        fig=progressive_carries_map(prepared,title="Tagged Carries",
                                                     theme_name=tn,pitch_mode=pm_tag,
                                                     pitch_width=pw_tag,min_distance=0.0)
                    else:
                        fig=touch_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn)
                        fig.axes[0].set_title("Tagged Events",color=THEMES[tn]["text"],fontsize=14,weight="bold")
                    st.image(_bytes(fig,dpi=150),use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Chart error: {e}")
        else:
            st.info("No events tagged yet. Click the pitch above to start tagging.")
    st.stop()
