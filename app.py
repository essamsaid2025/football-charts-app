"""
Football Analysis Suite  v2  —  Full modular Streamlit app
Each chart is fully standalone: own inputs, legend controls, image overlays, exports.
"""

import os, io, math, tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# ── core chart library ────────────────────────────────────────────────────────
from charts import (
    load_data, prepare_df_for_charts,
    pizza_chart, mpl_pizza_dark, athletic_pizza, shot_detail_card,
    defensive_regains_map, outcome_bar, start_location_heatmap,
    touch_map, pass_map, shot_map, defensive_actions_map,
    THEMES, make_pitch,
)
# ── extra charts ──────────────────────────────────────────────────────────────
from charts_extra import (
    register_extra_themes, overlay_image_on_fig,
    goal_location_map, goal_mouth_map,
    vertical_event_map, progressive_carries_map,
    pressure_map, xg_timeline, passing_network,
    draw_tagging_pitch,
)
register_extra_themes()

# ── scouting tools ────────────────────────────────────────────────────────────
try:
    from scouting_tools_v2 import (
        ROLE_TEMPLATES, load_player_data, standard_columns,
        numeric_metrics, coerce_numeric, match_template_metrics,
        add_percentiles_and_score, player_profile, profile_text,
        similar_players, auto_dataset_insights, comparison_chart,
        radar_chart, make_template_csv, recommendation_text,
    )
except Exception:
    ROLE_TEMPLATES = None; recommendation_text = None

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Football Analysis Suite", layout="wide",
                   initial_sidebar_state="expanded")

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
section[data-testid="stSidebar"] div[role="radiogroup"] label{background:rgba(13,31,53,.7)!important;border:1px solid var(--bdr)!important;border-radius:8px!important;padding:6px 10px!important;margin-bottom:5px!important;}
.hdr{display:flex;align-items:center;gap:14px;padding:16px 20px;background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(124,58,237,.06));border:1px solid rgba(0,212,255,.15);border-radius:16px;margin-bottom:18px;}
.hdr .ic{font-size:2rem;}.hdr .ti{font-size:1.5rem;font-weight:800;line-height:1.1;}
.hdr .su{font-size:.87rem;color:var(--mu);margin-top:3px;}
.card{background:var(--s1);border:1px solid var(--bdr);border-radius:14px;padding:16px;margin-bottom:12px;}
.ctitle{font-size:.85rem;font-weight:700;color:var(--ac);text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;}
.krow{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:12px;}
.kc{background:var(--s2);border:1px solid var(--bdr);border-radius:10px;padding:9px 13px;text-align:center;flex:1;min-width:80px;}
.kc .kv{font-size:1.1rem;font-weight:800;}.kc .kl{font-size:.73rem;color:var(--mu);margin-top:2px;}
.empty{border:1.5px dashed var(--bdr);background:rgba(13,31,53,.5);border-radius:14px;padding:36px 20px;text-align:center;color:var(--mu);}
.empty .ei{font-size:2.2rem;margin-bottom:8px;}.empty .et{font-size:1rem;font-weight:700;color:var(--tx);margin-bottom:5px;}
.divd{height:1px;background:linear-gradient(90deg,transparent,var(--bdr),transparent);margin:14px 0;}
.stButton>button{border-radius:10px;border:1px solid var(--bdr);background:linear-gradient(135deg,#0ea5e9,#2563eb);color:white;font-weight:700;padding:.55rem 1rem;width:100%;}
.stDownloadButton>button{border-radius:10px;border:1px solid var(--bdr);background:var(--s2);color:var(--tx);font-weight:600;width:100%;}
div[data-testid="stFileUploader"] section{background:var(--s2);border:1.5px dashed var(--bdr);border-radius:12px;}
.stExpander{background:var(--s2)!important;border:1px solid var(--bdr)!important;border-radius:12px!important;}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
MARKER_OPTS = {"None":None,"Circle":"o","Star":"*","Triangle ▲":"^","Triangle ▼":"v",
               "Square":"s","Diamond":"D","Plus":"+","X":"x","Pentagon":"p","Hexagon":"h"}
MKL = list(MARKER_OPTS.keys())

def _bytes(fig, dpi=240):
    b = io.BytesIO(); fig.savefig(b,format="png",dpi=dpi,bbox_inches="tight",pad_inches=.18); b.seek(0); return b.read()

def _pdf(fig):
    b = io.BytesIO()
    with PdfPages(b) as p: p.savefig(fig,bbox_inches="tight",pad_inches=.18)
    b.seek(0); return b.read()

def _clean(s): return pd.to_numeric(s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False).str.strip(),errors="coerce")

def hdr(icon, title, sub=""):
    s = f'<div class="su">{sub}</div>' if sub else ""
    st.markdown(f'<div class="hdr"><div class="ic">{icon}</div><div><div class="ti">{title}</div>{s}</div></div>',unsafe_allow_html=True)

def kpi(l,v): return f'<div class="kc"><div class="kv">{v}</div><div class="kl">{l}</div></div>'

def empty(icon="📊", title="Configure and generate", msg=""):
    st.markdown(f'<div class="empty"><div class="ei">{icon}</div><div class="et">{title}</div><div>{msg}</div></div>',unsafe_allow_html=True)

def dl_row(fig, name):
    c1,c2 = st.columns(2)
    with c1: st.download_button("⬇ PNG",_bytes(fig),f"{name}.png","image/png",key=f"p_{name}_{id(fig)}")
    with c2: st.download_button("⬇ PDF",_pdf(fig), f"{name}.pdf","application/pdf",key=f"f_{name}_{id(fig)}")
    plt.close(fig)

def preview(fig, name):
    st.image(_bytes(fig,dpi=180), use_container_width=True)
    dl_row(fig, name)

def load_ev(f):
    n = getattr(f,"name","x"); ext = n.lower().rsplit(".",1)[-1]
    if ext=="csv":
        for enc in ["utf-8","utf-8-sig","cp1256","latin1"]:
            try: f.seek(0); return pd.read_csv(f,encoding=enc)
            except: pass
    return pd.read_excel(f)

def ensure_outcome(df):
    if "outcome" in df.columns: return df
    for c in df.columns:
        if c.lower().strip() in ["event","result","type","event_type"]:
            df=df.copy(); df["outcome"]=df[c]; return df
    df=df.copy(); df["outcome"]="unknown"; return df

def load_img(f):
    if f is None: return None
    try: return Image.open(f).convert("RGBA")
    except: return None

def _pct(s, hi=True):
    x=pd.to_numeric(s,errors="coerce"); p=x.rank(pct=True,method="average")*100
    return (100-p if not hi else p).clip(0,100)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED IMAGE OVERLAY CONTROLS (returns dict of overlay params)
# ─────────────────────────────────────────────────────────────────────────────
def img_overlay_controls(prefix, label="Overlay Image", default_x=0.02, default_y=0.88,
                          default_w=0.10, default_h=0.10):
    f = st.file_uploader(label, type=["png","jpg","jpeg"], key=f"{prefix}_img_f")
    img = load_img(f)
    if img:
        col_x,col_y = st.columns(2)
        with col_x: x = st.slider("Image X (fig %)",  0,95, int(default_x*100), key=f"{prefix}_ix")/100
        with col_y: y = st.slider("Image Y (fig %)",  0,95, int(default_y*100), key=f"{prefix}_iy")/100
        col_w,col_h = st.columns(2)
        with col_w: w = st.slider("Image W (fig %)",  4,25,  int(default_w*100), key=f"{prefix}_iw")/100
        with col_h: h = st.slider("Image H (fig %)",  4,25,  int(default_h*100), key=f"{prefix}_ih")/100
        circle = st.checkbox("Circle crop", False, key=f"{prefix}_ic")
        bdr_lw = st.slider("Border width", 0.0, 5.0, 0.0, 0.5, key=f"{prefix}_ib")
    else:
        x,y,w,h,circle,bdr_lw = default_x,default_y,default_w,default_h,False,0.0
    return dict(img=img, x=x, y=y, w=w, h=h, circle=circle, bdr_lw=bdr_lw)

def apply_overlay(fig, ov):
    if ov["img"]:
        overlay_image_on_fig(fig, ov["img"], x=ov["x"], y=ov["y"], w=ov["w"], h=ov["h"],
                             circle_crop=ov["circle"], border_lw=ov["bdr_lw"])

# ─────────────────────────────────────────────────────────────────────────────
# SHARED LEGEND CONTROLS  (returns list of active legend keys)
# ─────────────────────────────────────────────────────────────────────────────
def legend_controls(prefix, all_items: list, default_active: list = None) -> list:
    if default_active is None: default_active = all_items
    st.markdown("**Legend items to show**")
    active = []
    cols = st.columns(min(3, len(all_items)))
    for i, item in enumerate(all_items):
        with cols[i % len(cols)]:
            if st.checkbox(item, value=(item in default_active), key=f"{prefix}_leg_{i}"):
                active.append(item)
    return active

# ─────────────────────────────────────────────────────────────────────────────
# EVENT DATA PREP  (used by all pitch-map sections)
# ─────────────────────────────────────────────────────────────────────────────
def ev_sidebar(prefix):
    with st.sidebar:
        st.markdown("### 📂 Event Data")
        f = st.file_uploader("CSV / Excel", type=["csv","xlsx","xls"], key=f"{prefix}_f")
        st.markdown('<div class="divd"></div>',unsafe_allow_html=True)
        st.markdown("### 🎨 Theme & Pitch")
        tn  = st.selectbox("Theme", ALL_THEMES, index=ALL_THEMES.index("The Athletic Dark") if "The Athletic Dark" in ALL_THEMES else 0, key=f"{prefix}_tn")
        adir= st.selectbox("Attack direction",["Left → Right","Right → Left"], key=f"{prefix}_ad")
        fy  = st.checkbox("Flip Y axis", False, key=f"{prefix}_fy")
        ps  = st.selectbox("Pitch shape",["Rectangular","Square"], key=f"{prefix}_ps")
        pm  = "rect" if ps=="Rectangular" else "square"
        pw  = st.slider("Pitch width",50.0,80.0,68.0,1.0,key=f"{prefix}_pw") if pm=="rect" else 100.0
    S   = dict(tn=tn, ad="ltr" if "Left" in adir else "rtl", fy=fy, pm=pm, pw=pw)
    if f is None: return None, S
    try:
        raw = load_ev(f); raw = ensure_outcome(raw)
        df  = prepare_df_for_charts(raw, attack_direction=S["ad"], flip_y=S["fy"],
                                    pitch_mode=S["pm"], pitch_width=S["pw"], xg_method="zone")
        return df, S
    except Exception as e:
        st.error(f"Load error: {e}"); return None, S

def nofile(): empty("📂","No file uploaded","Upload an event-data file from the sidebar.")

# ─────────────────────────────────────────────────────────────────────────────
# COLOR/MARKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def shot_cm(px):
    c1,c2=st.columns(2)
    with c1:
        co=st.color_picker("Off target","#FF8A00",key=f"{px}_co")
        con=st.color_picker("On target","#00C2FF",key=f"{px}_con")
    with c2:
        cg=st.color_picker("Goal","#00FF6A",key=f"{px}_cg")
        cb=st.color_picker("Blocked","#AAAAAA",key=f"{px}_cb")
    c3,c4=st.columns(2)
    with c3:
        mo=MKL.index("Triangle ▲"); mo=st.selectbox("Marker off-target",MKL,index=mo,key=f"{px}_mo")
        mon=MKL.index("Diamond"); mon=st.selectbox("Marker on-target",MKL,index=mon,key=f"{px}_mon")
    with c4:
        mg=MKL.index("Star"); mg=st.selectbox("Marker goal",MKL,index=mg,key=f"{px}_mg")
        mb=MKL.index("Square"); mb=st.selectbox("Marker blocked",MKL,index=mb,key=f"{px}_mb")
    colors ={"off target":co,"ontarget":con,"goal":cg,"blocked":cb}
    markers={"off target":MARKER_OPTS[mo],"ontarget":MARKER_OPTS[mon],"goal":MARKER_OPTS[mg],"blocked":MARKER_OPTS[mb]}
    return colors,markers

def pass_cm(px):
    c1,c2=st.columns(2)
    with c1:
        cs=st.color_picker("Successful","#00FF6A",key=f"{px}_cs")
        cu=st.color_picker("Unsuccessful","#FF4D4D",key=f"{px}_cu")
    with c2:
        ck=st.color_picker("Key pass","#00C2FF",key=f"{px}_ck")
        ca=st.color_picker("Assist","#FFD400",key=f"{px}_ca")
    c3,c4=st.columns(2)
    with c3:
        ms=st.selectbox("Marker succ",MKL,index=MKL.index("Circle"),key=f"{px}_ms")
        mu=st.selectbox("Marker unsucc",MKL,index=MKL.index("X"),key=f"{px}_mu")
    with c4:
        mk=st.selectbox("Marker key",MKL,index=MKL.index("Diamond"),key=f"{px}_mk")
        ma=st.selectbox("Marker assist",MKL,index=MKL.index("Star"),key=f"{px}_ma")
    colors ={"successful":cs,"unsuccessful":cu,"key pass":ck,"assist":ca}
    markers={"successful":MARKER_OPTS[ms],"unsuccessful":MARKER_OPTS[mu],"key pass":MARKER_OPTS[mk],"assist":MARKER_OPTS[ma]}
    return colors,markers

def def_cm(px):
    defs={"interception":("#00C2FF","Circle"),"tackle":("#FF8A00","Square"),
          "recovery":("#00FF6A","Diamond"),"aerial_duel":("#FFD400","Triangle ▲"),
          "ground_duel":("#FF4D4D","X"),"clearance":("#A78BFA","Star")}
    cols_o={}; mkrs={}
    c1,c2=st.columns(2)
    for i,(act,(dc,dm)) in enumerate(defs.items()):
        lbl=act.replace("_"," ").title()
        with (c1 if i%2==0 else c2):
            cols_o[act]=st.color_picker(lbl,dc,key=f"{px}_dc_{act}")
            ml=st.selectbox(f"{lbl} marker",MKL,index=MKL.index(dm),key=f"{px}_dm_{act}")
            mkrs[act]=MARKER_OPTS[ml]
    return cols_o,mkrs

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ Football Analysis Suite")
    st.markdown('<div class="divd"></div>',unsafe_allow_html=True)
    section = st.radio("Navigate", [
        "🏠 Home",
        "⚔️ Attacking Charts",
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
    hdr("⚽","Football Analysis Suite","Professional charts · scouting · tagging — fully modular")
    tiles=[
        ("⚔️","Attacking Charts","Shot maps, goal location (Opta-style), touch maps, xG timeline","x, y, outcome"),
        ("🛡️","Defensive Charts","Defensive action maps, ball regains heatmap, pressure map","x, y, outcome / pressure col"),
        ("🔄","Distribution Charts","Pass maps (all filters), progressive carries, passing network","x, y, x2, y2, outcome"),
        ("🎯","Specialist Charts","Goal mouth map (CannonStats-style), vertical pitch maps","x, y, y2/z for goal mouth"),
        ("🍕","Radars & Pizza","MPL pizza, Athletic style, percentile bars, scatter plots","Player col + numeric metrics"),
        ("🧠","Player Scouting","Role templates, scoring, shortlists, recommendations","Player col + numeric metrics"),
        ("🖱️","Tagging Tool","Click-to-tag events on an interactive pitch — no file needed","None"),
    ]
    cols=st.columns(3)
    for i,(ic,ti,de,req) in enumerate(tiles):
        with cols[i%3]:
            st.markdown(f"""<div class="card" style="min-height:130px;">
            <div style="font-size:1.5rem;mb-2">{ic}</div>
            <div style="font-weight:800;font-size:.98rem;color:var(--tx);margin:6px 0 4px">{ti}</div>
            <div style="font-size:.82rem;color:var(--mu);margin-bottom:6px">{de}</div>
            <div style="font-size:.75rem;color:var(--ac);">Requires: {req}</div>
            </div>""",unsafe_allow_html=True)
    st.markdown('<div class="divd"></div>',unsafe_allow_html=True)
    st.markdown("""
**File formats**
- **Event data** (CSV/Excel) — needs `outcome`, `x`, `y`. Used by all pitch maps.
- **Player metrics** (CSV/Excel) — needs a `Player` column + numeric stat columns. Used by radar/pizza & scouting.
- **Tagging tool** — no file needed, just click the pitch.

**New in v2**
- Goal Location Map (Opta Analyst style) — shows where goals were scored on a half-pitch with stats panel
- Goal Mouth Map (CannonStats style) — plots shots on the actual goal face, sized by xG
- Fully fixed vertical pitch (no more half-pitch clipping)
- Progressive Carries Map, Pressure Map, xG Timeline, Passing Network
- Full legend control (show/hide each item individually)
- Image overlay on any chart (position, size, circle-crop)
- 7 new themes: Opta Light, Athletic FC Dark/Light, Night Blue, Broadcast Green, Whoscored Dark, Statsbomb Light
""")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# ⚔️  ATTACKING CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "⚔️ Attacking Charts":
    df,S = ev_sidebar("atk")
    hdr("⚔️","Attacking Charts","Shot map · Goal location · Touch map · Heatmap · xG Timeline")

    tabs = st.tabs(["Shot Map","Goal Location Map","Touch Map","Start Heatmap","xG Timeline"])

    # ── Shot Map ──────────────────────────────────────────────────────────────
    with tabs[0]:
        if df is None: nofile()
        else:
            L,R = st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Shot Map Settings</div>',unsafe_allow_html=True)
                t = st.text_input("Title","Shot Map",key="sm_t")
                show_xg = st.checkbox("Show xG labels",True,key="sm_xg")
                sc,sm = shot_cm("sm")
                al = legend_controls("sm_leg",["off target","ontarget","goal","blocked"],["off target","ontarget","goal","blocked"])
                ov = img_overlay_controls("sm_ov")
                gen=st.button("Generate",key="sm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    sc_f={k:v for k,v in sc.items() if k in al}
                    sm_f={k:v for k,v in sm.items() if k in al}
                    fig=shot_map(df,shot_colors=sc_f,shot_markers=sm_f,pitch_mode=S["pm"],
                                 pitch_width=S["pw"],show_xg=show_xg,theme_name=S["tn"])
                    fig.axes[0].set_title(t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["atk_sm"]=fig
                if "atk_sm" in st.session_state: preview(st.session_state["atk_sm"],"shot_map")
                else: empty("🎯","Configure and generate")

    # ── Goal Location Map ─────────────────────────────────────────────────────
    with tabs[1]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Goal Location Map (Opta Style)</div>',unsafe_allow_html=True)
                gl_title  = st.text_input("Title","Goal Location Map",key="gl_t")
                gl_player = st.text_input("Player name","",key="gl_pn")
                gl_sub    = st.text_input("Subtitle","Premier League 2024-25",key="gl_su")
                gl_tn     = st.selectbox("Theme",ALL_THEMES,index=ALL_THEMES.index("Opta Light") if "Opta Light" in ALL_THEMES else 0,key="gl_tn")
                gl_color  = st.color_picker("Goal dot color","#C8102E",key="gl_gc")
                gl_edge   = st.color_picker("Goal dot edge","#8B0000",key="gl_ge")
                gl_ds     = st.slider("Dot size",60,400,160,key="gl_ds")
                gl_half   = st.checkbox("Show attacking half only",True,key="gl_half")
                st.markdown("**Stats panel** (up to 6 rows)")
                n_stats = st.number_input("Number of stat rows",1,6,3,key="gl_ns")
                stats=[]
                for i in range(int(n_stats)):
                    c1,c2=st.columns(2)
                    with c1: v=st.text_input(f"Value {i+1}","",key=f"gl_sv{i}")
                    with c2: lb=st.text_input(f"Label {i+1}","",key=f"gl_sl{i}")
                    if v or lb: stats.append((v,lb))
                logo_ov = img_overlay_controls("gl_logo","Logo image",default_x=0.72,default_y=0.88,default_w=0.12,default_h=0.10)
                player_ov=img_overlay_controls("gl_plyr","Player/club image",default_x=0.02,default_y=0.88,default_w=0.10,default_h=0.10)
                gen=st.button("Generate",key="gl_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=goal_location_map(df,title=gl_title,player_name=gl_player,subtitle=gl_sub,
                        stat_labels=stats or None,theme_name=gl_tn,pitch_mode=S["pm"],
                        pitch_width=S["pw"],goal_color=gl_color,goal_edge=gl_edge,dot_size=gl_ds,
                        show_pitch_half_only=gl_half,
                        logo_img=logo_ov["img"],
                        logo_x=logo_ov["x"],logo_y=logo_ov["y"],logo_w=logo_ov["w"],logo_h=logo_ov["h"],
                        player_img=player_ov["img"],
                        player_img_x=player_ov["x"],player_img_y=player_ov["y"],
                        player_img_w=player_ov["w"],player_img_h=player_ov["h"])
                    st.session_state["atk_gl"]=fig
                if "atk_gl" in st.session_state: preview(st.session_state["atk_gl"],"goal_location_map")
                else: empty("🥅","Configure and generate")

    # ── Touch Map ─────────────────────────────────────────────────────────────
    with tabs[2]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Touch Map</div>',unsafe_allow_html=True)
                tm_t =st.text_input("Title","Touch Map",key="tm_t")
                tm_c =st.color_picker("Dot color","#34D5FF",key="tm_c")
                tm_e =st.color_picker("Edge color","#0B0F14",key="tm_e")
                tm_s =st.slider("Dot size",60,500,220,key="tm_s")
                tm_a =st.slider("Opacity",20,100,90,key="tm_a")/100
                tm_ml=st.selectbox("Marker",MKL,index=1,key="tm_ml")
                tm_v =st.checkbox("Vertical pitch",False,key="tm_v")
                al   =legend_controls("tm_leg",["Touches"],["Touches"])
                ov   =img_overlay_controls("tm_ov")
                gen  =st.button("Generate",key="tm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=touch_map(df,pitch_mode=S["pm"],pitch_width=S["pw"],theme_name=S["tn"],
                                  dot_color=tm_c,edge_color=tm_e,dot_size=tm_s,alpha=tm_a,
                                  marker=MARKER_OPTS[tm_ml],vertical_pitch=tm_v)
                    fig.axes[0].set_title(tm_t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["atk_tm"]=fig
                if "atk_tm" in st.session_state: preview(st.session_state["atk_tm"],"touch_map")
                else: empty("👟","Configure and generate")

    # ── Start Heatmap ─────────────────────────────────────────────────────────
    with tabs[3]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Start Heatmap</div>',unsafe_allow_html=True)
                ht_t=st.text_input("Title","Start Location Heatmap",key="ht_t")
                ht_v=st.checkbox("Vertical pitch",False,key="ht_v")
                ov  =img_overlay_controls("ht_ov")
                gen =st.button("Generate",key="ht_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=start_location_heatmap(df,pitch_mode=S["pm"],pitch_width=S["pw"],
                                               theme_name=S["tn"],vertical_pitch=ht_v)
                    fig.axes[0].set_title(ht_t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["atk_ht"]=fig
                if "atk_ht" in st.session_state: preview(st.session_state["atk_ht"],"start_heatmap")
                else: empty("🌡️","Configure and generate")

    # ── xG Timeline ───────────────────────────────────────────────────────────
    with tabs[4]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">xG Timeline</div>',unsafe_allow_html=True)
                xt_t  =st.text_input("Title","xG Timeline",key="xt_t")
                xt_a  =st.text_input("Team A name","Home",key="xt_a")
                xt_b  =st.text_input("Team B name","Away",key="xt_b")
                tcol_opts=[None]+list(df.columns)
                xt_tc =st.selectbox("Team column (optional)",tcol_opts,key="xt_tc")
                mcol_opts=[None]+list(df.columns)
                xt_mc =st.selectbox("Minute column (optional)",mcol_opts,key="xt_mc")
                xt_ca =st.color_picker("Team A color","#00C2FF",key="xt_ca")
                xt_cb =st.color_picker("Team B color","#FF4060",key="xt_cb")
                ov    =img_overlay_controls("xt_ov",default_x=0.85,default_y=0.88)
                gen   =st.button("Generate",key="xt_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=xg_timeline(df,team_a=xt_a,team_b=xt_b,
                                    team_col=xt_tc,minute_col=xt_mc,
                                    title=xt_t,color_a=xt_ca,color_b=xt_cb,theme_name=S["tn"])
                    apply_overlay(fig,ov)
                    st.session_state["atk_xt"]=fig
                if "atk_xt" in st.session_state: preview(st.session_state["atk_xt"],"xg_timeline")
                else: empty("📈","Configure and generate")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 🛡️  DEFENSIVE CHARTS
# ─────────────────────────────────────────────────────────────────────────────
if section == "🛡️ Defensive Charts":
    df,S = ev_sidebar("def")
    hdr("🛡️","Defensive Charts","Actions map · Regains heatmap · Pressure map · Outcome bar")

    tabs=st.tabs(["Defensive Actions","Ball Regains Map","Pressure Map","Outcome Distribution"])

    with tabs[0]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Defensive Actions Map</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Defensive Actions",key="da_t")
                dc,dm=def_cm("da")
                all_acts=["interception","tackle","recovery","aerial_duel","ground_duel","clearance"]
                al=legend_controls("da_leg",all_acts,all_acts)
                dv=st.checkbox("Vertical pitch",False,key="da_v")
                ov=img_overlay_controls("da_ov")
                gen=st.button("Generate",key="da_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    dc_f={k:v for k,v in dc.items() if k in al}
                    dm_f={k:v for k,v in dm.items() if k in al}
                    fig=defensive_actions_map(df,def_colors=dc_f,def_markers=dm_f,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],
                                              theme_name=S["tn"],vertical_pitch=dv)
                    fig.axes[0].set_title(t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["def_da"]=fig
                if "def_da" in st.session_state: preview(st.session_state["def_da"],"defensive_actions")
                else: empty("🛡️","Configure and generate")

    with tabs[1]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Ball Regains Heatmap</div>',unsafe_allow_html=True)
                t   =st.text_input("Title","Ball Regains Map",key="br_t")
                bz  =st.checkbox("Show zone counts",True,key="br_z")
                bms =st.slider("Marker size",60,260,110,key="br_ms")
                bal =st.slider("Zone opacity",20,100,78,key="br_al")/100
                dv  =st.checkbox("Vertical pitch",False,key="br_v")
                dc,dm=def_cm("br")
                all_acts=["interception","tackle","recovery","aerial_duel","ground_duel","clearance"]
                al  =legend_controls("br_leg",all_acts,all_acts)
                ov  =img_overlay_controls("br_ov")
                gen =st.button("Generate",key="br_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    dc_f={k:v for k,v in dc.items() if k in al}
                    dm_f={k:v for k,v in dm.items() if k in al}
                    fig=defensive_regains_map(df,title=t,def_colors=dc_f,def_markers=dm_f,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],theme_name=S["tn"],
                                              vertical_pitch=dv,marker_size=bms,zone_alpha=bal,
                                              show_zone_values=bz)
                    apply_overlay(fig,ov)
                    st.session_state["def_br"]=fig
                if "def_br" in st.session_state: preview(st.session_state["def_br"],"ball_regains")
                else: empty("🗺️","Configure and generate")

    with tabs[2]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Pressure Map</div>',unsafe_allow_html=True)
                t  =st.text_input("Title","Pressure Map",key="pr_t")
                pc_opts=[None]+list(df.columns)
                pc =st.selectbox("Pressure column (bool/yes-no)",pc_opts,key="pr_pc")
                pv =st.checkbox("Vertical pitch",False,key="pr_v")
                ov =img_overlay_controls("pr_ov")
                gen=st.button("Generate",key="pr_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=pressure_map(df,title=t,theme_name=S["tn"],pitch_mode=S["pm"],
                                     pitch_width=S["pw"],vertical_pitch=pv,
                                     pressure_col=pc or "pressure")
                    apply_overlay(fig,ov)
                    st.session_state["def_pr"]=fig
                if "def_pr" in st.session_state: preview(st.session_state["def_pr"],"pressure_map")
                else: empty("⚡","Configure and generate","Needs a 'pressure' boolean column or select one above")

    with tabs[3]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Outcome Distribution</div>',unsafe_allow_html=True)
                t=st.text_input("Title","Outcome Distribution",key="ob_t")
                all_outc=["successful","unsuccessful","key pass","assist","goal","ontarget","off target","blocked"]
                al=legend_controls("ob_leg",all_outc,all_outc)
                bar_colors={
                    "successful":st.color_picker("Successful","#00FF6A",key="ob_cs"),
                    "unsuccessful":st.color_picker("Unsuccessful","#FF4D4D",key="ob_cu"),
                    "key pass":st.color_picker("Key pass","#00C2FF",key="ob_ck"),
                    "assist":st.color_picker("Assist","#FFD400",key="ob_ca"),
                    "goal":st.color_picker("Goal","#00FF6A",key="ob_cg"),
                    "ontarget":st.color_picker("On target","#00C2FF",key="ob_co"),
                    "off target":st.color_picker("Off target","#FF8A00",key="ob_cf"),
                    "blocked":st.color_picker("Blocked","#AAAAAA",key="ob_cb"),
                }
                ov=img_overlay_controls("ob_ov")
                gen=st.button("Generate",key="ob_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    df_f=df[df["outcome"].isin(al)].copy() if al else df.copy()
                    fig=outcome_bar(df_f,bar_colors={k:v for k,v in bar_colors.items() if k in al},theme_name=S["tn"])
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
    df,S=ev_sidebar("dist")
    hdr("🔄","Distribution Charts","Pass map · Progressive carries · Passing network")

    tabs=st.tabs(["Pass Map","Progressive Carries","Passing Network"])

    with tabs[0]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Pass Map</div>',unsafe_allow_html=True)
                t  =st.text_input("Title","Pass Map",key="pm_t")
                pv =st.selectbox("Pass view",["All passes","Into Final Third","Into Penalty Box","Line-breaking","Progressive passes"],key="pm_v")
                ps =st.selectbox("Result scope",["Attempts (all)","Successful only","Unsuccessful only"],key="pm_s")
                ppk=st.slider("Min packing",1,5,1,key="pm_pk")
                pc,pm=pass_cm("pm")
                all_pass=["successful","unsuccessful","key pass","assist"]
                al=legend_controls("pm_leg",all_pass,all_pass)
                dv=st.checkbox("Vertical pitch",False,key="pm_v2")
                ov=img_overlay_controls("pm_ov")
                gen=st.button("Generate",key="pm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    pc_f={k:v for k,v in pc.items() if k in al}
                    pm_f={k:v for k,v in pm.items() if k in al}
                    fig=pass_map(df,pass_colors=pc_f,pass_markers=pm_f,
                                 pitch_mode=S["pm"],pitch_width=S["pw"],theme_name=S["tn"],
                                 vertical_pitch=dv,pass_view=pv,result_scope=ps,min_packing=ppk)
                    fig.axes[0].set_title(t,color=THEMES[S["tn"]]["text"],fontsize=16,weight="bold")
                    apply_overlay(fig,ov)
                    st.session_state["dist_pm"]=fig
                if "dist_pm" in st.session_state: preview(st.session_state["dist_pm"],"pass_map")
                else: empty("➡️","Configure and generate")

    with tabs[1]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Progressive Carries Map</div>',unsafe_allow_html=True)
                st.caption("Requires: x, y, x2, y2 columns. Set event_type='carry' or all rows will be used.")
                t  =st.text_input("Title","Progressive Carries",key="pc_t")
                cc =st.color_picker("Carry color","#FF9300",key="pc_c")
                md =st.slider("Min distance (pitch units)",1.0,20.0,5.0,key="pc_md")
                dv =st.checkbox("Vertical pitch",False,key="pc_v")
                ov =img_overlay_controls("pc_ov")
                gen=st.button("Generate",key="pc_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=progressive_carries_map(df,title=t,theme_name=S["tn"],pitch_mode=S["pm"],
                                                pitch_width=S["pw"],carry_color=cc,min_distance=md,
                                                vertical_pitch=dv)
                    apply_overlay(fig,ov)
                    st.session_state["dist_pc"]=fig
                if "dist_pc" in st.session_state: preview(st.session_state["dist_pc"],"progressive_carries")
                else: empty("🏃","Configure and generate")

    with tabs[2]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Passing Network</div>',unsafe_allow_html=True)
                st.caption("Requires: player column, x, y. Optional: recipient column for edges.")
                t  =st.text_input("Title","Passing Network",key="pn_t")
                all_cols=[None]+list(df.columns)
                plcol=st.selectbox("Player column",all_cols,key="pn_pl")
                rccol=st.selectbox("Recipient column",all_cols,key="pn_rc")
                nc =st.color_picker("Node color","#00C2FF",key="pn_nc")
                ec =st.color_picker("Edge color","#AAAAAA",key="pn_ec")
                mp =st.slider("Min passes for edge",1,10,3,key="pn_mp")
                ov =img_overlay_controls("pn_ov")
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
    df,S=ev_sidebar("spec")
    hdr("🎯","Specialist Charts","Goal mouth map · Vertical pitch maps · Shot detail card")

    tabs=st.tabs(["Goal Mouth Map","Vertical Pitch Map","Shot Detail Card"])

    with tabs[0]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Goal Mouth Map (CannonStats style)</div>',unsafe_allow_html=True)
                st.caption("Shows shots on target plotted on the goal face.\nRequires: outcome, y. Optional: z/height column, xg.")
                t   =st.text_input("Title","West Ham",key="gm_t")
                sub =st.text_input("Subtitle","Shots on Target Map",key="gm_s")
                gtn =st.selectbox("Theme",ALL_THEMES,index=ALL_THEMES.index("Opta Light") if "Opta Light" in ALL_THEMES else 0,key="gm_tn")
                gc  =st.color_picker("Goal color","#7A2232",key="gm_gc")
                sc  =st.color_picker("Save color","#FFFFFF",key="gm_sc")
                ge  =st.color_picker("Goal edge","#3D0A18",key="gm_ge")
                se  =st.color_picker("Save edge","#1A2E5A",key="gm_se")
                sxg =st.checkbox("Size by xG",True,key="gm_sxg")
                st.markdown("**Stats row** (Goals / xG / SoT)")
                n_st=st.number_input("Stat columns",1,5,3,key="gm_ns")
                stats={}
                for i in range(int(n_st)):
                    c1,c2=st.columns(2)
                    with c1: k=st.text_input(f"Stat label {i+1}","",key=f"gm_sk{i}")
                    with c2: v=st.text_input(f"Value {i+1}","",key=f"gm_sv{i}")
                    if k: stats[k]=v
                fl =st.text_input("Footer left","Excluding Own Goals",key="gm_fl")
                fr =st.text_input("Footer right","CannonStats.com",key="gm_fr")
                logo_ov=img_overlay_controls("gm_logo","Logo image",default_x=0.84,default_y=0.88,default_w=0.12,default_h=0.10)
                gen=st.button("Generate",key="gm_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=goal_mouth_map(df,title=t,subtitle=sub,stats_row=stats or None,
                                       theme_name=gtn,goal_color=gc,save_color=sc,
                                       goal_edge=ge,save_edge=se,size_by_xg=sxg,
                                       logo_img=logo_ov["img"],
                                       logo_x=logo_ov["x"],logo_y=logo_ov["y"],
                                       logo_w=logo_ov["w"],logo_h=logo_ov["h"],
                                       footer_left=fl,footer_right=fr)
                    st.session_state["spec_gm"]=fig
                if "spec_gm" in st.session_state: preview(st.session_state["spec_gm"],"goal_mouth_map")
                else: empty("🥅","Configure and generate")

    with tabs[1]:
        if df is None: nofile()
        else:
            L,R=st.columns([1,2.2])
            with L:
                st.markdown('<div class="card"><div class="ctitle">Vertical Pitch Map (Fixed)</div>',unsafe_allow_html=True)
                t   =st.text_input("Title","Vertical Map",key="vp_t")
                et  =st.selectbox("Event type",["pass","shot","touch","defensive","all"],key="vp_et")
                dta =st.checkbox("Show arrows",True,key="vp_arr")
                pc,pm2=pass_cm("vp_pc")
                sc2,sm2=shot_cm("vp_sc")
                dc_col=st.color_picker("Dot color","#00C2FF",key="vp_dc")
                ds =st.slider("Dot size",30,300,80,key="vp_ds")
                all_et_items=["successful","unsuccessful","key pass","assist","off target","ontarget","goal","blocked"]
                al =legend_controls("vp_leg",all_et_items,all_et_items)
                ov =img_overlay_controls("vp_ov")
                gen=st.button("Generate",key="vp_gen")
                st.markdown('</div>',unsafe_allow_html=True)
            with R:
                if gen:
                    fig=vertical_event_map(df,event_type=et,title=t,theme_name=S["tn"],
                                           pitch_width=S["pw"],pass_colors=pc,pass_markers=pm2,
                                           shot_colors=sc2,shot_markers=sm2,dot_color=dc_col,
                                           dot_size=ds,show_arrows=dta,
                                           active_legend_items=al)
                    apply_overlay(fig,ov)
                    st.session_state["spec_vp"]=fig
                if "spec_vp" in st.session_state: preview(st.session_state["spec_vp"],"vertical_map")
                else: empty("📐","Configure and generate")

    with tabs[2]:
        if df is None: nofile()
        else:
            shots=df[df["event_type"]=="shot"].copy().reset_index(drop=True)
            if shots.empty: st.warning("No shots found.")
            else:
                def _sf(v):
                    try: return float(v)
                    except: return float("nan")
                shots["label"]=shots.apply(lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {_sf(r.get("xg")):.2f}',axis=1)
                L,R=st.columns([1,2.2])
                with L:
                    st.markdown('<div class="card"><div class="ctitle">Shot Detail Card</div>',unsafe_allow_html=True)
                    sel=st.selectbox("Select shot",shots["label"].tolist(),key="sc_sel")
                    sct=st.text_input("Card title","Shot Detail",key="sc_t")
                    sc3,sm3=shot_cm("sc3")
                    al3=legend_controls("sc3_leg",["off target","ontarget","goal","blocked"],["off target","ontarget","goal","blocked"])
                    ov=img_overlay_controls("sc3_ov")
                    gen=st.button("Generate",key="sc3_gen")
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    if gen:
                        idx=int(sel.split("|")[0].strip())-1
                        sc_f={k:v for k,v in sc3.items() if k in al3}
                        sm_f={k:v for k,v in sm3.items() if k in al3}
                        fig,_=shot_detail_card(df,shot_index=idx,title=sct,pitch_mode=S["pm"],
                                               pitch_width=S["pw"],shot_colors=sc_f,
                                               shot_markers=sm_f,theme_name=S["tn"])
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

    def _dpc(df):
        lw={str(c).strip().lower():c for c in df.columns}
        for k in ["player","player name","name"]:
            if k in lw: return lw[k]
        return df.columns[0]

    def _mc(df,excl,mv=1):
        return [c for c in df.columns if c not in excl and _clean(df[c]).notna().sum()>=mv]

    def _tbl(df,pcol,player,metrics):
        d=df.copy(); r=d[d[pcol].astype(str)==str(player)]
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
            pcol=_dpc(dfp)
            mc=_mc(dfp,{pcol},mv=1)
            if not mc: st.error("No numeric columns."); continue
            players=sorted(dfp[pcol].dropna().astype(str).unique().tolist())

            if ck in ["mpl","ath","pb"]:
                L,R=st.columns([1,2])
                with L:
                    st.markdown(f'<div class="card"><div class="ctitle">{tab}</div>',unsafe_allow_html=True)
                    sp=st.selectbox("Player",players,key=f"{ck}_p")
                    ct=st.text_input("Title",sp,key=f"{ck}_t")
                    cs=st.text_input("Subtitle","Percentile vs peers",key=f"{ck}_s")
                    sm=st.multiselect("Metrics",mc,default=mc[:min(12,len(mc))],key=f"{ck}_m")
                    cats=[]
                    if ck in ["mpl","ath"] and sm:
                        st.markdown("**Categories**")
                        for m in sm:
                            cats.append(st.selectbox(m,["Attacking","Possession","Defending"],key=f"{ck}_cat_{m}"))
                    if ck=="mpl":
                        atk_c=st.color_picker("Attacking","#1A78CF",key="mpl_ak")
                        pos_c=st.color_picker("Possession","#FF9300",key="mpl_po")
                        def_c=st.color_picker("Defending","#D70232",key="mpl_df")
                        bg_c =st.color_picker("Background","#222222",key="mpl_bg")
                        ci_f =st.file_uploader("Center image",type=["png","jpg","jpeg"],key="mpl_ci")
                        ci   =load_img(ci_f)
                        cs2  =st.slider("Center img scale",8,28,16,key="mpl_cs")/100
                    elif ck=="ath":
                        atk_c=st.color_picker("Attacking","#4B78B9",key="ath_ak")
                        pos_c=st.color_picker("Possession","#F0C987",key="ath_po")
                        def_c=st.color_picker("Defending","#9E374B",key="ath_df")
                        ci_f =st.file_uploader("Center image",type=["png","jpg","jpeg"],key="ath_ci")
                        ci   =load_img(ci_f)
                        cs2  =st.slider("Center img scale",8,28,14,key="ath_cs")/100
                    elif ck=="pb":
                        gc=st.color_picker("Good ≥70","#00e676",key="pb_g")
                        mc2=st.color_picker("Mid 50-70","#ffd060",key="pb_m")
                        lc=st.color_picker("Low <50","#ff4060",key="pb_l")
                    ov=img_overlay_controls(f"{ck}_ov","Extra overlay")
                    gen=st.button("Generate",key=f"{ck}_gen",disabled=not sm)
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    fk=f"piz_{ck}_fig"
                    if gen and sm:
                        try:
                            tbl=_tbl(dfp,pcol,sp,sm)
                            if ck=="mpl":
                                fig=mpl_pizza_dark(tbl,title=ct,subtitle=cs,categories=cats or None,
                                    attacking_color=atk_c,possession_color=pos_c,defending_color=def_c,
                                    center_image=ci,center_img_scale=cs2,bg_color=bg_c)
                            elif ck=="ath":
                                fig=athletic_pizza(tbl,title=ct,subtitle=cs,categories=cats or None,
                                    attacking_color=atk_c,possession_color=pos_c,defending_color=def_c)
                                if ci:
                                    overlay_image_on_fig(fig,ci,x=0.42,y=0.42,w=cs2,h=cs2,circle_crop=True)
                            else:
                                d2=tbl.sort_values("percentile",ascending=True)
                                vals=d2["percentile"].tolist()
                                clrs=[gc if v>=70 else mc2 if v>=50 else lc for v in vals]
                                fig,ax=plt.subplots(figsize=(10,max(4.5,len(d2)*.45)))
                                fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                                ax.barh(d2["metric"],vals,color=clrs,alpha=.92)
                                ax.set_xlim(0,100)
                                for i,v in enumerate(vals):
                                    ax.text(min(97,float(v)+1.2),i,f"{v:.0f}",va="center",color="#e8f0fe",fontsize=9,weight="bold")
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
            else:
                L,R=st.columns([1,2])
                with L:
                    st.markdown('<div class="card"><div class="ctitle">Scatter Plot</div>',unsafe_allow_html=True)
                    hi_p=st.selectbox("Highlight player",["(none)"]+players,key="sc_hp")
                    sx  =st.selectbox("X-axis metric",mc,index=0,key="sc_x")
                    sy  =st.selectbox("Y-axis metric",mc,index=min(1,len(mc)-1),key="sc_y")
                    sct2=st.text_input("Title","Scatter Plot",key="sc_t2")
                    sdc =st.color_picker("Dot color","#00d4ff",key="sc_dc")
                    shc =st.color_picker("Highlight","#ffd060",key="sc_hc")
                    ov  =img_overlay_controls("sc_ov")
                    gen =st.button("Generate",key="sc_gen")
                    st.markdown('</div>',unsafe_allow_html=True)
                with R:
                    if gen:
                        d2=dfp.copy(); d2[sx]=_clean(d2[sx]); d2[sy]=_clean(d2[sy])
                        d2=d2.dropna(subset=[sx,sy])
                        fig,ax=plt.subplots(figsize=(9,6))
                        fig.patch.set_facecolor("#07111f"); ax.set_facecolor("#07111f")
                        ax.scatter(d2[sx],d2[sy],s=75,color=sdc,alpha=.72,edgecolors="white",lw=.6)
                        if hi_p!="(none)":
                            h=d2[d2[pcol].astype(str)==str(hi_p)]
                            if not h.empty:
                                ax.scatter(h[sx],h[sy],s=200,color=shc,edgecolors="white",lw=1.5,zorder=5)
                                for _,rr in h.iterrows():
                                    ax.text(float(rr[sx]),float(rr[sy]),f"  {rr[pcol]}",color="#e8f0fe",fontsize=10,weight="bold")
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
        st.error("scouting_tools_v2.py not found."); st.stop()
    hdr("🧠","Player Scouting","Role templates · percentile scoring · shortlists · recommendations")

    def _cl(x): return str(x).strip().lower().replace("_"," ").replace("-"," ")
    def _sn(s): return pd.to_numeric(s.astype(str).str.replace("%","",regex=False).str.replace(",","",regex=False),errors="coerce")
    def _lbl(s):
        if pd.isna(s): return "No data"
        if s>=82: return "🟢 Elite"
        if s>=70: return "🟡 Strong"
        if s>=58: return "🟠 Watchlist"
        if s>=45: return "🔴 Average"
        return "⛔ Risky"
    def _ir(pos):
        p=_cl(pos)
        for k,v in {"gk":"Goalkeeper","goalkeeper":"Goalkeeper","cb":"Centre Back","rcb":"Centre Back",
                    "lcb":"Centre Back","rb":"Full Back / Wing Back","lb":"Full Back / Wing Back",
                    "rwb":"Full Back / Wing Back","lwb":"Full Back / Wing Back","dm":"Defensive Midfielder",
                    "cdm":"Defensive Midfielder","cm":"Central Midfielder","mc":"Central Midfielder",
                    "am":"Attacking Midfielder","cam":"Attacking Midfielder","rw":"Winger","lw":"Winger",
                    "st":"Striker","cf":"Striker","striker":"Striker"}.items():
            if k in p: return v
        return list(ROLE_TEMPLATES.keys())[0]

    def _ms(df_in,metrics,lb,gc=None,mc_arg=None,mf=900,ws=None):
        out=df_in.copy(); pcs=[]; ws=ws or {}
        for m in metrics:
            if m not in out.columns: continue
            out[m]=_sn(out[m]); pc=f"pct__{m}"; hi=m not in lb
            if gc and gc in out.columns:
                out[pc]=out.groupby(gc,dropna=False)[m].transform(lambda x: _pct(x,hi))
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

    with st.sidebar:
        st.markdown("### 📂 Scouting File")
        sf=st.file_uploader("CSV / Excel",type=["csv","xlsx","xls"],key="sc_f")
        if ROLE_TEMPLATES:
            st.download_button("⬇ Template CSV",data=make_template_csv(),file_name="scouting_template.csv",mime="text/csv",key="sc_tmpl")
        st.markdown('<div class="divd"></div>',unsafe_allow_html=True)
        st.markdown("### ⚙️ Settings")
        mmg=st.number_input("Default min minutes",0,value=300,step=50,key="sc_mmg")
        scp=st.radio("Percentile scope",["All filtered","Same position"],index=1,key="sc_scp")
        ur =st.checkbox("Reliability adjustment",True,key="sc_ur")
        rf =st.number_input("'Reliable' ≥ minutes",1,value=900,step=50,key="sc_rf")

    if sf is None:
        st.info("Upload a scouting file to begin."); st.stop()

    try: dfr=load_player_data(sf)
    except Exception as e: st.error(f"Load error: {e}"); st.stop()
    dfr.columns=[str(c).strip() for c in dfr.columns]
    cs_=standard_columns(dfr); ac=[None]+dfr.columns.tolist()

    with st.sidebar:
        st.markdown("### 🧩 Columns")
        pcol=st.selectbox("Player",ac,index=ac.index(cs_["player"]) if cs_["player"] in ac else 0,key="sc_pc")
        tcol=st.selectbox("Team",  ac,index=ac.index(cs_["team"])   if cs_["team"]   in ac else 0,key="sc_tc")
        poscol=st.selectbox("Position",ac,index=ac.index(cs_["position"]) if cs_["position"] in ac else 0,key="sc_pos")
        agcol=st.selectbox("Age",  ac,index=ac.index(cs_["age"])    if cs_["age"]    in ac else 0,key="sc_ag")
        mincol=st.selectbox("Minutes",ac,index=ac.index(cs_["minutes"]) if cs_["minutes"] in ac else 0,key="sc_min")
        valcol=st
