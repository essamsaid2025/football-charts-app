"""
Football Analysis Suite  v4  — Professional Analyst Platform
Upgrades: Cleaned and Ordered Dynamic Namespace Imports with Native Fallbacks
"""

import os, io, math, sys, importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from PIL import Image

# ── Dynamic module loader ─────────────────────────────────────────────────────
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

# 1. Safely register local modules in memory sequentially
charts = _load_local("charts", "charts.py", "charts (9).py", "charts__8_.py")
charts_extra = _load_local("charts_extra", "charts_extra.py", "charts_extra (2).py", "charts_extra__1_.py")
charts_pro = _load_local("charts_pro", "charts_pro.py")
scouting_tools_v2 = _load_local("scouting_tools_v2", "scouting_tools_v2.py", "scouting_tools_v2 (6).py", "scouting_tools_v2 (7).py", "scouting_tools_v2__6_.py")

# 2. Safely unpack features from core charts.py into the global namespace
if charts:
    load_data = getattr(charts, "load_data", None)
    prepare_df_for_charts = getattr(charts, "prepare_df_for_charts", None)
    pizza_chart = getattr(charts, "pizza_chart", None)
    mpl_pizza_dark = getattr(charts, "mpl_pizza_dark", None)
    athletic_pizza = getattr(charts, "athletic_pizza", None)
    shot_detail_card = getattr(charts, "shot_detail_card", None)
    defensive_regains_map = getattr(charts, "defensive_regains_map", None)
    outcome_bar = getattr(charts, "outcome_bar", None)
    start_location_heatmap = getattr(charts, "start_location_heatmap", None)
    touch_map = getattr(charts, "touch_map", None)
    pass_map = getattr(charts, "pass_map", None)
    shot_map = getattr(charts, "shot_map", None)
    defensive_actions_map = getattr(charts, "defensive_actions_map", None)
    THEMES = getattr(charts, "THEMES", {})
    make_pitch = getattr(charts, "make_pitch", None)

# 3. Safely unpack features from charts_extra.py
if charts_extra:
    draw_pass_sonar = getattr(charts_extra, "draw_pass_sonar", None)
    draw_defensive_territory = getattr(charts_extra, "draw_defensive_territory", None)
    overlay_image_on_fig = getattr(charts_extra, "overlay_image_on_fig", None)
    goal_location_map = getattr(charts_extra, "goal_location_map", None)
    goal_mouth_map = getattr(charts_extra, "goal_mouth_map", None)
    goal_shot_report_map = getattr(charts_extra, "goal_shot_report_map", None)
    vertical_event_map = getattr(charts_extra, "vertical_event_map", None)
    progressive_carries_map = getattr(charts_extra, "progressive_carries_map", None)
    pressure_map = getattr(charts_extra, "pressure_map", None)
    xg_timeline = getattr(charts_extra, "xg_timeline", None)

# ── NATIVE PASSING NETWORK POINTER + FALLBACK ─────────────────────────────────
# Checks draw_pass_network_advanced and passing_network, falls back to custom render if missing
passing_network = getattr(charts_extra, "draw_pass_network_advanced", None)
if passing_network is None:
    passing_network = getattr(charts_extra, "passing_network", None)
if passing_network is None:
    def passing_network(df, player_col="player_name", recipient_col="pass_recipient_name", 
                        title=None, theme_name="Dark", pitch_mode="horizontal", pitch_width=100,
                        node_color=None, edge_color=None, min_passes=3):
        from mplsoccer import Pitch
        t = THEMES.get(theme_name, {"bg": "#121212", "pitch": "#1e1e1e", "line": "#444444", "text": "#ffffff"})
        pitch = Pitch(pitch_type='statsbomb', pitch_color=t["pitch"], line_color=t["line"])
        fig, ax = pitch.draw(figsize=(10, 7), facecolor=t["bg"])
        ax.text(60, 40, f"{title or 'PASSING NETWORK'}\n(Rendered via Dynamic Fallback Engine)", 
                color=t["text"], fontsize=12, ha='center', va='center', weight="bold")
        return fig

# ── NATIVE DRAW TAGGING PITCH FALLBACK ────────────────────────────────────────
draw_tagging_pitch = getattr(charts_extra, "draw_tagging_pitch", None)
if draw_tagging_pitch is None and charts and hasattr(charts, "draw_tagging_pitch"):
    draw_tagging_pitch = charts.draw_tagging_pitch
if draw_tagging_pitch is None:
    def draw_tagging_pitch(
        theme_name="The Athletic Dark",
        pitch_mode="rect",
        pitch_width=68.0,
        display_width=900,
        show_thirds=True,
        events=None,
        start_point=None,
        current_marker="o",
        current_color="#22C55E",
        current_edge="#FFFFFF",
        current_size=9,
    ):
        """
        Renders an interactive tagging pitch as a PIL Image.
        Returns (PIL.Image, img_w, img_h, y_max, pad).
        """
        import math as _math
        from PIL import Image as _Image
        import io as _io

        t = THEMES.get(theme_name, THEMES.get("The Athletic Dark", {
            "bg": "#0E1117", "pitch": "#1f5f3b", "pitch_lines": "#E6E6E6",
            "text": "#FFFFFF", "muted": "#A0A7B4", "lines": "#2A3240",
        }))

        y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
        aspect = y_max / 100.0
        fig_w = display_width / 100.0
        fig_h = fig_w * aspect

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(t["bg"])
        ax.set_facecolor(t.get("pitch", "#1f5f3b"))

        lc = t.get("pitch_lines", "#E6E6E6")
        lw = 1.6
        mid = y_max / 2.0

        # ── Pitch outline ──────────────────────────────────────────────────
        ax.plot([0,100,100,0,0], [0,0,y_max,y_max,0], color=lc, lw=lw)
        # Halfway line
        ax.plot([50,50],[0,y_max], color=lc, lw=lw)
        # Centre circle
        theta = np.linspace(0, 2*np.pi, 100)
        rx = 9.15; ry = 9.15 / 100.0 * y_max
        ax.plot(50 + rx*np.cos(theta), mid + ry*np.sin(theta), color=lc, lw=lw)
        ax.plot(50, mid, "o", color=lc, ms=3)

        # ── Penalty areas ──────────────────────────────────────────────────
        pa_w = y_max * 40.32/68.0; sa_w = y_max * 18.32/68.0
        goal_w = y_max * 7.32/68.0
        pa_l = 16.5/105.0 * 100; sa_l = 5.5/105.0 * 100
        for x0, x1 in [(0, pa_l), (100-pa_l, 100)]:
            ax.plot([x0,x1,x1,x0,x0],
                    [mid-pa_w/2,mid-pa_w/2,mid+pa_w/2,mid+pa_w/2,mid-pa_w/2],
                    color=lc, lw=lw)
        for x0, x1 in [(0, sa_l), (100-sa_l, 100)]:
            ax.plot([x0,x1,x1,x0,x0],
                    [mid-sa_w/2,mid-sa_w/2,mid+sa_w/2,mid+sa_w/2,mid-sa_w/2],
                    color=lc, lw=lw)
        # Goals
        for x0, x1 in [(-2.5, 0), (100, 102.5)]:
            ax.plot([x0,x1,x1,x0,x0],
                    [mid-goal_w/2,mid-goal_w/2,mid+goal_w/2,mid+goal_w/2,mid-goal_w/2],
                    color=lc, lw=lw)
        # Penalty spots
        for px in [11.0/105.0*100, 100 - 11.0/105.0*100]:
            ax.plot(px, mid, "o", color=lc, ms=3)
        # Penalty arcs
        for cx, sign in [(pa_l, 1), (100-pa_l, -1)]:
            arc_r_x = rx; arc_r_y = ry
            th = np.linspace(0, 2*np.pi, 200)
            xs = cx + arc_r_x*np.cos(th)*sign
            ys = mid + arc_r_y*np.sin(th)
            mask = xs*sign >= cx*sign
            ax.plot(xs[mask], ys[mask], color=lc, lw=lw)

        # ── Thirds ────────────────────────────────────────────────────────
        if show_thirds:
            for x in [100/3, 200/3]:
                ax.plot([x,x],[0,y_max], color=lc, lw=0.8, ls="--", alpha=0.4)
            text_kw = dict(color=t.get("muted","#A0A7B4"), fontsize=8,
                           alpha=0.7, ha="center", va="top")
            ax.text(100/6, y_max*0.98, "Defensive Third", **text_kw)
            ax.text(50,    y_max*0.98, "Middle Third",    **text_kw)
            ax.text(500/6, y_max*0.98, "Attacking Third", **text_kw)

        # ── Attacking direction arrow ──────────────────────────────────────
        arrow_y = -y_max * 0.06
        ax.annotate("", xy=(75, arrow_y), xytext=(25, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color=t.get("muted","#A0A7B4"), lw=1.8),
                    annotation_clip=False)
        ax.text(50, arrow_y - y_max*0.04, "Attacking Direction",
                color=t.get("muted","#A0A7B4"), fontsize=8, ha="center",
                va="top", clip_on=False)

        # ── Draw existing events ───────────────────────────────────────────
        for ev in (events or []):
            try:
                ex = float(ev.get("x", 0)); ey = float(ev.get("y", 0))
                col  = str(ev.get("start_color",  current_color) or current_color)
                edge = str(ev.get("start_edge",   current_edge)  or current_edge)
                mk   = ev.get("start_marker", "o") or "o"
                sz   = float(ev.get("start_size", 9) or 9) * 15
                # Arrow for pass/carry/cross/dribble
                et = str(ev.get("event_type","")).lower()
                if et in ("pass","carry","cross","dribble"):
                    ex2 = ev.get("x2"); ey2 = ev.get("y2")
                    if ex2 is not None and ey2 is not None:
                        try:
                            if not (_math.isnan(float(ex2)) or _math.isnan(float(ey2))):
                                arr_col = str(ev.get("arrow_color", col) or col)
                                ax.annotate("", xy=(float(ex2), float(ey2)),
                                            xytext=(ex, ey),
                                            arrowprops=dict(arrowstyle="-|>",
                                                            color=arr_col, lw=1.8))
                        except Exception:
                            pass
                ax.scatter([ex], [ey], s=sz, marker=mk, color=col,
                           edgecolors=edge, linewidth=1.5, zorder=6)
            except Exception:
                pass

        # ── Pending start point ────────────────────────────────────────────
        if start_point is not None:
            try:
                spx, spy = float(start_point[0]), float(start_point[1])
                mk = current_marker if current_marker else "o"
                ax.scatter([spx], [spy], s=float(current_size)*15,
                           marker=mk, color=current_color, edgecolors=current_edge,
                           linewidth=2, zorder=7, alpha=0.6)
                # Crosshair
                ax.plot([spx-2, spx+2], [spy, spy], color=current_color, lw=1, alpha=0.5)
                ax.plot([spx, spx], [spy-2*aspect, spy+2*aspect], color=current_color, lw=1, alpha=0.5)
            except Exception:
                pass

        # Use FIXED axis limits — no tight bbox so pixel mapping is exact
        # xlim and ylim define the full coordinate space rendered
        X_MIN, X_MAX = -4.0, 104.0
        Y_MIN = -y_max * 0.18
        Y_MAX =  y_max * 1.05
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save at EXACT figure size — no bbox_inches='tight' to preserve mapping
        DPI = 100
        buf = _io.BytesIO()
        fig.savefig(buf, format="PNG", dpi=DPI,
                    bbox_inches=None, pad_inches=0,
                    facecolor=t["bg"])
        buf.seek(0)
        plt.close(fig)

        pil_img = _Image.open(buf).convert("RGB")
        img_w, img_h = pil_img.size

        # Return the axis coordinate bounds so the click mapper can use them
        # We pass X_MIN, X_MAX, Y_MIN, Y_MAX via the pad slot (as a tuple)
        coord_bounds = (X_MIN, X_MAX, Y_MIN, Y_MAX)
        return pil_img, img_w, img_h, y_max, coord_bounds

# 4. Safely unpack layout features from charts_pro.py
if charts_pro:
    PRO_THEMES = getattr(charts_pro, "PRO_THEMES", {})
    ALL_PRO_THEME_NAMES = getattr(charts_pro, "ALL_PRO_THEME_NAMES", [])
    default_layout_cfg = getattr(charts_pro, "default_layout_cfg", None)
    stat_block = getattr(charts_pro, "stat_block", None)
    draw_logo = getattr(charts_pro, "draw_logo", None)
    draw_title_block = getattr(charts_pro, "draw_title_block", None)
    draw_footer = getattr(charts_pro, "draw_footer", None)
    draw_attack_direction = getattr(charts_pro, "draw_attack_direction", None)
    # Wrap draw_attack_direction to expand the axis limits so the arrow is never clipped
    _raw_draw_attack_direction = draw_attack_direction
    if _raw_draw_attack_direction is not None:
        def draw_attack_direction(ax, cfg, theme, pitch_mode="rect", pitch_width=68.0, vertical=False):
            y_max = pitch_width if pitch_mode == "rect" else 100.0
            if not vertical:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(min(ymin, -y_max * 0.18), ymax)
                ax.set_clip_on(False)
            _raw_draw_attack_direction(ax, cfg, theme, pitch_mode, pitch_width, vertical)
    draw_opta_attack_arrows = getattr(charts_pro, "draw_opta_attack_arrows", None)
    draw_stat_blocks_bottom = getattr(charts_pro, "draw_stat_blocks_bottom", None)
    draw_stat_blocks_right = getattr(charts_pro, "draw_stat_blocks_right", None)
    draw_custom_legend = getattr(charts_pro, "draw_custom_legend", None)
    layout_controls_ui = getattr(charts_pro, "layout_controls_ui", None)
    image_overlay_controls_pro = getattr(charts_pro, "image_overlay_controls_pro", None)
    athletic_shot_map_pro = getattr(charts_pro, "athletic_shot_map_pro", None)
    opta_pass_map_pro = getattr(charts_pro, "opta_pass_map_pro", None)
    pro_pass_map = getattr(charts_pro, "pro_pass_map", None)
    pro_shot_map = getattr(charts_pro, "pro_shot_map", None)
    
    if "charts" in sys.modules:
        sys.modules["charts"].THEMES.update(PRO_THEMES)

# 5. Handle try-except imports safely for Optional modules
try:
    athletic_compact_shot_map = getattr(charts_pro, "athletic_compact_shot_map", None)
except Exception:
    athletic_compact_shot_map = None

try:
    if scouting_tools_v2:
        ROLE_TEMPLATES = getattr(scouting_tools_v2, "ROLE_TEMPLATES")
        load_player_data = getattr(scouting_tools_v2, "load_player_data")
        standard_columns = getattr(scouting_tools_v2, "standard_columns")
        numeric_metrics = getattr(scouting_tools_v2, "numeric_metrics")
        match_template_metrics = getattr(scouting_tools_v2, "match_template_metrics")
        add_percentiles_and_score = getattr(scouting_tools_v2, "add_percentiles_and_score")
        player_profile = getattr(scouting_tools_v2, "player_profile")
        similar_players = getattr(scouting_tools_v2, "similar_players")
        comparison_chart = getattr(scouting_tools_v2, "comparison_chart")
        radar_chart = getattr(scouting_tools_v2, "radar_chart")
        make_template_csv = getattr(scouting_tools_v2, "make_template_csv")
        recommendation_text = getattr(scouting_tools_v2, "recommendation_text")
        NEGATIVE_METRIC_WORDS = getattr(scouting_tools_v2, "NEGATIVE_METRIC_WORDS")
except Exception:
    ROLE_TEMPLATES = None; recommendation_text = None
    NEGATIVE_METRIC_WORDS = ["conceded","fouls","errors","turnovers","dispossessed"]

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_SIC = True
except Exception:
    streamlit_image_coordinates = None; HAS_SIC = False

# ── LEAVE ALL ORIGINAL LINES FROM ST.SET_PAGE_CONFIG DOWNWARDS UNTOUCHED ──────

# ── KEEP EVERYTHING BELOW THIS LINE EXACTLY AS IT WAS ─────────────────────────

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

def _store_fig(key, fig, dpi=180):
    """Store figure as PNG bytes in session state, then close it.
    Prevents matplotlib figures from being pickled by Streamlit (which crashes the app).
    """
    try:
        b = io.BytesIO()
        fig.canvas.draw()
        extras = list(fig.legends)
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                extras.append(leg)
        fig.savefig(b, format="png", dpi=dpi, bbox_inches="tight", pad_inches=.18,
                    bbox_extra_artists=extras or None)
        b.seek(0)
        st.session_state[key] = b.read()
    except Exception as e:
        st.error(f"Figure save error: {e}")
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass

def _preview_stored(key):
    """Show a stored PNG bytes figure."""
    val = st.session_state.get(key)
    if val is not None and isinstance(val, bytes):
        st.image(val, use_container_width=True)
        return val
    return None


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

def dl_row(fig_or_bytes, name):
    c1, c2 = st.columns(2)
    if isinstance(fig_or_bytes, bytes):
        # Already rendered bytes — create PDF from bytes via PIL
        with c1:
            st.download_button("⬇ PNG", fig_or_bytes, f"{name}.png",
                               "image/png", key=f"p_{name}_{hash(fig_or_bytes)}")
        with c2:
            # Build a quick PDF from the PNG bytes
            try:
                import io as _io
                from matplotlib.backends.backend_pdf import PdfPages as _PP
                from PIL import Image as _PILI
                import matplotlib.pyplot as _plt
                import numpy as _np
                pil_img = _PILI.open(_io.BytesIO(fig_or_bytes))
                w_in = pil_img.width / 180; h_in = pil_img.height / 180
                fig_tmp = _plt.figure(figsize=(w_in, h_in))
                ax_tmp = fig_tmp.add_axes([0,0,1,1])
                ax_tmp.imshow(_np.array(pil_img)); ax_tmp.axis("off")
                pdf_b = _io.BytesIO()
                with _PP(pdf_b) as pp:
                    pp.savefig(fig_tmp, bbox_inches="tight", pad_inches=0)
                _plt.close(fig_tmp)
                pdf_b.seek(0)
                st.download_button("⬇ PDF", pdf_b.read(), f"{name}.pdf",
                                   "application/pdf", key=f"f_{name}_{hash(fig_or_bytes)}")
            except Exception:
                pass
    else:
        with c1:
            st.download_button("⬇ PNG", _bytes(fig_or_bytes), f"{name}.png",
                               "image/png", key=f"p_{name}_{id(fig_or_bytes)}")
        with c2:
            st.download_button("⬇ PDF", _pdf(fig_or_bytes), f"{name}.pdf",
                               "application/pdf", key=f"f_{name}_{id(fig_or_bytes)}")
        plt.close(fig_or_bytes)

def preview(fig_or_bytes, name):
    if isinstance(fig_or_bytes, bytes):
        st.image(fig_or_bytes, use_container_width=True)
        dl_row(fig_or_bytes, name)
    else:
        st.image(_bytes(fig_or_bytes, dpi=180), use_container_width=True)
        dl_row(fig_or_bytes, name)

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
        show = st.checkbox("Show legend", True, key=f"{prefix}_leg_show")
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
        c4, c5 = st.columns(2)
        with c4:
            frame = st.checkbox("Show frame", False, key=f"{prefix}_leg_frame")
        with c5:
            ncol = st.number_input("Columns", 1, 8, min(4, max(1, len(all_items))), key=f"{prefix}_leg_ncol")
        markerscale = st.slider("Marker scale", 0.5, 3.0, 1.0, 0.1, key=f"{prefix}_leg_ms")
        # ── Extended legend controls ──────────────────────────────────────────
        st.markdown("**Extended style**")
        c6, c7 = st.columns(2)
        with c6:
            font_color = st.color_picker("Font color", "#FFFFFF", key=f"{prefix}_leg_fc")
            bg_color   = st.color_picker("Background", "#111111", key=f"{prefix}_leg_bg")
        with c7:
            opacity    = st.slider("Opacity", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_leg_op")
            labelspacing = st.slider("Row spacing", 0.1, 2.0, 0.5, 0.1, key=f"{prefix}_leg_ls")
        border_color = st.color_picker("Border color", "#333333", key=f"{prefix}_leg_bc")
        border_lw    = st.slider("Border width", 0.0, 3.0, 0.0, 0.25, key=f"{prefix}_leg_blw")
    return dict(show=show, active=active, pos=pos, fontsize=fs,
                title=title_txt or None, frame=frame, ncol=int(ncol),
                markerscale=markerscale,
                font_color=font_color, bg_color=bg_color,
                opacity=opacity, labelspacing=labelspacing,
                border_color=border_color, border_lw=border_lw)

def apply_legend_style(ax, legend_cfg, theme):
    """
    Pull handles from the axis legend (set by chart internals),
    filter by active items, then re-render with user-chosen style.
    """
    active = legend_cfg.get("active", [])
    # Prefer handles already stored in the axis legend
    existing_leg = ax.get_legend()
    if existing_leg is not None:
        handles = list(existing_leg.legend_handles)
        labels  = [txt.get_text() for txt in existing_leg.get_texts()]
        existing_leg.remove()
    else:
        handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    if not legend_cfg.get("show", True):
        return

    pairs = [(h, l) for h, l in zip(handles, labels) if l in active]

    if not pairs:
        return

    hs, ls = zip(*pairs)
    pos = legend_cfg.get("pos", "lower center")
    bbox = None
    if "upper" in pos and "center" in pos:
        bbox = (0.5, 0.98)
    elif "lower" in pos and "center" in pos:
        bbox = (0.5, 0.02)

    leg = ax.legend(list(hs), list(ls),
        loc=pos,
        fontsize=legend_cfg.get("fontsize", 9),
        title=legend_cfg.get("title"),
        frameon=legend_cfg.get("frame", False),
        bbox_to_anchor=bbox,
        ncol=int(legend_cfg.get("ncol") or min(4, len(hs))),
        markerscale=float(legend_cfg.get("markerscale", 1.0)),
        labelspacing=float(legend_cfg.get("labelspacing", 0.5)))
    leg.set_in_layout(True)
    fc = legend_cfg.get("font_color") or theme.get("text", "white")
    for txt in leg.get_texts():
        txt.set_color(fc)
    if leg.get_title():
        leg.get_title().set_color(legend_cfg.get("font_color") or theme.get("muted","#888888"))
    # Background / border / opacity
    frame_patch = leg.get_frame()
    if frame_patch is not None:
        bg_col = legend_cfg.get("bg_color", None)
        if bg_col:
            try:
                frame_patch.set_facecolor(bg_col)
            except Exception:
                pass
        blw = float(legend_cfg.get("border_lw", 0.0))
        bc  = legend_cfg.get("border_color", "#333333")
        frame_patch.set_linewidth(blw)
        if blw > 0:
            frame_patch.set_edgecolor(bc)
    op = float(legend_cfg.get("opacity", 1.0))
    leg.set_alpha(op)

# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED CUSTOMIZATION SYSTEM
# ─────────────────────────────────────────────────────────────────────────────

# Feature flags: which control groups are relevant for a given chart type.
# Charts call advanced_customization_controls(prefix, features=[...]) to get
# only the panels they need.  Everything not in `features` is silently omitted.
_ADV_FEATURES_ALL = ["pitch","figure","markers","arrows","lines","text","title"]

def advanced_customization_controls(prefix, features=None, theme=None):
    """
    Render professional-level customization expanders in the current column/
    sidebar context.  Returns an adv_cfg dict consumed by apply_advanced_customization().

    Parameters
    ----------
    prefix   : unique string key prefix (to avoid widget key collisions)
    features : list of feature groups to show; None = all groups.
               Valid values: "pitch", "figure", "markers", "arrows",
                             "lines", "text", "title"
    theme    : current theme dict (used as defaults for colors)
    """
    if features is None:
        features = _ADV_FEATURES_ALL
    features = set(features)
    theme = theme or {}

    cfg = {}

    # ── PITCH ─────────────────────────────────────────────────────────────────
    if "pitch" in features:
        with st.expander("🟩 Pitch Customization", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                pitch_color = st.color_picker("Pitch color",
                    theme.get("pitch", "#1f5f3b"), key=f"{prefix}_adv_pc")
                line_color  = st.color_picker("Line color",
                    theme.get("pitch_lines", "#E6E6E6"), key=f"{prefix}_adv_lc")
                goal_color  = st.color_picker("Goal color",
                    theme.get("goal", "#E6E6E6"), key=f"{prefix}_adv_gc")
            with c2:
                line_width  = st.slider("Line width", 0.5, 4.0, 1.5, 0.1,
                    key=f"{prefix}_adv_lw")
                pitch_pad   = st.slider("Pitch padding", 0.0, 10.0, 2.0, 0.5,
                    key=f"{prefix}_adv_pp")

            # Stripes
            stripe_on = st.checkbox("Show stripes", False, key=f"{prefix}_adv_st")
            if stripe_on:
                cs1, cs2, cs3 = st.columns(3)
                with cs1:
                    stripe_color = st.color_picker("Stripe color",
                        theme.get("pitch_stripe", "#1a5c38"), key=f"{prefix}_adv_sc")
                with cs2:
                    stripe_alpha = st.slider("Stripe alpha", 0.0, 1.0, 0.35, 0.05,
                        key=f"{prefix}_adv_sa")
                with cs3:
                    stripe_color2 = st.color_picker("Alt stripe color",
                        pitch_color, key=f"{prefix}_adv_sc2")
            else:
                stripe_color = None; stripe_alpha = 0.35; stripe_color2 = None

            st.markdown("**Show/hide pitch elements**")
            el1, el2, el3 = st.columns(3)
            with el1:
                show_center_circle = st.checkbox("Centre circle", True, key=f"{prefix}_adv_cc")
                show_penalty       = st.checkbox("Penalty areas", True, key=f"{prefix}_adv_pa")
            with el2:
                show_six_yard      = st.checkbox("Six-yard box", True, key=f"{prefix}_adv_sy")
                show_corner_arcs   = st.checkbox("Corner arcs", True, key=f"{prefix}_adv_ca")
            with el3:
                show_halfway       = st.checkbox("Halfway line", True, key=f"{prefix}_adv_hl")

        cfg["pitch"] = dict(
            pitch_color=pitch_color, line_color=line_color, goal_color=goal_color,
            line_width=line_width, pitch_pad=pitch_pad,
            stripe=stripe_on, stripe_color=stripe_color,
            stripe_alpha=stripe_alpha, stripe_color2=stripe_color2,
            show_center_circle=show_center_circle, show_penalty=show_penalty,
            show_six_yard=show_six_yard, show_corner_arcs=show_corner_arcs,
            show_halfway=show_halfway,
        )

    # ── FIGURE ────────────────────────────────────────────────────────────────
    if "figure" in features:
        with st.expander("📐 Figure Settings", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                fig_w   = st.slider("Width (in)", 6.0, 24.0, 12.0, 0.5, key=f"{prefix}_adv_fw")
                fig_dpi = st.slider("DPI", 72, 400, 180, 10, key=f"{prefix}_adv_dpi")
            with c2:
                fig_h   = st.slider("Height (in)", 4.0, 18.0, 8.0, 0.5, key=f"{prefix}_adv_fh")
                fig_bg  = st.color_picker("Background",
                    theme.get("bg", "#0E1117"), key=f"{prefix}_adv_fbg")
            with c3:
                transparent = st.checkbox("Transparent bg", False, key=f"{prefix}_adv_tr")

            st.markdown("**Margins (fraction 0–0.3)**")
            m1, m2, m3, m4 = st.columns(4)
            with m1: mg_l = st.slider("Left",   0.0, 0.3, 0.05, 0.01, key=f"{prefix}_adv_ml")
            with m2: mg_r = st.slider("Right",  0.0, 0.3, 0.05, 0.01, key=f"{prefix}_adv_mr")
            with m3: mg_t = st.slider("Top",    0.0, 0.3, 0.05, 0.01, key=f"{prefix}_adv_mt")
            with m4: mg_b = st.slider("Bottom", 0.0, 0.3, 0.05, 0.01, key=f"{prefix}_adv_mb")

            pad_inches = st.slider("Export pad (in)", 0.0, 0.5, 0.18, 0.02,
                key=f"{prefix}_adv_pad")

        cfg["figure"] = dict(
            width=fig_w, height=fig_h, dpi=fig_dpi,
            bg=fig_bg, transparent=transparent,
            margin_left=mg_l, margin_right=mg_r,
            margin_top=mg_t, margin_bottom=mg_b,
            pad_inches=pad_inches,
        )

    # ── MARKERS ───────────────────────────────────────────────────────────────
    if "markers" in features:
        with st.expander("🔵 Marker Style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                mk_size   = st.slider("Size", 10, 600, 120, 5, key=f"{prefix}_adv_mksz")
                mk_alpha  = st.slider("Alpha", 0.1, 1.0, 0.92, 0.02, key=f"{prefix}_adv_mkal")
            with c2:
                mk_fill   = st.color_picker("Fill color", "#00C2FF", key=f"{prefix}_adv_mkfc")
                mk_edge   = st.color_picker("Edge color", "#FFFFFF", key=f"{prefix}_adv_mkec")
            with c3:
                mk_ew     = st.slider("Edge width", 0.0, 4.0, 1.2, 0.1, key=f"{prefix}_adv_mkew")
                mk_zorder = st.slider("Z-order", 1, 10, 5, 1, key=f"{prefix}_adv_mkzo")
            mk_shape = st.selectbox("Default shape", MKL, index=MKL.index("Circle"),
                key=f"{prefix}_adv_mksh")
        cfg["markers"] = dict(
            size=mk_size, alpha=mk_alpha,
            fill=mk_fill, edge=mk_edge,
            edge_width=mk_ew, zorder=mk_zorder,
            shape=MARKER_OPTS[mk_shape],
        )

    # ── ARROWS ────────────────────────────────────────────────────────────────
    if "arrows" in features:
        with st.expander("➡️ Arrow Style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                arr_w      = st.slider("Width", 0.5, 6.0, 2.0, 0.1, key=f"{prefix}_adv_arw")
                arr_alpha  = st.slider("Alpha", 0.1, 1.0, 0.85, 0.05, key=f"{prefix}_adv_aral")
            with c2:
                arr_color  = st.color_picker("Color", "#FFFFFF", key=f"{prefix}_adv_arc")
                arr_hsize  = st.slider("Head size", 0.1, 1.5, 0.5, 0.05, key=f"{prefix}_adv_arhs")
            with c3:
                arr_hw     = st.slider("Head width", 0.1, 2.0, 0.8, 0.05, key=f"{prefix}_adv_arhw")
                arr_zorder = st.slider("Z-order", 1, 10, 4, 1, key=f"{prefix}_adv_arzo")
            arr_style  = st.selectbox("Line style",
                ["solid","dashed","dotted","dashdot"],
                key=f"{prefix}_adv_arls")
            arr_curve  = st.slider("Curvature (-1 to 1)", -1.0, 1.0, 0.0, 0.05,
                key=f"{prefix}_adv_arcv")
        cfg["arrows"] = dict(
            width=arr_w, alpha=arr_alpha, color=arr_color,
            head_size=arr_hsize, head_width=arr_hw,
            linestyle=arr_style, curvature=arr_curve, zorder=arr_zorder,
        )

    # ── LINES ─────────────────────────────────────────────────────────────────
    if "lines" in features:
        with st.expander("📏 Line Style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ln_w     = st.slider("Width", 0.5, 5.0, 1.5, 0.1, key=f"{prefix}_adv_lnw")
            with c2:
                ln_color = st.color_picker("Color", "#FFFFFF", key=f"{prefix}_adv_lnc")
            with c3:
                ln_alpha = st.slider("Alpha", 0.1, 1.0, 0.9, 0.05, key=f"{prefix}_adv_lnal")
            ln_style = st.selectbox("Style", ["solid","dashed","dotted","dashdot"],
                key=f"{prefix}_adv_lnst")
        cfg["lines"] = dict(
            width=ln_w, color=ln_color,
            alpha=ln_alpha, style=ln_style,
        )

    # ── TEXT ──────────────────────────────────────────────────────────────────
    if "text" in features:
        with st.expander("🔤 Text Style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                tx_family  = st.selectbox("Font family",
                    ["sans-serif","serif","monospace","DejaVu Sans","Arial","Georgia"],
                    key=f"{prefix}_adv_txfm")
                tx_size    = st.slider("Font size", 7, 24, 11, 1, key=f"{prefix}_adv_txsz")
            with c2:
                tx_weight  = st.selectbox("Weight", ["normal","bold","light","semibold"],
                    key=f"{prefix}_adv_txwt")
                tx_color   = st.color_picker("Color",
                    theme.get("text", "#FFFFFF"), key=f"{prefix}_adv_txc")
            with c3:
                tx_alpha   = st.slider("Opacity", 0.1, 1.0, 1.0, 0.05, key=f"{prefix}_adv_txal")
                tx_align   = st.selectbox("Alignment", ["left","center","right"],
                    key=f"{prefix}_adv_txali")
            tx_outline   = st.checkbox("Outline", False, key=f"{prefix}_adv_txol")
            tx_ol_width  = st.slider("Outline width", 0.5, 5.0, 2.0, 0.25,
                key=f"{prefix}_adv_txolw") if tx_outline else 0.0
        cfg["text"] = dict(
            family=tx_family, size=tx_size, weight=tx_weight,
            color=tx_color, alpha=tx_alpha, align=tx_align,
            outline=tx_outline, outline_width=tx_ol_width,
        )

    # ── TITLE ─────────────────────────────────────────────────────────────────
    if "title" in features:
        with st.expander("🏷️ Title Style", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                ti_size   = st.slider("Font size", 10, 36, 16, 1, key=f"{prefix}_adv_tisz")
                ti_weight = st.selectbox("Weight", ["bold","normal","light"],
                    key=f"{prefix}_adv_tiwt")
            with c2:
                ti_color  = st.color_picker("Color",
                    theme.get("text", "#FFFFFF"), key=f"{prefix}_adv_tic")
                ti_pad    = st.slider("Padding", 0.0, 30.0, 8.0, 1.0,
                    key=f"{prefix}_adv_tipd")
            with c3:
                ti_loc    = st.selectbox("Alignment", ["left","center","right"],
                    key=f"{prefix}_adv_tiloc")
                ti_y      = st.slider("Y position", 0.85, 1.05, 1.0, 0.01,
                    key=f"{prefix}_adv_tiy")
        cfg["title"] = dict(
            size=ti_size, weight=ti_weight, color=ti_color,
            pad=ti_pad, loc=ti_loc, y=ti_y,
        )

    return cfg


def apply_advanced_customization(fig, adv_cfg, theme=None):
    """
    Post-process a matplotlib figure with advanced customization settings.
    Safe to call with an empty or partial adv_cfg — only keys present are applied.
    Does NOT touch chart internals (pitch drawing, data plotting).
    """
    if not adv_cfg:
        return
    theme = theme or {}

    # ── Figure settings ───────────────────────────────────────────────────────
    fc = adv_cfg.get("figure")
    if fc:
        try:
            w = fc.get("width"); h = fc.get("height")
            if w and h:
                fig.set_size_inches(w, h)
            dpi = fc.get("dpi")
            if dpi:
                fig.set_dpi(dpi)
            bg = fc.get("bg")
            if bg:
                fig.patch.set_facecolor(bg)
            if fc.get("transparent"):
                fig.patch.set_alpha(0.0)
            # Apply margins if any were changed from defaults
            ml = fc.get("margin_left", 0.05)
            mr = fc.get("margin_right", 0.05)
            mt = fc.get("margin_top", 0.05)
            mb = fc.get("margin_bottom", 0.05)
            # Only adjust if user changed from defaults
            try:
                fig.subplots_adjust(left=ml, right=1-mr, top=1-mt, bottom=mb)
            except Exception:
                pass
        except Exception:
            pass

    # ── Pitch color overrides (applied to axes backgrounds) ──────────────────
    pc = adv_cfg.get("pitch")
    if pc:
        try:
            for ax in fig.axes:
                ax.set_facecolor(pc.get("pitch_color", ax.get_facecolor()))
                # Recolor all existing lines on the pitch
                lc = pc.get("line_color")
                lw = pc.get("line_width")
                if lc or lw:
                    for line in ax.get_lines():
                        if lc:
                            try: line.set_color(lc)
                            except Exception: pass
                        if lw:
                            try: line.set_linewidth(lw)
                            except Exception: pass
        except Exception:
            pass

    # ── Title style ───────────────────────────────────────────────────────────
    tc = adv_cfg.get("title")
    if tc:
        try:
            for ax in fig.axes:
                t = ax.title
                if t.get_text():
                    if tc.get("color"): t.set_color(tc["color"])
                    if tc.get("size"):  t.set_fontsize(tc["size"])
                    if tc.get("weight"): t.set_fontweight(tc["weight"])
                    if tc.get("pad") is not None: t.set_position((0.5, 1.0))
        except Exception:
            pass

    # ── Text style (axis labels, tick labels) ─────────────────────────────────
    txc = adv_cfg.get("text")
    if txc:
        import matplotlib.patheffects as pe
        effects = []
        if txc.get("outline") and txc.get("outline_width", 0) > 0:
            effects = [pe.withStroke(linewidth=txc["outline_width"], foreground="black")]
        try:
            for ax in fig.axes:
                for txt in ax.texts:
                    try:
                        if txc.get("color"): txt.set_color(txc["color"])
                        if txc.get("alpha") is not None: txt.set_alpha(txc["alpha"])
                        if txc.get("family"): txt.set_fontfamily(txc["family"])
                        if effects: txt.set_path_effects(effects)
                    except Exception:
                        pass
        except Exception:
            pass


def pitch_kw_from_adv(adv_cfg, theme):
    """
    Extract pitch-related kwargs from adv_cfg to pass to make_pitch / mplsoccer Pitch.
    Returns a dict of theme overrides.
    """
    pc = adv_cfg.get("pitch", {}) if adv_cfg else {}
    overrides = {}
    if pc.get("pitch_color"):
        overrides["pitch"] = pc["pitch_color"]
    if pc.get("line_color"):
        overrides["pitch_lines"] = pc["line_color"]
    if pc.get("stripe") and pc.get("stripe_color"):
        overrides["pitch_stripe"] = pc["stripe_color"]
    elif not pc.get("stripe"):
        overrides["pitch_stripe"] = None
    # Merge overrides into a copy of theme
    merged = {**theme, **overrides}
    return merged


def apply_figure_adv(fig, adv_cfg):
    """Apply only figure-level settings (size, DPI, bg, margins). Safe wrapper."""
    fc = (adv_cfg or {}).get("figure")
    if not fc:
        return
    try:
        w = fc.get("width"); h = fc.get("height")
        if w and h:
            fig.set_size_inches(w, h)
        dpi = fc.get("dpi")
        if dpi:
            fig.set_dpi(dpi)
        bg = fc.get("bg")
        if bg and not fc.get("transparent"):
            fig.patch.set_facecolor(bg)
        if fc.get("transparent"):
            fig.patch.set_alpha(0.0)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE OVERLAY CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
def tagged_events_map(events_df, pitch_mode="rect", pitch_width=68.0,
                      theme_name="The Athletic Dark", show_thirds=True,
                      legend_cfg=None, title="Tagged Events"):
    theme = THEMES.get(theme_name, THEMES.get("The Athletic Dark", {}))
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor(theme.get("bg", "#0E1117"))
    pitch.draw(ax=ax)
    ax.set_facecolor(theme.get("pitch", "#1f5f3b"))

    d = events_df.copy()
    for c in ["x", "y", "x2", "y2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x", "y"]).copy()

    active = None if legend_cfg is None else set(legend_cfg.get("active", []))
    if active is not None and "outcome" in d.columns:
        d = d[d["outcome"].astype(str).isin(active)].copy()

    handles = {}
    for _, row in d.iterrows():
        outcome = str(row.get("outcome", "tagged")).strip() or "tagged"
        event_type = str(row.get("event_type", "event")).strip().lower()
        color = str(row.get("start_color") or row.get("arrow_color") or theme.get("accent", "#22C55E"))
        edge = str(row.get("start_edge") or "#FFFFFF")
        marker = row.get("start_marker", "o")
        size = pd.to_numeric(row.get("start_size", 9), errors="coerce")
        size = 9 if pd.isna(size) else float(size)

        has_end = "x2" in d.columns and "y2" in d.columns and pd.notna(row.get("x2")) and pd.notna(row.get("y2"))
        if has_end:
            pitch.arrows([row["x"]], [row["y"]], [row["x2"]], [row["y2"]],
                         ax=ax, width=2, alpha=0.86,
                         color=str(row.get("arrow_color") or color), zorder=4)

        if str(marker).strip().lower() not in {"none", "no marker", ""}:
            pitch.scatter([row["x"]], [row["y"]], ax=ax, s=max(25, size * 12),
                          marker=marker, color=color, edgecolors=edge,
                          linewidth=1.3, alpha=0.95, zorder=6)

        label = outcome
        if label not in handles and str(marker).strip().lower() not in {"none", "no marker", ""}:
            handles[label] = Line2D([0], [0], marker=marker, color="none",
                                    markerfacecolor=color, markeredgecolor=edge,
                                    markersize=8, label=label)

    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    if show_thirds and charts and hasattr(charts, "_add_pitch_thirds"):
        charts._add_pitch_thirds(ax, pitch_mode, pitch_width, theme, False)
    ax.set_title(title, color=theme.get("text", "white"), fontsize=16, weight="bold")
    if charts and hasattr(charts, "_add_legend"):
        charts._add_legend(ax, list(handles.values()), theme, loc="upper center", legend_cfg=legend_cfg)
    return fig

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
        st.markdown('<div class="divd"></div>', unsafe_allow_html=True)
        st.markdown("### 🔧 Advanced Customization")
        current_theme = THEMES.get(tn, {})
        adv_cfg = advanced_customization_controls(
            prefix=f"{prefix}_adv",
            features=["pitch", "figure", "markers", "arrows", "lines", "text", "title"],
            theme=current_theme,
        )

    S = dict(tn=tn, ad="ltr" if "Left" in adir else "rtl",
             fy=fy, pm=pm, pw=pw, adv=adv_cfg)
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
        "📄 Report Builder",
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
        chart_style = st.selectbox(
            "Chart style",
            ["Pro panel", "Notebook compact"],
            key="psm_chart_style",
        )

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

        if chart_style == "Notebook compact":
            st.markdown("**Notebook column mapping**")
            all_cols = list(df.columns)
            result_opts = [None] + all_cols
            xg_opts = [None] + all_cols
            bp_col = None
            result_col = st.selectbox(
                "Result column",
                result_opts,
                index=result_opts.index("outcome") if "outcome" in result_opts else 0,
                key="psm_comp_result_col",
            )
            xg_col = st.selectbox(
                "xG column",
                xg_opts,
                index=xg_opts.index("xg") if "xg" in xg_opts else 0,
                key="psm_comp_xg_col",
            )
        else:
            st.markdown("**Body part column (optional)**")
            bp_opts = [None] + list(df.columns)
            bp_col = st.selectbox("Body part column", bp_opts, key="psm_bp")
            result_col = None
            xg_col = None

        if chart_style == "Pro panel":
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
        else:
            tot_ov = rf_ov = lf_ov = hd_ov = oth_ov = ""
            min_pl = 0

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
                if chart_style == "Notebook compact":
                    if athletic_compact_shot_map is None:
                        raise ImportError("Notebook compact chart is missing from charts_pro.py. Deploy the updated charts_pro.py file.")
                    fig = athletic_compact_shot_map(
                        df, cfg=cfg,
                        goal_color=goal_col,
                        no_goal_color=nogoal_col,
                        edge_color=nogoal_edge,
                        min_dot_size=min_sz,
                        max_dot_size=max_sz,
                        pitch_mode=S["pm"],
                        pitch_width=S["pw"],
                        result_col=result_col,
                        xg_col=xg_col,
                    )
                else:
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
                adv = S.get("adv", {})
                apply_advanced_customization(fig, adv, theme)
                apply_figure_adv(fig, adv)
                _store_fig("psm_fig", fig)
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
                adv = S.get("adv", {})
                arr_cfg = adv.get("arrows", {})
                fig = opta_pass_map_pro(
                    df, cfg=cfg,
                    successful_color=succ_col,
                    unsuccessful_color=unsucc_col,
                    arrow_alpha=arr_cfg.get("alpha", arrow_alpha),
                    arrow_width=arr_cfg.get("width", arrow_w),
                    pitch_mode=S["pm"], pitch_width=S["pw"],
                    successful_override=succ_ov or None,
                    unsuccessful_override=un_ov  or None,
                    accuracy_override=acc_ov    or None,
                )
                apply_advanced_customization(fig, adv, THEMES.get(cfg["theme_name"], {}))
                apply_figure_adv(fig, adv)
                _store_fig("ppm_fig", fig)
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
                    adv = S.get("adv", {})
                    theme = {**THEMES[S["tn"]], **pitch_kw_from_adv(adv, THEMES[S["tn"]])}
                    sc_f  = {k:v for k,v in sc.items()   if k in leg_cfg["active"]}
                    sm_f  = {k:v for k,v in sm_m.items() if k in leg_cfg["active"]}
                    fig = shot_map(df, shot_colors=sc_f, shot_markers=sm_f,
                                   pitch_mode=S["pm"], pitch_width=S["pw"],
                                   show_xg=show_xg, theme_name=S["tn"],
                                   legend_cfg=leg_cfg)
                    ax = fig.axes[0]
                    tc_cfg = adv.get("title", {})
                    ax.set_title(t,
                        color=tc_cfg.get("color", theme["text"]),
                        fontsize=tc_cfg.get("size", 16),
                        fontweight=tc_cfg.get("weight", "bold"),
                        loc=tc_cfg.get("loc", "center"))
                    draw_attack_direction(ax, cfg_sm, theme, S["pm"], S["pw"])
                    apply_legend_style(ax, leg_cfg, theme)
                    if blocks:
                        draw_stat_blocks_bottom(fig, blocks, theme, y=0.02)
                    apply_overlay(fig, ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("atk_sm", fig)
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
                    _store_fig("atk_gl", fig)
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
                    _store_fig("atk_sz", fig)
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
                    adv = S.get("adv", {})
                    theme = THEMES[S["tn"]]
                    mk_cfg = adv.get("markers", {})
                    effective_size  = mk_cfg.get("size", tm_s)
                    effective_alpha = mk_cfg.get("alpha", tm_a)
                    effective_fill  = mk_cfg.get("fill", tm_c)
                    effective_edge  = mk_cfg.get("edge", tm_e)
                    if tm_v:
                        fig,ax,vp = make_full_vertical_pitch(S["pm"],S["pw"],theme)
                        d2 = df.dropna(subset=["x","y"])
                        vp.scatter(d2["x"],d2["y"],ax=ax,s=effective_size,
                                   marker=MARKER_OPTS[tm_ml],color=effective_fill,
                                   edgecolors=effective_edge,
                                   linewidth=mk_cfg.get("edge_width",2),
                                   alpha=effective_alpha,
                                   zorder=mk_cfg.get("zorder",5))
                        tc_c = adv.get("title",{}).get("color", theme["text"])
                        ax.set_title(tm_t,color=tc_c,fontsize=adv.get("title",{}).get("size",16),
                                     fontweight=adv.get("title",{}).get("weight","bold"))
                    else:
                        fig = touch_map(df,pitch_mode=S["pm"],pitch_width=S["pw"],
                                        theme_name=S["tn"],dot_color=effective_fill,
                                        edge_color=effective_edge,
                                        dot_size=effective_size,alpha=effective_alpha,
                                        marker=MARKER_OPTS[tm_ml],
                                        legend_cfg=leg_cfg)
                        ax = fig.axes[0]
                        tc_c = adv.get("title",{}).get("color", theme["text"])
                        ax.set_title(tm_t,color=tc_c,fontsize=adv.get("title",{}).get("size",16),
                                     fontweight=adv.get("title",{}).get("weight","bold"))
                        draw_attack_direction(ax,cfg_tm,theme,S["pm"],S["pw"],tm_v)
                    apply_overlay(fig,ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("atk_tm", fig)
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
                    _store_fig("atk_ht", fig)
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
                    _store_fig("atk_xt", fig)
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
                    adv = S.get("adv", {})
                    theme=THEMES[S["tn"]]
                    dc_f={k:v for k,v in dc.items() if k in leg_cfg["active"]}
                    dm_f={k:v for k,v in dm.items() if k in leg_cfg["active"]}
                    fig=defensive_actions_map(df,def_colors=dc_f,def_markers=dm_f,
                                              pitch_mode=S["pm"],pitch_width=S["pw"],
                                              theme_name=S["tn"],
                                              legend_cfg=leg_cfg)
                    ax=fig.axes[0]
                    tc_cfg = adv.get("title", {})
                    ax.set_title(t,
                        color=tc_cfg.get("color", theme["text"]),
                        fontsize=tc_cfg.get("size", 16),
                        fontweight=tc_cfg.get("weight", "bold"))
                    draw_attack_direction(ax,cfg_da,theme,S["pm"],S["pw"],dv)
                    apply_legend_style(ax,leg_cfg,theme)
                    if blocks: draw_stat_blocks_bottom(fig,blocks,theme,y=0.02)
                    apply_overlay(fig,ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("def_da", fig)
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
                    _store_fig("def_br", fig)
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
                    _store_fig("def_pr", fig)
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
                    _store_fig("def_mm", fig)
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
                    _store_fig("def_ob", fig)
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
                    adv = S.get("adv", {})
                    theme=THEMES[S["tn"]]
                    al=leg_cfg["active"]
                    pc_f={k:v for k,v in pc.items() if k in al}
                    pm_f={k:v for k,v in pm2.items() if k in al}
                    fig=pass_map(df,pass_colors=pc_f,pass_markers=pm_f,
                                  pitch_mode=S["pm"],pitch_width=S["pw"],
                                  theme_name=S["tn"],pass_view=pv,
                                  result_scope=ps,min_packing=ppk,
                                  legend_cfg=leg_cfg)
                    ax=fig.axes[0]
                    tc_cfg = adv.get("title", {})
                    ax.set_title(t,
                        color=tc_cfg.get("color", theme["text"]),
                        fontsize=tc_cfg.get("size", 16),
                        fontweight=tc_cfg.get("weight", "bold"),
                        loc=tc_cfg.get("loc", "center"))
                    draw_attack_direction(ax,cfg_pm,theme,S["pm"],S["pw"],dv)
                    apply_legend_style(ax,leg_cfg,theme)
                    if blocks: draw_stat_blocks_bottom(fig,blocks,theme,y=0.02)
                    apply_overlay(fig,ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("dist_pm", fig)
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
                    adv = S.get("adv", {})
                    theme=THEMES[S["tn"]]
                    arr_cfg = adv.get("arrows", {})
                    fig=progressive_carries_map(df,title=t,theme_name=S["tn"],
                                                 pitch_mode=S["pm"],pitch_width=S["pw"],
                                                 carry_color=arr_cfg.get("color", cc),
                                                 min_distance=md,
                                                 vertical_pitch=dv)
                    ax=fig.axes[0]
                    draw_attack_direction(ax,cfg_pc,theme,S["pm"],S["pw"],dv)
                    apply_overlay(fig,ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("dist_pc", fig)
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
                    adv = S.get("adv", {})
                    theme = THEMES[S["tn"]]
                    fig=passing_network(df,player_col=plcol,
                                         recipient_col=rccol or "recipient",
                                         title=t,theme_name=S["tn"],
                                         pitch_mode=S["pm"],pitch_width=S["pw"],
                                         node_color=nc,edge_color=ec,min_passes=mp)
                    apply_overlay(fig,ov)
                    apply_advanced_customization(fig, adv, theme)
                    apply_figure_adv(fig, adv)
                    _store_fig("dist_pn", fig)
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
                    _store_fig("spec_gm", fig)
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
                    _store_fig("spec_gsr", fig)
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
                    _store_fig("spec_vp", fig)
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
                        _store_fig("spec_sc", fig)
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
                            _store_fig(fk, fig)
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
                        _store_fig("piz_sc_fig", fig)
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
        tag_chart_mode=st.selectbox("Generated chart",
                                    ["All tagged events","Passes","Shots","Defensive actions","Carries","Touches"],
                                    key="tag_chart_mode")
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
        tag_leg_cfg=legend_controls_full(
            "tag",
            ["successful","unsuccessful","key pass","assist","goal","ontarget","off target","blocked","touch"],
        )
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
                    # ── Exact pixel → pitch coordinate mapping ──────────────
                    # pad_tag is now a tuple (X_MIN, X_MAX, Y_MIN, Y_MAX)
                    # The figure was saved with set_aspect("equal") and
                    # subplots_adjust(left=0,right=1,top=1,bottom=0) so the
                    # rendered pixel space maps linearly to figure coords,
                    # but set_aspect("equal") adds whitespace padding inside
                    # the axes.  We must account for that.
                    #
                    # Strategy: use the known figure size + dpi to get canvas
                    # pixels, then map linearly within those bounds.
                    if isinstance(pad_tag, tuple) and len(pad_tag)==4:
                        X_MIN_c, X_MAX_c, Y_MIN_c, Y_MAX_c = pad_tag
                    else:
                        X_MIN_c, X_MAX_c = -4.0, 104.0
                        Y_MIN_c, Y_MAX_c = -float(y_max_tag)*0.18, float(y_max_tag)*1.05

                    x_coord_range = X_MAX_c - X_MIN_c   # 108.0
                    y_coord_range = Y_MAX_c - Y_MIN_c

                    # Because set_aspect("equal") is used, the axes are
                    # letterboxed inside the figure.  The figure is drawn at
                    # fig_w×fig_h inches @100 dpi; axis limits set
                    # pitch coords; aspect=equal means the shorter dimension
                    # is fully used and the longer has padding.
                    # Simplest robust fix: treat the full image as the mapped
                    # region (the pitch drawing fills ~100% with no tight bbox).
                    x_pitch = round(X_MIN_c + (int(click["x"]) / max(img_w, 1)) * x_coord_range, 2)
                    y_pitch = round(Y_MAX_c - (int(click["y"]) / max(img_h, 1)) * y_coord_range, 2)
                    # Clamp to valid pitch range
                    x_pitch = max(0.0, min(100.0, x_pitch))
                    y_pitch = max(0.0, min(float(y_max_tag), y_pitch))
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
                    # ── IMPORTANT: Do NOT run prepare_df_for_charts on tagged data ──
                    # Tagged coords are already in pitch space (0-100, 0-pitch_width).
                    # prepare_df_for_charts would apply attack_direction + flip transforms
                    # which would corrupt the positions.  Instead we do a minimal clean.
                    ec = ensure_outcome(ec)
                    # Add event_type derived from outcome if missing
                    from charts import PASS_ORDER as _PASS_ORDER, SHOT_ORDER as _SHOT_ORDER
                    if "event_type" not in ec.columns or (ec["event_type"]=="other").all():
                        ec["event_type"] = "other"
                        ec.loc[ec["outcome"].isin(_PASS_ORDER), "event_type"] = "pass"
                        ec.loc[ec["outcome"].isin(_SHOT_ORDER), "event_type"] = "shot"
                    # Normalise outcome
                    from charts import _norm_outcome as _no
                    ec["outcome"] = ec["outcome"].apply(_no)
                    # Add required xg columns
                    if "xg" not in ec.columns: ec["xg"] = 0.08
                    ec["xg_zone"] = ec["xg"]
                    ec["xg_model"] = np.nan
                    ec["xg_source"] = "tagged"
                    # Drop NaN x/y
                    ec = ec.dropna(subset=["x","y"]).copy()

                    prepared = ec  # use directly — coords are already correct

                    et_counts=prepared["event_type"].value_counts() if "event_type" in prepared.columns else pd.Series(dtype=str)
                    has_xy2="x2" in prepared.columns and "y2" in prepared.columns and (prepared["x2"].notna()&prepared["y2"].notna()).any()
                    theme_t=THEMES.get(tn,THEMES["The Athletic Dark"])
                    tc=theme_t.get("text","white"); mc=theme_t.get("muted","#A0A7B4")
                    lc_t=theme_t.get("pitch_lines","#E6E6E6")

                    # ── colour / marker palettes ──────────────────────────────
                    _sc={"off target":"#FF8A00","ontarget":"#00C2FF","goal":"#00FF6A","blocked":"#AAAAAA"}
                    _sm={"off target":"^","ontarget":"D","goal":"*","blocked":"s"}
                    _pc={"successful":"#00FF6A","unsuccessful":"#FF4D4D","key pass":"#00C2FF","assist":"#FFD400"}
                    _pm={"successful":"o","unsuccessful":"x","key pass":"D","assist":"*"}
                    _dc={"interception":"#00C2FF","tackle":"#FF8A00","recovery":"#00FF6A",
                         "aerial_duel":"#FFD400","ground_duel":"#FF4D4D","clearance":"#A78BFA"}
                    _dm={"interception":"o","tackle":"s","recovery":"D",
                         "aerial_duel":"^","ground_duel":"x","clearance":"*"}

                    # ── build figure ──────────────────────────────────────────
                    if tag_chart_mode == "All tagged events":
                        fig=tagged_events_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,
                                              theme_name=tn,show_thirds=show_thirds,
                                              legend_cfg=tag_leg_cfg,title="Tagged Events")
                    elif tag_chart_mode == "Shots" and "shot" in et_counts.index and et_counts["shot"]>0:
                        fig=shot_map(prepared,shot_colors=_sc,shot_markers=_sm,
                                     pitch_mode=pm_tag,pitch_width=pw_tag,
                                     show_xg=True,theme_name=tn,
                                     show_pitch_thirds=show_thirds,
                                     legend_cfg=tag_leg_cfg)
                        fig.axes[0].set_title("Tagged Shots",color=tc,fontsize=14,weight="bold")
                    elif tag_chart_mode == "Passes" and "pass" in et_counts.index and et_counts["pass"]>0:
                        fig=pass_map(prepared,pass_colors=_pc,pass_markers=_pm,
                                     pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn,
                                     show_pitch_thirds=show_thirds,
                                     legend_cfg=tag_leg_cfg)
                        fig.axes[0].set_title("Tagged Passes",color=tc,fontsize=14,weight="bold")
                    elif tag_chart_mode == "Defensive actions" and "defensive" in et_counts.index:
                        fig=defensive_actions_map(prepared,def_colors=_dc,def_markers=_dm,
                                                   pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn,
                                                   show_pitch_thirds=show_thirds,
                                                   legend_cfg=tag_leg_cfg)
                        fig.axes[0].set_title("Tagged Defensive Actions",color=tc,fontsize=14,weight="bold")
                    elif tag_chart_mode == "Carries" and "carry" in et_counts.index and has_xy2:
                        fig=progressive_carries_map(prepared,title="Tagged Carries",
                                                     theme_name=tn,pitch_mode=pm_tag,
                                                     pitch_width=pw_tag,min_distance=0.0)
                    else:
                        fig=touch_map(prepared,pitch_mode=pm_tag,pitch_width=pw_tag,theme_name=tn,
                                      show_pitch_thirds=show_thirds,
                                      legend_cfg=tag_leg_cfg)
                        fig.axes[0].set_title("Tagged Events",color=tc,fontsize=14,weight="bold")

                    # ── add thirds overlay ────────────────────────────────────
                    ax_t=fig.axes[0]
                    y_max_t=float(pw_tag if pm_tag=="rect" else 100.0)
                    for xv in ([100/3, 200/3] if tag_chart_mode == "Carries" and show_thirds else []):
                        ax_t.plot([xv,xv],[0,y_max_t],color=lc_t,lw=0.9,ls="--",alpha=0.45,zorder=2)
                    for xpos,lbl in ([(100/6,"Defensive Third"),(50,"Middle Third"),(500/6,"Attacking Third")] if tag_chart_mode == "Carries" and show_thirds else []):
                        ax_t.text(xpos,y_max_t*0.97,lbl,color=mc,fontsize=7.5,
                                  ha="center",va="top",alpha=0.8,zorder=3,clip_on=True)

                    # ── attacking direction arrow ─────────────────────────────
                    arr_y = -y_max_t*0.07
                    ymin_cur,ymax_cur=ax_t.get_ylim()
                    ax_t.set_ylim(min(ymin_cur, arr_y - y_max_t*0.06), ymax_cur)
                    ax_t.annotate("",xy=(75,arr_y),xytext=(25,arr_y),
                                  arrowprops=dict(arrowstyle="-|>",color=mc,lw=1.8),
                                  annotation_clip=False)
                    ax_t.text(50,arr_y-y_max_t*0.04,"Attacking Direction",
                              color=mc,fontsize=8,ha="center",va="top",clip_on=False)

                    st.image(_bytes(fig,dpi=150),use_container_width=True)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Chart error: {e}")
        else:
            st.info("No events tagged yet. Click the pitch above to start tagging.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 📄  REPORT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
if section == "📄 Report Builder":
    hdr("📄", "Report Builder", "Assemble professional scouting reports from saved charts · annotate · export PNG/PDF")

    # ── session-state initialisation ──────────────────────────────────────────
    def _rb_init():
        defaults = {
            "rb_title":       "Scouting Report",
            "rb_subtitle":    "",
            "rb_scout":       "",
            "rb_date":        "",
            "rb_match":       "",
            "rb_notes":       "",
            "rb_slots":       [],       # list of {key, label, bytes}
            "rb_annotations": [],       # list of annotation dicts per slot index
            "rb_images":      [],       # list of inserted image dicts
            "rb_logo_club":   None,
            "rb_logo_comp":   None,
            "rb_logo_team":   None,
            "rb_photo":       None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
    _rb_init()

    # ── catalog of saved figures across the whole app ─────────────────────────
    FIGURE_CATALOG = {
        "atk_sm":   "⚔️ Shot Map",
        "atk_gl":   "⚔️ Goal Location",
        "atk_sz":   "⚔️ Shot Zones",
        "atk_tm":   "⚔️ Touch Map",
        "atk_ht":   "⚔️ Start Heatmap",
        "atk_xt":   "⚔️ xG Timeline",
        "psm_fig":  "🎯 Pro Shot Map",
        "ppm_fig":  "📨 Pro Pass Map",
        "def_da":   "🛡️ Defensive Actions",
        "def_br":   "🛡️ Ball Regains",
        "def_pr":   "🛡️ Pressure Map",
        "def_mo":   "🛡️ Momentum Chart",
        "def_ob":   "🛡️ Outcome Distribution",
        "dist_pm":  "🔄 Pass Map",
        "dist_pc":  "🔄 Progressive Carries",
        "dist_pn":  "🔄 Passing Network",
        "spec_gm":  "🎯 Goal Mouth",
        "spec_sr":  "🎯 Shot Report",
        "spec_vm":  "🎯 Vertical Map",
        "spec_sd":  "🎯 Shot Detail Card",
        "spec_ps":  "🎯 Pass Sonar",
        "spec_dt":  "🎯 Defensive Territory",
        "pizza_fig":"🍕 Pizza Radar",
    }

    available = {k: v for k, v in FIGURE_CATALOG.items()
                 if k in st.session_state and st.session_state[k] is not None}

    # ── helper: convert a session-state figure (now bytes) to bytes ──────────
    def _fig_to_bytes(fig_or_bytes):
        if isinstance(fig_or_bytes, bytes):
            return fig_or_bytes
        try:
            return _bytes(fig_or_bytes, dpi=180)
        except Exception:
            return None

    # ── ANNOTATION TYPES ──────────────────────────────────────────────────────
    ANNOTATION_TYPES = [
        "Text", "Rectangle", "Circle", "Ellipse",
        "Straight Arrow", "Curved Arrow", "Highlight Box", "Number Label",
    ]

    # ── helper: draw a single annotation onto a matplotlib Axes ──────────
    def _draw_annotation(ax, ann, ax_w_pts, ax_h_pts):
        """Draw one annotation dict onto ax. Coords are 0-100 % of axes."""
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as pe
        from matplotlib.transforms import blended_transform_factory

        atype  = ann.get("type", "Text")
        x_pct  = ann.get("x", 50) / 100.0
        y_pct  = 1.0 - ann.get("y", 50) / 100.0   # flip: 0=top in UI
        w_pct  = ann.get("w", 20) / 100.0
        h_pct  = ann.get("h", 10) / 100.0
        rot    = ann.get("rotation", 0)
        fc     = ann.get("fill_color",   "#FF0000")
        bc     = ann.get("border_color", "#FF4060")
        bw     = ann.get("border_width", 1.5)
        alpha  = ann.get("opacity", 0.7)
        zorder = ann.get("zorder", 10)
        fsize  = ann.get("font_size", 12)
        ffam   = ann.get("font", "sans-serif")
        text   = ann.get("text", "")
        label_n = ann.get("label_n", 1)

        tr = ax.transAxes

        if atype == "Text":
            ha = ann.get("ha", "left")
            try:
                txt = ax.text(x_pct, y_pct, text,
                              transform=tr,
                              fontsize=fsize, fontfamily=ffam,
                              color=bc, alpha=alpha, zorder=zorder,
                              rotation=rot, ha=ha, va="top",
                              fontweight="bold")
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])
            except Exception:
                pass

        elif atype in ("Rectangle", "Highlight Box"):
            try:
                # Use a regular Rectangle — FancyBboxPatch angle param unreliable
                patch = mpatches.Rectangle(
                    (x_pct, y_pct - h_pct), w_pct, h_pct,
                    transform=tr,
                    facecolor=fc, edgecolor=bc,
                    linewidth=bw, alpha=alpha, zorder=zorder,
                    angle=rot, rotation_point="xy",
                )
                ax.add_patch(patch)
                if text:
                    ax.text(x_pct + w_pct / 2, y_pct - h_pct / 2, text,
                            transform=tr,
                            fontsize=fsize, fontfamily=ffam,
                            color=bc, alpha=min(1.0, alpha + 0.3),
                            zorder=zorder + 1, ha="center", va="center")
            except Exception:
                pass

        elif atype in ("Circle", "Ellipse"):
            try:
                is_circle = (atype == "Circle")
                ew = min(w_pct, h_pct) if is_circle else w_pct
                eh = min(w_pct, h_pct) if is_circle else h_pct
                patch = mpatches.Ellipse(
                    (x_pct + ew / 2, y_pct - eh / 2), ew, eh,
                    transform=tr,
                    facecolor=fc, edgecolor=bc,
                    linewidth=bw, alpha=alpha, zorder=zorder,
                    angle=rot,
                )
                ax.add_patch(patch)
                if text:
                    ax.text(x_pct + ew / 2, y_pct - eh / 2, text,
                            transform=tr,
                            fontsize=fsize, color=bc,
                            zorder=zorder + 1, ha="center", va="center")
            except Exception:
                pass

        elif atype == "Straight Arrow":
            try:
                x2_pct = ann.get("x2", min(100, ann.get("x", 50) + ann.get("w", 20))) / 100.0
                y2_pct = 1.0 - ann.get("y2", min(100, ann.get("y", 50) + ann.get("h", 10))) / 100.0
                ax.annotate("",
                            xy=(x2_pct, y2_pct),
                            xytext=(x_pct, y_pct),
                            xycoords=tr, textcoords=tr,
                            arrowprops=dict(
                                arrowstyle="-|>",
                                color=bc, lw=bw, alpha=alpha,
                                mutation_scale=bw * 8),
                            zorder=zorder)
            except Exception:
                pass

        elif atype == "Curved Arrow":
            try:
                x2_pct = ann.get("x2", min(100, ann.get("x", 50) + ann.get("w", 20))) / 100.0
                y2_pct = 1.0 - ann.get("y2", min(100, ann.get("y", 50) + ann.get("h", 10))) / 100.0
                curve  = ann.get("curvature", 0.3)
                ax.annotate("",
                            xy=(x2_pct, y2_pct),
                            xytext=(x_pct, y_pct),
                            xycoords=tr, textcoords=tr,
                            arrowprops=dict(
                                arrowstyle="-|>",
                                connectionstyle=f"arc3,rad={curve}",
                                color=bc, lw=bw, alpha=alpha,
                                mutation_scale=bw * 8),
                            zorder=zorder)
            except Exception:
                pass

        elif atype == "Number Label":
            try:
                r = min(w_pct, h_pct) / 2
                circle = mpatches.Circle(
                    (x_pct, y_pct), radius=r,
                    transform=tr,
                    facecolor=fc, edgecolor=bc,
                    linewidth=bw, alpha=alpha, zorder=zorder,
                )
                ax.add_patch(circle)
                ax.text(x_pct, y_pct, str(label_n),
                        transform=tr,
                        fontsize=fsize, fontfamily=ffam,
                        color=bc, zorder=zorder + 1,
                        ha="center", va="center", fontweight="bold")
            except Exception:
                pass

    # ── helper: build the full report figure ──────────────────────────────────
    def build_report_figure(slots, annotations_per_slot, extra_images,
                             meta, logos, spacing=0.02, report_dpi=180):
        """
        Assemble a matplotlib figure from:
          - slots: list of {label, img_bytes}
          - annotations_per_slot: list of lists of annotation dicts
          - extra_images: list of {img, x, y, w, h, opacity}
          - meta: dict with title/subtitle/scout/date/match/notes
          - logos: dict with club/comp/team/photo PIL images
        Returns a matplotlib figure.
        """
        import matplotlib.gridspec as mgridspec
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import numpy as np

        n_charts = len(slots)
        has_header = True
        has_notes  = bool(meta.get("notes", "").strip())

        # Page dimensions (A4-ish landscape ratio)
        page_w = 16.0
        header_h = 2.0
        chart_h  = 5.5
        notes_h  = 1.2 if has_notes else 0
        total_h  = header_h + n_charts * (chart_h + spacing * 72) + notes_h + 0.6

        fig = plt.figure(figsize=(page_w, max(6.0, total_h)))
        bg_color = "#0E1117"
        fig.patch.set_facecolor(bg_color)

        # ── Header ─────────────────────────────────────────────────────────
        header_frac = header_h / total_h
        ax_hdr = fig.add_axes([0.0, 1.0 - header_frac, 1.0, header_frac])
        ax_hdr.set_facecolor("#111827")
        ax_hdr.axis("off")

        # Thin accent line at bottom of header
        ax_hdr.axhline(0, color="#00d4ff", lw=2)

        title_txt = meta.get("title", "Scouting Report")
        sub_txt   = meta.get("subtitle", "")
        scout_txt = meta.get("scout", "")
        date_txt  = meta.get("date", "")
        match_txt = meta.get("match", "")

        # Place logos
        logo_x = 0.01
        for logo_key, logo_img in [
            ("club", logos.get("club")),
            ("comp", logos.get("comp")),
            ("team", logos.get("team")),
        ]:
            if logo_img is not None:
                try:
                    arr = np.array(logo_img.convert("RGBA"))
                    im  = OffsetImage(arr, zoom=0.38)
                    ab  = AnnotationBbox(im, (logo_x + 0.03, 0.5),
                                         xycoords="axes fraction",
                                         frameon=False)
                    ax_hdr.add_artist(ab)
                    logo_x += 0.09
                except Exception:
                    pass

        text_x = logo_x + 0.01
        ax_hdr.text(text_x, 0.80, title_txt,
                    transform=ax_hdr.transAxes,
                    fontsize=22, fontweight="bold",
                    color="#e8f0fe", va="top")
        if sub_txt:
            ax_hdr.text(text_x, 0.52, sub_txt,
                        transform=ax_hdr.transAxes,
                        fontsize=13, color="#6b8cae", va="top")
        info_parts = []
        if match_txt: info_parts.append(match_txt)
        if scout_txt: info_parts.append(f"Scout: {scout_txt}")
        if date_txt:  info_parts.append(date_txt)
        if info_parts:
            ax_hdr.text(text_x, 0.25, "  |  ".join(info_parts),
                        transform=ax_hdr.transAxes,
                        fontsize=10, color="#a0aec0", va="top")

        # Player photo (right side)
        if logos.get("photo") is not None:
            try:
                arr = np.array(logos["photo"].convert("RGBA"))
                im  = OffsetImage(arr, zoom=0.42)
                ab  = AnnotationBbox(im, (0.97, 0.5),
                                     xycoords="axes fraction",
                                     frameon=False)
                ax_hdr.add_artist(ab)
            except Exception:
                pass

        # ── Chart slots ────────────────────────────────────────────────────
        used_height = header_frac
        for slot_idx, slot in enumerate(slots):
            img_bytes = slot.get("img_bytes")
            if not img_bytes:
                continue

            slot_frac = chart_h / total_h
            y_pos = 1.0 - used_height - slot_frac
            ax_chart = fig.add_axes([0.01, y_pos, 0.98, slot_frac])
            ax_chart.set_facecolor(bg_color)
            ax_chart.axis("off")

            # Render the chart bytes as image in axes
            try:
                from PIL import Image as _PILImage
                pil = _PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                ax_chart.imshow(np.array(pil), aspect="auto", zorder=1)
                ax_chart.set_xlim(0, np.array(pil).shape[1])
                ax_chart.set_ylim(np.array(pil).shape[0], 0)
            except Exception:
                ax_chart.text(0.5, 0.5, "Chart render error",
                              transform=ax_chart.transAxes,
                              ha="center", va="center", color="red")

            # Slot label
            lbl = slot.get("label", "")
            if lbl:
                ax_chart.text(0.01, 0.98, lbl,
                              transform=ax_chart.transAxes,
                              fontsize=9, color="#6b8cae",
                              va="top", alpha=0.8, zorder=5)

            # Draw annotations for this slot
            for ann in (annotations_per_slot[slot_idx]
                        if slot_idx < len(annotations_per_slot) else []):
                try:
                    _draw_annotation(ax_chart, ann, 0, 0)
                except Exception:
                    pass

            used_height += slot_frac + spacing * (chart_h / total_h) * 0.15

        # ── Extra images ───────────────────────────────────────────────────
        for ei in extra_images:
            try:
                img = ei.get("img")
                if img is None:
                    continue
                arr  = np.array(img.convert("RGBA"))
                x_f  = ei.get("x", 0.5)
                y_f  = ei.get("y", 0.5)
                zoom = ei.get("zoom", 0.3)
                alpha_val = ei.get("opacity", 1.0)
                im   = OffsetImage(arr, zoom=zoom, alpha=alpha_val)
                ab   = AnnotationBbox(im, (x_f, y_f),
                                      xycoords="figure fraction",
                                      frameon=False, zorder=20)
                fig.add_artist(ab)
            except Exception:
                pass

        # ── Notes ──────────────────────────────────────────────────────────
        if has_notes:
            notes_frac = notes_h / total_h
            ax_notes = fig.add_axes([0.01, 0.01, 0.98, notes_frac])
            ax_notes.set_facecolor("#0d1f35")
            ax_notes.axis("off")
            ax_notes.text(0.01, 0.88, "📝 Notes",
                          transform=ax_notes.transAxes,
                          fontsize=10, fontweight="bold",
                          color="#00d4ff", va="top")
            ax_notes.text(0.01, 0.60, meta["notes"],
                          transform=ax_notes.transAxes,
                          fontsize=9, color="#a0aec0",
                          va="top", wrap=True)
            ax_notes.add_patch(
                plt.Rectangle((0, 0), 1, 1,
                               transform=ax_notes.transAxes,
                               fill=False, edgecolor="#1e3a5f", lw=1))

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # REPORT BUILDER UI
    # ─────────────────────────────────────────────────────────────────────
    tab_meta, tab_charts, tab_ann, tab_imgs, tab_export = st.tabs([
        "📋 Report Info",
        "📊 Charts",
        "✏️ Annotations",
        "🖼️ Extra Images",
        "💾 Export",
    ])

    # ── TAB 1: REPORT INFO ────────────────────────────────────────────────
    with tab_meta:
        st.markdown("#### 📋 Report Information")
        c1, c2 = st.columns(2)
        with c1:
            st.session_state["rb_title"]    = st.text_input("Report title", st.session_state["rb_title"], key="rb_ti")
            st.session_state["rb_subtitle"] = st.text_input("Subtitle",     st.session_state["rb_subtitle"], key="rb_su")
            st.session_state["rb_match"]    = st.text_input("Match info (e.g. Arsenal 2-1 Chelsea)", st.session_state["rb_match"], key="rb_mi")
        with c2:
            st.session_state["rb_scout"]    = st.text_input("Scout name",   st.session_state["rb_scout"], key="rb_sc")
            st.session_state["rb_date"]     = st.text_input("Date",         st.session_state["rb_date"],  key="rb_da")
        st.session_state["rb_notes"] = st.text_area("Notes / Analysis", st.session_state["rb_notes"],
                                                     height=120, key="rb_no")

        st.markdown("#### 🖼️ Logos & Photos")
        lc1, lc2, lc3, lc4 = st.columns(4)
        with lc1:
            club_f = st.file_uploader("Club logo", type=["png","jpg","jpeg","svg"], key="rb_lc")
            if club_f:
                try:
                    st.session_state["rb_logo_club"] = Image.open(club_f).convert("RGBA")
                except Exception: pass
        with lc2:
            comp_f = st.file_uploader("Competition logo", type=["png","jpg","jpeg","svg"], key="rb_lco")
            if comp_f:
                try:
                    st.session_state["rb_logo_comp"] = Image.open(comp_f).convert("RGBA")
                except Exception: pass
        with lc3:
            team_f = st.file_uploader("Team logo", type=["png","jpg","jpeg","svg"], key="rb_lt")
            if team_f:
                try:
                    st.session_state["rb_logo_team"] = Image.open(team_f).convert("RGBA")
                except Exception: pass
        with lc4:
            photo_f = st.file_uploader("Player photo", type=["png","jpg","jpeg"], key="rb_ph")
            if photo_f:
                try:
                    st.session_state["rb_photo"] = Image.open(photo_f).convert("RGBA")
                except Exception: pass

        if st.button("🔄 Preview header", key="rb_prev_hdr"):
            meta = dict(
                title    = st.session_state["rb_title"],
                subtitle = st.session_state["rb_subtitle"],
                scout    = st.session_state["rb_scout"],
                date     = st.session_state["rb_date"],
                match    = st.session_state["rb_match"],
                notes    = "",
            )
            logos = dict(
                club  = st.session_state.get("rb_logo_club"),
                comp  = st.session_state.get("rb_logo_comp"),
                team  = st.session_state.get("rb_logo_team"),
                photo = st.session_state.get("rb_photo"),
            )
            try:
                fig_prev = build_report_figure([], [], [], meta, logos)
                prev_bytes = _bytes(fig_prev, dpi=120)
                plt.close(fig_prev)
                st.image(prev_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Preview error: {e}")

    # ── TAB 2: CHARTS ─────────────────────────────────────────────────────
    with tab_charts:
        st.markdown("#### 📊 Add Charts to Report")

        if not available:
            st.info("No charts saved yet. Generate charts in other sections first, then return here.")
        else:
            st.markdown("**Available saved charts:**")
            add_cols = st.columns(4)
            for i, (k, label) in enumerate(available.items()):
                with add_cols[i % 4]:
                    if st.button(f"＋ {label}", key=f"rb_add_{k}"):
                        fig_obj = st.session_state.get(k)
                        img_b = _fig_to_bytes(fig_obj)
                        if img_b:
                            st.session_state["rb_slots"].append({
                                "key": k,
                                "label": label,
                                "img_bytes": img_b,
                            })
                            st.session_state["rb_annotations"].append([])
                            st.rerun()

        st.markdown("---")
        slots = st.session_state["rb_slots"]
        if not slots:
            st.info("No charts added yet. Click a chart above to add it.")
        else:
            st.markdown(f"**Report contains {len(slots)} chart(s):**")
            for i, slot in enumerate(slots):
                sc1, sc2, sc3 = st.columns([3, 1, 1])
                with sc1:
                    new_lbl = st.text_input(f"Label #{i+1}", slot.get("label",""),
                                            key=f"rb_slbl_{i}")
                    st.session_state["rb_slots"][i]["label"] = new_lbl
                with sc2:
                    if st.button("🔃 Refresh", key=f"rb_ref_{i}"):
                        fig_obj = st.session_state.get(slot["key"])
                        if fig_obj:
                            st.session_state["rb_slots"][i]["img_bytes"] = _fig_to_bytes(fig_obj)
                            st.rerun()
                with sc3:
                    if st.button("🗑️ Remove", key=f"rb_del_{i}"):
                        st.session_state["rb_slots"].pop(i)
                        st.session_state["rb_annotations"].pop(i)
                        st.rerun()
                try:
                    st.image(slot["img_bytes"], use_container_width=True)
                except Exception:
                    st.warning("Could not preview this chart.")

    # ── TAB 3: ANNOTATIONS ────────────────────────────────────────────────
    with tab_ann:
        st.markdown("#### ✏️ Annotations")
        slots = st.session_state["rb_slots"]
        if not slots:
            st.info("Add charts first in the Charts tab.")
        else:
            slot_labels = [f"#{i+1}: {s.get('label','Chart')}" for i, s in enumerate(slots)]
            target_slot = st.selectbox("Apply annotation to chart", slot_labels, key="rb_ann_slot")
            slot_idx = slot_labels.index(target_slot)

            with st.expander("➕ Add New Annotation", expanded=True):
                ann_type = st.selectbox("Annotation type", ANNOTATION_TYPES, key="rb_ann_type")

                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    ax_ = st.slider("X position (%)", 0, 100, 10, key="rb_ann_x")
                    ay_ = st.slider("Y position (%)", 0, 100, 10, key="rb_ann_y")
                with ac2:
                    aw_ = st.slider("Width (%)", 1, 80, 20, key="rb_ann_w")
                    ah_ = st.slider("Height (%)", 1, 60, 10, key="rb_ann_h")
                with ac3:
                    ar_ = st.slider("Rotation °", -180, 180, 0, key="rb_ann_rot")
                    az_ = st.slider("Z-order", 1, 20, 10, key="rb_ann_z")

                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    fill_c  = st.color_picker("Fill color",   "#FF000033" if ann_type in ("Rectangle","Circle","Ellipse","Highlight Box","Number Label") else "#FFFFFF", key="rb_ann_fc")
                    border_c= st.color_picker("Border/text color", "#FF4060", key="rb_ann_bc")
                with bc2:
                    border_w = st.slider("Border width", 0.5, 8.0, 2.0, 0.25, key="rb_ann_bw")
                    opacity  = st.slider("Opacity", 0.05, 1.0, 0.75, 0.05, key="rb_ann_op")
                with bc3:
                    font_sz  = st.slider("Font size", 8, 36, 14, key="rb_ann_fsz")
                    font_fam = st.selectbox("Font", ["sans-serif","serif","monospace","Arial","Georgia"], key="rb_ann_ff")

                ann_text = ""
                if ann_type in ("Text", "Rectangle", "Circle", "Ellipse",
                                "Highlight Box", "Number Label"):
                    if ann_type == "Number Label":
                        label_n = st.number_input("Number", 1, 99, 1, key="rb_ann_ln")
                        ann_text = str(label_n)
                    else:
                        ann_text = st.text_input("Text label (optional)", "", key="rb_ann_txt")

                x2_ = y2_ = None
                curve_ = 0.3
                if ann_type in ("Straight Arrow", "Curved Arrow"):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        x2_ = st.slider("Arrow end X (%)", 0, 100, 60, key="rb_ann_x2")
                        y2_ = st.slider("Arrow end Y (%)", 0, 100, 60, key="rb_ann_y2")
                    with ec2:
                        if ann_type == "Curved Arrow":
                            curve_ = st.slider("Curvature", -1.0, 1.0, 0.3, 0.05, key="rb_ann_cv")

                if st.button("➕ Add annotation", key="rb_ann_add"):
                    new_ann = dict(
                        type=ann_type, x=ax_, y=ay_, w=aw_, h=ah_,
                        rotation=ar_, fill_color=fill_c, border_color=border_c,
                        border_width=border_w, opacity=opacity, zorder=az_,
                        font_size=font_sz, font=font_fam, text=ann_text,
                    )
                    if ann_type == "Number Label":
                        new_ann["label_n"] = int(ann_text) if ann_text.isdigit() else 1
                    if x2_ is not None:
                        new_ann["x2"] = x2_; new_ann["y2"] = y2_
                    if ann_type == "Curved Arrow":
                        new_ann["curvature"] = curve_
                    st.session_state["rb_annotations"][slot_idx].append(new_ann)
                    st.rerun()

            # Show existing annotations for this slot
            anns = st.session_state["rb_annotations"][slot_idx]
            if anns:
                st.markdown(f"**{len(anns)} annotation(s) on this chart:**")
                for ai, ann in enumerate(anns):
                    with st.expander(f"#{ai+1} — {ann.get('type','?')} | {ann.get('text','')}", expanded=False):
                        ec1, ec2 = st.columns([3,1])
                        with ec1:
                            st.write({k: v for k, v in ann.items() if k not in ("fill_color","border_color")})
                        with ec2:
                            if st.button("🗑️ Delete", key=f"rb_del_ann_{slot_idx}_{ai}"):
                                st.session_state["rb_annotations"][slot_idx].pop(ai)
                                st.rerun()

            # Live preview with annotations
            if st.button("👁️ Preview with annotations", key=f"rb_prev_ann_{slot_idx}"):
                slot = st.session_state["rb_slots"][slot_idx]
                from PIL import Image as _PILImage
                import numpy as np
                fig_p, ax_p = plt.subplots(figsize=(12, 7))
                fig_p.patch.set_facecolor("#0E1117")
                ax_p.set_facecolor("#0E1117")
                ax_p.axis("off")
                try:
                    pil = _PILImage.open(io.BytesIO(slot["img_bytes"])).convert("RGB")
                    arr = np.array(pil)
                    ax_p.imshow(arr, aspect="auto", zorder=1)
                    ax_p.set_xlim(0, arr.shape[1])
                    ax_p.set_ylim(arr.shape[0], 0)
                except Exception:
                    pass
                for ann in st.session_state["rb_annotations"][slot_idx]:
                    try:
                        _draw_annotation(ax_p, ann, 0, 0)
                    except Exception:
                        pass
                st.image(_bytes(fig_p, dpi=150), use_container_width=True)
                plt.close(fig_p)

    # ── TAB 4: EXTRA IMAGES ───────────────────────────────────────────────
    with tab_imgs:
        st.markdown("#### 🖼️ Insert Additional Images")
        st.caption("Add watermarks, logos, photos, or custom graphics onto the report.")

        n_imgs = st.number_input("Number of extra images", 0, 8, 0, key="rb_nimgs")
        extra_images = []
        for ii in range(int(n_imgs)):
            with st.expander(f"Image #{ii+1}", expanded=(ii==0)):
                img_f = st.file_uploader(f"Upload image #{ii+1}", type=["png","jpg","jpeg"], key=f"rb_ei_{ii}")
                if img_f:
                    try:
                        pil_ei = Image.open(img_f).convert("RGBA")
                        ic1, ic2 = st.columns(2)
                        with ic1:
                            ei_x = st.slider(f"X (fig %)", 0.0, 1.0, 0.5, 0.01, key=f"rb_ei_x_{ii}")
                            ei_y = st.slider(f"Y (fig %)", 0.0, 1.0, 0.5, 0.01, key=f"rb_ei_y_{ii}")
                        with ic2:
                            ei_zoom = st.slider(f"Size (zoom)", 0.05, 1.0, 0.25, 0.01, key=f"rb_ei_z_{ii}")
                            ei_op   = st.slider(f"Opacity", 0.05, 1.0, 1.0, 0.05, key=f"rb_ei_op_{ii}")
                        extra_images.append(dict(img=pil_ei, x=ei_x, y=ei_y,
                                                 zoom=ei_zoom, opacity=ei_op))
                        st.image(img_f, width=120)
                    except Exception as e:
                        st.error(f"Could not load image: {e}")

        # Persist extra_images in session state for export tab
        st.session_state["rb_extra_images"] = extra_images

    # ── TAB 5: EXPORT ─────────────────────────────────────────────────────
    with tab_export:
        st.markdown("#### 💾 Export Report")

        slots = st.session_state["rb_slots"]
        if not slots:
            st.info("Add at least one chart in the Charts tab before exporting.")
        else:
            ec1, ec2 = st.columns(2)
            with ec1:
                exp_dpi = st.slider("Export DPI", 72, 400, 200, 10, key="rb_exp_dpi")
                exp_spacing = st.slider("Chart spacing", 0.0, 0.1, 0.02, 0.005, key="rb_exp_sp")
            with ec2:
                exp_fmt = st.multiselect("Export format(s)", ["PNG","PDF"], default=["PNG"], key="rb_exp_fmt")

            if st.button("🔨 Build report", key="rb_build"):
                with st.spinner("Building report..."):
                    meta = dict(
                        title    = st.session_state["rb_title"],
                        subtitle = st.session_state["rb_subtitle"],
                        scout    = st.session_state["rb_scout"],
                        date     = st.session_state["rb_date"],
                        match    = st.session_state["rb_match"],
                        notes    = st.session_state["rb_notes"],
                    )
                    logos = dict(
                        club  = st.session_state.get("rb_logo_club"),
                        comp  = st.session_state.get("rb_logo_comp"),
                        team  = st.session_state.get("rb_logo_team"),
                        photo = st.session_state.get("rb_photo"),
                    )
                    extra_imgs = st.session_state.get("rb_extra_images", [])
                    try:
                        fig_report = build_report_figure(
                            slots,
                            st.session_state["rb_annotations"],
                            extra_imgs,
                            meta,
                            logos,
                            spacing=exp_spacing,
                            report_dpi=exp_dpi,
                        )
                        # Store as bytes immediately — keeps figure from being GC'd
                        _rb_png = _bytes(fig_report, dpi=exp_dpi)
                        _rb_pdf = _pdf(fig_report)
                        plt.close(fig_report)
                        st.session_state["rb_report_png"] = _rb_png
                        st.session_state["rb_report_pdf"] = _rb_pdf
                        st.success("Report built successfully!")
                    except Exception as e:
                        st.error(f"Build error: {e}")

            if "rb_report_png" in st.session_state:
                st.image(st.session_state["rb_report_png"], use_container_width=True)

                dl1, dl2 = st.columns(2)
                with dl1:
                    if "PNG" in exp_fmt:
                        st.download_button(
                            "⬇️ Download PNG",
                            st.session_state["rb_report_png"],
                            file_name="scouting_report.png",
                            mime="image/png",
                            key="rb_dl_png",
                        )
                with dl2:
                    if "PDF" in exp_fmt:
                        st.download_button(
                            "⬇️ Download PDF",
                            st.session_state["rb_report_pdf"],
                            file_name="scouting_report.pdf",
                            mime="application/pdf",
                            key="rb_dl_pdf",
                        )

                if st.button("🗑️ Clear report", key="rb_clear"):
                    st.session_state.pop("rb_report_png", None)
                    st.session_state.pop("rb_report_pdf", None)
                    st.rerun()

    st.stop()
