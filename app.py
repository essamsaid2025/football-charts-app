import os
import io
import tempfile

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

from charts import (
    load_data,
    clean_numeric_value,
    numeric_metric_columns,
    build_player_metric_table,
    generic_scatter_plot,
    generic_bar_chart,
    percentile_bar_chart,
    mpl_pizza_dark,
    athletic_pizza,
    prepare_df_for_charts,
    build_report_from_prepared_df,
    pizza_chart,
    shot_detail_card,
    defensive_regains_map,
    make_pitch,
    THEMES,
)


try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

from scouting_tools import (
    ROLE_TEMPLATES,
    standard_columns,
    numeric_metrics as scouting_numeric_metrics,
    coerce_numeric,
    infer_role_from_position,
    match_template_metrics,
    add_percentiles_and_score,
    player_profile,
    profile_text,
    recommendation_text,
    similar_players,
    auto_dataset_insights,
    comparison_chart,
    radar_chart,
    make_template_csv,
)

st.set_page_config(page_title='Football Charts Generator', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
/* ================================
   OPTA / WYSCOUT DARK UI THEME
   ================================ */
:root{
    --bg:#06101f;
    --bg2:#0b1220;
    --sidebar:#070f1d;
    --panel:#0f1b2d;
    --panel2:#111827;
    --border:#25344a;
    --border2:#334155;
    --text:#f8fafc;
    --muted:#94a3b8;
    --accent:#38bdf8;
    --accent2:#22c55e;
    --danger:#ef4444;
}

/* App background */
.stApp{
    background:
      radial-gradient(circle at top left, rgba(56,189,248,.13), transparent 28%),
      linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%) !important;
    color:var(--text) !important;
}
.block-container{
    padding-top:1rem !important;
    padding-left:1.4rem !important;
    padding-right:1.4rem !important;
    max-width:100% !important;
}

/* Force global text colors */
h1,h2,h3,h4,h5,h6,p,span,div,label,small{
    color:var(--text) !important;
}

/* Header */
.app-header{
    background:linear-gradient(135deg,rgba(56,189,248,.18),rgba(34,197,94,.10));
    border:1px solid rgba(148,163,184,.20);
    padding:22px 24px;
    border-radius:22px;
    margin-bottom:18px;
    box-shadow:0 18px 45px rgba(0,0,0,.24);
}
.app-title{
    font-size:2.05rem;
    font-weight:950;
    margin:0;
    line-height:1.08;
    letter-spacing:-.03em;
}
.app-subtitle{
    color:var(--muted) !important;
    margin-top:8px;
    font-size:.98rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#050b16 0%,#081426 100%) !important;
    border-right:1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div{
    background:transparent !important;
}
section[data-testid="stSidebar"] *{
    color:var(--text) !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaptionContainer{
    color:var(--text) !important;
    opacity:1 !important;
}

/* Cards / panels */
.panel{
    background:linear-gradient(180deg,rgba(15,27,45,.98),rgba(9,18,32,.98)) !important;
    border:1px solid var(--border) !important;
    border-radius:18px !important;
    padding:16px !important;
    margin-bottom:14px !important;
    box-shadow:0 12px 32px rgba(0,0,0,.22) !important;
}
.preview{
    background:linear-gradient(180deg,rgba(15,27,45,.96),rgba(9,18,32,.96)) !important;
    border:1px solid var(--border) !important;
    border-radius:20px !important;
    padding:18px !important;
    min-height:74vh !important;
    box-shadow:0 12px 32px rgba(0,0,0,.22) !important;
}
.small{color:var(--muted) !important;font-size:.9rem;}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea{
    background:#0b1728 !important;
    color:var(--text) !important;
    border:1px solid var(--border2) !important;
    border-radius:12px !important;
}
.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus{
    border-color:var(--accent) !important;
    box-shadow:0 0 0 1px rgba(56,189,248,.45) !important;
}

/* Select boxes */
div[data-baseweb="select"] > div{
    background:#0b1728 !important;
    border:1px solid var(--border2) !important;
    border-radius:12px !important;
    color:var(--text) !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] div{
    color:var(--text) !important;
}
ul[role="listbox"], div[role="listbox"]{
    background:#0b1728 !important;
    color:var(--text) !important;
}


/* Multiselect / dropdown menu fix */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"]{
    background:#0b1728 !important;
    color:#f8fafc !important;
    border:1px solid #334155 !important;
    border-radius:12px !important;
}
div[data-baseweb="popover"] *,
div[data-baseweb="menu"] *,
ul[role="listbox"] *,
div[role="listbox"] *{
    color:#f8fafc !important;
    opacity:1 !important;
}
li[role="option"],
div[role="option"]{
    background:#0b1728 !important;
    color:#f8fafc !important;
}
li[role="option"]:hover,
div[role="option"]:hover,
li[aria-selected="true"],
div[aria-selected="true"]{
    background:#1e293b !important;
    color:#ffffff !important;
}
span[data-baseweb="tag"]{
    background:#ef4444 !important;
    color:#ffffff !important;
}
span[data-baseweb="tag"] *{
    color:#ffffff !important;
}

/* Radio / checkbox */
.stRadio label,
.stCheckbox label{
    color:var(--text) !important;
    opacity:1 !important;
}
.stRadio div[role="radiogroup"] label > div:first-child,
.stCheckbox label > div:first-child{
    border-color:#64748b !important;
}

/* File uploader */
div[data-testid="stFileUploader"] section{
    background:#0b1728 !important;
    border:1px dashed #3b82f6 !important;
    border-radius:16px !important;
}
div[data-testid="stFileUploader"] section *{
    color:var(--text) !important;
}
div[data-testid="stFileUploader"] button{
    background:#111827 !important;
    color:var(--text) !important;
    border:1px solid var(--border2) !important;
    border-radius:12px !important;
}

/* Buttons */
.stButton > button{
    width:100% !important;
    border-radius:13px !important;
    border:1px solid rgba(56,189,248,.35) !important;
    background:linear-gradient(135deg,#0ea5e9,#2563eb) !important;
    color:white !important;
    font-weight:900 !important;
    padding:.65rem 1rem !important;
    box-shadow:0 10px 26px rgba(37,99,235,.22) !important;
}
.stButton > button:hover{
    border-color:#7dd3fc !important;
    filter:brightness(1.08) !important;
}
.stDownloadButton > button{
    width:100% !important;
    border-radius:13px !important;
    border:1px solid var(--border2) !important;
    background:#101d31 !important;
    color:white !important;
    font-weight:850 !important;
    padding:.65rem 1rem !important;
}

/* Alerts */
div[data-testid="stAlert"]{
    background:#0f2340 !important;
    border:1px solid #1e40af !important;
    border-radius:14px !important;
    color:var(--text) !important;
}
div[data-testid="stAlert"] *{color:var(--text) !important;}

/* Expanders / dataframes */
.streamlit-expanderHeader{
    background:#0b1728 !important;
    color:var(--text) !important;
    border-radius:12px !important;
}
[data-testid="stExpander"]{
    background:#0b1728 !important;
    border:1px solid var(--border) !important;
    border-radius:14px !important;
}
[data-testid="stDataFrame"]{
    border:1px solid var(--border) !important;
    border-radius:14px !important;
    overflow:hidden !important;
}

/* Sliders */
.stSlider label{color:var(--text) !important;}
.stSlider [data-baseweb="slider"] div{
    color:var(--text) !important;
}

/* Make collapsed sidebar button visible */
button[kind="header"]{
    color:var(--text) !important;
}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="app-header">
  <div class="app-title">⚽ Football Charts Generator</div>
  <div class="app-subtitle">Choose every chart separately: Match Report / Scatter / Bar / Percentile Bar / MPL Pizza / Athletic Pizza</div>
</div>
''', unsafe_allow_html=True)


def fig_to_bytes(fig, fmt='png', dpi=300):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', pad_inches=.25)
    buf.seek(0)
    return buf.getvalue()


def save_outputs(fig, base_name):
    png = fig_to_bytes(fig, 'png', 300)
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=.25)
    pdf_buf.seek(0)
    return [(f'{base_name}.png', png, 'image/png'), (f'{base_name}.pdf', pdf_buf.getvalue(), 'application/pdf')]


def clean_metric_df(df):
    d = df.copy()
    for c in d.columns:
        s = pd.to_numeric(d[c].astype(str).str.replace('%','', regex=False).str.replace(',','', regex=False).str.strip(), errors='coerce')
        if s.notna().sum() > 0:
            d[c] = s
    return d


def find_default_player_col(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    for k in ['player', 'player name', 'name']:
        if k in lower:
            return lower[k]
    return df.columns[0]


def show_downloads(files):
    for fname, data, mime in files:
        st.download_button(f'⬇️ Download {fname}', data=data, file_name=fname, mime=mime, key=f'dl_{fname}')


# =========================
# TAGGING TOOL HELPERS
# =========================
def theme_value(theme_name, key, fallback):
    try:
        return THEMES.get(theme_name, THEMES.get('The Athletic Dark', {})).get(key, fallback)
    except Exception:
        return fallback


def make_pitch_background_image(theme_name='The Athletic Dark', pitch_mode='rect', pitch_width=64.0, width_px=900, height_px=576):
    """Create a pitch image using the same theme colors used in charts.py."""
    theme = THEMES.get(theme_name, THEMES.get('The Athletic Dark', {}))
    fig_w = 10
    fig_h = fig_w * (height_px / max(width_px, 1))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor(theme.get('bg', '#0E1117'))
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme, vertical_pitch=False)
    pitch.draw(ax=ax)
    ax.set_facecolor(theme.get('pitch', '#1f5f3b'))
    y_max = pitch_width if pitch_mode == 'rect' else 100.0
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, y_max + 1)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize((width_px, height_px))
    return img


def _pixel_to_pitch(px, py, width_px, height_px, pitch_mode='rect', pitch_width=64.0):
    y_max = pitch_width if pitch_mode == 'rect' else 100.0
    x = max(0.0, min(100.0, float(px) / max(width_px, 1) * 100.0))
    y = max(0.0, min(y_max, float(py) / max(height_px, 1) * y_max))
    return round(x, 2), round(y, 2)


def _fabric_line_points(obj):
    left = float(obj.get('left', 0) or 0)
    top = float(obj.get('top', 0) or 0)
    width = float(obj.get('width', 0) or 0)
    height = float(obj.get('height', 0) or 0)

    # Most stable fallback for streamlit-drawable-canvas line objects.
    x1, y1 = left, top
    x2, y2 = left + width, top + height

    # If fabric gives direct coordinates, try to preserve direction.
    try:
        ox1, oy1 = float(obj.get('x1')), float(obj.get('y1'))
        ox2, oy2 = float(obj.get('x2')), float(obj.get('y2'))
        if all(abs(v) < 2000 for v in [ox1, oy1, ox2, oy2]):
            # Fabric often stores x1/x2 relative to line center; left/top is bbox top-left.
            if min(ox1, ox2, oy1, oy2) < 0:
                x1 = left + (ox1 - min(ox1, ox2))
                x2 = left + (ox2 - min(ox1, ox2))
                y1 = top + (oy1 - min(oy1, oy2))
                y2 = top + (oy2 - min(oy1, oy2))
    except Exception:
        pass
    return x1, y1, x2, y2


def canvas_object_to_event(obj, event_type, player, team, minute, outcome, tag, note, width_px, height_px, pitch_mode, pitch_width):
    typ = str(obj.get('type', '')).lower()
    line_events = {'pass', 'carry', 'dribble', 'cross'}

    if event_type.lower() in line_events or typ == 'line':
        x1p, y1p, x2p, y2p = _fabric_line_points(obj)
        x, y = _pixel_to_pitch(x1p, y1p, width_px, height_px, pitch_mode, pitch_width)
        x2, y2 = _pixel_to_pitch(x2p, y2p, width_px, height_px, pitch_mode, pitch_width)
    else:
        left = float(obj.get('left', 0) or 0)
        top = float(obj.get('top', 0) or 0)
        radius = float(obj.get('radius', 0) or 0)
        width = float(obj.get('width', 0) or 0)
        height = float(obj.get('height', 0) or 0)
        cx = left + (radius if radius else width / 2)
        cy = top + (radius if radius else height / 2)
        x, y = _pixel_to_pitch(cx, cy, width_px, height_px, pitch_mode, pitch_width)
        x2, y2 = np.nan, np.nan

    return {
        'event_id': None,
        'event_type': event_type.lower().strip(),
        'player': player,
        'team': team,
        'minute': minute,
        'x': x,
        'y': y,
        'x2': x2,
        'y2': y2,
        'outcome': outcome.lower().strip(),
        'tag': tag,
        'note': note,
    }


def tagged_events_map(df_events, theme_name='The Athletic Dark', pitch_mode='rect', pitch_width=64.0, title='Tagged Events Map'):
    theme = THEMES.get(theme_name, THEMES.get('The Athletic Dark', {}))
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme, vertical_pitch=False)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(theme.get('bg', '#0E1117'))
    pitch.draw(ax=ax)
    ax.set_facecolor(theme.get('pitch', '#1f5f3b'))
    y_max = pitch_width if pitch_mode == 'rect' else 100.0

    d = df_events.copy()
    for c in ['x', 'y', 'x2', 'y2']:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors='coerce')

    colors = {
        'pass': '#00C2FF',
        'carry': '#FF9300',
        'dribble': '#A78BFA',
        'cross': '#FFD400',
        'shot': '#00FF6A',
        'touch': '#FFFFFF',
        'defensive action': '#FF4D4D',
        'recovery': '#FF4D4D',
    }
    markers = {
        'shot': '*',
        'touch': 'o',
        'defensive action': 's',
        'recovery': 'D',
    }
    handles = []
    line_events = ['pass', 'carry', 'dribble', 'cross']
    for et in line_events:
        sub = d[d['event_type'].astype(str).str.lower() == et].dropna(subset=['x', 'y', 'x2', 'y2'])
        if len(sub):
            pitch.arrows(sub['x'], sub['y'], sub['x2'], sub['y2'], ax=ax, width=2.2, headwidth=5, headlength=5,
                         color=colors.get(et, theme.get('text', 'white')), alpha=0.9, zorder=4)
            handles.append(Line2D([0], [0], color=colors.get(et), lw=3, label=et.title()))

    for et in ['shot', 'touch', 'defensive action', 'recovery']:
        sub = d[d['event_type'].astype(str).str.lower() == et].dropna(subset=['x', 'y'])
        if len(sub):
            pitch.scatter(sub['x'], sub['y'], ax=ax, s=160 if et == 'shot' else 95,
                          marker=markers.get(et, 'o'), color=colors.get(et, theme.get('text', 'white')),
                          edgecolors=theme.get('bg', '#0E1117'), linewidth=1.5, alpha=0.95, zorder=6)
            handles.append(Line2D([0], [0], marker=markers.get(et, 'o'), color='none',
                                  markerfacecolor=colors.get(et), markeredgecolor=theme.get('bg', '#0E1117'),
                                  markersize=9, label=et.title()))

    ax.set_title(title, color=theme.get('text', 'white'), fontsize=18, weight='bold')
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    if handles:
        leg = ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.06), ncol=min(5, len(handles)), frameon=False)
        for t in leg.get_texts():
            t.set_color(theme.get('text', 'white'))
    return fig


with st.sidebar:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('App Section')
    app_section = st.radio('Choose App Section', ['Charts Generator', 'Player Scouting', 'Tagging Tool'], index=0)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('1) Output Type')
    mode = st.radio('Choose chart', [
        'Match Report',
        'Scatter Plot',
        'Bar Chart',
        'Percentile Bar',
        'MPL Pizza',
        'Athletic Pizza',
        'Old Simple Pizza',
        'Shot Detail Card',
        'Defensive Actions Map',
    ])
    uploaded = st.file_uploader('Upload CSV / Excel', type=['csv','xlsx','xls'])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('2) Global Style')
    theme_name = st.selectbox('Theme', list(THEMES.keys()), index=list(THEMES.keys()).index('The Athletic Dark') if 'The Athletic Dark' in THEMES else 0)
    title = st.text_input('Title', 'Chart Title')
    subtitle = st.text_input('Subtitle', '')
    bg_color = st.color_picker('Background color', '#0E1117')
    text_color = st.color_picker('Text color', '#FFFFFF')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('3) Colors')
    attacking_color = st.color_picker('Attacking', '#1A78CF')
    possession_color = st.color_picker('Possession / Progression', '#FF9300')
    defending_color = st.color_picker('Defending', '#D70232')
    bar_color = st.color_picker('Bar / Scatter main color', '#38BDF8')
    highlight_color = st.color_picker('Highlight color', '#FF9300')
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('4) Images')
    center_img_file = st.file_uploader('Center image for pizza / header image', type=['png','jpg','jpeg'], key='centerimg')
    center_scale = st.slider('Center image size', 8, 24, 13) / 100
    st.markdown('</div>', unsafe_allow_html=True)

center_img = None
if center_img_file is not None:
    try:
        center_img = Image.open(center_img_file).convert('RGBA')
    except Exception:
        center_img = None


# =========================
# INTERACTIVE TAGGING TOOL SECTION
# =========================
if app_section == 'Tagging Tool':
    st.markdown("""
    <div class="app-header">
      <div class="app-title">🖱️ Interactive Tagging Tool</div>
      <div class="app-subtitle">Draw passes / carries / shots on the pitch → save events → download CSV or generate a themed map</div>
    </div>
    """, unsafe_allow_html=True)

    if 'tagged_events' not in st.session_state:
        st.session_state.tagged_events = []

    tag_left, tag_right = st.columns([1.0, 1.65], gap='large')

    with tag_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('Tagging Settings')
        tag_pitch_mode_ui = st.selectbox('Pitch shape', ['Rectangular', 'Square'], key='tag_pitch_mode_ui')
        tag_pitch_mode = 'rect' if tag_pitch_mode_ui == 'Rectangular' else 'square'
        tag_pitch_width = st.slider('Rect pitch width', 50.0, 80.0, 64.0, 1.0, key='tag_pitch_width')
        event_type = st.selectbox('Event type', ['pass', 'carry', 'dribble', 'cross', 'shot', 'touch', 'defensive action', 'recovery'])
        outcome = st.selectbox('Outcome', ['successful', 'unsuccessful', 'goal', 'ontarget', 'off target', 'blocked', 'neutral'])
        action_tag = st.selectbox('Action tag', ['progressive', 'line breaking', 'into final third', 'into box', 'key pass', 'chance created', 'under pressure', 'turnover', 'recovery', 'duel', 'dangerous action', 'other'])
        player_name = st.text_input('Player', '')
        team_name = st.text_input('Team', '')
        minute = st.number_input('Minute', min_value=0, max_value=130, value=0, step=1)
        note = st.text_area('Scout note', '', height=90)
        stroke = st.color_picker('Drawing color', '#00C2FF')
        st.caption('For pass/carry/dribble/cross: draw a line from start to end. For shot/touch/defensive action: click one point.')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('Actions')
        if st.button('Undo last saved event') and st.session_state.tagged_events:
            st.session_state.tagged_events = st.session_state.tagged_events[:-1]
            st.rerun()
        if st.button('Clear all saved events'):
            st.session_state.tagged_events = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with tag_right:
        st.markdown('<div class="preview">', unsafe_allow_html=True)
        st.subheader('Interactive Pitch')
        width_px = 900
        height_px = 576 if tag_pitch_mode == 'rect' else 700
        bg_img = make_pitch_background_image(theme_name=theme_name, pitch_mode=tag_pitch_mode, pitch_width=tag_pitch_width, width_px=width_px, height_px=height_px)
        drawing_mode = 'line' if event_type in ['pass', 'carry', 'dribble', 'cross'] else 'point'

        if st_canvas is None:
            st.error('streamlit-drawable-canvas is not installed. Add it to requirements.txt then redeploy.')
            st.code('streamlit-drawable-canvas')
            st.info('Fallback: you can still use the saved CSV/map after installing the package.')
        else:
            canvas_result = st_canvas(
                fill_color='rgba(255, 255, 255, 0.0)',
                stroke_width=4,
                stroke_color=stroke,
                background_image=bg_img,
                update_streamlit=True,
                height=height_px,
                width=width_px,
                drawing_mode=drawing_mode,
                point_display_radius=6,
                key=f'tag_canvas_{event_type}_{theme_name}_{tag_pitch_mode}_{tag_pitch_width}',
            )

            objects = []
            if canvas_result.json_data and 'objects' in canvas_result.json_data:
                objects = canvas_result.json_data['objects']

            c1, c2 = st.columns(2)
            with c1:
                if st.button('Save last drawn action', disabled=(len(objects) == 0)):
                    try:
                        ev = canvas_object_to_event(
                            objects[-1], event_type, player_name, team_name, int(minute), outcome, action_tag, note,
                            width_px, height_px, tag_pitch_mode, tag_pitch_width
                        )
                        ev['event_id'] = len(st.session_state.tagged_events) + 1
                        st.session_state.tagged_events.append(ev)
                        st.success('Event saved.')
                    except Exception as e:
                        st.error(f'Could not save event: {e}')
            with c2:
                st.caption(f'Drawn objects on canvas: {len(objects)}')

        if st.session_state.tagged_events:
            df_tagged = pd.DataFrame(st.session_state.tagged_events)
            st.subheader('Saved Events')
            st.dataframe(df_tagged, use_container_width=True)
            st.download_button('⬇️ Download Tagged Events CSV', data=df_tagged.to_csv(index=False).encode('utf-8-sig'), file_name='tagged_events.csv', mime='text/csv')

            if st.button('Generate Map From Tagged Events'):
                fig_tag = tagged_events_map(df_tagged, theme_name=theme_name, pitch_mode=tag_pitch_mode, pitch_width=tag_pitch_width, title=title or 'Tagged Events Map')
                tag_files = save_outputs(fig_tag, 'tagged_events_map')
                st.image(tag_files[0][1], use_container_width=True)
                show_downloads(tag_files)
        else:
            st.info('Draw an action on the pitch, then click Save last drawn action.')
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# =========================
# PLAYER SCOUTING SECTION
# =========================
if app_section == 'Player Scouting':
    st.markdown("""
    <div class="app-header">
      <div class="app-title">🧠 Player Scouting</div>
      <div class="app-subtitle">Role templates • Position presets • Auto insights • Player comparison</div>
    </div>
    """, unsafe_allow_html=True)

    scout_left, scout_right = st.columns([1.0, 1.65], gap='large')

    if uploaded is None:
        with scout_right:
            st.markdown('<div class="preview">', unsafe_allow_html=True)
            st.info('Upload a player metrics CSV / Excel from the sidebar to start scouting.')
            st.download_button('⬇️ Download scouting template CSV', data=make_template_csv(), file_name='scouting_template.csv', mime='text/csv')
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, 'wb') as f:
            f.write(uploaded.getbuffer())
        df_scout_raw = load_data(path)

    cols_guess = standard_columns(df_scout_raw)

    with scout_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader('Scouting Settings')

        cols_list = df_scout_raw.columns.tolist()
        default_player = cols_guess.get('player') if cols_guess.get('player') in cols_list else cols_list[0]
        player_col = st.selectbox('Player column', cols_list, index=cols_list.index(default_player))

        optional_cols = ['None'] + cols_list
        def _optional_index(name):
            val = cols_guess.get(name)
            return optional_cols.index(val) if val in optional_cols else 0

        team_col_sel = st.selectbox('Team column', optional_cols, index=_optional_index('team'))
        position_col_sel = st.selectbox('Position column', optional_cols, index=_optional_index('position'))
        age_col_sel = st.selectbox('Age column', optional_cols, index=_optional_index('age'))
        minutes_col_sel = st.selectbox('Minutes column', optional_cols, index=_optional_index('minutes'))

        team_col = None if team_col_sel == 'None' else team_col_sel
        position_col = None if position_col_sel == 'None' else position_col_sel
        age_col = None if age_col_sel == 'None' else age_col_sel
        minutes_col = None if minutes_col_sel == 'None' else minutes_col_sel

        df_scout = df_scout_raw.copy()
        exclude_cols = [player_col, team_col, position_col, age_col, minutes_col]
        metric_cols_all = scouting_numeric_metrics(df_scout, exclude=exclude_cols)
        df_scout = coerce_numeric(df_scout, metric_cols_all)

        if minutes_col and minutes_col in df_scout.columns:
            min_minutes = st.number_input('Min minutes filter', min_value=0, value=0, step=100)
            if min_minutes > 0:
                df_scout = df_scout[pd.to_numeric(df_scout[minutes_col], errors='coerce').fillna(0) >= min_minutes].copy()
        else:
            min_minutes = 0

        role_options = list(ROLE_TEMPLATES.keys())
        inferred_role = None
        if position_col and position_col in df_scout.columns and len(df_scout):
            first_pos = df_scout[position_col].dropna().astype(str).head(1)
            if len(first_pos):
                inferred_role = infer_role_from_position(first_pos.iloc[0])
        default_role_index = role_options.index(inferred_role) if inferred_role in role_options else 0
        role = st.selectbox('Player role / position preset', role_options, index=default_role_index)

        preset_metrics, missing_preset, preset_mapping = match_template_metrics(df_scout, role)
        metric_mode = st.radio('Metrics source', ['Position Template', 'Manual'], horizontal=True)

        if metric_mode == 'Position Template':
            selected_metrics = st.multiselect('Preset metrics found in your file', metric_cols_all, default=[m for m in preset_metrics if m in metric_cols_all][:12])
            if missing_preset:
                with st.expander('Missing preset metrics', expanded=False):
                    st.write(missing_preset)
        else:
            selected_metrics = st.multiselect('Choose metrics manually', metric_cols_all, default=metric_cols_all[:min(12, len(metric_cols_all))])

        group_by_position = st.checkbox('Compare percentiles within same position', value=True)
        group_col = position_col if group_by_position and position_col else None

        if selected_metrics:
            df_scored = add_percentiles_and_score(df_scout, selected_metrics, group_col=group_col)
        else:
            df_scored = df_scout.copy()
            df_scored['Scouting Score'] = pd.NA

        players = sorted(df_scored[player_col].dropna().astype(str).unique().tolist()) if player_col in df_scored.columns else []
        selected_player = st.selectbox('Choose player', players) if players else None
        compare_options = [p for p in players if p != selected_player]
        compare_player = st.selectbox('Compare with', compare_options) if compare_options else None

        st.markdown('</div>', unsafe_allow_html=True)

    with scout_right:
        st.markdown('<div class="preview">', unsafe_allow_html=True)
        st.subheader('Scouting Preview')

        with st.expander('File preview / columns', expanded=False):
            st.write(list(df_scout_raw.columns))
            st.dataframe(df_scout_raw.head(30), use_container_width=True)

        if not selected_metrics:
            st.warning('Choose at least one metric to generate scouting score and insights.')
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric('Players', len(df_scored))
        with k2:
            st.metric('Metrics', len(selected_metrics))
        with k3:
            score_avg = pd.to_numeric(df_scored['Scouting Score'], errors='coerce').mean()
            st.metric('Avg Score', 'NA' if pd.isna(score_avg) else f'{score_avg:.1f}')

        st.subheader('Auto Insights')
        insights = auto_dataset_insights(df_scored, player_col=player_col, metrics=selected_metrics, team_col=team_col, age_col=age_col, minutes_col=minutes_col)
        for ins in insights:
            st.write('• ' + ins)

        if selected_player:
            st.subheader('Player Role Profile')
            prof = player_profile(df_scored, player_col, selected_player, selected_metrics, top_n=5)
            st.info(profile_text(selected_player, prof))

            st.subheader('Recommendation Text Generator')
            try:
                rec_txt = recommendation_text(
                    df_scored, player_col=player_col, player_name=selected_player, metrics=selected_metrics,
                    role=role, team_col=team_col, position_col=position_col, age_col=age_col, minutes_col=minutes_col
                )
                st.markdown(rec_txt)
            except Exception as e:
                st.warning(f'Recommendation text error: {e}')
            player_rows = df_scored[df_scored[player_col].astype(str) == str(selected_player)].copy()
            show_cols = [c for c in [player_col, team_col, position_col, age_col, minutes_col, 'Scouting Score'] if c and c in df_scored.columns]
            st.dataframe(player_rows[show_cols + selected_metrics].head(1), use_container_width=True)

            st.subheader('Similar Players')
            try:
                sims = similar_players(df_scored, player_col, selected_player, selected_metrics, top_n=8)
                sim_cols = [c for c in [player_col, team_col, position_col, age_col, minutes_col, 'Scouting Score', 'Similarity Distance'] if c and c in sims.columns]
                st.dataframe(sims[sim_cols], use_container_width=True)
            except Exception as e:
                st.warning(f'Could not calculate similar players: {e}')

        if selected_player and compare_player:
            st.subheader('Player Comparison')
            use_pct = st.radio('Comparison chart values', ['Percentiles', 'Raw values'], horizontal=True) == 'Percentiles'
            try:
                fig_cmp = comparison_chart(df_scored, player_col, selected_player, compare_player, selected_metrics, use_percentiles=use_pct)
                cmp_files = save_outputs(fig_cmp, 'player_comparison')
                st.image(cmp_files[0][1], use_container_width=True)
                show_downloads(cmp_files)
            except Exception as e:
                st.warning(f'Comparison chart error: {e}')

            try:
                fig_rad = radar_chart(df_scored, player_col, selected_player, compare_player, selected_metrics[:8])
                radar_files = save_outputs(fig_rad, 'player_radar')
                st.image(radar_files[0][1], use_container_width=True)
                show_downloads(radar_files)
            except Exception as e:
                st.warning(f'Radar chart error: {e}')

        st.subheader('Ranked Shortlist')
        shortlist_cols = [c for c in [player_col, team_col, position_col, age_col, minutes_col, 'Scouting Score'] if c and c in df_scored.columns]
        ranked = df_scored.sort_values('Scouting Score', ascending=False).copy()
        st.dataframe(ranked[shortlist_cols + selected_metrics].head(50), use_container_width=True)
        st.download_button('⬇️ Download ranked shortlist CSV', data=ranked[shortlist_cols + selected_metrics].to_csv(index=False).encode('utf-8-sig'), file_name='ranked_scouting_shortlist.csv', mime='text/csv')
        st.download_button('⬇️ Download scouting template CSV', data=make_template_csv(), file_name='scouting_template.csv', mime='text/csv')
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


left, right = st.columns([1.0, 1.65], gap='large')

with right:
    st.markdown('<div class="preview">', unsafe_allow_html=True)
    st.subheader('Preview & Downloads')

    if uploaded is None:
        st.info('Upload your file first.')
        st.markdown('</div>', unsafe_allow_html=True)
        st.stop()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, uploaded.name)
        with open(path, 'wb') as f:
            f.write(uploaded.getbuffer())
        df_raw = load_data(path)

    with st.expander('File preview / columns', expanded=False):
        st.write(list(df_raw.columns))
        st.dataframe(df_raw.head(30), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader('Chart Settings')

# =========================
# PLAYER METRIC CHARTS
# =========================
player_metric_modes = {'Scatter Plot','Bar Chart','Percentile Bar','MPL Pizza','Athletic Pizza','Old Simple Pizza'}

if mode in player_metric_modes:
    with left:
        player_col = st.selectbox('Player column', df_raw.columns.tolist(), index=df_raw.columns.tolist().index(find_default_player_col(df_raw)) if find_default_player_col(df_raw) in df_raw.columns else 0)
        df_metrics = clean_metric_df(df_raw)
        meta_cols = [player_col]
        for c in df_metrics.columns:
            if str(c).strip().lower() in ['team','squad','position','pos','age','minutes','mins','league','season','id']:
                meta_cols.append(c)
        metric_cols = numeric_metric_columns(df_metrics, exclude_cols=meta_cols, min_valid=1)
        if not metric_cols:
            st.error('No numeric metric columns found.')
            st.stop()
        players = sorted(df_metrics[player_col].dropna().astype(str).unique().tolist())
        selected_player = st.selectbox('Choose player', players)

        if mode == 'Scatter Plot':
            x_metric = st.selectbox('X-axis metric', metric_cols, index=0)
            y_metric = st.selectbox('Y-axis metric', metric_cols, index=min(1, len(metric_cols)-1))
            show_labels = st.checkbox('Show labels for highlighted player only', value=True)
            generate = st.button('Generate Scatter Plot')
        else:
            selected_metrics = st.multiselect('Choose metrics', metric_cols, default=metric_cols[:min(12, len(metric_cols))])
            value_mode = st.radio('Use values or percentiles?', ['Percentile', 'Raw Value'], horizontal=True)
            categories = []
            if mode in ['MPL Pizza','Athletic Pizza'] and selected_metrics:
                st.markdown('Metric categories')
                cat_defaults = ['Attacking','Possession','Defending']
                for m in selected_metrics:
                    categories.append(st.selectbox(str(m), cat_defaults, key=f'cat_{m}'))
            horizontal = st.checkbox('Horizontal bars', value=True) if mode == 'Bar Chart' else True
            generate = st.button(f'Generate {mode}', disabled=not selected_metrics)
    with right:
        st.markdown('<div class="preview">', unsafe_allow_html=True)
        if not generate:
            st.info('Choose settings then click Generate.')
        else:
            if mode == 'Scatter Plot':
                fig = generic_scatter_plot(
                    df_metrics, x_metric=x_metric, y_metric=y_metric, label_col=player_col,
                    highlight_value=selected_player if show_labels else None,
                    title=title or f'{x_metric} vs {y_metric}', x_color=bar_color,
                    highlight_color=highlight_color, bg_color=bg_color, text_color=text_color,
                )
                files = save_outputs(fig, 'scatter_plot')
            else:
                table = build_player_metric_table(df_metrics, player_col, selected_player, selected_metrics, value_mode=value_mode)
                if mode == 'Bar Chart':
                    fig = generic_bar_chart(table, title=title or selected_player, value_col='plot_value', bar_color=bar_color, bg_color=bg_color, text_color=text_color, horizontal=horizontal)
                    files = save_outputs(fig, 'bar_chart')
                elif mode == 'Percentile Bar':
                    fig = percentile_bar_chart(table, title=title or f'{selected_player} Percentile Bar', good_color=attacking_color, mid_color=possession_color, low_color=defending_color, bg_color=bg_color, text_color=text_color)
                    files = save_outputs(fig, 'percentile_bar')
                elif mode == 'MPL Pizza':
                    fig = mpl_pizza_dark(table, title=title or selected_player, subtitle=subtitle, categories=categories, attacking_color=attacking_color, possession_color=possession_color, defending_color=defending_color, center_image=center_img, center_img_scale=center_scale, footer_text='data: uploaded file')
                    files = save_outputs(fig, 'mpl_pizza')
                elif mode == 'Athletic Pizza':
                    fig = athletic_pizza(table, title=title or selected_player, subtitle=subtitle, categories=categories, attacking_color=attacking_color, possession_color=possession_color, defending_color=defending_color)
                    files = save_outputs(fig, 'athletic_pizza')
                else:
                    # Old simple pizza from previous app
                    simple = table.rename(columns={'metric':'metric','value':'value','percentile':'percentile'})[['metric','value','percentile']]
                    # Old Simple Pizza scale only:
                    # Elite = Blue, Above average = Green, Average = Orange, Below average = Red
                    colors = [
                        '#1A78CF' if float(p) >= 85 else
                        '#2ECC71' if float(p) >= 70 else
                        '#FF9300' if float(p) >= 50 else
                        '#D70232'
                        for p in simple['percentile']
                    ]
                    fig = pizza_chart(simple, title=title or selected_player, subtitle=subtitle, slice_colors=colors, center_image=center_img, center_img_scale=center_scale, show_values_legend=False)
                    files = save_outputs(fig, 'simple_pizza')
            st.image(files[0][1], use_container_width=True)
            show_downloads(files)
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# =========================
# EVENT-DATA CHARTS
# =========================
with left:
    attack_dir_ui = st.selectbox('Attack direction', ['Left → Right', 'Right → Left'])
    attack_dir = 'ltr' if attack_dir_ui == 'Left → Right' else 'rtl'
    flip_y = st.checkbox('Flip Y axis', value=False)
    pitch_mode_ui = st.selectbox('Pitch shape', ['Rectangular', 'Square'])
    pitch_mode = 'rect' if pitch_mode_ui == 'Rectangular' else 'square'
    pitch_width = st.slider('Rect pitch width', 50.0, 80.0, 64.0, 1.0)
    vertical_pitch = st.checkbox('Vertical pitch', value=False)

    pass_colors = {'successful':'#00FF6A','unsuccessful':'#FF4D4D','key pass':'#00C2FF','assist':'#FFD400'}
    shot_colors = {'off target':'#FF8A00','ontarget':'#00C2FF','goal':'#00FF6A','blocked':'#AAAAAA'}
    def_colors = {'interception':'#00C2FF','tackle':'#FF8A00','recovery':'#00FF6A','aerial_duel':'#FFD400','ground_duel':'#FF4D4D','clearance':'#A78BFA'}
    bar_colors = {**pass_colors, **shot_colors}
    pass_markers = {'successful':'o','unsuccessful':'x','key pass':'D','assist':'*'}
    shot_markers = {'off target':'^','ontarget':'D','goal':'*','blocked':'s'}
    def_markers = {'interception':'o','tackle':'s','recovery':'D','aerial_duel':'^','ground_duel':'x','clearance':'*'}

    if mode == 'Match Report':
        all_charts = ['Outcome Bar','Start Heatmap','Touch Map (Scatter)','Pass Map','Shot Map','Defensive Actions Map']
        selected_charts = st.multiselect('Choose report charts', all_charts, default=all_charts)
        generate = st.button('Generate Match Report')
    elif mode == 'Shot Detail Card':
        generate = st.button('Prepare Shots')
    else:
        generate = st.button('Generate Defensive Map')
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="preview">', unsafe_allow_html=True)
    try:
        df2 = prepare_df_for_charts(df_raw, attack_direction=attack_dir, flip_y=flip_y, pitch_mode=pitch_mode, pitch_width=pitch_width, xg_method='zone')
    except Exception as e:
        st.error(f'Event-data error: {e}')
        st.info('For Scatter / Bar / Pizza use player metrics file. For Match Report use event file with outcome, x, y.')
        st.stop()

    if not generate:
        st.info('Choose settings then click Generate.')
    else:
        if mode == 'Match Report':
            with tempfile.TemporaryDirectory() as tmp:
                pdf_path, png_paths = build_report_from_prepared_df(
                    df2, out_dir=tmp, title=title, subtitle=subtitle,
                    header_image=center_img, theme_name=theme_name, pitch_mode=pitch_mode, pitch_width=pitch_width,
                    pass_colors=pass_colors, pass_markers=pass_markers,
                    shot_colors=shot_colors, shot_markers=shot_markers,
                    def_colors=def_colors, def_markers=def_markers,
                    bar_colors=bar_colors, charts_to_include=selected_charts,
                    vertical_pitch=vertical_pitch,
                )
                files = [('report.pdf', open(pdf_path,'rb').read(), 'application/pdf')]
                for p in png_paths:
                    files.append((os.path.basename(p), open(p,'rb').read(), 'image/png'))
                for p in png_paths:
                    st.image(open(p,'rb').read(), use_container_width=True)
                show_downloads(files)
        elif mode == 'Shot Detail Card':
            shots = df2[df2['event_type'] == 'shot'].copy().reset_index(drop=True)
            if shots.empty:
                st.error('No shots found.')
            else:
                labels = [f'{i+1} | {r.outcome} | xG {float(r.xg):.2f}' for i, r in shots.iterrows()]
                selected = st.selectbox('Select shot', labels)
                shot_idx = int(selected.split('|')[0].strip()) - 1
                if st.button('Generate Shot Card Now'):
                    fig, _ = shot_detail_card(df2, shot_idx, title=title, pitch_mode=pitch_mode, pitch_width=pitch_width, shot_colors=shot_colors, shot_markers=shot_markers, theme_name=theme_name)
                    files = save_outputs(fig, 'shot_card')
                    st.image(files[0][1], use_container_width=True)
                    show_downloads(files)
        else:
            fig = defensive_regains_map(df2, title=title or 'Ball Regains Map', def_colors=def_colors, def_markers=def_markers, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, vertical_pitch=vertical_pitch)
            files = save_outputs(fig, 'defensive_actions_map')
            st.image(files[0][1], use_container_width=True)
            show_downloads(files)
    st.markdown('</div>', unsafe_allow_html=True)
