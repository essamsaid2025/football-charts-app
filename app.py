import os
import io
import tempfile

import streamlit as st
import pandas as pd
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
    THEMES,
)

st.set_page_config(page_title='Football Charts Generator', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.stApp {background:#0b1220;color:#f3f4f6;}
.block-container {padding-top:1rem; max-width:100%;}
h1,h2,h3,h4,h5,h6,p,span,div,label {color:#f3f4f6;}
.app-header{background:linear-gradient(135deg,rgba(56,189,248,.18),rgba(16,185,129,.10));border:1px solid rgba(255,255,255,.08);padding:20px 22px;border-radius:20px;margin-bottom:16px;}
.app-title{font-size:2rem;font-weight:900;margin:0;line-height:1.1;}
.app-subtitle{color:#9ca3af;margin-top:8px;font-size:.95rem;}
.panel{background:rgba(17,24,39,.94);border:1px solid #243041;border-radius:18px;padding:15px;margin-bottom:14px;}
.preview{background:rgba(17,24,39,.94);border:1px solid #243041;border-radius:18px;padding:16px;min-height:74vh;}
.small{color:#9ca3af;font-size:.9rem;}
.stButton>button,.stDownloadButton>button{width:100%;border-radius:12px;font-weight:800;}
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


with st.sidebar:
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
            if str(c).strip().lower() in ['team','squad','position','pos','age','minutes','mins','league','season']:
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
                    colors = [attacking_color if p >= 75 else possession_color if p >= 50 else defending_color for p in simple['percentile']]
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
