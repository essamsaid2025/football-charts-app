"""
charts_extra.py  —  Fully self-contained extra chart functions.
Only imports PUBLIC names from charts.py (THEMES, make_pitch).
No private underscore functions imported from charts.py.
"""
from __future__ import annotations
import math
import io
import os
import sys
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch, Arc
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from mplsoccer import Pitch, VerticalPitch
from scipy.spatial import ConvexHull

# ── Dynamic Import Fix for base 'charts' module ───────────────────────────
try:
    from charts import THEMES, make_pitch
except ImportError:
    # Fallback search if sys.modules contains the dynamically loaded module
    charts_mod = sys.modules.get("charts")
    if charts_mod is not None:
        THEMES = getattr(charts_mod, "THEMES", {})
        make_pitch = getattr(charts_mod, "make_pitch", None)
    else:
        # Emergency local file fallback detection
        possible_names = ["charts", "charts (9)", "charts__8_"]
        loaded = False
        for name in possible_names:
            if name in sys.modules:
                THEMES = getattr(sys.modules[name], "THEMES", {})
                make_pitch = getattr(sys.modules[name], "make_pitch", None)
                loaded = True
                break
        if not loaded:
            # Absolute baseline defaults to prevent script crash
            THEMES = {
                "Dark": {"bg": "#121212", "pitch": "#1e1e1e", "line": "#444444", "text": "#ffffff", "accent": "#00ffcc"},
                "Light": {"bg": "#ffffff", "pitch": "#f0f0f0", "line": "#cccccc", "text": "#000000", "accent": "#ff0055"}
            }
            def make_pitch(theme_name="Dark", pitch_type="statsbomb", orientation="horizontal", view="full"):
                t = THEMES.get(theme_name, THEMES["Dark"])
                return Pitch(pitch_color=t["pitch"], line_color=t["line"])
# ──────────────────────────────────────────────────────────────────────────

# Setup fonts
try:
    font_path = os.path.join(os.path.dirname(__file__), 'GeistMono-Bold.otf')
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        FONT_NAME = fm.FontProperties(fname=font_path).get_name()
    else:
        FONT_NAME = 'DejaVu Sans'
except Exception:
    FONT_NAME = 'DejaVu Sans'

# LOCAL COPIES of helpers (avoids importing private _ functions from charts.py)
PASS_ORDER_X = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER_X  = ["off target", "ontarget", "goal", "blocked"]
DEF_COLS_X    = ["interception", "tackle", "recovery", "aerial_duel", "ground_duel", "clearance"]


def _is_no_mk(m) -> bool:
    return m is None or str(m).strip().lower() in {"", "none", "no marker", "null"}


def _yes_col(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype=bool)
    x = s.copy().replace("", pd.NA).fillna(False)
    if pd.api.types.is_numeric_dtype(x):
        return (pd.to_numeric(x, errors="coerce").fillna(0) == 1)
    xs = x.astype(str).str.strip().str.lower()
    return xs.isin({"yes", "y", "true", "t", "1", "نعم"}).astype(bool)


def draw_pass_sonar(df, player_name, theme_name="Dark"):
    """Draws a Pass Sonar chart for a specific player."""
    t = THEMES.get(theme_name, THEMES.get("Dark", {"bg": "#121212", "text": "#ffffff", "accent": "#00ffcc"}))
    
    p_df = df[(df['type_name'] == 'Pass') & (df['player_name'] == player_name)].copy()
    if p_df.empty:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=t["bg"])
        ax.set_facecolor(t["bg"])
        ax.text(0.5, 0.5, "No pass data available\nfor this player.", 
                color=t["text"], fontname=FONT_NAME, fontsize=14, ha='center', va='center')
        ax.axis('off')
        return fig
        
    p_df['dx'] = p_df['pass_end_x'] - p_df['x']
    p_df['dy'] = p_df['pass_end_y'] - p_df['y']
    p_df['angle'] = np.arctan2(p_df['dy'], p_df['dx'])
    p_df['angle_deg'] = np.degrees(p_df['angle']) % 360
    p_df['length'] = np.sqrt(p_df['dx']**2 + p_df['dy']**2)
    
    num_bins = 16
    bin_edges = np.linspace(0, 360, num_bins + 1)
    p_df['angle_bin'] = pd.cut(p_df['angle_deg'], bins=bin_edges, labels=False, include_lowest=True)
    
    grouped = p_df.groupby('angle_bin').agg(
        count=('id', 'count'),
        avg_len=('length', 'mean'),
        prog_count=('pass_progressive', lambda x: sorted(x)[-1] if len(x)>0 else 0)
    ).reindex(range(num_bins), fill_value=0)
    
    angles = np.linspace(0, 2*np.pi, num_bins, endpoint=False)
    counts = grouped['count'].values
    lengths = grouped['avg_len'].values
    
    fig = plt.figure(figsize=(7, 7), facecolor=t["bg"])
    ax = fig.add_subplot(111, projection='polar')
    ax.set_facecolor(t["bg"])
    
    widths = np.ones(num_bins) * (2 * np.pi / num_bins) * 0.9
    cmap = LinearSegmentedColormap.from_list("sonar", [t.get("line", "#444444"), t["accent"]])
    norm = plt.Normalize(vmin=0, vmax=max(lengths) if max(lengths)>0 else 40)
    
    ax.bar(angles, counts, width=widths, bottom=0.0, color=cmap(norm(lengths)), edgecolor=t["bg"], linewidth=1)
    
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    
    ax.set_xticklabels(['Forward', 'Run/Left', 'Backward', 'Run/Right'], color=t["text"], fontname=FONT_NAME, fontsize=10)
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    
    ax.spines['polar'].set_visible(False)
    ax.grid(True, color=t.get("line", "#444444"), alpha=0.5, linestyle='--')
    ax.tick_params(colors=t["text"], labelsize=9)
    
    ax.set_title(f"PASS SONAR: {player_name.upper()}\nBar length = Frequency | Color = Avg Distance", 
                 color=t["text"], fontname=FONT_NAME, fontsize=12, pad=20, va='bottom')
    
    plt.tight_layout()
    return fig


def draw_pass_network_advanced(df, team_name, theme_name="Dark"):
    """Draws an advanced Pass Network with convex hulls / positioning variance."""
    t = THEMES.get(theme_name, THEMES.get("Dark", {"bg": "#121212", "pitch": "#1e1e1e", "line": "#444444", "text": "#ffffff", "accent": "#00ffcc"}))
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=t["pitch"], line_color=t["line"], goal_type='line')
    fig, ax = pitch.draw(figsize=(10, 7), facecolor=t["bg"])
    
    team_df = df[df['team_name'] == team_name].copy()
    if team_df.empty:
        ax.text(60, 40, f"No data found for {team_name}", color=t["text"], fontname=FONT_NAME, fontsize=14, ha='center')
        return fig
        
    players = []
    for p in team_df['player_name'].dropna().unique():
        p_events = team_df[team_df['player_name'] == p]
        first_loc = p_events.index[0]
        players.append({'name': p, 'first_idx': first_loc, 'count': len(p_events)})
    
    top_players = pd.DataFrame(players).sort_values('first_idx').head(11)['name'].tolist()
    net_df = team_df[(team_df['player_name'].isin(top_players)) & (team_df['type_name'] == 'Pass')].copy()
    avg_locs = net_df.groupby('player_name').agg(x=('x', 'mean'), y=('y', 'mean')).reset_index()
    
    for p in top_players[:3]:
        p_locs = net_df[net_df['player_name'] == p][['x', 'y']].dropna()
        if len(p_locs) > 5:
            try:
                hull = ConvexHull(p_locs.values)
                hull_pts = p_locs.values[hull.vertices]
                poly = plt.Polygon(hull_pts, facecolor=t["accent"], alpha=0.05, edgecolor=t["accent"], linestyle=':', linewidth=1)
                ax.add_patch(poly)
            except:
                pass

    net_df['next_player'] = net_df['pass_recipient_name']
    pairs = net_df[net_df['next_player'].isin(top_players)].groupby(['player_name', 'next_player']).size().reset_index(name='count')
    pairs = pairs[pairs['count'] > 2]
    
    loc_dict = avg_locs.set_index('player_name').to_dict(orient='index')
    max_count = pairs['count'].max() if not pairs.empty else 1
    
    for _, row in pairs.iterrows():
        p1, p2 = row['player_name'], row['next_player']
        if p1 in loc_dict and p2 in loc_dict:
            x1, y1 = loc_dict[p1]['x'], loc_dict[p1]['y']
            x2, y2 = loc_dict[p2]['x'], loc_dict[p2]['y']
            alpha = max(0.2, row['count'] / max_count)
            lw = (row['count'] / max_count) * 5 + 1
            pitch.lines(x1, y1, x2, y2, color=t["accent"], alpha=alpha, lw=lw, ax=ax, zorder=2)
            
    for _, row in avg_locs.iterrows():
        p = row['player_name']
        p_events_count = len(net_df[net_df['player_name'] == p])
        size = max(100, min(600, p_events_count * 10))
        pitch.scatter(row['x'], row['y'], s=size, color=t["bg"], edgecolor=t["accent"], linewidth=2, zorder=3, ax=ax)
        
        display_name = p.split(' ')[-1]
        ax.text(row['x'], row['y'] - 3, display_name, color=t["text"], fontname=FONT_NAME, fontsize=9, ha='center', zorder=4)
        
    ax.set_title(f"ADVANCED PASSING NETWORK & TERRITORY\n{team_name.upper()} Starting Lineup", 
                 color=t["text"], fontname=FONT_NAME, fontsize=14, pad=15)
                 
    return fig


def draw_defensive_territory(df, team_name, theme_name="Dark"):
    """Draws a visual heatmap overlay showcasing defensive interventions."""
    t = THEMES.get(theme_name, THEMES.get("Dark", {"bg": "#121212", "pitch": "#1e1e1e", "line": "#444444", "text": "#ffffff", "accent": "#00ffcc"}))
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color=t["pitch"], line_color=t["line"])
    fig, ax = pitch.draw(figsize=(10, 7), facecolor=t["bg"])
    
    def_actions = ['Ball Recovery', 'Block', 'Clearance', 'Interception', 'Tackle']
    def_df = df[(df['team_name'] == team_name) & (df['type_name'].isin(def_actions))].copy()
    
    if def_df.empty or len(def_df) < 3:
        ax.text(60, 40, "Insufficient defensive event data", color=t["text"], fontname=FONT_NAME, fontsize=14, ha='center')
        return fig
        
    try:
        pitch.kdeplot(def_df['x'], def_df['y'], ax=ax,
                      cmap=LinearSegmentedColormap.from_list("def", [t["pitch"], t["accent"]]),
                      fill=True, alpha=0.4, levels=8, thresh=0.1)
    except:
        pitch.scatter(def_df['x'], def_df['y'], color=t["accent"], alpha=0.5, s=60, ax=ax)
        
    markers = {'Tackle': 'o', 'Interception': 's', 'Block': '^', 'Clearance': 'x', 'Ball Recovery': 'D'}
    for act in def_actions:
        act_df = def_df[def_df['type_name'] == act]
        if not act_df.empty:
            pitch.scatter(act_df['x'], act_df['y'], alpha=0.8, s=40, 
                          color=t["text"], marker=markers.get(act, 'o'), edgecolors=t["bg"], 
                          label=act, ax=ax, zorder=3)
            
    avg_x = def_df['x'].mean()
    avg_y = def_df['y'].mean()
    pitch.scatter(avg_x, avg_y, color='#ff0055', marker='*', s=300, edgecolors=t["text"], zorder=5, ax=ax, label='Defensive Center')
    
    ax.legend(facecolor=t["bg"], edgecolor=t["line"], labelcolor=t["text"], loc='lower left', prop={'size': 9})
    ax.set_title(f"DEFENSIVE ACTIONS TERRITORY & ENGAGEMENT BLOCKS\n{team_name.upper()} Heatmap Overlay", 
                 color=t["text"], fontname=FONT_NAME, fontsize=13, pad=15)
                 
    return fig


def draw_tactical_board(events: Optional[List[Dict[str, Any]]] = None, 
                       bg_color: str = "#121212", pitch_color: str = "#1e1e1e", 
                       line_color: str = "#444444", text_color: str = "#ffffff") -> io.BytesIO:
    """Fallback tactical vector generator rendering direct shape buffers."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (1200, 800), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw field outline
    draw.rectangle([50, 50, 1150, 750], fill=pitch_color, outline=line_color, width=4)
    draw.line([600, 50, 600, 750], fill=line_color, width=4)
    draw.ellipse([500, 300, 700, 500], outline=line_color, width=4)
    
    # Boxes
    draw.rectangle([50, 200, 215, 600], fill=pitch_color, outline=line_color, width=4)
    draw.rectangle([985, 200, 1150, 600], fill=pitch_color, outline=line_color, width=4)
    
    def P(x, y):
        # Scale 0-100 to canvas boundaries safely
        nx = 50 + (float(x) / 100.0) * 1100
        ny = 50 + (float(y) / 100.0) * 700
        return nx, ny

    def _ev_col(et: str) -> str:
        if et == "pass": return "#00FFCC"
        if et == "shot": return "#FF0055"
        return "#FFFF00"

    def _arrow(p1, p2, color, width=5):
        draw.line([p1[0], p1[1], p2[0], p2[1]], fill=color, width=width)
        # Cap end indicator
        draw.ellipse([p2[0]-4, p2[1]-4, p2[0]+4, p2[1]+4], fill=color)

    def _dot(x, y, color, edge="#FFF", r=10, marker="o"):
        nx, ny = P(x, y)
        box = [nx - r, ny - r, nx + r, ny + r]
        box2 = [nx - r + 2, ny - r + 2, nx + r - 2, ny + r - 2]
        if marker == "x":
            draw.line([nx-r, ny-r, nx+r, ny+r], fill=color, width=4)
            draw.line([nx-r, ny+r, nx+r, ny-r], fill=color, width=4)
        elif marker == "^":
            draw.polygon([nx, ny-r, nx-r, ny+r, nx+r, ny+r], fill=color, outline=edge)
        elif marker == "s":
            draw.rectangle(box, fill=color, outline=edge, width=3)
        elif marker == "D":
            draw.polygon([nx, ny-r, nx+r, ny, nx, ny+r, nx-r, ny], fill=color, outline=edge, width=5)
        else:
            draw.ellipse(box2, fill=color, outline=edge, width=3)

    for ev in (events or []):
        try:
            et   = str(ev.get("event_type", "")).lower()
            col  = str(ev.get("start_color",  _ev_col(et)) or _ev_col(et))
            edge = str(ev.get("start_edge",   "#FFF") or "#FFF")
            mk   = ev.get("start_marker", "o")
            sz   = int(float(ev.get("start_size", 9) or 9))
            ac   = str(ev.get("arrow_color", col) or col)
            ex   = float(ev.get("x", 0))
            ey   = float(ev.get("y", 0))
            ex2  = ev.get("x2")
            ey2  = ev.get("y2")
            if et in ["pass", "carry", "dribble", "cross"]:
                try:
                    if ex2 is not None and ey2 is not None:
                        if not (isinstance(ex2, float) and math.isnan(ex2)) and \
                           not (isinstance(ey2, float) and math.isnan(ey2)):
                            _arrow(P(ex, ey), P(float(ex2), float(ey2)), ac, width=5)
                except Exception:
                    pass
            _dot(ex, ey, col, edge=edge, r=max(4, sz), marker=mk)
        except Exception:
            pass

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
