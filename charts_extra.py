"""
charts_extra.py  —  Fully self-contained extra chart functions.
Only imports PUBLIC names from charts.py (THEMES, make_pitch).
No private underscore functions imported from charts.py.
"""
from __future__ import annotations
import math
import io
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D
from mplsoccer import Pitch, VerticalPitch

# Only public names from charts
try:
    from charts import THEMES, make_pitch
except Exception:
    # Fallbacks if charts isn't available or partially-initialized (prevents circular import failures)
    THEMES = {}
    def make_pitch(**kw):
        return Pitch()

# ────────────────────────────────────────────────────────────────��[...]
# LOCAL COPIES of helpers (avoids importing private _ functions from charts.py)
# ────────────────────────────────────────────────────────────────��[...]
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
    return xs.map(lambda v: True if v in {"yes","y","true","t","1","نعم"} else False).astype(bool)


def _add_leg(ax, handles, theme: dict, loc: str = "lower center"):
    if not handles:
        return
    if "upper" in loc:
        anchor = (0.5, 0.98)
    elif "lower" in loc:
        anchor = (0.5, 0.02)
    else:
        anchor = None
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor,
                    ncol=min(4, len(handles)), frameon=False, fontsize=9)
    leg.set_in_layout(True)
    for t in leg.get_texts():
        t.set_color(theme.get("text", "white"))


def _set_pitch_bounds(ax, pitch_mode: str = "rect", pitch_width: float = 68.0, vertical_pitch: bool = False):
    y_max = pitch_width if pitch_mode == "rect" else 100.0
    if vertical_pitch:
        ax.set_xlim(-2, y_max + 2)
        ax.set_ylim(-2, 102)
    else:
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, y_max + 2)


# ────────────────────────────────────────────────────────────────��[...]
# EXTRA THEMES
# ────────────────────────────────────────────────────────────────��[...]
EXTRA_THEMES: Dict[str, dict] = {
    "Opta Analyst Light": {
        "bg": "#F3F3F4", "panel": "#ECEDEF", "pitch": "#F3F3F4",
        "pitch_lines": "#9B9B9B", "pitch_stripe": None,
        "text": "#151326", "muted": "#77727F", "lines": "#D4D4D7", "goal": "#4A4A4A",
    },
    "Opta Analyst Pink": {
        "bg": "#F4F2F4", "panel": "#EEE9EE", "pitch": "#F4F2F4",
        "pitch_lines": "#9B9B9B", "pitch_stripe": None,
        "text": "#171329", "muted": "#807985", "lines": "#D7D2D7", "goal": "#4A4A4A",
    },
    "The Athletic FC Cream": {
        "bg": "#F7F3EA", "panel": "#EFE8DB", "pitch": "#376B49",
        "pitch_lines": "#FDFBF4", "pitch_stripe": "#3F7551",
        "text": "#161616", "muted": "#6F6A61", "lines": "#CCC1AD", "goal": "#222222",
    },
    "The Athletic FC Paper": {
        "bg": "#FBFAF6", "panel": "#F0EEE6", "pitch": "#FBFAF6",
        "pitch_lines": "#8E8E88", "pitch_stripe": None,
        "text": "#222222", "muted": "#62615C", "lines": "#D9D5C9", "goal": "#2B2B2B",
    },
    "Opta Light": {
        "bg": "#F0F0F0", "panel": "#E8E8E8", "pitch": "#F0F0F0",
        "pitch_lines": "#888888", "pitch_stripe": None,
        "text": "#1A1A2E", "muted": "#666666", "lines": "#CCCCCC", "goal": "#444444",
    },
    "Athletic FC Dark": {
        "bg": "#0A0A0A", "panel": "#111111", "pitch": "#0D2818",
        "pitch_lines": "#CCCCCC", "pitch_stripe": "#0F2E1A",
        "text": "#F5F5F0", "muted": "#999999", "lines": "#333333", "goal": "#DDDDDD",
    },
    "Athletic FC Light": {
        "bg": "#F4F1E8", "panel": "#EDE9DC", "pitch": "#4A7C59",
        "pitch_lines": "#FFFFFF", "pitch_stripe": None,
        "text": "#1A1A1A", "muted": "#666666", "lines": "#CCBFA0", "goal": "#333333",
    },
    "Whoscored Dark": {
        "bg": "#1C1C2E", "panel": "#252540", "pitch": "#1A3A2A",
        "pitch_lines": "#E0E0E0", "pitch_stripe": None,
        "text": "#FFFFFF", "muted": "#A0A0C0", "lines": "#303060", "goal": "#FFFFFF",
    },
    "Statsbomb Light": {
        "bg": "#FAFAFA", "panel": "#F0F0F0", "pitch": "#68BB59",
        "pitch_lines": "#FFFFFF", "pitch_stripe": None,
        "text": "#111111", "muted": "#555555", "lines": "#DDDDDD", "goal": "#222222",
    },
    "Night Blue": {
        "bg": "#060D1F", "panel": "#0A1628", "pitch": "#0F3460",
        "pitch_lines": "#E8F4FD", "pitch_stripe": None,
        "text": "#E8F4FD", "muted": "#7BA7C2", "lines": "#1A2D50", "goal": "#E8F4FD",
    },
    "Broadcast Green": {
        "bg": "#0A1A0A", "panel": "#111C11", "pitch": "#1A4A1A",
        "pitch_lines": "#FFFFFF", "pitch_stripe": "#1E501E",
        "text": "#FFFFFF", "muted": "#88BB88", "lines": "#224422", "goal": "#FFFFFF",
    },
}


def register_extra_themes():
    """Add extra themes into the global THEMES dict from charts.py."""
    for k, v in EXTRA_THEMES.items():
        if k not in THEMES:
            THEMES[k] = v


register_extra_themes()


# ────────────────────────────────────────────────────────────────�[...]
# IMAGE OVERLAY
# ────────────────────────────────────────────────────────────────�[...]
def overlay_image_on_fig(fig, img_obj, x=0.02, y=0.88, w=0.10, h=0.10,
                          circle_crop=False, border_color="white", border_lw=0.0):
    """Paste a PIL Image onto any figure at figure-fraction position."""
    if img_obj is None:
        return
    try:
        img = img_obj.convert("RGBA")
        arr = np.asarray(img)
        ax_img = fig.add_axes([x, y, w, h], zorder=50)
        ax_img.imshow(arr)
        ax_img.axis("off")
        ax_img.set_facecolor("none")
        if circle_crop:
            circ = plt.Circle((0.5, 0.5), 0.5, transform=ax_img.transAxes,
                               facecolor="none", edgecolor=border_color, linewidth=border_lw)
            ax_img.add_patch(circ)
            ax_img.set_clip_path(circ)
        if border_lw > 0 and not circle_crop:
            for sp in ax_img.spines.values():
                sp.set_visible(True)
                sp.set_color(border_color)
                sp.set_linewidth(border_lw)
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────�[...]
# 1. GOAL LOCATION MAP  (Opta Analyst style — Image 1 reference)
# ────────────────────────────────────────────────────────────────�[...]
def goal_location_map(
    df: pd.DataFrame,
    title: str = "Goal Location Map",
    player_name: str = "",
    subtitle: str = "",
    stat_labels: Optional[List[Tuple[str, str]]] = None,
    theme_name: str = "Opta Light",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    goal_color: str = "#C8102E",
    goal_edge: str = "#8B0000",
    dot_size: int = 160,
    penalty_label_col: Optional[str] = None,
    logo_img=None,
    logo_x: float = 0.72, logo_y: float = 0.88,
    logo_w: float = 0.14, logo_h: float = 0.10,
    player_img=None,
    player_img_x: float = 0.02, player_img_y: float = 0.88,
    player_img_w: float = 0.10, player_img_h: float = 0.10,
    show_pitch_half_only: bool = True,
    attack_direction: str = "ltr",
):
    """
    Opta Analyst-style goal location map.
    Required columns: x, y, outcome
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme.get("bg", "#F0F0F0")
    text_col = theme.get("text", "#1A1A1A")
    muted_col = theme.get("muted", "#666666")
    line_col = theme.get("pitch_lines", "#888888")

    df = df.copy()
    goals = df[df["outcome"].astype(str).str.lower() == "goal"].copy()

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(bg)

    ax_pitch = fig.add_axes([0.02, 0.06, 0.60, 0.78])
    ax_stats  = fig.add_axes([0.64, 0.06, 0.34, 0.78])

    ax_pitch.set_facecolor(bg)
    ax_pitch.set_xlim(-2, 104)
    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax_pitch.set_ylim(-2, y_max + 2)
    ax_pitch.axis("off")

    lc = line_col
    lw = 2.0
    mid = y_max / 2.0

    def _rect(x0, y0, x1, y1):
        ax_pitch.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                      color=lc, lw=lw, solid_capstyle="round")

    x_start = 50.0 if show_pitch_half_only else 0.0
    _rect(x_start, 0, 100, y_max)

    pa_w = y_max * 40.32 / 68.0
    pa_l = 16.5 / 105.0 * 100.0
    _rect(100 - pa_l, mid - pa_w / 2, 100, mid + pa_w / 2)

    sa_w = y_max * 18.32 / 68.0
    sa_l = 5.5 / 105.0 * 100.0
    _rect(100 - sa_l, mid - sa_w / 2, 100, mid + sa_w / 2)

    goal_w = y_max * 7.32 / 68.0
    ax_pitch.plot([100, 103], [mid - goal_w / 2, mid - goal_w / 2], color=lc, lw=lw)
    ax_pitch.plot([100, 103], [mid + goal_w / 2, mid + goal_w / 2], color=lc, lw=lw)
    ax_pitch.plot([103, 103], [mid - goal_w / 2, mid + goal_w / 2], color=lc, lw=lw)

    pen_x = 100 - 11.0 / 105.0 * 100.0
    ax_pitch.plot(pen_x, mid, "o", color=lc, ms=3)

    # Penalty arc
    arc_cx = 100 - pa_l
    arc_rx = 9.15 / 105.0 * 100.0
    arc_ry = 9.15 / 68.0 * y_max
    theta_arc = np.linspace(np.pi * 0.62, np.pi * 1.38, 80)
    arc_xs = arc_cx + arc_rx * np.cos(theta_arc)
    arc_ys = mid + arc_ry * np.sin(theta_arc)
    outside = arc_xs <= (100 - pa_l + 0.5)
    ax_pitch.plot(arc_xs[outside], arc_ys[outside], color=lc, lw=lw)

    if not show_pitch_half_only:
        ax_pitch.plot([50, 50], [0, y_max], color=lc, lw=lw)

    if not goals.empty:
        ax_pitch.scatter(goals["x"], goals["y"], s=dot_size,
                         color=goal_color, edgecolors=goal_edge,
                         linewidth=1.2, zorder=6, alpha=0.92)
        if penalty_label_col and penalty_label_col in goals.columns:
            for _, r in goals.iterrows():
                try:
                    val = int(float(r.get(penalty_label_col, 0)))
                    if val > 0:
                        ax_pitch.text(float(r["x"]), float(r["y"]), str(val),
                                      ha="center", va="center", fontsize=7,
                                      color="white", weight="bold", zorder=7)
                except Exception:
                    pass

    # Stats panel
    ax_stats.set_facecolor(bg)
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    if stat_labels:
        y_pos = 0.90
        for i, (val, label) in enumerate(stat_labels):
            ax_stats.text(0.05, y_pos, str(val), fontsize=28, weight="900",
                          color=text_col, va="top", transform=ax_stats.transAxes)
            ax_stats.text(0.05, y_pos - 0.08, str(label), fontsize=12,
                          color=muted_col, va="top", transform=ax_stats.transAxes)
            if i < len(stat_labels) - 1:
                y_sep = y_pos - 0.14
                ax_stats.plot([0.0, 0.85], [y_sep, y_sep],
                              color=theme.get("lines", "#CCCCCC"), lw=1.0,
                              transform=ax_stats.transAxes)
            y_pos -= 0.22

    fig.text(0.02, 0.955, player_name or title, fontsize=22, weight="900",
             color=text_col, va="top")
    fig.text(0.02, 0.912, subtitle, fontsize=11, color=muted_col, va="top")

    overlay_image_on_fig(fig, logo_img,   x=logo_x,      y=logo_y,      w=logo_w,      h=logo_h)
    overlay_image_on_fig(fig, player_img, x=player_img_x, y=player_img_y,
                         w=player_img_w, h=player_img_h, circle_crop=True)
    return fig
