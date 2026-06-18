"""
charts_pro.py — Professional Football Analytics Layout Engine
Shared components: title blocks, attack direction, legend, stat blocks,
logo/image overlays, and two flagship charts:
  - athletic_shot_map_pro()   → The Athletic style shot map
  - opta_pass_map_pro()       → Opta Analyst style pass map
"""
from __future__ import annotations
import io, math
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import to_rgba
from mplsoccer import Pitch, VerticalPitch

# ── Import from existing modules ──────────────────────────────────────────────
try:
    from charts import THEMES, make_pitch
except Exception:
    THEMES = {}
    def make_pitch(**kw): return Pitch()

# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED THEME REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
PRO_THEMES: Dict[str, dict] = {
    # ── Existing ──────────────────────────────────────────────────────────────
    "The Athletic Dark": {
        "bg": "#0E1117", "panel": "#111827", "pitch": "#1f5f3b",
        "text": "#FFFFFF", "muted": "#A0A7B4", "lines": "#2A3240",
        "goal": "#E6E6E6", "pitch_lines": "#E6E6E6",
        "accent": "#C8102E", "accent2": "#00C2FF",
    },
    "Opta Analyst Light": {
        "bg": "#F3F3F4", "panel": "#ECEDEF", "pitch": "#F3F3F4",
        "pitch_lines": "#9B9B9B", "text": "#151326", "muted": "#77727F",
        "lines": "#D4D4D7", "goal": "#4A4A4A",
        "accent": "#C8102E", "accent2": "#7B2D8B",
    },
    "Opta Analyst Dark": {
        "bg": "#0E1117", "panel": "#141A22", "pitch": "#0E1117",
        "pitch_lines": "#4A4A5A", "text": "#FFFFFF", "muted": "#9B9BB0",
        "lines": "#2A2A3A", "goal": "#CCCCCC",
        "accent": "#C8102E", "accent2": "#7B2D8B",
    },
    "The Athletic Light": {
        "bg": "#F7F3EA", "panel": "#EFE8DB", "pitch": "#F7F3EA",
        "pitch_lines": "#8E8E88", "text": "#161616", "muted": "#6F6A61",
        "lines": "#CCC1AD", "goal": "#2B2B2B",
        "accent": "#C8102E", "accent2": "#1A1A1A",
    },
    "StatsBomb": {
        "bg": "#FAFAFA", "panel": "#F0F0F0", "pitch": "#68BB59",
        "pitch_lines": "#FFFFFF", "text": "#111111", "muted": "#555555",
        "lines": "#DDDDDD", "goal": "#222222",
        "accent": "#E03A3E", "accent2": "#1A78CF",
    },
    "FotMob Dark": {
        "bg": "#10121C", "panel": "#181B2B", "pitch": "#1A3A2A",
        "pitch_lines": "#CCDDCC", "text": "#FFFFFF", "muted": "#9090A0",
        "lines": "#252840", "goal": "#DDDDDD",
        "accent": "#FF7F00", "accent2": "#00C0FF",
    },
    "Wyscout": {
        "bg": "#1C2232", "panel": "#222A3C", "pitch": "#1A3A2A",
        "pitch_lines": "#E0E8E0", "text": "#FFFFFF", "muted": "#8090A0",
        "lines": "#2A3448", "goal": "#DDDDDD",
        "accent": "#00A8E8", "accent2": "#FF6B35",
    },
    "SofaScore": {
        "bg": "#FFFFFF", "panel": "#F5F5F5", "pitch": "#4CAF50",
        "pitch_lines": "#FFFFFF", "text": "#1A1A1A", "muted": "#666666",
        "lines": "#E0E0E0", "goal": "#333333",
        "accent": "#FF5722", "accent2": "#2196F3",
    },
    "Night Blue": {
        "bg": "#060D1F", "panel": "#0A1628", "pitch": "#0F3460",
        "pitch_lines": "#E8F4FD", "text": "#E8F4FD", "muted": "#7BA7C2",
        "lines": "#1A2D50", "goal": "#E8F4FD",
        "accent": "#00D4FF", "accent2": "#FF6B6B",
    },
    "Broadcast Green": {
        "bg": "#0A1A0A", "panel": "#111C11", "pitch": "#1A4A1A",
        "pitch_lines": "#FFFFFF", "text": "#FFFFFF", "muted": "#88BB88",
        "lines": "#224422", "goal": "#FFFFFF",
        "accent": "#FFD700", "accent2": "#00FF88",
    },
}

# Merge into global THEMES
for _k, _v in PRO_THEMES.items():
    if _k not in THEMES:
        THEMES[_k] = _v

ALL_PRO_THEME_NAMES = list(PRO_THEMES.keys())


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT CONFIG  — dataclass-style dict for a chart
# ─────────────────────────────────────────────────────────────────────────────
def default_layout_cfg() -> dict:
    return dict(
        # Title block
        title="", subtitle="", competition="", season="",
        team_name="", player_name="", match_info="", analyst_credit="",
        footer_note="",
        # Title styling
        title_fontsize=22, title_weight="bold", title_color=None,
        subtitle_fontsize=13, subtitle_color=None,
        meta_fontsize=11, meta_color=None,
        # Attack direction
        show_attack_dir=True, attack_dir="ltr",
        attack_dir_color="#888888", attack_dir_size=12,
        attack_dir_label="Attacking Direction",
        attack_dir_pos="bottom",        # "bottom" | "top" | "left" | "right"
        attack_dir_show_label=True,
        # Legend
        show_legend=True, legend_pos="lower center",
        legend_fontsize=9, legend_markerscale=1.0,
        legend_title="", legend_ncol=None,
        legend_frame=False,
        # Logo / Player image
        logo_img=None, logo_x=0.84, logo_y=0.88, logo_w=0.12, logo_h=0.10,
        logo_circle=False,
        player_img=None, player_x=0.02, player_y=0.88,
        player_w=0.10, player_h=0.10, player_circle=True,
        # Stat blocks
        stat_blocks=[],           # list of StatBlock dicts
        # Theme
        theme_name="The Athletic Dark",
    )


def stat_block(value: str, label: str, color: str = None,
               fontsize_val: int = 28, fontsize_lbl: int = 11,
               bold_val: bool = True) -> dict:
    return dict(value=value, label=label, color=color,
                fontsize_val=fontsize_val, fontsize_lbl=fontsize_lbl,
                bold_val=bold_val)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE DRAWING PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def draw_logo(fig, img, x: float, y: float, w: float, h: float,
              circle_crop: bool = False, border_lw: float = 0.0,
              border_color: str = "white"):
    """Paste a PIL Image onto any figure at figure-fraction coordinates."""
    if img is None:
        return
    try:
        arr = np.asarray(img.convert("RGBA"))
        ax_img = fig.add_axes([x, y, w, h], zorder=50)
        ax_img.imshow(arr)
        ax_img.axis("off")
        ax_img.set_facecolor("none")
        if circle_crop:
            circ = plt.Circle((0.5, 0.5), 0.5, transform=ax_img.transAxes,
                               facecolor="none", edgecolor=border_color, linewidth=border_lw)
            ax_img.add_patch(circ)
            ax_img.set_clip_path(circ)
        elif border_lw > 0:
            for sp in ax_img.spines.values():
                sp.set_visible(True); sp.set_color(border_color); sp.set_linewidth(border_lw)
    except Exception:
        pass


def draw_title_block(fig, cfg: dict, theme: dict,
                     title_y: float = 0.965, line_height: float = 0.034):
    """
    Draw a multi-line title block:
      player_name / title  (large bold)
      subtitle / competition + season  (medium)
      match_info / team_name  (small muted)
    Returns the y position consumed (for spacing below).
    """
    text_col = cfg.get("title_color") or theme.get("text", "#111111")
    muted_col = cfg.get("subtitle_color") or theme.get("muted", "#666666")
    meta_col = cfg.get("meta_color") or theme.get("muted", "#666666")

    # Player name or title
    main = cfg.get("player_name") or cfg.get("title") or ""
    if main:
        fig.text(0.04, title_y, main,
                 fontsize=cfg.get("title_fontsize", 22),
                 weight=cfg.get("title_weight", "bold"),
                 color=text_col, va="top")
        title_y -= line_height * 0.9

    # Subtitle: competition | match
    sub_parts = [p for p in [
        cfg.get("subtitle", ""),
        cfg.get("competition", ""),
        cfg.get("season", ""),
        cfg.get("match_info", ""),
    ] if p.strip()]
    sub_line = "  |  ".join(sub_parts[:2])
    if sub_line:
        fig.text(0.04, title_y, sub_line,
                 fontsize=cfg.get("subtitle_fontsize", 13),
                 color=muted_col, va="top")
        title_y -= line_height * 0.75

    # Team / analyst
    meta_parts = [p for p in [cfg.get("team_name", ""), cfg.get("analyst_credit", "")] if p.strip()]
    meta_line = "  ·  ".join(meta_parts)
    if meta_line:
        fig.text(0.04, title_y, meta_line,
                 fontsize=cfg.get("meta_fontsize", 11),
                 color=meta_col, va="top")
        title_y -= line_height * 0.7

    return title_y


def draw_footer(fig, cfg: dict, theme: dict):
    """Draw footer note at bottom."""
    note = cfg.get("footer_note", "").strip()
    if not note:
        return
    col = theme.get("muted", "#888888")
    fig.text(0.04, 0.018, note, fontsize=8.5, color=col, va="bottom")


def draw_attack_direction(ax, cfg: dict, theme: dict,
                          pitch_mode: str = "rect", pitch_width: float = 68.0,
                          vertical: bool = False):
    """
    Draw attacking direction arrow + label onto a pitch axis.
    pos can be 'bottom', 'top', 'left', 'right'.
    """
    if not cfg.get("show_attack_dir", True):
        return

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    col = cfg.get("attack_dir_color", "#888888")
    lbl = cfg.get("attack_dir_label", "Attacking Direction") if cfg.get("attack_dir_show_label", True) else ""
    fsize = cfg.get("attack_dir_size", 11)
    pos = cfg.get("attack_dir_pos", "bottom")
    direction = cfg.get("attack_dir", "ltr")  # ltr or rtl

    if vertical:
        # Arrow at left side pointing up/down
        ax_xmin, ax_xmax = -2, y_max + 2
        ax_ymin, ax_ymax = -2, 102
        mid_x = ax_xmin - 4
        y0, y1 = (20, 80) if direction == "ltr" else (80, 20)
        ax.annotate("", xy=(mid_x, y1), xytext=(mid_x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
        if lbl:
            ax.text(mid_x - 2, (y0 + y1) / 2, lbl, color=col, fontsize=fsize - 1,
                    rotation=90, ha="right", va="center")
    else:
        # Horizontal arrow below pitch
        ax_ymin = -2
        arrow_y = ax_ymin - 3.5
        x0, x1 = (20, 80) if direction == "ltr" else (80, 20)
        ax.annotate("", xy=(x1, arrow_y), xytext=(x0, arrow_y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0),
                    annotation_clip=False)
        if lbl:
            ax.text((x0 + x1) / 2, arrow_y - 2.5, lbl,
                    color=col, fontsize=fsize, ha="center", va="top")


def draw_opta_attack_arrows(ax, theme: dict, y_max: float = 68.0,
                             direction: str = "ltr",
                             label: str = "Attacking Direction",
                             show_label: bool = True,
                             color: str = "#888888",
                             y_pos: float = None):
    """
    Opta-style attacking direction indicator — multiple small triangles in a row.
    Placed below the pitch.
    """
    col = color or theme.get("muted", "#888888")
    base_y = (y_pos or -5.5)
    n = 5
    xs = np.linspace(44, 56, n)
    for i, x in enumerate(xs):
        alpha = 0.3 + 0.14 * i
        ax.annotate("", xy=(x + 1.4, base_y), xytext=(x, base_y),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5, alpha=alpha),
                    annotation_clip=False)
    if show_label:
        ax.text(50, base_y - 2.8, label, color=col, fontsize=9.5,
                ha="center", va="top")


def draw_stat_blocks_bottom(fig, blocks: List[dict], theme: dict,
                             y: float = 0.055, spacing: float = 0.18):
    """
    Draw stat blocks in a horizontal row at the bottom of the figure.
    Each block: {value, label, color, ...}
    """
    if not blocks:
        return
    text_col = theme.get("text", "#111111")
    muted_col = theme.get("muted", "#888888")
    n = len(blocks)
    xs = np.linspace(0.05, 0.95, n)

    for i, blk in enumerate(blocks):
        val = str(blk.get("value", ""))
        lbl = str(blk.get("label", ""))
        col = blk.get("color") or theme.get("accent", "#C8102E")
        fv = blk.get("fontsize_val", 26)
        fl = blk.get("fontsize_lbl", 10)
        x = xs[i]
        fig.text(x, y + 0.025, val, ha="center", va="bottom",
                 fontsize=fv, weight="900", color=col)
        fig.text(x, y, lbl, ha="center", va="top",
                 fontsize=fl, color=muted_col)


def draw_stat_blocks_right(ax_stats, blocks: List[dict], theme: dict):
    """
    Draw stat blocks in a vertical column in a dedicated axis panel.
    The Athletic style: label (bold, dark) above value (bold, accent color).
    """
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1); ax_stats.axis("off")
    text_col = theme.get("text", "#111111")
    muted_col = theme.get("muted", "#888888")
    n = max(1, len(blocks))
    y_start = 0.92
    step = min(0.22, 0.88 / n)

    for i, blk in enumerate(blocks):
        val = str(blk.get("value", ""))
        lbl = str(blk.get("label", ""))
        col = blk.get("color") or theme.get("accent", "#C8102E")
        fv = blk.get("fontsize_val", 28)
        fl = blk.get("fontsize_lbl", 11)
        y = y_start - i * step
        ax_stats.text(0.05, y, lbl, ha="left", va="top",
                      fontsize=fl, weight="bold", color=text_col,
                      transform=ax_stats.transAxes)
        ax_stats.text(0.05, y - step * 0.42, val, ha="left", va="top",
                      fontsize=fv, weight="900", color=col,
                      transform=ax_stats.transAxes)


def draw_custom_legend(ax, handles: list, cfg: dict, theme: dict):
    """Draw a fully configurable legend onto an axis."""
    if not cfg.get("show_legend", True) or not handles:
        return
    active_labels = None  # all
    title = cfg.get("legend_title", "") or None
    pos = cfg.get("legend_pos", "lower center")
    fs = cfg.get("legend_fontsize", 9)
    ms = cfg.get("legend_markerscale", 1.0)
    ncol = cfg.get("legend_ncol") or min(4, len(handles))
    frame = cfg.get("legend_frame", False)

    bbox = None
    if "upper" in pos and "center" in pos:
        bbox = (0.5, 0.98)
    elif "lower" in pos and "center" in pos:
        bbox = (0.5, 0.02)

    leg = ax.legend(
        handles=handles, loc=pos, fontsize=fs,
        markerscale=ms, ncol=ncol, frameon=frame,
        title=title, bbox_to_anchor=bbox,
    )
    leg.set_in_layout(True)
    text_col = theme.get("text", "white")
    for t in leg.get_texts():
        t.set_color(text_col)
    if leg.get_title():
        leg.get_title().set_color(theme.get("muted", "#888888"))
    if frame:
        leg.get_frame().set_facecolor(theme.get("panel", "#111827"))
        leg.get_frame().set_edgecolor(theme.get("lines", "#333333"))


# ─────────────────────────────────────────────────────────────────────────────
# HALF-PITCH DRAWING HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _draw_half_pitch_lines(ax, y_max: float, line_color: str, lw: float = 1.8,
                            show_full: bool = False, attack_right: bool = True):
    """Draw pitch lines for a half-pitch (or full) on a plain axis."""
    mid = y_max / 2.0
    x_start = 50.0 if not show_full else 0.0

    def rect(x0, y0, x1, y1):
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                color=line_color, lw=lw, solid_capstyle="round", zorder=3)

    rect(x_start, 0, 100, y_max)
    if not show_full:
        # Centre arc
        arc_t = np.linspace(-np.pi / 2, np.pi / 2, 60)
        ax.plot(50 + 9.15 * np.cos(arc_t), mid + (9.15 / 68.0 * y_max) * np.sin(arc_t),
                color=line_color, lw=lw, zorder=3)
    else:
        ax.plot([50, 50], [0, y_max], color=line_color, lw=lw, zorder=3)
        r_x = 9.15 / 105.0 * 100.0
        r_y = 9.15 / 68.0 * y_max
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(50 + r_x * np.cos(theta), mid + r_y * np.sin(theta),
                color=line_color, lw=lw, zorder=3)
        ax.plot(50, mid, "o", color=line_color, ms=3, zorder=3)

    pa_w = y_max * 40.32 / 68.0; pa_l = 16.5 / 105.0 * 100.0
    sa_w = y_max * 18.32 / 68.0; sa_l = 5.5 / 105.0 * 100.0
    goal_w = y_max * 7.32 / 68.0

    # Attacking end
    rect(100 - pa_l, mid - pa_w / 2, 100, mid + pa_w / 2)
    rect(100 - sa_l, mid - sa_w / 2, 100, mid + sa_w / 2)
    ax.plot([100, 103, 103, 100], [mid - goal_w / 2, mid - goal_w / 2,
                                   mid + goal_w / 2, mid + goal_w / 2],
            color=line_color, lw=lw, zorder=3)
    pen_x = 100 - 11.0 / 105.0 * 100.0
    ax.plot(pen_x, mid, "o", color=line_color, ms=3, zorder=3)
    # Penalty arc
    arc_cx = 100 - pa_l
    arc_rx = 9.15 / 105.0 * 100.0
    arc_ry = 9.15 / 68.0 * y_max
    theta = np.linspace(np.pi * 0.62, np.pi * 1.38, 80)
    axs = arc_cx + arc_rx * np.cos(theta)
    ays = mid + arc_ry * np.sin(theta)
    mask = axs <= (100 - pa_l + 0.5)
    ax.plot(axs[mask], ays[mask], color=line_color, lw=lw, zorder=3)

    if show_full:
        # Defending end
        rect(0, mid - pa_w / 2, pa_l, mid + pa_w / 2)
        rect(0, mid - sa_w / 2, sa_l, mid + sa_w / 2)
        ax.plot([0, -3, -3, 0], [mid - goal_w / 2, mid - goal_w / 2,
                                  mid + goal_w / 2, mid + goal_w / 2],
                color=line_color, lw=lw, zorder=3)


def _draw_vertical_half_pitch_lines(ax, pitch_w: float, line_color: str, lw: float = 1.8):
    """Draw The Athletic style vertical attacking half with the goal at the top."""
    mid = pitch_w / 2.0

    def line(xs, ys):
        ax.plot(xs, ys, color=line_color, lw=lw, solid_capstyle="round", zorder=3)

    def rect(x0, y0, x1, y1):
        line([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])

    rect(0, 50, pitch_w, 100)

    # Centre circle arc at halfway.
    r_x = 9.15 / 68.0 * pitch_w
    r_y = 9.15 / 105.0 * 100.0
    theta = np.linspace(0, np.pi, 80)
    line(mid + r_x * np.cos(theta), 50 + r_y * np.sin(theta))

    pa_w = pitch_w * 40.32 / 68.0
    sa_w = pitch_w * 18.32 / 68.0
    pa_l = 16.5 / 105.0 * 100.0
    sa_l = 5.5 / 105.0 * 100.0
    goal_w = pitch_w * 7.32 / 68.0

    rect(mid - pa_w / 2, 100 - pa_l, mid + pa_w / 2, 100)
    rect(mid - sa_w / 2, 100 - sa_l, mid + sa_w / 2, 100)
    rect(mid - goal_w / 2, 100, mid + goal_w / 2, 103)
    ax.plot(mid, 100 - 11.0 / 105.0 * 100.0, "o", color=line_color, ms=3, zorder=3)

    arc_y = 100 - pa_l
    theta = np.linspace(np.pi * 1.10, np.pi * 1.90, 80)
    xs = mid + r_x * np.cos(theta)
    ys = arc_y + r_y * np.sin(theta)
    mask = ys <= arc_y + 0.5
    line(xs[mask], ys[mask])


# ─────────────────────────────────────────────────────────────────────────────
# 1.  THE ATHLETIC SHOT MAP  (reference image 2)
# ─────────────────────────────────────────────────────────────────────────────

def athletic_shot_map_pro(
    df: pd.DataFrame,
    cfg: dict = None,
    # Shot styling
    goal_color: str = "#C8102E",
    no_goal_color: str = "#FFFFFF",
    goal_edge: str = "#C8102E",
    no_goal_edge: str = "#555555",
    # xG size range
    min_dot_size: int = 30,
    max_dot_size: int = 700,
    # Layout
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    # Body part columns (optional)
    body_part_col: str = None,
    # Minute column (for stats)
    minute_col: str = None,
    # Stats overrides
    total_shots_override: str = None,
    right_foot_override: str = None,
    left_foot_override: str = None,
    head_override: str = None,
    other_override: str = None,
    goals_override: str = None,
    xg_override: str = None,
    avg_dist_override: str = None,
    minutes_played: float = None,
):
    """
    Exact replica of The Athletic non-penalty shot map.
    Layout: half-pitch (vertical orientation) + right stats panel + bottom stats row.
    """
    cfg = cfg or default_layout_cfg()
    theme_name = cfg.get("theme_name", "The Athletic Light")
    theme = THEMES.get(theme_name, THEMES.get("The Athletic Light", {}))
    if not theme:
        theme = PRO_THEMES.get("The Athletic Light", PRO_THEMES["The Athletic Dark"])

    bg = theme.get("bg", "#F7F3EA")
    text_col = theme.get("text", "#161616")
    muted_col = theme.get("muted", "#6F6A61")
    line_col = theme.get("pitch_lines", "#8E8E88")
    accent = theme.get("accent", "#C8102E")

    d = df.copy()
    if "event_type" in d.columns:
        shots = d[d["event_type"].astype(str).str.lower().str.strip().str.contains("shot", na=False)].copy()
        if shots.empty:
            shots = d.copy()
    else:
        shots = d.copy()

    shots["xg"] = pd.to_numeric(shots.get("xg", pd.Series(0.1, index=shots.index)), errors="coerce").fillna(0.1)
    shots["outcome"] = shots.get("outcome", pd.Series("", index=shots.index)).astype(str).str.lower()
    for c in ["x", "y"]:
        if c in shots.columns:
            shots[c] = pd.to_numeric(shots[c], errors="coerce")
    shots = shots.dropna(subset=["x", "y"]).copy()

    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    mid = y_max / 2.0

    if not shots.empty:
        if shots["x"].max(skipna=True) <= 1.5:
            shots["x"] = shots["x"] * 100.0
        if shots["y"].max(skipna=True) <= 1.5:
            shots["y"] = shots["y"] * y_max
        elif y_max != 100.0 and shots["y"].max(skipna=True) > y_max + 1:
            shots["y"] = shots["y"] / 100.0 * y_max
        # If the data is from the opposite attacking direction, mirror it so shots
        # appear in the attacking half used by the reference.
        if shots["x"].median(skipna=True) < 50:
            shots["x"] = 100.0 - shots["x"]

    # ── Compute stats ──────────────────────────────────────────────────────
    n_total = len(shots)
    n_goals = int((shots["outcome"] == "goal").sum())
    total_xg = float(shots["xg"].sum())
    xg_per_shot = total_xg / max(1, n_total)

    # Body part breakdown
    if body_part_col and body_part_col in shots.columns:
        bp = shots[body_part_col].astype(str).str.lower()
        n_right = int(bp.str.contains("right").sum())
        n_left  = int(bp.str.contains("left").sum())
        n_head  = int(bp.str.contains("head").sum())
        n_other = n_total - n_right - n_left - n_head
    else:
        n_right = n_total; n_left = 0; n_head = 0; n_other = 0

    # Average distance (metres approx)
    def _dist(x, y):
        gx, gy = 105.0, 34.0
        xm = x / 100.0 * 105.0; ym = y / y_max * 68.0
        return math.sqrt((gx - xm) ** 2 + (gy - ym) ** 2) * 1.094  # m → yards

    shots["_dist_yd"] = shots.apply(lambda r: _dist(float(r["x"]), float(r["y"])), axis=1)
    avg_dist = float(shots["_dist_yd"].mean()) if not shots.empty else 0.0

    # Overrides
    if total_shots_override: n_total = total_shots_override
    if goals_override: n_goals = goals_override
    if right_foot_override: n_right = right_foot_override
    if left_foot_override: n_left = left_foot_override
    if head_override: n_head = head_override
    if other_override: n_other = other_override
    if xg_override: total_xg = float(xg_override)
    if avg_dist_override: avg_dist = float(avg_dist_override)

    xg_per_shot_str = f"{xg_per_shot:.2f}"
    avg_dist_str = f"{avg_dist:.1f}"

    if minutes_played:
        goals_p90 = n_goals / minutes_played * 90.0 if isinstance(n_goals, (int, float)) else "—"
        xg_p90 = total_xg / minutes_played * 90.0 if isinstance(total_xg, (int, float)) else "—"
        goals_lbl = f"{n_goals} ({goals_p90:.2f} p90)" if isinstance(goals_p90, float) else str(n_goals)
        xg_lbl = f"{total_xg:.0f} ({xg_p90:.2f})" if isinstance(xg_p90, float) else f"{total_xg:.2f}"
    else:
        goals_lbl = str(n_goals)
        xg_lbl = f"{total_xg:.2f}"

    # ── Figure layout ──────────────────────────────────────────────────────
    # fig_w = 12, divided: pitch col 0.62, stats col 0.20, right margin 0.18
    fig = plt.figure(figsize=(12, 14))
    fig.patch.set_facecolor(bg)

    # Top title strip
    ax_title = fig.add_axes([0.0, 0.88, 1.0, 0.12])
    ax_title.set_facecolor(bg); ax_title.axis("off")

    # Pitch panel
    ax_pitch = fig.add_axes([0.08, 0.27, 0.60, 0.58])
    ax_pitch.set_facecolor(bg)
    ax_pitch.set_xlim(-4, y_max + 4); ax_pitch.set_ylim(48, 108)
    ax_pitch.set_aspect("equal", adjustable="box")
    ax_pitch.axis("off")

    # Right stats panel
    ax_right = fig.add_axes([0.65, 0.30, 0.32, 0.58])
    ax_right.set_facecolor(bg); ax_right.axis("off")
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)

    # Bottom stats strip
    ax_bottom = fig.add_axes([0.02, 0.07, 0.96, 0.16])
    ax_bottom.set_facecolor(bg); ax_bottom.axis("off")
    ax_bottom.set_xlim(0, 1); ax_bottom.set_ylim(0, 1)

    # ── Title ──────────────────────────────────────────────────────────────
    player_name = cfg.get("player_name") or cfg.get("title", "")
    subtitle = "  |  ".join(p for p in [
        cfg.get("subtitle", ""), cfg.get("competition", ""), cfg.get("season", "")
    ] if p.strip())

    fig.text(0.50, 0.975, player_name, ha="center", va="top",
             fontsize=30, weight="900", color=text_col)
    if subtitle:
        fig.text(0.50, 0.942, subtitle, ha="center", va="top",
                 fontsize=13, color=muted_col)

    # ── Pitch lines ────────────────────────────────────────────────────────
    lw = 1.8
    _draw_vertical_half_pitch_lines(ax_pitch, y_max, line_col, lw=lw)

    # ── Goal/No Goal legend (top of pitch) ─────────────────────────────────
    lgnd_y = 106.0
    ax_pitch.scatter([mid - 8], [lgnd_y], s=200, color=goal_color, edgecolors=goal_edge,
                     linewidth=1.5, zorder=5)
    ax_pitch.text(mid - 5.7, lgnd_y, "Goal", va="center", color=text_col, fontsize=11, weight="bold")
    ax_pitch.scatter([mid + 9], [lgnd_y], s=200, facecolors=no_goal_color,
                     edgecolors=no_goal_edge, linewidth=1.5, zorder=5)
    ax_pitch.text(mid + 11.3, lgnd_y, "No Goal", va="center", color=text_col, fontsize=11, weight="bold")

    # xG size legend
    lgnd_y2 = 102.8
    ax_pitch.text(0.5, lgnd_y2, "Low-quality chance", va="center",
                  color=text_col, fontsize=9, weight="bold")
    for j, xgv in enumerate([0.03, 0.08, 0.15, 0.25, 0.45]):
        xpos = mid - 3 + j * 5.8
        sz = min_dot_size + (xgv / 0.65) * (max_dot_size - min_dot_size)
        ax_pitch.scatter([xpos], [lgnd_y2], s=sz, facecolors=no_goal_color,
                         edgecolors=no_goal_edge, linewidth=1.2, zorder=5)
    ax_pitch.text(y_max - 0.5, lgnd_y2, "High-quality chance", va="center",
                  ha="right", color=text_col, fontsize=9, weight="bold")

    # ── Plot shots ─────────────────────────────────────────────────────────
    for _, row in shots.iterrows():
        outc = str(row.get("outcome", "")).lower()
        is_goal = outc == "goal"
        xgv = float(row.get("xg", 0.1))
        sz = min_dot_size + (xgv / 0.65) * (max_dot_size - min_dot_size)
        sz = max(min_dot_size, min(max_dot_size, sz))
        fc = goal_color if is_goal else no_goal_color
        ec = goal_edge if is_goal else no_goal_edge
        lw_dot = 0 if is_goal else 1.5
        ax_pitch.scatter([float(row["y"])], [float(row["x"])],
                         s=sz, color=fc, edgecolors=ec,
                         linewidth=lw_dot, alpha=0.92, zorder=6)

    # ── Average distance indicator ─────────────────────────────────────────
    if not shots.empty and avg_dist > 0:
        # Convert yards back to pitch units
        avg_dist_m = avg_dist / 1.094
        avg_x = 100 - (avg_dist_m / 105.0 * 100.0)
        avg_x = max(52, min(97, avg_x))
        ax_pitch.scatter([mid], [avg_x], s=180, color=muted_col,
                         edgecolors=muted_col, alpha=0.85, zorder=4)
        ax_pitch.plot([mid, mid], [avg_x, 100], color=muted_col,
                      lw=2, ls="-", alpha=0.6, zorder=3)
        ax_pitch.text(mid - 4.5, avg_x - 1.5,
                      f"Average distance:\n{avg_dist_str} yards",
                      color=muted_col, fontsize=8.5, ha="right", va="top")

        # Attack arrow on left side
        ax_pitch.annotate("", xy=(-2.2, 72),
                          xytext=(-2.2, 60),
                          arrowprops=dict(arrowstyle="-|>", color=muted_col, lw=1.5))
        ax_pitch.text(-3.3, 68, "Attack", color=muted_col,
                      fontsize=8, rotation=90, va="bottom")

    # ── Right stats panel — The Athletic style ─────────────────────────────
    right_stats = [
        ("Total shots", str(n_total)),
        ("Right Foot",  str(n_right)),
        ("Left Foot",   str(n_left)),
        ("Head",        str(n_head)),
        ("Other",       str(n_other)),
    ]
    y_s = 0.95
    step_s = 0.18
    for lbl, val in right_stats:
        ax_right.text(0.0, y_s, lbl, ha="left", va="top", fontsize=13,
                      weight="bold", color=text_col, transform=ax_right.transAxes)
        ax_right.text(0.0, y_s - 0.065, val, ha="left", va="top",
                      fontsize=28, weight="900", color=accent,
                      transform=ax_right.transAxes)
        y_s -= step_s

    # ── Bottom stats — Goals, xG, xG per Shot ─────────────────────────────
    bottom_stats = [
        ("Goals", goals_lbl),
        ("xG", xg_lbl),
        ("xG per Shot", xg_per_shot_str),
    ]
    xs_b = [0.12, 0.40, 0.68]
    ax_bottom.plot([0.01, 0.99], [0.88, 0.88], color=line_col, lw=1.2, alpha=0.5)
    for i, (lbl, val) in enumerate(bottom_stats):
        xb = xs_b[i]
        ax_bottom.text(xb, 0.78, lbl, ha="left", va="top",
                       fontsize=14, weight="bold", color=text_col)
        ax_bottom.text(xb, 0.32, val, ha="left", va="top",
                       fontsize=26, weight="900", color=accent)

    # Logo / player image
    if cfg.get("logo_img"):
        draw_logo(fig, cfg["logo_img"],
                  cfg.get("logo_x", 0.78), cfg.get("logo_y", 0.89),
                  cfg.get("logo_w", 0.14), cfg.get("logo_h", 0.10),
                  circle_crop=cfg.get("logo_circle", False))
    if cfg.get("player_img"):
        draw_logo(fig, cfg["player_img"],
                  cfg.get("player_x", 0.02), cfg.get("player_y", 0.88),
                  cfg.get("player_w", 0.10), cfg.get("player_h", 0.10),
                  circle_crop=True)

    # Footer
    footer = cfg.get("footer_note", "")
    if footer:
        fig.text(0.04, 0.022, footer, fontsize=8.5,
                 color=theme.get("muted", "#888888"), va="bottom")
    # Analyst credit bottom-right
    credit = cfg.get("analyst_credit", "")
    if credit:
        fig.text(0.96, 0.022, credit, fontsize=10, weight="bold",
                 ha="right", va="bottom", color=muted_col)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  OPTA ANALYST PASS MAP  (reference image 1)
# ─────────────────────────────────────────────────────────────────────────────

def opta_pass_map_pro(
    df: pd.DataFrame,
    cfg: dict = None,
    # Colors
    successful_color: str = "#C8102E",
    unsuccessful_color: str = "#AAAAAA",
    # Arrow styling
    arrow_alpha: float = 0.82,
    arrow_width: float = 1.8,
    arrow_head_width: float = 5,
    # Layout
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    # Stats overrides
    successful_override: str = None,
    unsuccessful_override: str = None,
    accuracy_override: str = None,
):
    """
    Exact Opta Analyst style pass map:
    - Light pitch background
    - Red = successful, grey = unsuccessful
    - Bottom stats row: N successful, N unsuccessful, arrows attack dir, total passes, accuracy%
    - Player photo top-left, logo top-right
    - Title: player name + match info
    """
    cfg = cfg or default_layout_cfg()
    theme_name = cfg.get("theme_name", "Opta Analyst Light")
    theme = THEMES.get(theme_name, PRO_THEMES.get("Opta Analyst Light", PRO_THEMES["The Athletic Dark"]))

    bg = theme.get("bg", "#F3F3F4")
    pitch_bg = theme.get("pitch", bg)
    text_col = theme.get("text", "#151326")
    muted_col = theme.get("muted", "#77727F")
    line_col = theme.get("pitch_lines", "#9B9B9B")

    d = df.copy()
    if "event_type" in d.columns:
        passes = d[d["event_type"].astype(str).str.lower() == "pass"].copy()
    else:
        passes = d.copy()

    passes["outcome"] = passes.get("outcome", pd.Series("successful", index=passes.index)).astype(str).str.lower()
    for c in ["x", "y", "x2", "y2"]:
        if c in passes.columns:
            passes[c] = pd.to_numeric(passes[c], errors="coerce")

    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)

    # Stats
    succ = passes[passes["outcome"].isin(["successful", "key pass", "assist"])]
    unsucc = passes[passes["outcome"] == "unsuccessful"]
    n_succ = len(succ)
    n_unsucc = len(unsucc)
    n_total = n_succ + n_unsucc
    accuracy_pct = round(n_succ / max(1, n_total) * 100, 1)

    if successful_override:   n_succ = successful_override
    if unsuccessful_override: n_unsucc = unsuccessful_override
    if accuracy_override:     accuracy_pct = accuracy_override

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 12))
    fig.patch.set_facecolor(bg)
    if cfg.get("player_img"):
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax_bg.imshow(np.asarray(cfg["player_img"].convert("RGB")), aspect="auto")
        ax_bg.axis("off")
    else:
        ax_bg = fig.add_axes([0, 0, 1, 1], zorder=0)
        ax_bg.set_facecolor(bg)
        ax_bg.axis("off")

    card_bg = "#F5F5F6"
    ax_card = fig.add_axes([0.055, 0.065, 0.89, 0.57], zorder=1)
    ax_card.set_facecolor(card_bg)
    for sp in ax_card.spines.values():
        sp.set_visible(False)
    ax_card.set_xticks([]); ax_card.set_yticks([])

    # Header strip — white/light card with subtle border
    ax_hdr = fig.add_axes([0.095, 0.545, 0.81, 0.07], zorder=2)
    ax_hdr.set_facecolor(card_bg); ax_hdr.axis("off")

    # Pitch area
    ax_pitch = fig.add_axes([0.135, 0.185, 0.73, 0.335], zorder=2)
    ax_pitch.set_facecolor(pitch_bg)
    ax_pitch.set_xlim(-3, 103); ax_pitch.set_ylim(-3, y_max + 3)
    ax_pitch.axis("off")

    # Bottom stats bar
    ax_foot = fig.add_axes([0.12, 0.075, 0.76, 0.09], zorder=2)
    ax_foot.set_facecolor(card_bg); ax_foot.axis("off")
    ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1)

    # ── Pitch background card ───────────────────────────────────────────────
    ax_pitch.add_patch(FancyBboxPatch((-2, -2), 104, y_max + 4,
                                      boxstyle="square,pad=0",
                                      facecolor=pitch_bg,
                                      edgecolor=line_col, linewidth=1.25,
                                      zorder=0))

    # ── Full pitch lines ────────────────────────────────────────────────────
    _draw_half_pitch_lines(ax_pitch, y_max, line_col, lw=1.8, show_full=True)

    # ── Plot passes ─────────────────────────────────────────────────────────
    def _plot_pass_group(pdf, color, alpha):
        pdf = pdf.dropna(subset=["x", "y"])
        has_end = ("x2" in pdf.columns and "y2" in pdf.columns and
                   pdf["x2"].notna().any())
        if has_end:
            valid = pdf.dropna(subset=["x2", "y2"])
            for _, row in valid.iterrows():
                ax_pitch.annotate(
                    "", xy=(float(row["x2"]), float(row["y2"])),
                    xytext=(float(row["x"]), float(row["y"])),
                    arrowprops=dict(
                        arrowstyle=f"-|>,head_width={arrow_head_width * 0.055},head_length=0.08",
                        color=color, lw=arrow_width, alpha=alpha
                    ), zorder=4
                )
        else:
            ax_pitch.scatter(pdf["x"], pdf["y"], s=40, color=color, alpha=alpha, zorder=4)

    _plot_pass_group(succ, successful_color, arrow_alpha)
    _plot_pass_group(unsucc, unsuccessful_color, arrow_alpha * 0.7)

    # ── Header ──────────────────────────────────────────────────────────────
    player_name = cfg.get("player_name") or cfg.get("title", "")
    sub_parts = [p for p in [
        cfg.get("competition", ""), cfg.get("match_info", ""), cfg.get("season", "")
    ] if p.strip()]
    sub_line = "  |  ".join(sub_parts[:3])

    title_x = 0.145

    fig.text(title_x, 0.610, player_name, fontsize=18, weight="900",
             color=text_col, va="top")
    if sub_line:
        fig.text(title_x, 0.585, sub_line, fontsize=9, color=muted_col, va="top")

    if cfg.get("logo_img"):
        draw_logo(fig, cfg["logo_img"],
                  cfg.get("logo_x", 0.755), cfg.get("logo_y", 0.568),
                  cfg.get("logo_w", 0.13), cfg.get("logo_h", 0.055),
                  circle_crop=cfg.get("logo_circle", True), border_lw=2.0)

    # ── Footer stats row (Opta-style) ───────────────────────────────────────
    # Divider line
    ax_foot.plot([0.02, 0.98], [0.94, 0.94], color=line_col, lw=1.0, alpha=0.35)

    # Successful count
    ax_foot.text(0.03, 0.70, str(n_succ), ha="left", va="top",
                 fontsize=17, weight="900", color=text_col)
    ax_foot.text(0.08, 0.68, "successful", ha="left", va="top",
                 fontsize=8, color=muted_col)
    ax_foot.scatter([0.048], [0.28], s=70, color=successful_color,
                    edgecolors="none", zorder=5)
    ax_foot.annotate("", xy=(0.09, 0.28), xytext=(0.055, 0.28),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=successful_color, lw=1.5))

    # Unsuccessful count
    ax_foot.text(0.21, 0.70, str(n_unsucc), ha="left", va="top",
                 fontsize=17, weight="900", color=text_col)
    ax_foot.text(0.255, 0.68, "unsuccessful", ha="left", va="top",
                 fontsize=8, color=muted_col)
    ax_foot.scatter([0.218], [0.28], s=70, facecolors="none",
                    edgecolors=unsuccessful_color, linewidth=1.5, zorder=5)
    ax_foot.annotate("", xy=(0.262, 0.28), xytext=(0.225, 0.28),
                     xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=unsuccessful_color, lw=1.5))

    # Attacking direction arrows (5 chevrons)
    adir_col = cfg.get("attack_dir_color", muted_col)
    adir_label = cfg.get("attack_dir_label", "Attacking Direction")
    if cfg.get("show_attack_dir", True):
        for i, xf in enumerate(np.linspace(0.43, 0.55, 5)):
            alpha = 0.25 + 0.15 * i
            ax_foot.annotate("", xy=(xf + 0.022, 0.34), xytext=(xf, 0.34),
                             xycoords="axes fraction",
                             arrowprops=dict(arrowstyle="-|>", color=adir_col,
                                             lw=1.6, alpha=alpha))
        ax_foot.text(0.49, 0.04, adir_label, ha="center", va="bottom",
                     fontsize=7, color=adir_col, transform=ax_foot.transAxes)

    # Total passes count (circled)
    total_passes = n_succ + n_unsucc if not isinstance(n_succ, str) else "—"
    total_str = str(total_passes) if isinstance(total_passes, int) else str(n_total)
    circ = plt.Circle((0.68, 0.42), 0.075, transform=ax_foot.transAxes,
                       facecolor="none", edgecolor=text_col, linewidth=1.8, zorder=3)
    ax_foot.add_patch(circ)
    ax_foot.text(0.68, 0.42, total_str, ha="center", va="center",
                 fontsize=12, weight="900", color=text_col,
                 transform=ax_foot.transAxes)
    ax_foot.text(0.735, 0.42, "passes", ha="left", va="center",
                 fontsize=8, color=muted_col, transform=ax_foot.transAxes)

    # Accuracy %
    acc_str = f"{accuracy_pct}%" if isinstance(accuracy_pct, (int, float)) else str(accuracy_pct)
    ax_foot.text(0.88, 0.62, acc_str, ha="center", va="top",
                 fontsize=18, weight="900", color=text_col,
                 transform=ax_foot.transAxes)
    ax_foot.text(0.94, 0.40, "accuracy", ha="left", va="center",
                 fontsize=8, color=muted_col, transform=ax_foot.transAxes)

    # Footer note
    footer = cfg.get("footer_note", "")
    if footer:
        fig.text(0.08, 0.070, footer, fontsize=7, color=muted_col, va="bottom")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PROFESSIONAL PASS MAP (generic, all themes)
# ─────────────────────────────────────────────────────────────────────────────

def pro_pass_map(
    df: pd.DataFrame,
    cfg: dict = None,
    pass_colors: dict = None,
    pass_markers: dict = None,
    show_arrows: bool = True,
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    vertical: bool = False,
):
    """Universal pass map with full layout system."""
    cfg = cfg or default_layout_cfg()
    theme = THEMES.get(cfg.get("theme_name", "The Athletic Dark"), {})
    if not theme:
        theme = PRO_THEMES["The Athletic Dark"]

    pass_colors = pass_colors or {
        "successful": "#00FF6A", "unsuccessful": "#FF4D4D",
        "key pass": "#00C2FF", "assist": "#FFD400",
    }
    pass_markers = pass_markers or {
        "successful": "o", "unsuccessful": "x",
        "key pass": "D", "assist": "*",
    }

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme,
                       vertical_pitch=vertical)
    fig_w = 8.0 if vertical else 12.0
    fig_h = fig_w * (100.0 / max(float(pitch_width), 1.0)) * 0.95 if vertical else 8.0

    # Reserve space for title and footer
    fig = plt.figure(figsize=(fig_w, fig_h + 2.0))
    fig.patch.set_facecolor(theme["bg"])

    # Title strip
    _title_h = 0.15
    ax_pitch = fig.add_axes([0.04, 0.08, 0.92, 0.76])
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"].astype(str).str.lower() == "pass"].copy()
    for c in ["x", "y", "x2", "y2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
    handles = []
    for outc in PASS_ORDER:
        sub = d[d["outcome"] == outc] if "outcome" in d.columns else pd.DataFrame()
        if sub.empty:
            continue
        col = pass_colors.get(outc, theme.get("muted", "#AAAAAA"))
        mk = pass_markers.get(outc, "o")
        if show_arrows and "x2" in sub.columns:
            valid = sub.dropna(subset=["x2", "y2"])
            if not valid.empty:
                if vertical:
                    pitch.arrows(valid["x"], valid["y"], valid["x2"], valid["y2"],
                                 ax=ax_pitch, color=col, width=1.8, alpha=0.82)
                else:
                    pitch.arrows(valid["x"], valid["y"], valid["x2"], valid["y2"],
                                 ax=ax_pitch, color=col, width=1.8, alpha=0.82)
        if mk and str(mk).lower() not in {"none", ""}:
            pitch.scatter(sub["x"], sub["y"], ax=ax_pitch, s=70, marker=mk,
                          color=col, edgecolors="white", linewidth=0.8, zorder=5)
            handles.append(Line2D([0], [0], marker=mk, color="none",
                markerfacecolor=col, markeredgecolor="white",
                markersize=8, label=outc))

    draw_attack_direction(ax_pitch, cfg, theme, pitch_mode, pitch_width, vertical)
    draw_custom_legend(ax_pitch, handles, cfg, theme)

    # Title
    draw_title_block(fig, cfg, theme, title_y=0.965)
    draw_footer(fig, cfg, theme)
    if cfg.get("logo_img"):
        draw_logo(fig, cfg["logo_img"], cfg.get("logo_x", 0.84),
                  cfg.get("logo_y", 0.88), cfg.get("logo_w", 0.12),
                  cfg.get("logo_h", 0.10))
    if cfg.get("player_img"):
        draw_logo(fig, cfg["player_img"], cfg.get("player_x", 0.02),
                  cfg.get("player_y", 0.88), cfg.get("player_w", 0.10),
                  cfg.get("player_h", 0.10), circle_crop=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROFESSIONAL SHOT MAP (generic)
# ─────────────────────────────────────────────────────────────────────────────

def pro_shot_map(
    df: pd.DataFrame,
    cfg: dict = None,
    shot_colors: dict = None,
    shot_markers: dict = None,
    show_xg: bool = True,
    size_by_xg: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    vertical: bool = False,
):
    """Universal shot map with full layout system."""
    cfg = cfg or default_layout_cfg()
    theme = THEMES.get(cfg.get("theme_name", "The Athletic Dark"), {})
    if not theme:
        theme = PRO_THEMES["The Athletic Dark"]

    shot_colors = shot_colors or {
        "off target": "#FF8A00", "ontarget": "#00C2FF",
        "goal": "#00FF6A", "blocked": "#AAAAAA",
    }
    shot_markers = shot_markers or {
        "off target": "^", "ontarget": "D",
        "goal": "*", "blocked": "s",
    }

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme,
                       vertical_pitch=vertical)
    fig_w = 8.0 if vertical else 12.0
    fig_h = fig_w * (100.0 / max(float(pitch_width), 1.0)) * 0.95 if vertical else 8.0

    fig = plt.figure(figsize=(fig_w, fig_h + 2.0))
    fig.patch.set_facecolor(theme["bg"])

    ax_pitch = fig.add_axes([0.04, 0.08, 0.92, 0.76])
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"].astype(str).str.lower() == "shot"].copy()

    SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
    handles = []

    for outc in SHOT_ORDER:
        sub = d[d["outcome"] == outc] if "outcome" in d.columns else pd.DataFrame()
        if sub.empty:
            continue
        col = shot_colors.get(outc, theme.get("muted", "#AAAAAA"))
        mk = shot_markers.get(outc, "o")
        if not mk or str(mk).lower() in {"none", ""}:
            continue

        if size_by_xg and "xg" in sub.columns:
            sizes = (pd.to_numeric(sub["xg"], errors="coerce").fillna(0.1) * 800 + 60).clip(40, 900)
        else:
            sizes = 160

        pitch.scatter(sub["x"], sub["y"], ax=ax_pitch, s=sizes, marker=mk,
                      color=col, edgecolors="white", linewidth=1.4, alpha=0.95, zorder=5)

        if show_xg and "xg" in sub.columns:
            for _, row in sub.iterrows():
                ax_pitch.text(float(row["x"]) + 1.0, float(row["y"]) + 1.0,
                              f'{float(row.get("xg", 0)):.2f}',
                              fontsize=8, color="white", weight="bold")

        handles.append(Line2D([0], [0], marker=mk, color="none",
            markerfacecolor=col, markeredgecolor="white",
            markersize=10, label=outc))

    draw_attack_direction(ax_pitch, cfg, theme, pitch_mode, pitch_width, vertical)
    draw_custom_legend(ax_pitch, handles, cfg, theme)
    draw_title_block(fig, cfg, theme, title_y=0.965)
    draw_footer(fig, cfg, theme)
    if cfg.get("logo_img"):
        draw_logo(fig, cfg["logo_img"], cfg.get("logo_x", 0.84),
                  cfg.get("logo_y", 0.88), cfg.get("logo_w", 0.12),
                  cfg.get("logo_h", 0.10))
    if cfg.get("player_img"):
        draw_logo(fig, cfg["player_img"], cfg.get("player_x", 0.02),
                  cfg.get("player_y", 0.88), cfg.get("player_w", 0.10),
                  cfg.get("player_h", 0.10), circle_crop=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LAYOUT CONTROLS for Streamlit
# ─────────────────────────────────────────────────────────────────────────────

def layout_controls_ui(prefix: str, st_module, defaults: dict = None) -> dict:
    """
    Render Streamlit UI for universal layout controls.
    Returns a cfg dict ready to pass to any pro chart.
    st_module = streamlit (passed in to avoid circular import in the chart library)
    """
    import streamlit as st
    cfg = defaults or default_layout_cfg()

    with st.expander("📝 Title & Info Block", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cfg["player_name"] = st.text_input("Player name", cfg.get("player_name", ""), key=f"{prefix}_pn")
            cfg["subtitle"]    = st.text_input("Subtitle",    cfg.get("subtitle", ""),     key=f"{prefix}_sub")
            cfg["match_info"]  = st.text_input("Match info",  cfg.get("match_info", ""),   key=f"{prefix}_mi")
            cfg["team_name"]   = st.text_input("Team",        cfg.get("team_name", ""),    key=f"{prefix}_tn")
        with c2:
            cfg["competition"] = st.text_input("Competition", cfg.get("competition", ""),  key=f"{prefix}_cp")
            cfg["season"]      = st.text_input("Season",      cfg.get("season", ""),       key=f"{prefix}_se")
            cfg["analyst_credit"] = st.text_input("Credit",   cfg.get("analyst_credit", ""), key=f"{prefix}_ac")
            cfg["footer_note"] = st.text_input("Footer note", cfg.get("footer_note", ""),  key=f"{prefix}_fn")
        c3, c4 = st.columns(2)
        with c3:
            cfg["title_fontsize"] = st.slider("Title size", 14, 40, cfg.get("title_fontsize", 22), key=f"{prefix}_tfs")
        with c4:
            cfg["subtitle_fontsize"] = st.slider("Subtitle size", 9, 20, cfg.get("subtitle_fontsize", 13), key=f"{prefix}_sfs")

    with st.expander("⬆ Attacking Direction", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cfg["show_attack_dir"] = st.checkbox("Show arrow", cfg.get("show_attack_dir", True), key=f"{prefix}_sad")
            cfg["attack_dir"]      = st.selectbox("Direction", ["ltr", "rtl"], key=f"{prefix}_ad")
            cfg["attack_dir_pos"]  = st.selectbox("Position",  ["bottom", "top"], key=f"{prefix}_adp")
        with c2:
            cfg["attack_dir_color"] = st.color_picker("Arrow color", cfg.get("attack_dir_color", "#888888"), key=f"{prefix}_adc")
            cfg["attack_dir_label"] = st.text_input("Label",         cfg.get("attack_dir_label", "Attacking Direction"), key=f"{prefix}_adl")
            cfg["attack_dir_show_label"] = st.checkbox("Show label", cfg.get("attack_dir_show_label", True), key=f"{prefix}_adsl")
            cfg["attack_dir_size"]  = st.slider("Arrow size", 8, 16, cfg.get("attack_dir_size", 11), key=f"{prefix}_ads")

    with st.expander("📖 Legend", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cfg["show_legend"]    = st.checkbox("Show legend", cfg.get("show_legend", True), key=f"{prefix}_sl")
            cfg["legend_pos"]     = st.selectbox("Position", [
                "lower center","upper center","lower left","upper left",
                "lower right","upper right","center left","center right","best"
            ], key=f"{prefix}_lp")
            cfg["legend_title"]   = st.text_input("Legend title", cfg.get("legend_title", ""), key=f"{prefix}_lt")
        with c2:
            cfg["legend_fontsize"]    = st.slider("Font size", 6, 16, cfg.get("legend_fontsize", 9), key=f"{prefix}_lfs")
            cfg["legend_markerscale"] = st.slider("Marker scale", 0.5, 3.0, float(cfg.get("legend_markerscale", 1.0)), 0.1, key=f"{prefix}_lms")
            cfg["legend_ncol"]        = st.number_input("Columns", 1, 8, int(cfg.get("legend_ncol") or 4), key=f"{prefix}_lnc")
            cfg["legend_frame"]       = st.checkbox("Show frame", cfg.get("legend_frame", False), key=f"{prefix}_lf")

    return cfg


def image_overlay_controls_pro(prefix: str, label: str,
                                 dx: float, dy: float, dw: float, dh: float,
                                 circle: bool = False) -> dict:
    """Return overlay config dict from Streamlit UI."""
    import streamlit as st
    from PIL import Image
    with st.expander(f"🖼️ {label}", expanded=False):
        f = st.file_uploader(label, type=["png", "jpg", "jpeg"],
                             key=f"{prefix}_f", label_visibility="collapsed")
        img = None
        if f:
            try:
                img = Image.open(f).convert("RGBA")
            except Exception:
                pass
        if img:
            c1, c2 = st.columns(2)
            with c1:
                x = st.slider("X %", 0, 95, int(dx * 100), key=f"{prefix}_x") / 100
                w = st.slider("W %", 4, 30, int(dw * 100), key=f"{prefix}_w") / 100
            with c2:
                y = st.slider("Y %", 0, 95, int(dy * 100), key=f"{prefix}_y") / 100
                h = st.slider("H %", 4, 30, int(dh * 100), key=f"{prefix}_h") / 100
            circ = st.checkbox("Circle crop", circle, key=f"{prefix}_circ")
        else:
            x, y, w, h, circ = dx, dy, dw, dh, circle
    return dict(img=img, x=x, y=y, w=w, h=h, circle=circ)
