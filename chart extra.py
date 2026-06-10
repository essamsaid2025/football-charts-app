"""
charts_extra.py  —  additional chart functions for Football Analysis Suite
Imports from charts.py (the original file) for shared helpers.
"""
from __future__ import annotations
import math
import io
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mplsoccer import Pitch, VerticalPitch

# ── re-use theme dict from charts.py ─────────────────────────────────────────
from charts import THEMES, make_pitch, _add_legend, _is_no_marker, _yes_only, \
                   DEF_ACTION_COLS, _standardize_defensive_columns, \
                   _goal_mouth_bounds, SHOT_ORDER, PASS_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# EXTRA THEMES
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_THEMES = {
    "Opta Light": {
        "bg": "#F0F0F0", "panel": "#E8E8E8", "pitch": "#F0F0F0",
        "pitch_lines": "#888888", "pitch_stripe": None,
        "text": "#1A1A2E", "muted": "#666666", "lines": "#CCCCCC",
        "goal": "#444444",
    },
    "Athletic FC Dark": {
        "bg": "#0A0A0A", "panel": "#111111", "pitch": "#0D2818",
        "pitch_lines": "#CCCCCC", "pitch_stripe": "#0F2E1A",
        "text": "#F5F5F0", "muted": "#999999", "lines": "#333333",
        "goal": "#DDDDDD",
    },
    "Athletic FC Light": {
        "bg": "#F4F1E8", "panel": "#EDE9DC", "pitch": "#4A7C59",
        "pitch_lines": "#FFFFFF", "pitch_stripe": None,
        "text": "#1A1A1A", "muted": "#666666", "lines": "#CCBFA0",
        "goal": "#333333",
    },
    "Whoscored Dark": {
        "bg": "#1C1C2E", "panel": "#252540", "pitch": "#1A3A2A",
        "pitch_lines": "#E0E0E0", "pitch_stripe": None,
        "text": "#FFFFFF", "muted": "#A0A0C0", "lines": "#303060",
        "goal": "#FFFFFF",
    },
    "Statsbomb Light": {
        "bg": "#FAFAFA", "panel": "#F0F0F0", "pitch": "#68BB59",
        "pitch_lines": "#FFFFFF", "pitch_stripe": None,
        "text": "#111111", "muted": "#555555", "lines": "#DDDDDD",
        "goal": "#222222",
    },
    "Night Blue": {
        "bg": "#060D1F", "panel": "#0A1628", "pitch": "#0F3460",
        "pitch_lines": "#E8F4FD", "pitch_stripe": None,
        "text": "#E8F4FD", "muted": "#7BA7C2", "lines": "#1A2D50",
        "goal": "#E8F4FD",
    },
    "Broadcast Green": {
        "bg": "#0A1A0A", "panel": "#111C11", "pitch": "#1A4A1A",
        "pitch_lines": "#FFFFFF", "pitch_stripe": "#1E501E",
        "text": "#FFFFFF", "muted": "#88BB88", "lines": "#224422",
        "goal": "#FFFFFF",
    },
}

def register_extra_themes():
    """Call once at app startup to add extra themes into the global THEMES dict."""
    for k, v in EXTRA_THEMES.items():
        if k not in THEMES:
            THEMES[k] = v

register_extra_themes()


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY IMAGE HELPER
# ─────────────────────────────────────────────────────────────────────────────
def overlay_image_on_fig(
    fig,
    img_obj,                    # PIL Image
    x: float = 0.02,           # figure-fraction x
    y: float = 0.88,           # figure-fraction y
    w: float = 0.10,           # figure-fraction width
    h: float = 0.10,           # figure-fraction height
    circle_crop: bool = False,
    border_color: str = "white",
    border_lw: float = 0.0,
):
    """Paste a PIL image onto any figure at the given figure-fraction position."""
    if img_obj is None:
        return
    try:
        import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# LEGEND HELPER  (full manual control)
# ─────────────────────────────────────────────────────────────────────────────
def build_legend_handles(
    items: Dict[str, dict],   # {"label": {"color":..,"marker":..,"linestyle":..}}
    active_labels: List[str],
) -> List:
    """Build legend handle list only for labels in active_labels."""
    handles = []
    for label in active_labels:
        spec = items.get(label, {})
        color = spec.get("color", "#AAAAAA")
        marker = spec.get("marker", "o")
        ls = spec.get("linestyle", "none")
        if _is_no_marker(marker) and ls == "none":
            handles.append(Patch(facecolor=color, label=label))
        else:
            handles.append(Line2D([0], [0],
                marker=marker if not _is_no_marker(marker) else "none",
                color=color if ls != "none" else "none",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=9,
                linewidth=2 if ls != "none" else 0,
                linestyle=ls,
                label=label))
    return handles


# ─────────────────────────────────────────────────────────────────────────────
# VERTICAL PITCH  (fixed full-pitch sizing)
# ─────────────────────────────────────────────────────────────────────────────
def make_vertical_pitch_fig(theme: dict, pitch_width: float = 68.0,
                             figsize: Tuple = (7, 11)) -> Tuple:
    """Returns (fig, ax) with a properly-sized full vertical pitch."""
    pitch_color = theme.get("pitch", "#1f5f3b")
    line_color = theme.get("pitch_lines", "#FFFFFF")
    stripe = theme.get("pitch_stripe")

    vp = VerticalPitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        pitch_color=pitch_color,
        line_color=line_color,
        line_zorder=2,
        stripe=bool(stripe),
        stripe_color=stripe or pitch_color,
    )
    fig, ax = vp.draw(figsize=figsize)
    fig.patch.set_facecolor(theme.get("bg", "#0E1117"))
    ax.set_facecolor(pitch_color)
    return fig, ax, vp


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GOAL LOCATION MAP  (Opta-style — half pitch, vertical, shot locations)
#     Inspired by Image 1 (Mohamed Salah Opta Analyst style)
# ─────────────────────────────────────────────────────────────────────────────
def goal_location_map(
    df: pd.DataFrame,
    title: str = "Goal Map",
    player_name: str = "",
    subtitle: str = "",
    stat_labels: Optional[List[Tuple[str, str]]] = None,
    # stat_labels = [("27", "goals"), ("0.94", "goals per 90"), ...]
    theme_name: str = "Opta Light",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    goal_color: str = "#C8102E",
    goal_edge: str = "#8B0000",
    dot_size: int = 160,
    penalty_label_col: Optional[str] = None,   # col with penalty count per shot
    logo_img=None,
    logo_x: float = 0.72, logo_y: float = 0.88, logo_w: float = 0.14, logo_h: float = 0.10,
    player_img=None,
    player_img_x: float = 0.02, player_img_y: float = 0.88,
    player_img_w: float = 0.10, player_img_h: float = 0.10,
    show_pitch_half_only: bool = True,
    attack_direction: str = "ltr",
):
    """
    Opta Analyst-style goal location map.
    Shows only the attacking half (or full pitch).
    Stats panel on the right.
    Requires columns: x, y, outcome
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme.get("bg", "#F0F0F0")
    text_col = theme.get("text", "#1A1A1A")
    muted_col = theme.get("muted", "#666666")
    line_col = theme.get("pitch_lines", "#888888")

    df = df.copy()
    goals = df[df["outcome"].astype(str).str.lower() == "goal"].copy()

    # Figure layout: pitch left, stats right
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor(bg)

    # Axes: pitch occupies left ~62%, stats right ~38%
    ax_pitch = fig.add_axes([0.02, 0.06, 0.60, 0.78])
    ax_stats = fig.add_axes([0.64, 0.06, 0.34, 0.78])

    # ── Draw half pitch ───────────────────────────────────────────────────────
    ax_pitch.set_facecolor(bg)
    ax_pitch.set_xlim(0, 100)
    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax_pitch.set_ylim(-2, y_max + 2)
    ax_pitch.axis("off")

    lc = line_col; lw = 2.0

    def _draw_rect(x0, y0, x1, y1):
        ax_pitch.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=lc, lw=lw, solid_capstyle="round")

    mid = y_max / 2.0
    # Only show attacking half (x=50 to 100) if show_pitch_half_only
    x_start = 50.0 if show_pitch_half_only else 0.0
    _draw_rect(x_start, 0, 100, y_max)

    # Penalty area
    pa_w = y_max * 40.32 / 68.0; pa_l = 16.5 / 105.0 * 100.0
    _draw_rect(100 - pa_l, mid - pa_w / 2, 100, mid + pa_w / 2)

    # Six-yard box
    sa_w = y_max * 18.32 / 68.0; sa_l = 5.5 / 105.0 * 100.0
    _draw_rect(100 - sa_l, mid - sa_w / 2, 100, mid + sa_w / 2)

    # Goal
    goal_w = y_max * 7.32 / 68.0
    ax_pitch.plot([100, 100 + 3], [mid - goal_w / 2, mid - goal_w / 2], color=lc, lw=lw)
    ax_pitch.plot([100, 100 + 3], [mid + goal_w / 2, mid + goal_w / 2], color=lc, lw=lw)

    # Penalty spot
    pen_x = 100 - 11.0 / 105.0 * 100.0
    ax_pitch.plot(pen_x, mid, "o", color=lc, ms=3)

    # Centre circle arc (if full pitch)
    if not show_pitch_half_only:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        r = 9.15 / 105.0 * 100.0
        ry = 9.15 / 68.0 * y_max
        ax_pitch.plot(50 + r * np.cos(theta), mid + ry * np.sin(theta), color=lc, lw=lw)

    # Half-way line
    if not show_pitch_half_only:
        ax_pitch.plot([50, 50], [0, y_max], color=lc, lw=lw)

    # Penalty arc
    arc_cx = 100 - pa_l
    arc_r_x = 9.15 / 105.0 * 100.0; arc_r_y = 9.15 / 68.0 * y_max
    theta_arc = np.linspace(np.pi * 0.62, np.pi * 1.38, 80)
    arc_xs = arc_cx + arc_r_x * np.cos(theta_arc)
    arc_ys = mid + arc_r_y * np.sin(theta_arc)
    outside = arc_xs <= (100 - pa_l + 0.5)
    ax_pitch.plot(arc_xs[outside], arc_ys[outside], color=lc, lw=lw)

    # ── Plot goals ────────────────────────────────────────────────────────────
    if not goals.empty:
        ax_pitch.scatter(
            goals["x"], goals["y"],
            s=dot_size, color=goal_color, edgecolors=goal_edge, linewidth=1.2,
            zorder=6, alpha=0.92
        )
        if penalty_label_col and penalty_label_col in goals.columns:
            for _, r in goals.iterrows():
                val = r.get(penalty_label_col, 0)
                try:
                    val = int(float(val))
                except Exception:
                    val = 0
                if val > 0:
                    ax_pitch.text(float(r["x"]), float(r["y"]), str(val),
                                  ha="center", va="center", fontsize=7,
                                  color="white", weight="bold", zorder=7)

    # ── Stats panel ───────────────────────────────────────────────────────────
    ax_stats.set_facecolor(bg)
    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)

    if stat_labels:
        y_pos = 0.88
        for i, (val, label) in enumerate(stat_labels):
            ax_stats.text(0.05, y_pos, val, fontsize=26, weight="900",
                          color=text_col, va="top", transform=ax_stats.transAxes)
            ax_stats.text(0.05, y_pos - 0.075, label, fontsize=13,
                          color=muted_col, va="top", transform=ax_stats.transAxes)
            if i < len(stat_labels) - 1:
                ax_stats.axhline(y_pos - 0.13, xmin=0.0, xmax=0.85,
                                 color=theme.get("lines", "#CCCCCC"), lw=1.0,
                                 transform=ax_stats.transAxes)
            y_pos -= 0.20

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(0.02, 0.95, player_name or title, fontsize=22, weight="900",
             color=text_col, va="top")
    fig.text(0.02, 0.905, subtitle, fontsize=11, color=muted_col, va="top")

    overlay_image_on_fig(fig, logo_img, x=logo_x, y=logo_y, w=logo_w, h=logo_h, circle_crop=False)
    overlay_image_on_fig(fig, player_img, x=player_img_x, y=player_img_y,
                         w=player_img_w, h=player_img_h, circle_crop=True)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  GOAL MOUTH MAP  (CannonStats-style — shots on target plotted on goal face)
#     Inspired by Image 2 (West Ham Cannon Stats style)
# ─────────────────────────────────────────────────────────────────────────────
def goal_mouth_map(
    df: pd.DataFrame,
    title: str = "Shots on Target",
    subtitle: str = "Shots on Target Map",
    stats_row: Optional[Dict[str, str]] = None,
    # stats_row = {"Goals": "2", "Post-Shot xG": "1.8", "Shots on Target": "3"}
    theme_name: str = "Opta Light",
    goal_color: str = "#7A2232",
    save_color: str = "#FFFFFF",
    goal_edge: str = "#3D0A18",
    save_edge: str = "#1A2E5A",
    size_by_xg: bool = True,
    base_size: int = 600,
    logo_img=None,
    logo_x: float = 0.84, logo_y: float = 0.88, logo_w: float = 0.12, logo_h: float = 0.10,
    footer_left: str = "",
    footer_right: str = "",
):
    """
    CannonStats-style: plots shots on target as circles on the actual goal face.
    Size of circle = xG value (if size_by_xg=True).
    Requires: outcome (goal/ontarget/save), y, y2 columns.
    y2 and a height column (or z column) represent where the shot hit the goal.
    Falls back to y-only if z/height not available.
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme.get("bg", "#F0F0F0")
    text_col = theme.get("text", "#1A1A1A")
    muted_col = theme.get("muted", "#666666")
    panel_col = theme.get("panel", "#E8E8E8")

    df = df.copy()
    on_target = df[df["outcome"].astype(str).str.lower().isin(["goal", "ontarget", "save", "saved"])].copy()

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(bg)

    # Stats row area
    ax_stats = fig.add_axes([0.0, 0.72, 1.0, 0.22])
    ax_stats.set_facecolor(bg); ax_stats.axis("off")
    ax_stats.set_xlim(0, 1); ax_stats.set_ylim(0, 1)

    # Title
    fig.text(0.04, 0.97, title, fontsize=36, weight="900", color=text_col, va="top",
             fontfamily="serif")
    fig.text(0.04, 0.84, subtitle, fontsize=14, color=muted_col, va="top")

    # Stats numbers
    if stats_row:
        keys = list(stats_row.keys())
        n = len(keys)
        xs = np.linspace(0.28, 0.75, n)
        for i, (k, v) in enumerate(stats_row.items()):
            ax_stats.text(xs[i], 0.82, f"{k}:", fontsize=13, weight="700",
                          color=text_col, ha="center", va="top")
            ax_stats.text(xs[i], 0.40, str(v), fontsize=30, weight="900",
                          color=text_col, ha="center", va="top")

    # Horizontal divider
    ax_stats.axhline(0.02, xmin=0.02, xmax=0.98, color=theme.get("lines","#CCCCCC"), lw=1.5)

    # Goal face area
    ax_goal = fig.add_axes([0.06, 0.13, 0.88, 0.56])
    ax_goal.set_facecolor(bg); ax_goal.axis("off")

    # Outer frame (posts + crossbar with depth effect)
    frame_bg = "#888888"
    frame_outer = FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                                  boxstyle="square,pad=0", linewidth=0,
                                  facecolor=frame_bg, zorder=1)
    ax_goal.add_patch(frame_outer)

    # White goal interior
    goal_interior = FancyBboxPatch((0.09, 0.09), 0.82, 0.82,
                                    boxstyle="square,pad=0", linewidth=2,
                                    edgecolor="#555555", facecolor="#F5F5F5", zorder=2)
    ax_goal.add_patch(goal_interior)
    ax_goal.set_xlim(0, 1); ax_goal.set_ylim(0, 1)

    # Grid lines inside goal
    for gx in np.linspace(0.09, 0.91, 22):
        ax_goal.plot([gx, gx], [0.09, 0.91], color="#DDDDDD", lw=0.5, alpha=0.6, zorder=3)
    for gy in np.linspace(0.09, 0.91, 10):
        ax_goal.plot([0.09, 0.91], [gy, gy], color="#DDDDDD", lw=0.5, alpha=0.6, zorder=3)

    # ── Plot shots ────────────────────────────────────────────────────────────
    if not on_target.empty:
        # Map y → horizontal goal position (0.09..0.91)
        y_max_p = 68.0  # assume standard
        if "y" in on_target.columns:
            y_vals = pd.to_numeric(on_target["y"], errors="coerce").fillna(34.0)
        else:
            y_vals = pd.Series([34.0] * len(on_target))

        # Map z/height → vertical goal position (0.09..0.91)
        z_col = None
        for cand in ["z", "height", "shot_height", "goal_height", "y2_z"]:
            if cand in on_target.columns:
                z_col = cand; break

        if z_col:
            z_vals = pd.to_numeric(on_target[z_col], errors="coerce").fillna(1.2)
            z_max = 2.44  # goal height in metres
            z_norm = (z_vals / z_max).clip(0.05, 0.95)
        else:
            # Use random vertical scatter if no height column
            rng = np.random.default_rng(42)
            z_norm = pd.Series(rng.uniform(0.15, 0.75, len(on_target)), index=on_target.index)

        gx_norm = 0.09 + (y_vals / y_max_p) * 0.82

        for i, (idx, row) in enumerate(on_target.iterrows()):
            outc = str(row.get("outcome", "")).lower()
            is_goal = outc == "goal"
            color = goal_color if is_goal else save_color
            edge  = goal_edge  if is_goal else save_edge

            if size_by_xg and "xg" in row.index and pd.notna(row.get("xg")):
                sz = max(100, min(2000, float(row["xg"]) * 3500))
            else:
                sz = base_size if is_goal else base_size * 0.8

            ax_goal.scatter([gx_norm.iloc[i]], [z_norm.iloc[i]],
                             s=sz, color=color, edgecolors=edge,
                             linewidth=2.5 if not is_goal else 0,
                             zorder=6, alpha=0.92)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax_leg = fig.add_axes([0.0, 0.0, 1.0, 0.13])
    ax_leg.set_facecolor(bg); ax_leg.axis("off")
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)

    # Outcome legend
    ax_leg.text(0.15, 0.85, "Shot Outcome:", fontsize=12, weight="700",
                color=text_col, ha="center", va="top")
    ax_leg.scatter([0.06], [0.35], s=400, color=save_color,
                   edgecolors=save_edge, linewidth=2.5, zorder=5)
    ax_leg.text(0.09, 0.35, "Save", fontsize=11, color=text_col, va="center")
    ax_leg.scatter([0.19], [0.35], s=600, color=goal_color,
                   edgecolors=goal_edge, linewidth=0, zorder=5)
    ax_leg.text(0.22, 0.35, "Goal", fontsize=11, color=text_col, va="center")

    # xG size legend
    if size_by_xg:
        ax_leg.text(0.67, 0.85, "Post-Shot xG Value:", fontsize=12, weight="700",
                    color=text_col, ha="center", va="top")
        xg_vals = [0.03, 0.08, 0.18, 0.35, 0.65]
        xg_xs = np.linspace(0.53, 0.82, len(xg_vals))
        for xgv, xgx in zip(xg_vals, xg_xs):
            ax_leg.scatter([xgx], [0.35], s=max(30, xgv * 3500 * 0.3),
                           color=muted_col, edgecolors=text_col, linewidth=1, alpha=0.7)

    # Footer
    if footer_left:
        ax_leg.text(0.02, 0.05, footer_left, fontsize=9, color=muted_col, va="bottom")
    if footer_right:
        ax_leg.text(0.98, 0.05, footer_right, fontsize=9, color=muted_col,
                    va="bottom", ha="right")

    overlay_image_on_fig(fig, logo_img, x=logo_x, y=logo_y, w=logo_w, h=logo_h,
                         circle_crop=True, border_color=text_col, border_lw=2.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FULL VERTICAL PITCH MAP  (fixed sizing)
# ─────────────────────────────────────────────────────────────────────────────
def vertical_event_map(
    df: pd.DataFrame,
    event_type: str = "pass",   # "pass","shot","touch","defensive","all"
    title: str = "Vertical Map",
    theme_name: str = "The Athletic Dark",
    pitch_width: float = 68.0,
    pass_colors: Optional[dict] = None,
    pass_markers: Optional[dict] = None,
    shot_colors: Optional[dict] = None,
    shot_markers: Optional[dict] = None,
    dot_color: str = "#00C2FF",
    dot_size: int = 80,
    show_arrows: bool = True,
    overlay_img=None,
    img_x: float = 0.02, img_y: float = 0.88, img_w: float = 0.10, img_h: float = 0.10,
    active_legend_items: Optional[List[str]] = None,
):
    """Full vertical pitch — correctly sized, no half-pitch clipping."""
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch_color = theme.get("pitch", "#1f5f3b")
    line_color = theme.get("pitch_lines", "#FFFFFF")
    stripe = theme.get("pitch_stripe")

    vp = VerticalPitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        pitch_color=pitch_color,
        line_color=line_color,
        line_zorder=2,
        stripe=bool(stripe),
        stripe_color=stripe or pitch_color,
    )

    fig, ax = vp.draw(figsize=(8, 12))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(pitch_color)

    text_col = theme.get("text", "white")

    et = (event_type or "all").lower()
    if et == "all":
        d = df.copy()
    elif "event_type" in df.columns:
        d = df[df["event_type"].astype(str).str.lower() == et].copy()
    else:
        d = df.copy()

    handles = []
    pass_colors = pass_colors or {}
    pass_markers = pass_markers or {}
    shot_colors  = shot_colors  or {}
    shot_markers = shot_markers or {}

    if et in ["pass"] and "outcome" in d.columns:
        from charts import PASS_ORDER
        for outc in PASS_ORDER:
            sub = d[d["outcome"] == outc]
            if sub.empty: continue
            col = pass_colors.get(outc, theme.get("muted","#AAAAAA"))
            mk  = pass_markers.get(outc, "o")
            if show_arrows and "x2" in sub.columns and "y2" in sub.columns:
                valid = sub.dropna(subset=["x2","y2"])
                if not valid.empty:
                    vp.arrows(valid["x"], valid["y"], valid["x2"], valid["y2"],
                              ax=ax, color=col, width=1.8, headwidth=5, alpha=0.85)
            if not _is_no_marker(mk):
                vp.scatter(sub["x"], sub["y"], ax=ax, s=60, marker=mk,
                           color=col, edgecolors="white", linewidth=0.8, zorder=5)
                handles.append(Line2D([0],[0], marker=mk, color="none",
                    markerfacecolor=col, markeredgecolor="white", markersize=7, label=outc))

    elif et in ["shot"] and "outcome" in d.columns:
        for outc in SHOT_ORDER:
            sub = d[d["outcome"] == outc]
            if sub.empty: continue
            col = shot_colors.get(outc, theme.get("muted","#AAAAAA"))
            mk  = shot_markers.get(outc, "o")
            if not _is_no_marker(mk):
                vp.scatter(sub["x"], sub["y"], ax=ax, s=120, marker=mk,
                           color=col, edgecolors="white", linewidth=1.2, zorder=5)
                handles.append(Line2D([0],[0], marker=mk, color="none",
                    markerfacecolor=col, markeredgecolor="white", markersize=8, label=outc))
    else:
        vp.scatter(d["x"], d["y"], ax=ax, s=dot_size,
                   color=dot_color, edgecolors="white", linewidth=0.8, alpha=0.9, zorder=5)
        handles.append(Line2D([0],[0], marker="o", color="none",
            markerfacecolor=dot_color, markeredgecolor="white", markersize=7, label=et))

    if active_legend_items is not None:
        handles = [h for h in handles if h.get_label() in active_legend_items]

    if handles:
        leg = ax.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, -0.04), ncol=min(4, len(handles)),
                        frameon=True, facecolor=theme.get("panel","#111827"),
                        edgecolor=theme.get("lines","#333"), fontsize=9)
        for t in leg.get_texts(): t.set_color(text_col)

    ax.set_title(title, color=text_col, fontsize=16, weight="bold", pad=14)
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PROGRESSIVE CARRIES MAP
# ─────────────────────────────────────────────────────────────────────────────
def progressive_carries_map(
    df: pd.DataFrame,
    title: str = "Progressive Carries",
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    carry_color: str = "#FF9300",
    min_distance: float = 5.0,   # minimum x-distance to count as progressive
    overlay_img=None,
    img_x: float = 0.02, img_y: float = 0.88, img_w: float = 0.09, img_h: float = 0.09,
    active_legend_items: Optional[List[str]] = None,
    vertical_pitch: bool = False,
):
    """
    Draws arrows for carries that gain ≥ min_distance x-units.
    Requires: x, y, x2, y2 columns.
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width,
                       theme=theme, vertical_pitch=vertical_pitch)

    figsize = (8, 12) if vertical_pitch else (11, 7)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"].astype(str).str.lower() == "carry"].copy()
    for c in ["x","y","x2","y2"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x","y","x2","y2"])

    if not d.empty:
        prog = d[d["x2"] - d["x"] >= min_distance].copy()
        if not prog.empty:
            pitch.arrows(prog["x"], prog["y"], prog["x2"], prog["y2"],
                         ax=ax, color=carry_color, width=2.0, headwidth=6,
                         headlength=6, alpha=0.88, zorder=4)
            pitch.scatter(prog["x"], prog["y"], ax=ax, s=60,
                          color=carry_color, edgecolors="white",
                          linewidth=0.8, zorder=5)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=theme["text"], fontsize=16, weight="bold")

    handles = [Line2D([0],[0], color=carry_color, lw=2, label="Progressive carry",
                      marker="o", markerfacecolor=carry_color, markeredgecolor="white", markersize=6)]
    if active_legend_items is not None:
        handles = [h for h in handles if h.get_label() in active_legend_items]
    leg = ax.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(theme["text"])

    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PRESSURE MAP  (heatmap of pressing actions)
# ─────────────────────────────────────────────────────────────────────────────
def pressure_map(
    df: pd.DataFrame,
    title: str = "Pressure Map",
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    vertical_pitch: bool = False,
    pressure_col: str = "pressure",   # boolean/yes-no column
    overlay_img=None,
    img_x=0.02, img_y=0.88, img_w=0.09, img_h=0.09,
):
    """
    Heatmap of pressure events.
    If pressure_col exists: filters those rows. Otherwise uses all rows as pressures.
    Requires: x, y
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width,
                       theme=theme, vertical_pitch=vertical_pitch)
    figsize = (8, 12) if vertical_pitch else (11, 7)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme["bg"]); pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])

    d = df.copy()
    if pressure_col in d.columns:
        d = d[_yes_only(d[pressure_col])].copy()
    for c in ["x","y"]:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x","y"])

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    if not d.empty:
        try:
            pitch.kdeplot(d["x"], d["y"], ax=ax, fill=True, levels=50,
                          cmap="Reds", alpha=0.72)
        except Exception:
            pitch.scatter(d["x"], d["y"], ax=ax, s=40, alpha=0.5,
                          color="#FF4060", edgecolors="none")

    ax.set_xlim(-2, 102); ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=theme["text"], fontsize=16, weight="bold")
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6.  xG TIMELINE  (cumulative xG over minutes)
# ─────────────────────────────────────────────────────────────────────────────
def xg_timeline(
    df: pd.DataFrame,
    team_a: str = "Home",
    team_b: str = "Away",
    team_col: Optional[str] = None,
    minute_col: Optional[str] = None,
    xg_col: str = "xg",
    title: str = "xG Timeline",
    color_a: str = "#00C2FF",
    color_b: str = "#FF4060",
    theme_name: str = "The Athletic Dark",
    overlay_img=None,
    img_x=0.85, img_y=0.88, img_w=0.10, img_h=0.08,
):
    """
    Cumulative xG step-chart over match minutes.
    Requires: xg, minute (or time) column, team column (optional — if absent treats all as team_a).
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme["bg"]; text_col = theme["text"]; muted = theme["muted"]

    d = df[df["event_type"] == "shot"].copy() if "event_type" in df.columns else df.copy()

    # Find minute column
    min_c = minute_col
    if not min_c:
        for cand in ["minute","min","time","match_time","minute_label"]:
            if cand in d.columns: min_c = cand; break

    d[xg_col] = pd.to_numeric(d.get(xg_col, 0), errors="coerce").fillna(0)
    if min_c:
        d[min_c] = pd.to_numeric(d[min_c], errors="coerce").fillna(0)
        d = d.sort_values(min_c)
    else:
        d = d.reset_index(drop=True)
        d["_idx"] = d.index
        min_c = "_idx"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    if team_col and team_col in d.columns:
        for team_name, color in [(team_a, color_a), (team_b, color_b)]:
            td = d[d[team_col].astype(str) == str(team_name)].copy()
            if td.empty: continue
            minutes = [0] + td[min_c].tolist() + [90]
            cumxg   = [0] + td[xg_col].cumsum().tolist() + [td[xg_col].sum()]
            ax.step(minutes, cumxg, where="post", color=color, lw=2.5, label=team_name)
            ax.fill_between(minutes, 0, cumxg, step="post", color=color, alpha=0.15)
            for m, x in zip(td[min_c], td[xg_col].cumsum()):
                if x > 0:
                    ax.scatter([m], [x], s=60, color=color, edgecolors="white", lw=1, zorder=5)
    else:
        minutes = [0] + d[min_c].tolist() + [90]
        cumxg   = [0] + d[xg_col].cumsum().tolist() + [d[xg_col].sum()]
        ax.step(minutes, cumxg, where="post", color=color_a, lw=2.5, label=team_a)
        ax.fill_between(minutes, 0, cumxg, step="post", color=color_a, alpha=0.15)

    ax.axvline(45, color=muted, ls="--", lw=1, alpha=0.5)
    ax.set_title(title, color=text_col, fontsize=16, weight="bold")
    ax.set_xlabel("Minute", color=muted); ax.set_ylabel("Cumulative xG", color=muted)
    ax.tick_params(colors=muted)
    for sp in ax.spines.values(): sp.set_color(theme["lines"])
    leg = ax.legend(frameon=False)
    for t in leg.get_texts(): t.set_color(text_col)

    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PASSING NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def passing_network(
    df: pd.DataFrame,
    player_col: str = "player",
    recipient_col: str = "recipient",
    x_col: str = "avg_x",
    y_col: str = "avg_y",
    title: str = "Passing Network",
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    node_color: str = "#00C2FF",
    edge_color: str = "#AAAAAA",
    min_passes: int = 3,
    overlay_img=None,
    img_x=0.02, img_y=0.88, img_w=0.09, img_h=0.09,
):
    """
    Passing network chart.
    Requires two modes:
      Mode A (event-level): player, recipient columns + x, y
      Mode B (pre-aggregated): player, avg_x, avg_y, (plus passes_to_recipient, recipient)

    Required columns: player, x, y (start location).
    Optional but useful: recipient, x2, y2
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(theme["bg"]); pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])

    text_col = theme["text"]
    d = df.copy()
    passes = d[d["event_type"] == "pass"].copy() if "event_type" in d.columns else d.copy()

    if player_col not in passes.columns:
        ax.set_title("No player column found", color=text_col); return fig

    for c in ["x","y","x2","y2"]:
        if c in passes.columns: passes[c] = pd.to_numeric(passes[c], errors="coerce")

    # Average positions
    avg_pos = passes.groupby(player_col)[["x","y"]].mean().reset_index()
    touch_count = passes.groupby(player_col).size().reset_index(name="touches")
    avg_pos = avg_pos.merge(touch_count, on=player_col)

    # Pass pair counts
    if recipient_col in passes.columns and "x2" in passes.columns:
        pairs = passes.groupby([player_col, recipient_col]).size().reset_index(name="n_passes")
        pairs = pairs[pairs["n_passes"] >= min_passes]
    else:
        pairs = pd.DataFrame()

    # Draw edges
    if not pairs.empty:
        pos_map = avg_pos.set_index(player_col)[["x","y"]].to_dict("index")
        max_passes = pairs["n_passes"].max()
        for _, row in pairs.iterrows():
            p1 = pos_map.get(row[player_col]); p2 = pos_map.get(row[recipient_col])
            if p1 and p2:
                lw = 0.8 + (row["n_passes"] / max_passes) * 5.0
                alpha = 0.3 + (row["n_passes"] / max_passes) * 0.55
                ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]],
                        color=edge_color, lw=lw, alpha=alpha, zorder=3)

    # Draw nodes
    max_t = avg_pos["touches"].max()
    for _, row in avg_pos.iterrows():
        sz = 60 + (row["touches"] / max_t) * 800
        ax.scatter(row["x"], row["y"], s=sz, color=node_color,
                   edgecolors="white", linewidth=1.8, zorder=5)
        ax.text(row["x"], row["y"] + 2.8, str(row[player_col])[:10],
                ha="center", va="bottom", fontsize=8, color=text_col,
                weight="bold", zorder=6)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax.set_xlim(-2, 102); ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=text_col, fontsize=16, weight="bold")
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TAGGING TOOL PITCH DRAWING  (fixed — returns PIL Image with correct sizing)
# ─────────────────────────────────────────────────────────────────────────────
def draw_tagging_pitch(
    theme_name: str,
    pitch_mode: str,
    pitch_width: float,
    display_w: int,
    show_thirds: bool,
    events: list,
    start_point=None,
    current_marker="o",
    current_color="#22C55E",
    current_edge="#FFFFFF",
    current_size: int = 9,
) -> tuple:
    """
    Returns (PIL Image, img_w, img_h, y_max, pad).
    Fixed vertical sizing so the full pitch is always visible.
    """
    from PIL import Image as PilImage, ImageDraw as PilDraw
    import math as _math

    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    pad = 24

    # Correct aspect ratio: 105m × 68m standard pitch
    aspect = y_max / 100.0  # width/length ratio
    inner_w = display_w - 2 * pad
    inner_h = int(round(inner_w * aspect))
    display_h = inner_h + 2 * pad

    bg_col  = theme.get("pitch", "#1f5f3b")
    lc      = theme.get("pitch_lines", "#E6E6E6")
    lw_px   = 3

    img = PilImage.new("RGB", (display_w, display_h), bg_col)
    draw = PilDraw.Draw(img)

    def P(x, y):
        """pitch coords (0-100, 0-y_max) → pixel coords"""
        px = pad + int(round((float(x) / 100.0) * inner_w))
        py = pad + int(round((1.0 - float(y) / y_max) * inner_h))
        return px, py

    def line(x0, y0, x1, y1, color=None, width=None):
        draw.line([P(x0, y0), P(x1, y1)], fill=color or lc, width=width or lw_px)

    def rect(x0, y0, x1, y1):
        p0 = P(x0, y0); p1 = P(x1, y1)
        draw.rectangle([min(p0[0],p1[0]), min(p0[1],p1[1]),
                        max(p0[0],p1[0]), max(p0[1],p1[1])],
                        outline=lc, width=lw_px)

    # Pitch border
    rect(0, 0, 100, y_max)

    # Halfway line
    line(50, 0, 50, y_max)

    # Thirds
    if show_thirds:
        tc = theme.get("lines", "#446688")
        for tx in (100/3, 200/3):
            draw.line([P(tx, 0), P(tx, y_max)], fill=tc, width=2)

    mid = y_max / 2.0
    pa_w = y_max * 40.32 / 68.0; pa_l = 16.5 / 105.0 * 100.0
    sa_w = y_max * 18.32 / 68.0; sa_l = 5.5  / 105.0 * 100.0

    # Penalty boxes
    rect(0, mid - pa_w/2, pa_l, mid + pa_w/2)
    rect(100 - pa_l, mid - pa_w/2, 100, mid + pa_w/2)
    rect(0, mid - sa_w/2, sa_l, mid + sa_w/2)
    rect(100 - sa_l, mid - sa_w/2, 100, mid + sa_w/2)

    # Goals
    goal_w = y_max * 7.32 / 68.0
    for gx in [0, 100]:
        off = -10 if gx == 0 else 10
        draw.line([P(gx, mid - goal_w/2), (P(gx, mid - goal_w/2)[0] + off, P(gx, mid - goal_w/2)[1])], fill=lc, width=lw_px)
        draw.line([P(gx, mid + goal_w/2), (P(gx, mid + goal_w/2)[0] + off, P(gx, mid + goal_w/2)[1])], fill=lc, width=lw_px)

    # Centre circle (ellipse scaled to pitch aspect)
    cx, cy = P(50, mid)
    r_px_x = int(inner_w * 9.15 / 105.0)
    r_px_y = int(inner_h * 9.15 / 68.0)
    draw.ellipse([cx - r_px_x, cy - r_px_y, cx + r_px_x, cy + r_px_y], outline=lc, width=lw_px)
    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=lc)

    # Penalty spots
    for sx in [11.0/105.0*100.0, 100-11.0/105.0*100.0]:
        spx, spy = P(sx, mid)
        draw.ellipse([spx-3, spy-3, spx+3, spy+3], fill=lc)

    # ── Draw existing events ────────────────────────────────────────────────
    def ev_color(et):
        return {"pass":"#00C2FF","carry":"#FF9300","dribble":"#A78BFA","cross":"#FFD400",
                "shot":"#00FF6A","touch":"#FFFFFF","defensive action":"#FF4D4D",
                "recovery":"#EF4444"}.get(str(et).lower(), "#FFFFFF")

    def draw_arrow_px(p1, p2, color, width=5):
        draw.line([p1, p2], fill=color, width=width)
        dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
        ang = _math.atan2(dy, dx); sz = 14
        for a in (ang + _math.pi*0.82, ang - _math.pi*0.82):
            tip = (p2[0] + int(sz*_math.cos(a)), p2[1] + int(sz*_math.sin(a)))
            draw.polygon([p2, tip], fill=color)

    def dot(x, y, color, edge="#FFF", r=9, marker="o"):
        if marker is None or str(marker).lower() in {"none",""}: return
        px, py = P(x, y)
        box = [px-r, py-r, px+r, py+r]
        m = str(marker)
        if m == "s":
            draw.rectangle(box, fill=color, outline=edge, width=3)
        elif m == "D":
            draw.polygon([(px, py-r),(px+r,py),(px,py+r),(px-r,py)], fill=color)
            draw.line([(px,py-r),(px+r,py),(px,py+r),(px-r,py),(px,py-r)], fill=edge, width=2)
        elif m == "^":
            draw.polygon([(px,py-r),(px+r,py+r),(px-r,py+r)], fill=color)
            draw.line([(px,py-r),(px+r,py+r),(px-r,py+r),(px,py-r)], fill=edge, width=2)
        elif m == "v":
            draw.polygon([(px-r,py-r),(px+r,py-r),(px,py+r)], fill=color)
            draw.line([(px-r,py-r),(px+r,py-r),(px,py+r),(px-r,py-r)], fill=edge, width=2)
        elif m == "*":
            for dx2,dy2 in [(r,0),(0,r),(r,r),(-r,r)]:
                draw.line([(px-dx2,py-dy2),(px+dx2,py+dy2)], fill=edge, width=3)
            draw.ellipse([px-r//2,py-r//2,px+r//2,py+r//2], fill=color)
        elif m in {"+","x"}:
            if m=="+":
                draw.line([(px-r,py),(px+r,py)], fill=color, width=5)
                draw.line([(px,py-r),(px,py+r)], fill=color, width=5)
            else:
                draw.line([(px-r,py-r),(px+r,py+r)], fill=color, width=5)
                draw.line([(px-r,py+r),(px+r,py-r)], fill=color, width=5)
        else:
            draw.ellipse(box, fill=color, outline=edge, width=3)

    for ev in (events or []):
        try:
            et   = str(ev.get("event_type","")).lower()
            col  = str(ev.get("start_color", ev_color(et)) or ev_color(et))
            edge = str(ev.get("start_edge","#FFF") or "#FFF")
            mk   = ev.get("start_marker","o")
            sz   = int(float(ev.get("start_size",9) or 9))
            ac   = str(ev.get("arrow_color", col) or col)
            x, y = float(ev.get("x",0)), float(ev.get("y",0))
            x2, y2 = ev.get("x2"), ev.get("y2")
            if et in ["pass","carry","dribble","cross"] and x2 is not None and y2 is not None:
                try:
                    if not (pd.isna(x2) or pd.isna(y2)):
                        draw_arrow_px(P(x,y), P(float(x2),float(y2)), ac, width=5)
                except Exception:
                    pass
            dot(x, y, col, edge=edge, r=max(4, sz), marker=mk)
        except Exception:
            pass

    if start_point:
        sx, sy = start_point
        dot(sx, sy, current_color, edge=current_edge, r=int(current_size), marker=current_marker)

    return img, display_w, display_h, y_max, pad
