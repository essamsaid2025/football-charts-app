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
from charts import THEMES, make_pitch

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL COPIES of helpers (avoids importing private _ functions from charts.py)
# ─────────────────────────────────────────────────────────────────────────────
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
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=(0.5, -0.02),
                    ncol=min(4, len(handles)), frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(theme.get("text", "white"))


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA THEMES
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_THEMES: Dict[str, dict] = {
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


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 1. GOAL LOCATION MAP  (Opta Analyst style — Image 1 reference)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 2. GOAL MOUTH MAP  (CannonStats style — Image 2 reference)
# ─────────────────────────────────────────────────────────────────────────────
def goal_mouth_map(
    df: pd.DataFrame,
    title: str = "Team",
    subtitle: str = "Shots on Target Map",
    stats_row: Optional[Dict[str, str]] = None,
    theme_name: str = "Opta Light",
    goal_color: str = "#7A2232",
    save_color: str = "#FFFFFF",
    goal_edge: str = "#3D0A18",
    save_edge: str = "#1A2E5A",
    size_by_xg: bool = True,
    base_size: int = 600,
    logo_img=None,
    logo_x: float = 0.84, logo_y: float = 0.88,
    logo_w: float = 0.12, logo_h: float = 0.10,
    footer_left: str = "",
    footer_right: str = "",
):
    """
    CannonStats-style goal mouth map.
    Required columns: outcome (goal/ontarget/save/saved), y
    Optional: xg, z/height (for vertical shot position)
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme.get("bg", "#F0F0F0")
    text_col = theme.get("text", "#1A1A1A")
    muted_col = theme.get("muted", "#666666")

    df = df.copy()
    on_target = df[df["outcome"].astype(str).str.lower().isin(
        ["goal", "ontarget", "save", "saved"])].copy()

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(bg)

    # ── Stats strip ───────────────────────────────────────────────────────────
    ax_top = fig.add_axes([0.0, 0.74, 1.0, 0.22])
    ax_top.set_facecolor(bg)
    ax_top.axis("off")
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    fig.text(0.04, 0.97, title, fontsize=38, weight="900",
             color=text_col, va="top", fontfamily="serif")
    fig.text(0.04, 0.84, subtitle, fontsize=13, color=muted_col, va="top")

    if stats_row:
        keys = list(stats_row.keys())
        n = len(keys)
        xs = np.linspace(0.25, 0.78, n)
        for i, (k, v) in enumerate(stats_row.items()):
            ax_top.text(xs[i], 0.78, f"{k}:", fontsize=12, weight="700",
                        color=text_col, ha="center", va="top")
            ax_top.text(xs[i], 0.35, str(v), fontsize=32, weight="900",
                        color=text_col, ha="center", va="top")

    ax_top.plot([0.02, 0.98], [0.03, 0.03], color=theme.get("lines", "#CCCCCC"),
                lw=1.5, transform=ax_top.transAxes)

    # ── Goal face ─────────────────────────────────────────────────────────────
    ax_g = fig.add_axes([0.06, 0.14, 0.88, 0.58])
    ax_g.set_facecolor(bg)
    ax_g.axis("off")
    ax_g.set_xlim(0, 1)
    ax_g.set_ylim(0, 1)

    frame = FancyBboxPatch((0.04, 0.04), 0.92, 0.92,
                            boxstyle="square,pad=0", linewidth=0,
                            facecolor="#888888", zorder=1)
    ax_g.add_patch(frame)

    interior = FancyBboxPatch((0.09, 0.09), 0.82, 0.82,
                               boxstyle="square,pad=0", linewidth=2,
                               edgecolor="#555555", facecolor="#F5F5F5", zorder=2)
    ax_g.add_patch(interior)

    for gx in np.linspace(0.09, 0.91, 22):
        ax_g.plot([gx, gx], [0.09, 0.91], color="#DDDDDD", lw=0.45, alpha=0.55, zorder=3)
    for gy in np.linspace(0.09, 0.91, 10):
        ax_g.plot([0.09, 0.91], [gy, gy], color="#DDDDDD", lw=0.45, alpha=0.55, zorder=3)

    if not on_target.empty:
        y_max_p = 68.0
        y_vals = pd.to_numeric(on_target.get("y", 34.0), errors="coerce").fillna(34.0)
        gx_norm = 0.09 + (y_vals / y_max_p) * 0.82

        z_col = next((c for c in ["z", "height", "shot_height", "goal_height"] if c in on_target.columns), None)
        if z_col:
            z_raw = pd.to_numeric(on_target[z_col], errors="coerce").fillna(1.2)
            z_norm = (z_raw / 2.44).clip(0.08, 0.90)
        else:
            rng = np.random.default_rng(42)
            z_norm = pd.Series(rng.uniform(0.18, 0.78, len(on_target)), index=on_target.index)

        for i, (idx, row) in enumerate(on_target.iterrows()):
            outc = str(row.get("outcome", "")).lower()
            is_goal = outc == "goal"
            color = goal_color if is_goal else save_color
            edge  = goal_edge  if is_goal else save_edge

            if size_by_xg and "xg" in row.index and pd.notna(row.get("xg")):
                sz = max(100, min(2200, float(row["xg"]) * 3500))
            else:
                sz = base_size if is_goal else int(base_size * 0.75)

            ax_g.scatter([float(gx_norm.iloc[i])], [float(z_norm.iloc[i])],
                         s=sz, color=color, edgecolors=edge,
                         linewidth=2.5 if not is_goal else 0,
                         zorder=6, alpha=0.92)

    # ── Legend strip ──────────────────────────────────────────────────────────
    ax_l = fig.add_axes([0.0, 0.0, 1.0, 0.14])
    ax_l.set_facecolor(bg)
    ax_l.axis("off")
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)

    ax_l.text(0.14, 0.88, "Shot Outcome:", fontsize=12, weight="700",
              color=text_col, ha="center", va="top")
    ax_l.scatter([0.06], [0.38], s=350, color=save_color, edgecolors=save_edge, linewidth=2.5, zorder=5)
    ax_l.text(0.09, 0.38, "Save", fontsize=11, color=text_col, va="center")
    ax_l.scatter([0.18], [0.38], s=550, color=goal_color, edgecolors=goal_edge, linewidth=0, zorder=5)
    ax_l.text(0.21, 0.38, "Goal", fontsize=11, color=text_col, va="center")

    if size_by_xg:
        ax_l.text(0.65, 0.88, "Post-Shot xG Value:", fontsize=12, weight="700",
                  color=text_col, ha="center", va="top")
        for xgv, xgx in zip([0.03, 0.08, 0.18, 0.35, 0.65], np.linspace(0.52, 0.80, 5)):
            ax_l.scatter([xgx], [0.38], s=max(25, xgv * 3500 * 0.28),
                         color=muted_col, edgecolors=text_col, linewidth=1, alpha=0.7)

    if footer_left:
        ax_l.text(0.02, 0.06, footer_left, fontsize=9, color=muted_col, va="bottom")
    if footer_right:
        ax_l.text(0.98, 0.06, footer_right, fontsize=9, color=muted_col,
                  va="bottom", ha="right")

    overlay_image_on_fig(fig, logo_img, x=logo_x, y=logo_y, w=logo_w, h=logo_h,
                         circle_crop=True, border_lw=2.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. VERTICAL EVENT MAP  (fully correct sizing — no clipping)
# ─────────────────────────────────────────────────────────────────────────────
def vertical_event_map(
    df: pd.DataFrame,
    event_type: str = "pass",
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
    img_x=0.02, img_y=0.88, img_w=0.10, img_h=0.10,
    active_legend_items: Optional[List[str]] = None,
):
    """Full vertical pitch — correct aspect ratio, no clipping."""
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pc = theme.get("pitch", "#1f5f3b")
    lc = theme.get("pitch_lines", "#FFFFFF")
    stripe = theme.get("pitch_stripe")

    vp = VerticalPitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        pitch_color=pc,
        line_color=lc,
        line_zorder=2,
        stripe=bool(stripe),
        stripe_color=stripe or pc,
    )
    fig, ax = vp.draw(figsize=(8, 12))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(pc)

    text_col = theme.get("text", "white")
    et = (event_type or "all").lower()

    if et == "all":
        d = df.copy()
    elif "event_type" in df.columns:
        d = df[df["event_type"].astype(str).str.lower() == et].copy()
    else:
        d = df.copy()

    pass_colors  = pass_colors  or {}
    pass_markers = pass_markers or {}
    shot_colors  = shot_colors  or {}
    shot_markers = shot_markers or {}
    handles = []

    if et == "pass" and "outcome" in d.columns:
        for outc in PASS_ORDER_X:
            sub = d[d["outcome"] == outc]
            if sub.empty:
                continue
            col = pass_colors.get(outc, theme.get("muted", "#AAAAAA"))
            mk  = pass_markers.get(outc, "o")
            if show_arrows and "x2" in sub.columns and "y2" in sub.columns:
                valid = sub.dropna(subset=["x2", "y2"])
                if not valid.empty:
                    vp.arrows(valid["x"], valid["y"], valid["x2"], valid["y2"],
                              ax=ax, color=col, width=1.8, headwidth=5, alpha=0.85)
            if not _is_no_mk(mk):
                vp.scatter(sub["x"], sub["y"], ax=ax, s=60, marker=mk,
                           color=col, edgecolors="white", linewidth=0.8, zorder=5)
                handles.append(Line2D([0], [0], marker=mk, color="none",
                    markerfacecolor=col, markeredgecolor="white", markersize=7, label=outc))

    elif et == "shot" and "outcome" in d.columns:
        for outc in SHOT_ORDER_X:
            sub = d[d["outcome"] == outc]
            if sub.empty:
                continue
            col = shot_colors.get(outc, theme.get("muted", "#AAAAAA"))
            mk  = shot_markers.get(outc, "o")
            if not _is_no_mk(mk):
                vp.scatter(sub["x"], sub["y"], ax=ax, s=120, marker=mk,
                           color=col, edgecolors="white", linewidth=1.2, zorder=5)
                handles.append(Line2D([0], [0], marker=mk, color="none",
                    markerfacecolor=col, markeredgecolor="white", markersize=8, label=outc))
    else:
        vp.scatter(d["x"], d["y"], ax=ax, s=dot_size,
                   color=dot_color, edgecolors="white", linewidth=0.8, alpha=0.9, zorder=5)
        handles.append(Line2D([0], [0], marker="o", color="none",
            markerfacecolor=dot_color, markeredgecolor="white", markersize=7, label=et))

    if active_legend_items is not None:
        handles = [h for h in handles if h.get_label() in active_legend_items]

    if handles:
        leg = ax.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, -0.04), ncol=min(4, len(handles)),
                        frameon=True, facecolor=theme.get("panel", "#111827"),
                        edgecolor=theme.get("lines", "#333"), fontsize=9)
        for t in leg.get_texts():
            t.set_color(text_col)

    ax.set_title(title, color=text_col, fontsize=16, weight="bold", pad=14)
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    fig.tight_layout(pad=1.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. PROGRESSIVE CARRIES MAP
# ─────────────────────────────────────────────────────────────────────────────
def progressive_carries_map(
    df: pd.DataFrame,
    title: str = "Progressive Carries",
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    carry_color: str = "#FF9300",
    min_distance: float = 5.0,
    overlay_img=None,
    img_x=0.02, img_y=0.88, img_w=0.09, img_h=0.09,
    active_legend_items: Optional[List[str]] = None,
    vertical_pitch: bool = False,
):
    """Progressive carries arrow map. Requires: x, y, x2, y2."""
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
    for c in ["x", "y", "x2", "y2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x", "y", "x2", "y2"])

    if not d.empty:
        prog = d[d["x2"] - d["x"] >= min_distance].copy()
        if not prog.empty:
            pitch.arrows(prog["x"], prog["y"], prog["x2"], prog["y2"],
                         ax=ax, color=carry_color, width=2.0,
                         headwidth=6, headlength=6, alpha=0.88, zorder=4)
            pitch.scatter(prog["x"], prog["y"], ax=ax, s=55,
                          color=carry_color, edgecolors="white",
                          linewidth=0.8, zorder=5)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=theme["text"], fontsize=16, weight="bold")

    handles = [Line2D([0], [0], color=carry_color, lw=2, label="Progressive carry",
                      marker="o", markerfacecolor=carry_color,
                      markeredgecolor="white", markersize=6)]
    if active_legend_items is not None:
        handles = [h for h in handles if h.get_label() in active_legend_items]
    _add_leg(ax, handles, theme, loc="lower center")

    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRESSURE MAP
# ─────────────────────────────────────────────────────────────────────────────
def pressure_map(
    df: pd.DataFrame,
    title: str = "Pressure Map",
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 68.0,
    vertical_pitch: bool = False,
    pressure_col: str = "pressure",
    overlay_img=None,
    img_x=0.02, img_y=0.88, img_w=0.09, img_h=0.09,
):
    """Pressure heatmap. Optional pressure column (bool/yes-no)."""
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width,
                       theme=theme, vertical_pitch=vertical_pitch)
    figsize = (8, 12) if vertical_pitch else (11, 7)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])

    d = df.copy()
    if pressure_col in d.columns:
        d = d[_yes_col(d[pressure_col])].copy()
    for c in ["x", "y"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x", "y"])

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    if not d.empty:
        try:
            pitch.kdeplot(d["x"], d["y"], ax=ax, fill=True,
                          levels=50, cmap="Reds", alpha=0.72)
        except Exception:
            pitch.scatter(d["x"], d["y"], ax=ax, s=40, alpha=0.5,
                          color="#FF4060", edgecolors="none")

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=theme["text"], fontsize=16, weight="bold")
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. xG TIMELINE
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
    """Cumulative xG step chart over match minutes. Requires xg column."""
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    bg = theme["bg"]
    text_col = theme["text"]
    muted = theme["muted"]

    d = df[df["event_type"] == "shot"].copy() if "event_type" in df.columns else df.copy()

    min_c = minute_col
    if not min_c:
        for cand in ["minute", "min", "time", "match_time"]:
            if cand in d.columns:
                min_c = cand
                break

    d[xg_col] = pd.to_numeric(d.get(xg_col, pd.Series(0, index=d.index)), errors="coerce").fillna(0)
    if min_c and min_c in d.columns:
        d[min_c] = pd.to_numeric(d[min_c], errors="coerce").fillna(0)
        d = d.sort_values(min_c)
    else:
        d = d.reset_index(drop=True)
        d["_idx"] = d.index
        min_c = "_idx"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    if team_col and team_col in d.columns:
        for team_name, color in [(team_a, color_a), (team_b, color_b)]:
            td = d[d[team_col].astype(str) == str(team_name)].copy()
            if td.empty:
                continue
            minutes = [0] + td[min_c].tolist() + [90]
            cumxg   = [0] + td[xg_col].cumsum().tolist() + [td[xg_col].sum()]
            ax.step(minutes, cumxg, where="post", color=color, lw=2.5, label=team_name)
            ax.fill_between(minutes, 0, cumxg, step="post", color=color, alpha=0.15)
            for m, x in zip(td[min_c], td[xg_col].cumsum()):
                if x > 0:
                    ax.scatter([m], [x], s=55, color=color,
                               edgecolors="white", lw=1, zorder=5)
    else:
        minutes = [0] + d[min_c].tolist() + [90]
        cumxg   = [0] + d[xg_col].cumsum().tolist() + [d[xg_col].sum()]
        ax.step(minutes, cumxg, where="post", color=color_a, lw=2.5, label=team_a)
        ax.fill_between(minutes, 0, cumxg, step="post", color=color_a, alpha=0.15)

    ax.axvline(45, color=muted, ls="--", lw=1, alpha=0.5)
    ax.set_title(title, color=text_col, fontsize=16, weight="bold")
    ax.set_xlabel("Minute", color=muted)
    ax.set_ylabel("Cumulative xG", color=muted)
    ax.tick_params(colors=muted)
    for sp in ax.spines.values():
        sp.set_color(theme.get("lines", "#2A3240"))
    leg = ax.legend(frameon=False)
    for t in leg.get_texts():
        t.set_color(text_col)

    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. PASSING NETWORK
# ─────────────────────────────────────────────────────────────────────────────
def passing_network(
    df: pd.DataFrame,
    player_col: str = "player",
    recipient_col: str = "recipient",
    x_col: str = "x",
    y_col: str = "y",
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
    Passing network. Requires: player col + x, y.
    Optional: recipient col for edges.
    """
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(theme["bg"])
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])
    text_col = theme["text"]

    d = df.copy()
    if "event_type" in d.columns:
        d = d[d["event_type"].astype(str).str.lower() == "pass"].copy()

    if player_col not in d.columns:
        ax.set_title("No player column found", color=text_col)
        return fig

    for c in ["x", "y", "x2", "y2"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    avg_pos = d.groupby(player_col)[["x", "y"]].mean().reset_index()
    touch_c = d.groupby(player_col).size().reset_index(name="touches")
    avg_pos = avg_pos.merge(touch_c, on=player_col)

    if recipient_col in d.columns:
        pairs = d.groupby([player_col, recipient_col]).size().reset_index(name="n")
        pairs = pairs[pairs["n"] >= min_passes]
    else:
        pairs = pd.DataFrame()

    if not pairs.empty:
        pos_map = avg_pos.set_index(player_col)[["x", "y"]].to_dict("index")
        max_n = pairs["n"].max()
        for _, row in pairs.iterrows():
            p1 = pos_map.get(row[player_col])
            p2 = pos_map.get(row[recipient_col])
            if p1 and p2:
                lw = 0.8 + (row["n"] / max_n) * 5.0
                alpha = 0.3 + (row["n"] / max_n) * 0.55
                ax.plot([p1["x"], p2["x"]], [p1["y"], p2["y"]],
                        color=edge_color, lw=lw, alpha=alpha, zorder=3)

    max_t = avg_pos["touches"].max()
    for _, row in avg_pos.iterrows():
        sz = 60 + (row["touches"] / max_t) * 800
        ax.scatter(row["x"], row["y"], s=sz, color=node_color,
                   edgecolors="white", linewidth=1.8, zorder=5)
        ax.text(row["x"], row["y"] + 2.8, str(row[player_col])[:10],
                ha="center", va="bottom", fontsize=8,
                color=text_col, weight="bold", zorder=6)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, y_max + 2)
    ax.set_title(title, color=text_col, fontsize=16, weight="bold")
    overlay_image_on_fig(fig, overlay_img, x=img_x, y=img_y, w=img_w, h=img_h)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. DRAW TAGGING PITCH  (PIL-based, fixed sizing)
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
    Full pitch correctly sized — no clipping.
    """
    from PIL import Image as PilImage, ImageDraw as PilDraw

    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    y_max = float(pitch_width if pitch_mode == "rect" else 100.0)
    pad = 24

    # Correct aspect: pitch is 105m long × pitch_width m wide
    # x-axis = 0..100 (length), y-axis = 0..y_max (width)
    inner_w = display_w - 2 * pad
    inner_h = int(round(inner_w * (y_max / 100.0)))
    display_h = inner_h + 2 * pad

    bg_col = theme.get("pitch", "#1f5f3b")
    lc     = theme.get("pitch_lines", "#E6E6E6")
    lw_px  = 3

    img  = PilImage.new("RGB", (display_w, display_h), bg_col)
    draw = PilDraw.Draw(img)

    def P(x, y):
        """pitch (0-100, 0-y_max) → pixel"""
        px2 = pad + int(round((float(x) / 100.0) * inner_w))
        py2 = pad + int(round((1.0 - float(y) / y_max) * inner_h))
        return px2, py2

    def line(x0, y0, x1, y1, color=None, wid=None):
        draw.line([P(x0, y0), P(x1, y1)], fill=color or lc, width=wid or lw_px)

    def box(x0, y0, x1, y1):
        p0, p1 = P(x0, y0), P(x1, y1)
        draw.rectangle([min(p0[0], p1[0]), min(p0[1], p1[1]),
                        max(p0[0], p1[0]), max(p0[1], p1[1])],
                       outline=lc, width=lw_px)

    # Border
    box(0, 0, 100, y_max)
    # Halfway line
    line(50, 0, 50, y_max)

    mid = y_max / 2.0

    # Thirds
    if show_thirds:
        tc = theme.get("lines", "#446688")
        for tx in (100 / 3, 200 / 3):
            draw.line([P(tx, 0), P(tx, y_max)], fill=tc, width=2)

    # Penalty boxes
    pa_w = y_max * 40.32 / 68.0; pa_l = 16.5 / 105.0 * 100.0
    sa_w = y_max * 18.32 / 68.0; sa_l = 5.5  / 105.0 * 100.0
    box(0, mid - pa_w / 2, pa_l, mid + pa_w / 2)
    box(100 - pa_l, mid - pa_w / 2, 100, mid + pa_w / 2)
    box(0, mid - sa_w / 2, sa_l, mid + sa_w / 2)
    box(100 - sa_l, mid - sa_w / 2, 100, mid + sa_w / 2)

    # Goals
    goal_w = y_max * 7.32 / 68.0
    for gx in [0, 100]:
        off = -10 if gx == 0 else 10
        p1g = P(gx, mid - goal_w / 2)
        p2g = P(gx, mid + goal_w / 2)
        draw.line([p1g, (p1g[0] + off, p1g[1])], fill=lc, width=lw_px)
        draw.line([p2g, (p2g[0] + off, p2g[1])], fill=lc, width=lw_px)

    # Centre circle
    cx, cy = P(50, mid)
    rx = int(inner_w * 9.15 / 100.0)
    ry = int(inner_h * 9.15 / y_max)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=lc, width=lw_px)
    draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=lc)

    # Penalty spots
    for sx in [11.0 / 105.0 * 100.0, 100 - 11.0 / 105.0 * 100.0]:
        spx, spy = P(sx, mid)
        draw.ellipse([spx - 3, spy - 3, spx + 3, spy + 3], fill=lc)

    # ── Draw events ──────────────────────────────────────────────────────────
    def _ev_col(et):
        return {"pass": "#00C2FF", "carry": "#FF9300", "dribble": "#A78BFA",
                "cross": "#FFD400", "shot": "#00FF6A", "touch": "#FFFFFF",
                "defensive action": "#FF4D4D", "recovery": "#EF4444"}.get(str(et).lower(), "#FFFFFF")

    def _arrow(s, e, color, width=5):
        draw.line([s, e], fill=color, width=width)
        dx = e[0] - s[0]; dy = e[1] - s[1]
        ang = math.atan2(dy, dx); sz = 14
        for a in (ang + math.pi * 0.82, ang - math.pi * 0.82):
            tip = (e[0] + int(sz * math.cos(a)), e[1] + int(sz * math.sin(a)))
            draw.polygon([e, tip], fill=color)

    def _dot(x, y, color, edge="#FFF", r=9, marker="o"):
        if marker is None or str(marker).lower() in {"none", ""}:
            return
        ppx, ppy = P(x, y)
        box2 = [ppx - r, ppy - r, ppx + r, ppy + r]
        m = str(marker)
        if m == "s":
            draw.rectangle(box2, fill=color, outline=edge, width=3)
        elif m == "D":
            draw.polygon([(ppx, ppy - r), (ppx + r, ppy), (ppx, ppy + r), (ppx - r, ppy)],
                         fill=color)
            draw.line([(ppx, ppy - r), (ppx + r, ppy), (ppx, ppy + r), (ppx - r, ppy), (ppx, ppy - r)],
                      fill=edge, width=2)
        elif m == "^":
            draw.polygon([(ppx, ppy - r), (ppx + r, ppy + r), (ppx - r, ppy + r)], fill=color)
            draw.line([(ppx, ppy - r), (ppx + r, ppy + r), (ppx - r, ppy + r), (ppx, ppy - r)],
                      fill=edge, width=2)
        elif m == "v":
            draw.polygon([(ppx - r, ppy - r), (ppx + r, ppy - r), (ppx, ppy + r)], fill=color)
            draw.line([(ppx - r, ppy - r), (ppx + r, ppy - r), (ppx, ppy + r), (ppx - r, ppy - r)],
                      fill=edge, width=2)
        elif m == "*":
            for ddx, ddy in [(r, 0), (0, r), (r, r), (-r, r)]:
                draw.line([(ppx - ddx, ppy - ddy), (ppx + ddx, ppy + ddy)], fill=edge, width=3)
            draw.ellipse([ppx - r // 2, ppy - r // 2, ppx + r // 2, ppy + r // 2], fill=color)
        elif m in {"+", "x"}:
            if m == "+":
                draw.line([(ppx - r, ppy), (ppx + r, ppy)], fill=color, width=5)
                draw.line([(ppx, ppy - r), (ppx, ppy + r)], fill=color, width=5)
            else:
                draw.line([(ppx - r, ppy - r), (ppx + r, ppy + r)], fill=color, width=5)
                draw.line([(ppx - r, ppy + r), (ppx + r, ppy - r)], fill=color, width=5)
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

    if start_point is not None:
        try:
            sx, sy = start_point
            _dot(float(sx), float(sy), current_color, edge=current_edge,
                 r=int(current_size), marker=current_marker)
        except Exception:
            pass

    return img, display_w, display_h, y_max, pad
