# ===========================
# Football Charts Generator
# (Full code with fixes 1-2-3-4)
# ===========================

import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza

# ----------------------------
# Constants
# ----------------------------
PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]

# Shots: add blocked
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)

REQUIRED = ["outcome", "x", "y"]  # x2,y2 optional

# ----------------------------
# Theme System (optional use)
# ----------------------------
THEMES = {
    "Opta Dark": {
        "bg": "#0E1117",
        "panel": "#141A22",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
    },
    "Sofa Light": {
        "bg": "white",
        "panel": "#F5F7FA",
        "pitch": "#2f6b3a",
        "text": "#111111",
        "muted": "#5A6572",
        "lines": "#DDE3EA",
        "goal": "#444444",
    },
    "StatsBomb Dark": {
        "bg": "#111111",
        "panel": "#1B1B1B",
        "pitch": "#245a3a",
        "text": "white",
        "muted": "#B3B3B3",
        "lines": "#2E2E2E",
        "goal": "#DDDDDD",
    },
}

# ----------------------------
# Outcome Normalization (Fix #1)
# ----------------------------
def _norm_outcome(s: str) -> str:
    """
    Fixes cases like '1ontarget', weird separators, typos.
    Normalizes to:
      - shots: goal / ontarget / off target / blocked
      - passes: successful / unsuccessful / key pass / assist
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""

    s = str(s).strip().lower()

    # remove leading digits like "1ontarget"
    s = re.sub(r"^\d+", "", s).strip()

    # normalize separators
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)

    aliases = {
        # shots
        "on target": "ontarget",
        "ontarget": "ontarget",
        "offtarget": "off target",
        "off target": "off target",
        "block": "blocked",
        "blocked shot": "blocked",
        "blk": "blocked",
        # passes
        "keypass": "key pass",
        "key pass": "key pass",
        "assist": "assist",
        "successful": "successful",
        "unsuccessful": "unsuccessful",
        "unsucssesful": "unsuccessful",
        "unsuccessfull": "unsuccessful",
        "un successful": "unsuccessful",
        "un-successful": "unsuccessful",
    }
    return aliases.get(s, s)


# ----------------------------
# Load data
# ----------------------------
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1256", "cp1252", "latin1", "utf-16"]
        for enc in encodings_to_try:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        return pd.read_csv(path, encoding="latin1", encoding_errors="replace")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported file type. Use CSV or Excel.")


# ----------------------------
# Validate & Clean (Fix #1)
# ----------------------------
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")

    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()

    # classify
    df["event_type"] = "pass"
    df.loc[df["outcome"].isin(SHOT_TYPES), "event_type"] = "shot"

    # drop weird outcomes per type (prevents "outcome not showing")
    df_pass = df[df["event_type"] == "pass"].copy()
    df_shot = df[df["event_type"] == "shot"].copy()

    if not df_pass.empty:
        df_pass = df_pass[df_pass["outcome"].isin(PASS_ORDER)]
    if not df_shot.empty:
        df_shot = df_shot[df_shot["outcome"].isin(SHOT_ORDER)]

    return pd.concat([df_pass, df_shot], ignore_index=True)


# ----------------------------
# Pitch transforms
# ----------------------------
def apply_pitch_transforms(
    df: pd.DataFrame,
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
) -> pd.DataFrame:
    df = df.copy()

    # Flip Y in 0-100 space
    if flip_y:
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    # Flip X in 0-100 space
    if attack_direction == "rtl":
        for c in ["x", "x2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    # Scale Y for rect
    if pitch_mode == "rect":
        scale = pitch_width / 100.0
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = df[c] * scale

    return df


def make_pitch(pitch_mode: str = "rect", pitch_width: float = 64.0):
    if pitch_mode == "square":
        return Pitch(
            pitch_type="custom",
            pitch_length=100,
            pitch_width=100,
            line_zorder=2,
        )

    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        line_zorder=2,
    )


# ----------------------------
# Helpers for goal mouth
# ----------------------------
def _goal_mouth_bounds(pitch_mode="rect", pitch_width=64.0):
    gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    # goal width ratio ~7.32/68 = 0.10765
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    return gy - goal_mouth / 2.0, gy + goal_mouth / 2.0


# ----------------------------
# Zone-based xG (Fix #2)
# ----------------------------
def _shot_angle_and_distance(x: float, y: float, pitch_mode: str, pitch_width: float):
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0

    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    left_post_y = goal_y - goal_mouth / 2.0
    right_post_y = goal_y + goal_mouth / 2.0

    dx = goal_x - x
    dy = goal_y - y
    distance = math.sqrt(dx * dx + dy * dy) + 1e-9

    a = math.atan2(right_post_y - y, goal_x - x)
    b = math.atan2(left_post_y - y, goal_x - x)
    angle = abs(a - b)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle, distance


def _meters_distance_approx(x, y, pitch_mode="rect", pitch_width=64.0):
    # rough mapping to meters to bin distance like Sofa/Opta-ish
    length_m = 105.0
    width_m = 68.0 if pitch_mode == "rect" else 105.0
    y_max = pitch_width if pitch_mode == "rect" else 100.0

    xm = (x / 100.0) * length_m
    ym = (y / y_max) * width_m
    goal_xm = length_m
    goal_ym = width_m / 2.0

    dx = goal_xm - xm
    dy = goal_ym - ym
    return math.sqrt(dx * dx + dy * dy)


def zone_based_xg(x, y, pitch_mode="rect", pitch_width=64.0):
    angle, _ = _shot_angle_and_distance(x, y, pitch_mode, pitch_width)
    dist_m = _meters_distance_approx(x, y, pitch_mode, pitch_width)

    # Angle bins
    if angle < 0.35:
        a_bin = "small"
    elif angle < 0.75:
        a_bin = "mid"
    else:
        a_bin = "big"

    # Distance bins
    if dist_m <= 6:
        d_bin = "0-6"
    elif dist_m <= 12:
        d_bin = "6-12"
    elif dist_m <= 18:
        d_bin = "12-18"
    elif dist_m <= 25:
        d_bin = "18-25"
    else:
        d_bin = "25+"

    # tuned lookup table (reasonable Sofa-like)
    table = {
        ("0-6", "big"): 0.55,
        ("0-6", "mid"): 0.45,
        ("0-6", "small"): 0.32,
        ("6-12", "big"): 0.32,
        ("6-12", "mid"): 0.22,
        ("6-12", "small"): 0.12,
        ("12-18", "big"): 0.18,
        ("12-18", "mid"): 0.10,
        ("12-18", "small"): 0.05,
        ("18-25", "big"): 0.08,
        ("18-25", "mid"): 0.05,
        ("18-25", "small"): 0.03,
        ("25+", "big"): 0.04,
        ("25+", "mid"): 0.025,
        ("25+", "small"): 0.015,
    }
    xg = table.get((d_bin, a_bin), 0.02)
    return float(max(0.01, min(0.85, xg)))


def estimate_xg_from_location(x: float, y: float, pitch_mode: str = "rect", pitch_width: float = 64.0) -> float:
    # Zone-based (no logistic)
    return zone_based_xg(x, y, pitch_mode=pitch_mode, pitch_width=pitch_width)


def estimate_xg_for_shots(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0) -> pd.DataFrame:
    df = df.copy()
    if "xg" not in df.columns:
        df["xg"] = pd.NA

    mask = df["event_type"] == "shot"
    if mask.any():
        xs = df.loc[mask, "x"].astype(float)
        ys = df.loc[mask, "y"].astype(float)
        df.loc[mask, "xg"] = [
            round(estimate_xg_from_location(x, y, pitch_mode=pitch_mode, pitch_width=pitch_width), 2)
            for x, y in zip(xs, ys)
        ]
    return df


# ----------------------------
# End location rendering (Fix #3)
# ----------------------------
def fix_shot_end_location(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    """
    Opta-ish:
      - goal/ontarget end at goal line x2=100 and y2 within goal mouth (clamped)
      - off target pushes outside mouth
      - blocked short deflection if missing
    """
    df = df.copy()
    if "x2" not in df.columns:
        df["x2"] = pd.NA
    if "y2" not in df.columns:
        df["y2"] = pd.NA

    y_low, y_high = _goal_mouth_bounds(pitch_mode, pitch_width)
    mid = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0

    s_mask = df["event_type"] == "shot"
    for i, r in df.loc[s_mask].iterrows():
        outc = str(r.get("outcome", "")).lower()
        x, y = float(r["x"]), float(r["y"])

        x2 = r.get("x2")
        y2 = r.get("y2")

        if pd.isna(x2) or pd.isna(y2):
            if outc in ("goal", "ontarget"):
                x2 = 100.0
                y2 = mid
            elif outc == "blocked":
                x2 = min(100.0, x + 3.0)
                y2 = y
            else:  # off target
                x2 = 100.5
                y2 = (y_high + 2.0) if (y > mid) else (y_low - 2.0)
        else:
            x2 = float(x2)
            y2 = float(y2)

        # clamps
        if outc == "goal":
            x2 = 100.0
            y2 = max(y_low, min(y_high, y2))
        elif outc == "ontarget":
            x2 = 100.0
            y2 = max(y_low - 1.0, min(y_high + 1.0, y2))
        elif outc == "off target":
            x2 = max(100.0, x2)
        elif outc == "blocked":
            x2 = min(100.0, x2)

        df.at[i, "x2"] = x2
        df.at[i, "y2"] = y2

    return df


# ----------------------------
# Charts
# ----------------------------
def outcome_bar(df: pd.DataFrame, title: str = "", bar_colors: dict | None = None):
    bar_colors = bar_colors or {}
    counts = df["outcome"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [bar_colors.get(k, None) for k in counts.index.astype(str)]
    ax.bar(counts.index.astype(str), counts.values, color=colors)

    ax.set_title((title + "\nOutcome Distribution").strip())
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    return fig


def start_location_heatmap(df: pd.DataFrame, title: str = "", pitch_mode: str = "rect", pitch_width: float = 64.0):
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    pitch.kdeplot(df["x"], df["y"], ax=ax, fill=True, levels=50)
    ax.set_title((title + "\nStart Locations Heatmap").strip())
    return fig


def pass_map(
    df: pd.DataFrame,
    title: str = "",
    pass_colors: dict | None = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
):
    pass_colors = pass_colors or {}
    d = df[df["event_type"] == "pass"].copy()

    if not {"x2", "y2"}.issubset(d.columns):
        d["x2"] = pd.NA
        d["y2"] = pd.NA
    d = d.dropna(subset=["x2", "y2"])

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))

    for t in PASS_ORDER:
        dt = d[d["outcome"] == t]
        if len(dt) == 0:
            continue
        pitch.arrows(
            dt["x"],
            dt["y"],
            dt["x2"],
            dt["y2"],
            ax=ax,
            width=2,
            alpha=0.85,
            color=pass_colors.get(t, None),
        )

    ax.set_title((title + "\nPass Map (successful / unsuccessful / key pass / assist)").strip())
    return fig


def shot_map(
    df: pd.DataFrame,
    title: str = "",
    shot_colors: dict | None = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    show_xg: bool = False,
):
    """
    Standard shot map.
    If show_xg=True => writes xG beside each shot (estimated if not present).
    """
    shot_colors = shot_colors or {}
    s = df[df["event_type"] == "shot"].copy()

    if show_xg:
        if "xg" not in s.columns or s["xg"].isna().all():
            tmp = estimate_xg_for_shots(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
            s = tmp[tmp["event_type"] == "shot"].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))

    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue
        pitch.scatter(
            stt["x"],
            stt["y"],
            ax=ax,
            s=90,
            alpha=0.95,
            color=shot_colors.get(t, None),
        )

        if show_xg and "xg" in stt.columns:
            for _, r in stt.iterrows():
                try:
                    ax.text(
                        float(r["x"]) + 1.0,
                        float(r["y"]) + 1.0,
                        f'{float(r["xg"]):.2f}',
                        fontsize=9,
                        color="white",
                        weight="bold",
                    )
                except Exception:
                    pass

    ax.set_title((title + "\nShot Map (off target / on target / goal / blocked)").strip())
    return fig


def build_report_from_df(
    df: pd.DataFrame,
    out_dir: str,
    title: str = "Match Report",
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    pass_colors: dict | None = None,
    shot_colors: dict | None = None,
    bar_colors: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    df2 = apply_pitch_transforms(
        df,
        attack_direction=attack_direction,
        flip_y=flip_y,
        pitch_mode=pitch_mode,
        pitch_width=pitch_width,
    )

    # Fix #3: End locations Opta-ish
    df2 = fix_shot_end_location(df2, pitch_mode=pitch_mode, pitch_width=pitch_width)

    # Fix #2: Zone-based xG
    df2 = estimate_xg_for_shots(df2, pitch_mode=pitch_mode, pitch_width=pitch_width)

    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title, bar_colors=bar_colors)),
            ("start_heatmap", start_location_heatmap(df2, title=title, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("pass_map", pass_map(df2, title=title, pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("shot_map", shot_map(df2, title=title, shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
        ]

        pngs = []
        for name, fig in figs:
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pngs.append(png_path)

    return pdf_path, pngs


# ----------------------------
# Shot detail card (Fix #4)
# - Outcome only (NO Shot Type)
# - Mini goal indicator uses y2 mapping correctly
# - Supports theme
# ----------------------------
def shot_detail_card(
    df: pd.DataFrame,
    shot_index: int,
    title: str = "",
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    shot_colors: dict | None = None,
    theme_name: str = "Opta Dark",
):
    shot_colors = shot_colors or {
        "off target": "#FF8A00",
        "ontarget": "#00C2FF",
        "goal": "#00FF6A",
        "blocked": "#AAAAAA",
    }
    theme = THEMES.get(theme_name, THEMES["Opta Dark"])

    shots = df[df["event_type"] == "shot"].copy().reset_index(drop=True)
    if shots.empty:
        raise ValueError("No shots found in the file (event_type == shot).")

    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError("Shot index out of range.")

    # ensure end locations and xG (safe)
    shots = fix_shot_end_location(shots, pitch_mode=pitch_mode, pitch_width=pitch_width)
    shots = estimate_xg_for_shots(shots, pitch_mode=pitch_mode, pitch_width=pitch_width)

    r = shots.iloc[shot_index]

    # xG text
    try:
        xg_txt = f"{float(r.get('xg')):.2f}"
    except Exception:
        xg_txt = "NA"

    outcome = str(r.get("outcome", "")).lower()
    display_outcome = "On target" if outcome == "ontarget" else outcome.title()
    c = shot_colors.get(outcome, "#00C2FF")

    # Layout
    fig = plt.figure(figsize=(12, 6), facecolor=theme["bg"])
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.35, 1.0],
        height_ratios=[0.25, 1.0],
        wspace=0.08, hspace=0.05
    )

    ax_goal = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    # ---------- mini goal panel ----------
    ax_goal.set_facecolor(theme["panel"])
    ax_goal.set_xlim(0, 100)
    ax_goal.set_ylim(0, 30)
    ax_goal.axis("off")

    # goal frame
    ax_goal.plot([25, 75], [5, 5], lw=2, color=theme["goal"])
    ax_goal.plot([25, 25], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([75, 75], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([25, 75], [22, 22], lw=2, color=theme["goal"])

    # ---------- pitch ----------
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])

    x, y = float(r["x"]), float(r["y"])

    # ball marker
    pitch.scatter([x], [y], ax=ax_pitch, s=520, color=c, edgecolors="white", linewidth=2, zorder=5)
    pitch.scatter([x], [y], ax=ax_pitch, s=170, color="white", alpha=0.30, zorder=6)

    ax_pitch.text(
        x + 1.2, y + 1.2,
        f"xG {xg_txt}",
        color="white",
        fontsize=12,
        weight="bold",
        zorder=10,
    )

    # line to end
    has_end = ("x2" in shots.columns and "y2" in shots.columns and pd.notna(r.get("x2")) and pd.notna(r.get("y2")))
    y_low, y_high = _goal_mouth_bounds(pitch_mode, pitch_width)

    if has_end:
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color="white", alpha=0.9, zorder=4)
        pitch.scatter([x2], [y2], ax=ax_pitch, s=130, color="white", alpha=0.9, zorder=6)

        # map y2 into mini goal x-range [25..75]
        def map_to_mini_goal(y_val):
            y_val = float(y_val)
            y_clamped = max(y_low, min(y_high, y_val))
            t = (y_clamped - y_low) / (y_high - y_low + 1e-9)
            return 25 + t * 50

        gx = map_to_mini_goal(y2)
        ax_goal.scatter([gx], [12], s=240, color=c, edgecolors="white", linewidth=2, zorder=5)
    else:
        gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
        ax_pitch.plot([x, 100], [y, gy], linestyle=":", linewidth=3, color="white", alpha=0.6, zorder=4)

    # ---------- info panel ----------
    ax_info.set_facecolor(theme["panel"])
    ax_info.axis("off")

    if title:
        ax_info.text(0.02, 0.96, title, color=theme["text"], fontsize=18, weight="bold", transform=ax_info.transAxes)

    ax_info.text(0.02, 0.83, "xG", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.75, xg_txt, color=theme["text"], fontsize=24, weight="bold", transform=ax_info.transAxes)

    ax_info.plot([0.02, 0.98], [0.70, 0.70], color=theme["lines"], lw=2, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.60, "Outcome", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.53, display_outcome, color=theme["text"], fontsize=24, weight="bold", transform=ax_info.transAxes)

    # NO "Shot type" section (as requested)

    return fig, shots


# ----------------------------
# Pizza chart
# ----------------------------
def pizza_chart(
    df_pizza: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    slice_colors: list | None = None,
    show_values_legend: bool = True,
):
    dfp = df_pizza.copy()
    dfp.columns = [c.strip().lower() for c in dfp.columns]

    required = {"metric", "value", "percentile"}
    if not required.issubset(set(dfp.columns)):
        raise ValueError("Pizza input لازم يحتوي أعمدة: metric, value, percentile")

    params = dfp["metric"].astype(str).tolist()
    values = pd.to_numeric(dfp["percentile"], errors="coerce").fillna(0).tolist()
    value_text = dfp["value"].astype(str).tolist()

    if slice_colors is None or len(slice_colors) != len(values):
        slice_colors = ["#1f77b4"] * len(values)

    pizza = PyPizza(
        params=params,
        background_color="#111111",
        straight_line_color="#000000",
        straight_line_lw=1,
        last_circle_lw=1,
        last_circle_color="#000000",
    )

    try:
        fig, ax = pizza.make_pizza(
            values,
            figsize=(10, 10),
            blank_alpha=0.25,
            slice_colors=slice_colors,
            kwargs_slices=dict(edgecolor="#000000", linewidth=1),
            kwargs_params=dict(color="white", fontsize=12),
            kwargs_values=dict(color="white", fontsize=12),
        )
    except TypeError:
        fig, ax = pizza.make_pizza(
            values,
            figsize=(10, 10),
            blank_alpha=0.25,
            value_bck_colors=["#1f77b4"] * len(values),
            kwargs_slices=dict(edgecolor="#000000", linewidth=1),
            kwargs_params=dict(color="white", fontsize=12),
            kwargs_values=dict(color="white", fontsize=12),
        )

    fig.text(0.5, 0.985, title, ha="center", va="top", color="white", fontsize=18)
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", color="white", fontsize=12)

    if show_values_legend:
        lines = []
        for m, v, p in zip(params, value_text, values):
            lines.append(f"{m}: {v}   (pct {p:.1f})")

        legend_text = "\n".join(lines)
        fig.text(
            0.02, 0.02,
            legend_text,
            ha="left", va="bottom",
            color="white",
            fontsize=10,
            family="monospace",
        )

    return fig


# ----------------------------
# Convenience: full pipeline helper
# ----------------------------
def prepare_df_for_charts(
    df_raw: pd.DataFrame,
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
) -> pd.DataFrame:
    """
    1) validate+clean
    2) apply transforms
    3) fix end locations
    4) compute zone-based xG
    """
    df = validate_and_clean(df_raw)
    df = apply_pitch_transforms(df, attack_direction=attack_direction, flip_y=flip_y, pitch_mode=pitch_mode, pitch_width=pitch_width)
    df = fix_shot_end_location(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    df = estimate_xg_for_shots(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    return df


# ----------------------------
# Streamlit dropdown snippet (copy into your app)
# ----------------------------
"""
# shots = df2[df2["event_type"]=="shot"].copy().reset_index(drop=True)
# shots["label"] = shots.apply(
#     lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {float(r["xg"]):.2f} | ({r["x"]:.1f},{r["y"]:.1f})',
#     axis=1
# )
# selected = st.selectbox("Select a shot", shots["label"].tolist())
# shot_index = shots.index[shots["label"] == selected][0]
# fig, _ = shot_detail_card(df2, shot_index=shot_index, title="Shot Detail", pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme)
# st.pyplot(fig)
"""
