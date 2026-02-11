# charts.py  (FULL UPDATED) — Zone + Model xG + THEMES (Athletic)
# =========================================
# - Fix outcome like "1ontarget"
# - Prepare once (clean + transforms + end-location + xG)
# - xG: computes BOTH xg_zone + xg_model (if model_pipe provided)
# - Final xg column based on xg_method ("zone" / "model") with fallback
# - Adds xg_source column for transparency
# - Charts + Shot Detail Card + Pizza chart
# - Theme support (Opta / Sofa / StatsBomb + The Athletic Dark/Light)
# - Compatible with Python 3.9 (NO dict|None / list|None)
# =========================================

import os
import re
import math
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza


# ----------------------------
# Constants
# ----------------------------
PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]  # x2,y2 optional


# ----------------------------
# Theme System
# ----------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "Opta Dark": {
        "bg": "#0E1117",
        "panel": "#141A22",
        "pitch": "#1f5f3b",
        "text": "#FFFFFF",
        "muted": "#A0A7B4",
        "lines": "#E6E9EF",
        "goal": "#E6E6E6",
    },
    "Sofa Light": {
        "bg": "#FFFFFF",
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
        "text": "#FFFFFF",
        "muted": "#B3B3B3",
        "lines": "#E6E9EF",
        "goal": "#DDDDDD",
    },
    # ✅ Athletic-like themes
    "The Athletic Dark": {
        "bg": "#0B0F14",
        "panel": "#101721",
        "pitch": "#0F3D2E",
        "text": "#F5F7FA",
        "muted": "#B8C0CC",
        "lines": "#E6E9EF",
        "goal": "#E6E9EF",
    },
    "The Athletic Light": {
        "bg": "#FFFFFF",
        "panel": "#F3F5F7",
        "pitch": "#2F7D4C",
        "text": "#111827",
        "muted": "#4B5563",
        "lines": "#FFFFFF",
        "goal": "#111827",
    },
}


def _get_theme(theme_name: str) -> Dict[str, str]:
    return THEMES.get(theme_name, THEMES["Opta Dark"])


def _style_figure_axes(fig, ax, theme_name: str):
    theme = _get_theme(theme_name)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])
    for spine in ax.spines.values():
        spine.set_color(theme["panel"])
    ax.title.set_color(theme["text"])
    ax.xaxis.label.set_color(theme["text"])
    ax.yaxis.label.set_color(theme["text"])
    ax.tick_params(axis="x", colors=theme["muted"])
    ax.tick_params(axis="y", colors=theme["muted"])
    return theme


def _style_legend(ax, theme_name: str):
    theme = _get_theme(theme_name)
    leg = ax.get_legend()
    if leg is None:
        return
    leg.get_frame().set_facecolor(theme["panel"])
    leg.get_frame().set_edgecolor(theme["panel"])
    for t in leg.get_texts():
        t.set_color(theme["text"])


# ----------------------------
# 1) Outcome Normalization (Fix "1ontarget")
# ----------------------------
def _norm_outcome(s) -> str:
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
# Validate & Clean
# ----------------------------
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize outcome col name if weird spacing/case
    if "outcome" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "outcome":
                df.rename(columns={c: "outcome"}, inplace=True)
                break

    # common uppercase columns
    for c in ["x", "y", "x2", "y2"]:
        if c not in df.columns and c.upper() in df.columns:
            df.rename(columns={c.upper(): c}, inplace=True)

    # allow trailing space columns
    if "x" not in df.columns and "x " in df.columns:
        df.rename(columns={"x ": "x"}, inplace=True)
    if "y" not in df.columns and "y " in df.columns:
        df.rename(columns={"y ": "y"}, inplace=True)

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

    # keep only supported outcomes per type
    df_pass = df[df["event_type"] == "pass"].copy()
    df_shot = df[df["event_type"] == "shot"].copy()

    if not df_pass.empty:
        df_pass = df_pass[df_pass["outcome"].isin(PASS_ORDER)]
    if not df_shot.empty:
        df_shot = df_shot[df_shot["outcome"].isin(SHOT_ORDER)]

    return pd.concat([df_pass, df_shot], ignore_index=True)


# ----------------------------
# Pitch transforms (run ONCE only)
# ----------------------------
def apply_pitch_transforms(
    df: pd.DataFrame,
    attack_direction: str = "ltr",  # "ltr" or "rtl"
    flip_y: bool = False,
    pitch_mode: str = "rect",       # "rect" or "square"
    pitch_width: float = 64.0,
) -> pd.DataFrame:
    df = df.copy()

    if flip_y:
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    if attack_direction == "rtl":
        for c in ["x", "x2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    if pitch_mode == "rect":
        scale = pitch_width / 100.0
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = df[c] * scale

    return df


def make_pitch(pitch_mode: str = "rect", pitch_width: float = 64.0, theme_name: str = "Opta Dark") -> Pitch:
    theme = _get_theme(theme_name)
    if pitch_mode == "square":
        return Pitch(
            pitch_type="custom",
            pitch_length=100,
            pitch_width=100,
            line_zorder=2,
            pitch_color=theme["pitch"],
            line_color=theme["lines"],
        )
    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        line_zorder=2,
        pitch_color=theme["pitch"],
        line_color=theme["lines"],
    )


# ----------------------------
# Goal mouth helpers
# ----------------------------
def _goal_mouth_bounds(pitch_mode="rect", pitch_width=64.0) -> Tuple[float, float]:
    gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    return gy - goal_mouth / 2.0, gy + goal_mouth / 2.0


# ----------------------------
# Angle / Distance helpers (used by zone + model features)
# ----------------------------
def _shot_angle_radians(x: float, y: float, pitch_mode: str, pitch_width: float) -> float:
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0

    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    left_post_y = goal_y - goal_mouth / 2.0
    right_post_y = goal_y + goal_mouth / 2.0

    a = math.atan2(right_post_y - y, goal_x - x)
    b = math.atan2(left_post_y - y, goal_x - x)
    angle = abs(a - b)
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return float(angle)


def _meters_distance_approx(x, y, pitch_mode="rect", pitch_width=64.0) -> float:
    length_m = 105.0
    width_m = 68.0 if pitch_mode == "rect" else 105.0
    y_max = pitch_width if pitch_mode == "rect" else 100.0

    xm = (x / 100.0) * length_m
    ym = (y / y_max) * width_m
    goal_xm = length_m
    goal_ym = width_m / 2.0

    dx = goal_xm - xm
    dy = goal_ym - ym
    return float(math.sqrt(dx * dx + dy * dy))


def calculate_angle_degrees(x: float, y: float, pitch_mode="rect", pitch_width=64.0) -> float:
    return float(np.degrees(_shot_angle_radians(x, y, pitch_mode, pitch_width)))


def calculate_distance_units(x: float, y: float, pitch_mode="rect", pitch_width=64.0) -> float:
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    dx = goal_x - x
    dy = goal_y - y
    return float(math.sqrt(dx * dx + dy * dy) + 1e-9)


# ----------------------------
# 2) Zone-based xG (distance + angle bins)
# ----------------------------
def zone_based_xg(x, y, pitch_mode="rect", pitch_width=64.0):
    angle = _shot_angle_radians(float(x), float(y), pitch_mode, pitch_width)
    dist_m = _meters_distance_approx(float(x), float(y), pitch_mode, pitch_width)

    if angle < 0.35:
        a_bin = "small"
    elif angle < 0.75:
        a_bin = "mid"
    else:
        a_bin = "big"

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

    table = {
        ("0-6", "big"): 0.55,   ("0-6", "mid"): 0.45,   ("0-6", "small"): 0.32,
        ("6-12", "big"): 0.32,  ("6-12", "mid"): 0.22,  ("6-12", "small"): 0.12,
        ("12-18", "big"): 0.18, ("12-18", "mid"): 0.10, ("12-18", "small"): 0.05,
        ("18-25", "big"): 0.08, ("18-25", "mid"): 0.05, ("18-25", "small"): 0.03,
        ("25+", "big"): 0.04,   ("25+", "mid"): 0.025,  ("25+", "small"): 0.015,
    }
    xg = table.get((d_bin, a_bin), 0.02)
    return float(max(0.01, min(0.85, xg)))


def estimate_xg_zone(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    df = df.copy()
    if "xg_zone" not in df.columns:
        df["xg_zone"] = pd.NA

    mask = df["event_type"] == "shot"
    if mask.any():
        df.loc[mask, "xg_zone"] = [
            round(zone_based_xg(float(x), float(y), pitch_mode=pitch_mode, pitch_width=pitch_width), 2)
            for x, y in zip(df.loc[mask, "x"], df.loc[mask, "y"])
        ]
    return df


# ----------------------------
# Model xG support
# ----------------------------
def _is_preferable_side(y, body_part_name, pitch_mode="rect", pitch_width=64.0) -> int:
    try:
        y = float(y)
    except Exception:
        return 0

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    center = y_max / 2.0

    side = "center"
    if y < center:
        side = "left"
    elif y > center:
        side = "right"

    bp = ("" if body_part_name is None else str(body_part_name)).strip().lower()
    if (side == "left" and "right" in bp) or (side == "right" and "left" in bp):
        return 1
    return 0


def _header_flag(body_part_name) -> int:
    bp = ("" if body_part_name is None else str(body_part_name)).strip().lower()
    return 1 if "head" in bp else 0


def build_model_features(df_prepared: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    shots = df_prepared[df_prepared["event_type"] == "shot"].copy()

    shots["x"] = pd.to_numeric(shots["x"], errors="coerce")
    shots["y"] = pd.to_numeric(shots["y"], errors="coerce")

    shots["angle"] = shots.apply(lambda r: calculate_angle_degrees(r["x"], r["y"], pitch_mode, pitch_width), axis=1)
    shots["distance"] = shots.apply(lambda r: calculate_distance_units(r["x"], r["y"], pitch_mode, pitch_width), axis=1)

    if "under_pressure" not in shots.columns:
        shots["under_pressure"] = 0
    shots["under_pressure"] = pd.to_numeric(shots["under_pressure"], errors="coerce").fillna(0).astype(int)

    if "body_part_name" not in shots.columns:
        shots["body_part_name"] = ""
    if "technique_name" not in shots.columns:
        shots["technique_name"] = "Normal"
    if "sub_type_name" not in shots.columns:
        shots["sub_type_name"] = "Open Play"

    shots["preferable_side"] = shots.apply(
        lambda r: _is_preferable_side(r["y"], r.get("body_part_name", ""), pitch_mode, pitch_width),
        axis=1
    )
    shots["header"] = shots.apply(lambda r: _header_flag(r.get("body_part_name", "")), axis=1)

    cols = [
        "x", "y",
        "angle", "distance",
        "under_pressure",
        "preferable_side", "header",
        "technique_name", "sub_type_name", "body_part_name",
    ]
    for c in cols:
        if c not in shots.columns:
            shots[c] = pd.NA

    return shots[cols].copy()


def estimate_xg_model(df: pd.DataFrame, model_pipe=None, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    df = df.copy()
    if "xg_model" not in df.columns:
        df["xg_model"] = pd.NA

    if model_pipe is None:
        return df

    mask = df["event_type"] == "shot"
    if not mask.any():
        return df

    try:
        X = build_model_features(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

        if hasattr(model_pipe, "predict_proba"):
            preds = model_pipe.predict_proba(X)[:, 1]
        else:
            preds = model_pipe.predict(X)

        preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
        preds = np.round(preds, 3)

        df.loc[mask, "xg_model"] = preds.tolist()
        return df

    except Exception:
        return df


# ----------------------------
# 3) Opta-ish End Location
# ----------------------------
def fix_shot_end_location(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
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
            else:
                x2 = 100.5
                y2 = (y_high + 2.0) if (y > mid) else (y_low - 2.0)
        else:
            x2 = float(x2)
            y2 = float(y2)

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
# Prepare DF (RUN ONCE)
# ----------------------------
def prepare_df_for_charts(
    df_raw: pd.DataFrame,
    attack_direction="ltr",
    flip_y=False,
    pitch_mode="rect",
    pitch_width=64.0,
    xg_method: str = "zone",     # "zone" or "model"
    model_pipe=None,
) -> pd.DataFrame:
    df = validate_and_clean(df_raw)
    df = apply_pitch_transforms(
        df,
        attack_direction=attack_direction,
        flip_y=flip_y,
        pitch_mode=pitch_mode,
        pitch_width=pitch_width
    )
    df = fix_shot_end_location(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

    df = estimate_xg_zone(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    df = estimate_xg_model(df, model_pipe=model_pipe, pitch_mode=pitch_mode, pitch_width=pitch_width)

    df["xg_source"] = "zone"
    df["xg"] = df["xg_zone"]

    xg_method = (xg_method or "zone").strip().lower()
    if xg_method == "model":
        if df["xg_model"].notna().any():
            df["xg"] = df["xg_model"]
            df["xg_source"] = "model"
        else:
            df["xg"] = df["xg_zone"]
            df["xg_source"] = "zone (fallback)"

    df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
    return df


# ----------------------------
# Charts (with theme_name)
# ----------------------------
def outcome_bar(df: pd.DataFrame, title: str = "", theme_name: str = "Opta Dark", bar_colors: Optional[Dict[str, str]] = None):
    bar_colors = bar_colors or {}
    counts = df["outcome"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 4))
    theme = _style_figure_axes(fig, ax, theme_name)

    colors = [bar_colors.get(k, None) for k in counts.index.astype(str)]
    ax.bar(counts.index.astype(str), counts.values, color=colors)

    ax.set_title((title + "\nOutcome Distribution").strip(), color=theme["text"])
    ax.set_ylabel("Count", color=theme["text"])
    ax.tick_params(axis="x", rotation=25, colors=theme["muted"])
    ax.tick_params(axis="y", colors=theme["muted"])

    # Legend (only if colors provided)
    if bar_colors:
        for k, v in bar_colors.items():
            ax.bar(0, 0, color=v, label=k)
        ax.legend(loc="upper right", frameon=True)
        _style_legend(ax, theme_name)

    return fig


def start_location_heatmap(df: pd.DataFrame, title: str = "", theme_name: str = "Opta Dark", pitch_mode="rect", pitch_width=64.0):
    # heatmap for START locations (shots only)
    s = df[df["event_type"] == "shot"].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    theme = _get_theme(theme_name)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["pitch"])

    if not s.empty:
        pitch.kdeplot(s["x"], s["y"], ax=ax, fill=True, levels=50)

    ax.set_title((title + "\nStart Locations Heatmap (Shots)").strip(), color=theme["text"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig


def pass_map(df: pd.DataFrame, title: str = "", theme_name: str = "Opta Dark", pass_colors: Optional[Dict[str, str]] = None,
             pitch_mode="rect", pitch_width=64.0):
    pass_colors = pass_colors or {}
    d = df[df["event_type"] == "pass"].copy()

    if not {"x2", "y2"}.issubset(d.columns):
        d["x2"] = pd.NA
        d["y2"] = pd.NA
    d = d.dropna(subset=["x2", "y2"])

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    theme = _get_theme(theme_name)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["pitch"])

    for t in PASS_ORDER:
        dt = d[d["outcome"] == t]
        if len(dt) == 0:
            continue
        pitch.arrows(
            dt["x"], dt["y"], dt["x2"], dt["y2"],
            ax=ax, width=2, alpha=0.85,
            color=pass_colors.get(t, theme["lines"])
        )

    ax.set_title((title + "\nPass Map").strip(), color=theme["text"])

    if pass_colors:
        for k, v in pass_colors.items():
            ax.plot([], [], color=v, label=k)
        ax.legend(loc="upper right", frameon=True)
        _style_legend(ax, theme_name)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig


def shot_map(df: pd.DataFrame, title: str = "", theme_name: str = "Opta Dark", shot_colors: Optional[Dict[str, str]] = None,
             pitch_mode="rect", pitch_width=64.0, show_xg=False):
    shot_colors = shot_colors or {}
    s = df[df["event_type"] == "shot"].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    theme = _get_theme(theme_name)
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["pitch"])

    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue
        pitch.scatter(stt["x"], stt["y"], ax=ax, s=90, alpha=0.95, color=shot_colors.get(t, theme["lines"]))

        if show_xg and "xg" in stt.columns:
            for _, r in stt.iterrows():
                try:
                    ax.text(float(r["x"]) + 1.0, float(r["y"]) + 1.0,
                            f'{float(r["xg"]):.2f}', fontsize=9, color=theme["text"], weight="bold")
                except Exception:
                    pass

    xg_src = ""
    if "xg_source" in df.columns and len(df):
        xg_src = str(df["xg_source"].iloc[0])

    ax.set_title((title + f"\nShot Map — xG: {xg_src}").strip(), color=theme["text"])

    if shot_colors:
        for k, v in shot_colors.items():
            ax.scatter([], [], color=v, label=k)
        ax.legend(loc="upper right", frameon=True)
        _style_legend(ax, theme_name)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig


# ----------------------------
# Report builder (expects prepared df) + theme_name
# ----------------------------
def build_report_from_prepared_df(
    df_prepared: pd.DataFrame,
    out_dir: str,
    title: str = "Match Report",
    theme_name: str = "Opta Dark",
    pitch_mode="rect",
    pitch_width=64.0,
    pass_colors: Optional[Dict[str, str]] = None,
    shot_colors: Optional[Dict[str, str]] = None,
    bar_colors: Optional[Dict[str, str]] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    df2 = df_prepared.copy()

    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title, theme_name=theme_name, bar_colors=bar_colors)),
            ("start_heatmap", start_location_heatmap(df2, title=title, theme_name=theme_name, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("pass_map", pass_map(df2, title=title, theme_name=theme_name, pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("shot_map", shot_map(df2, title=title, theme_name=theme_name, shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width, show_xg=True)),
        ]

        pngs: List[str] = []
        for name, fig in figs:
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pngs.append(png_path)

    return pdf_path, pngs


# ----------------------------
# Shot Detail Card (NO shot type) + theme pitch
# ----------------------------
def shot_detail_card(
    df_prepared: pd.DataFrame,
    shot_index: int,
    title: str = "Shot Detail",
    pitch_mode="rect",
    pitch_width=64.0,
    shot_colors: Optional[Dict[str, str]] = None,
    theme_name: str = "Opta Dark",
):
    shot_colors = shot_colors or {
        "off target": "#FF8A00",
        "ontarget": "#00C2FF",
        "goal": "#00FF6A",
        "blocked": "#AAAAAA",
    }
    theme = _get_theme(theme_name)

    shots = df_prepared[df_prepared["event_type"] == "shot"].copy().reset_index(drop=True)
    if shots.empty:
        raise ValueError("No shots found.")
    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError("Shot index out of range.")

    r = shots.iloc[shot_index]

    xg_txt = "NA"
    try:
        xg_txt = f"{float(r.get('xg')):.2f}"
    except Exception:
        pass

    outcome = str(r.get("outcome", "")).lower()
    display_outcome = "On target" if outcome == "ontarget" else outcome.title()
    c = shot_colors.get(outcome, "#00C2FF")

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

    # mini goal
    ax_goal.set_facecolor(theme["panel"])
    ax_goal.set_xlim(0, 100)
    ax_goal.set_ylim(0, 30)
    ax_goal.axis("off")
    ax_goal.plot([25, 75], [5, 5], lw=2, color=theme["goal"])
    ax_goal.plot([25, 25], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([75, 75], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([25, 75], [22, 22], lw=2, color=theme["goal"])

    # pitch
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])

    ax_pitch.set_xlim(-2, 102)
    ax_pitch.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    x, y = float(r["x"]), float(r["y"])
    pitch.scatter([x], [y], ax=ax_pitch, s=520, color=c, edgecolors="white", linewidth=2, zorder=5, clip_on=False)
    pitch.scatter([x], [y], ax=ax_pitch, s=170, color="white", alpha=0.30, zorder=6, clip_on=False)

    ax_pitch.text(x + 1.2, y + 1.2, f"xG {xg_txt}",
                  color=theme["text"], fontsize=12, weight="bold", zorder=10)

    has_end = ("x2" in shots.columns and "y2" in shots.columns and pd.notna(r.get("x2")) and pd.notna(r.get("y2")))
    y_low, y_high = _goal_mouth_bounds(pitch_mode, pitch_width)

    if has_end:
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color=theme["text"], alpha=0.9, zorder=4)
        pitch.scatter([x2], [y2], ax=ax_pitch, s=130, color=theme["text"], alpha=0.9, zorder=6, clip_on=False)

        def map_to_mini_goal(y_val):
            y_val = float(y_val)
            y_clamped = max(y_low, min(y_high, y_val))
            t = (y_clamped - y_low) / (y_high - y_low + 1e-9)
            return 25 + t * 50

        gx = map_to_mini_goal(y2)
        ax_goal.scatter([gx], [12], s=240, color=c, edgecolors="white", linewidth=2, zorder=5)

    else:
        gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
        ax_pitch.plot([x, 100], [y, gy], linestyle=":", linewidth=3, color=theme["text"], alpha=0.6, zorder=4)

    # info panel
    ax_info.set_facecolor(theme["panel"])
    ax_info.axis("off")

    ax_info.text(0.02, 0.94, title, color=theme["text"], fontsize=18, weight="bold", transform=ax_info.transAxes)

    ax_info.text(0.02, 0.80, "xG", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.72, xg_txt, color=theme["text"], fontsize=26, weight="bold", transform=ax_info.transAxes)
    ax_info.plot([0.02, 0.98], [0.67, 0.67], color=theme["lines"], lw=2, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.55, "Outcome", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.47, display_outcome, color=theme["text"], fontsize=26, weight="bold", transform=ax_info.transAxes)

    return fig, shots


# ----------------------------
# Pizza chart (unchanged)
# ----------------------------
def pizza_chart(
    df_pizza: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    slice_colors: Optional[List[str]] = None,
    show_values_legend: bool = True
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
        lines = [f"{m}: {v}   (pct {p:.1f})" for m, v, p in zip(params, value_text, values)]
        fig.text(0.02, 0.02, "\n".join(lines), ha="left", va="bottom",
                 color="white", fontsize=10, family="monospace")

    return fig
