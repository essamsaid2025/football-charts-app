# app.py
# ===========================
# Streamlit Football Charts Generator
# Zone + Model xG (default auto)
# - Outcome normalize like "1ontarget"
# - Pitch transforms ONCE
# - Zone xG lookup + Logistic Model xG (pipeline)
# - Opta-ish end location
# - Report PDF + PNGs + Shot Detail Card
# ===========================

import os
import re
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Constants
# ----------------------------
PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]  # x2,y2 optional

MODEL_FILE_DEFAULT = "xg_pipeline.joblib"

# ----------------------------
# Theme System (for Shot Card)
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
# 1) Outcome Normalization (Fix "1ontarget")
# ----------------------------
def _norm_outcome(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"^\d+", "", s).strip()  # remove leading digits
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
# Load data (uploader bytes or path)
# ----------------------------
def load_data_any(file_or_path) -> pd.DataFrame:
    if hasattr(file_or_path, "name"):  # streamlit uploader
        name = file_or_path.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(file_or_path)
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(file_or_path)
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    else:
        path = str(file_or_path)
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

    # normalize column names (keep original, but also handle trailing spaces)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # common uppercase columns
    for c in ["x", "y", "x2", "y2"]:
        if c not in df.columns and c.upper() in df.columns:
            df.rename(columns={c.upper(): c}, inplace=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")

    # numeric
    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # outcome normalize
    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()

    # Optional cols for model xG defaults
    if "under_pressure" not in df.columns:
        df["under_pressure"] = 0
    df["under_pressure"] = pd.to_numeric(df["under_pressure"], errors="coerce").fillna(0).astype(int)

    if "body_part_name" not in df.columns:
        df["body_part_name"] = "Right Foot"
    df["body_part_name"] = df["body_part_name"].fillna("Right Foot").astype(str)

    if "technique_name" not in df.columns:
        df["technique_name"] = "Normal"
    df["technique_name"] = df["technique_name"].fillna("Normal").astype(str)

    if "sub_type_name" not in df.columns:
        df["sub_type_name"] = "Open Play"
    df["sub_type_name"] = df["sub_type_name"].fillna("Open Play").astype(str)

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
    """
    Your tag tool coords assumed 0-100 for both axes.
    - flip_y: flips Y in original 0-100 space
    - attack_direction == "rtl": flips X in original 0-100 space
    - rect: scales Y to 0..pitch_width
    """
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

def make_pitch(pitch_mode: str = "rect", pitch_width: float = 64.0):
    if pitch_mode == "square":
        return Pitch(pitch_type="custom", pitch_length=100, pitch_width=100, line_zorder=2)
    return Pitch(pitch_type="custom", pitch_length=100, pitch_width=pitch_width, line_zorder=2)

# ----------------------------
# Goal mouth helpers
# ----------------------------
def _goal_mouth_bounds(pitch_mode="rect", pitch_width=64.0):
    gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    return gy - goal_mouth / 2.0, gy + goal_mouth / 2.0

# ----------------------------
# Preferable side + engineered model features
# ----------------------------
def is_preferable_side(y: float, body_part_name: str) -> int:
    side = "center"
    if y < 40:
        side = "left"
    elif y > 40:
        side = "right"
    if (side == "left" and body_part_name == "Right Foot") or (side == "right" and body_part_name == "Left Foot"):
        return 1
    return 0

def compute_angle_distance_for_row(x: float, y: float, pitch_mode="rect", pitch_width=64.0):
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    left_post_y = goal_y - goal_mouth / 2.0
    right_post_y = goal_y + goal_mouth / 2.0

    # angle in radians
    a = math.atan2(right_post_y - y, goal_x - x)
    b = math.atan2(left_post_y - y, goal_x - x)
    angle = abs(a - b)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    dx = goal_x - x
    dy = goal_y - y
    dist_units = math.sqrt(dx * dx + dy * dy)
    return float(angle), float(dist_units)

def add_model_engineered_features(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    df = df.copy()
    mask = df["event_type"] == "shot"
    if not mask.any():
        df["angle"] = pd.NA
        df["distance"] = pd.NA
        df["preferable_side"] = pd.NA
        df["header"] = pd.NA
        return df

    angles = []
    dists = []
    prefs = []
    headers = []
    for _, r in df.iterrows():
        if r["event_type"] != "shot":
            angles.append(pd.NA)
            dists.append(pd.NA)
            prefs.append(pd.NA)
            headers.append(pd.NA)
            continue

        x = float(r["x"]); y = float(r["y"])
        angle, dist_units = compute_angle_distance_for_row(x, y, pitch_mode=pitch_mode, pitch_width=pitch_width)
        body = str(r.get("body_part_name", "Right Foot"))
        angles.append(angle)
        dists.append(dist_units)
        prefs.append(is_preferable_side(y if pitch_mode=="square" else float(r["y"]) , body))
        headers.append(1 if body == "Head" else 0)

    df["angle"] = angles
    df["distance"] = dists
    df["preferable_side"] = prefs
    df["header"] = headers
    return df

# ----------------------------
# 2) Zone-based xG (distance + angle bins) - your original
# ----------------------------
def _shot_angle_and_distance_units(x: float, y: float, pitch_mode: str, pitch_width: float):
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0

    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    left_post_y = goal_y - goal_mouth / 2.0
    right_post_y = goal_y + goal_mouth / 2.0

    dx = goal_x - x
    dy = goal_y - y
    dist_units = math.sqrt(dx * dx + dy * dy) + 1e-9

    a = math.atan2(right_post_y - y, goal_x - x)
    b = math.atan2(left_post_y - y, goal_x - x)
    angle = abs(a - b)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle, dist_units

def _meters_distance_approx(x, y, pitch_mode="rect", pitch_width=64.0):
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
    angle, _ = _shot_angle_and_distance_units(x, y, pitch_mode, pitch_width)
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

    table = {
        ("0-6", "big"): 0.55,   ("0-6", "mid"): 0.45,   ("0-6", "small"): 0.32,
        ("6-12", "big"): 0.32,  ("6-12", "mid"): 0.22,  ("6-12", "small"): 0.12,
        ("12-18", "big"): 0.18, ("12-18", "mid"): 0.10, ("12-18", "small"): 0.05,
        ("18-25", "big"): 0.08, ("18-25", "mid"): 0.05, ("18-25", "small"): 0.03,
        ("25+", "big"): 0.04,   ("25+", "mid"): 0.025,  ("25+", "small"): 0.015,
    }
    xg = table.get((d_bin, a_bin), 0.02)
    return float(max(0.01, min(0.85, xg)))

def estimate_zone_xg(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
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
# Model xG (Logistic Regression pipeline like article, but on YOUR pitch)
# ----------------------------
def build_xg_model_pipeline(df_prepared: pd.DataFrame) -> Pipeline:
    """
    Train model on *your* dataset shots.
    Needs enough samples + both classes (goal & non-goal).
    """
    shots = df_prepared[df_prepared["event_type"] == "shot"].copy()
    if shots.empty:
        raise ValueError("No shots to train on.")

    # label
    shots["goal"] = (shots["outcome"] == "goal").astype(int)

    # engineered features needed
    needed = ["under_pressure", "angle", "distance", "preferable_side", "header", "technique_name", "sub_type_name"]
    for c in needed:
        if c not in shots.columns:
            raise ValueError(f"Missing engineered feature: {c}")

    # must have both classes
    if shots["goal"].nunique() < 2:
        raise ValueError("Training needs at least 1 goal and 1 non-goal.")

    # If very small, warn (still can train, but not reliable)
    if len(shots) < 50:
        raise ValueError(f"Too few shots to train a reliable model ({len(shots)}).")

    numeric_features = ["under_pressure", "angle", "distance", "preferable_side", "header"]
    categorical_features = ["technique_name", "sub_type_name"]

    X = shots[numeric_features + categorical_features].copy()
    y = shots["goal"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=2000)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", clf)])
    pipe.fit(X, y)
    return pipe

def predict_model_xg(df_prepared: pd.DataFrame, model_pipe: Pipeline) -> pd.DataFrame:
    df = df_prepared.copy()
    if "xg_model" not in df.columns:
        df["xg_model"] = pd.NA
    shots = df["event_type"] == "shot"
    if shots.any():
        X = df.loc[shots, ["under_pressure", "angle", "distance", "preferable_side", "header",
                           "technique_name", "sub_type_name"]].copy()
        proba = model_pipe.predict_proba(X)[:, 1]
        df.loc[shots, "xg_model"] = np.round(proba, 2)
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
# Prepare DF (RUN ONCE)
# ----------------------------
def prepare_df_for_charts(
    df_raw: pd.DataFrame,
    attack_direction="ltr",
    flip_y=False,
    pitch_mode="rect",
    pitch_width=64.0,
    xg_method="zone",          # "zone" or "model"
    model_pipe: Pipeline | None = None,
) -> pd.DataFrame:
    df = validate_and_clean(df_raw)
    df = apply_pitch_transforms(
        df,
        attack_direction=attack_direction,
        flip_y=flip_y,
        pitch_mode=pitch_mode,
        pitch_width=pitch_width,
    )
    df = fix_shot_end_location(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

    # compute engineered model features always (cheap)
    df = add_model_engineered_features(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

    # compute both xGs
    df = estimate_zone_xg(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    if model_pipe is not None:
        try:
            df = predict_model_xg(df, model_pipe=model_pipe)
        except Exception:
            df["xg_model"] = pd.NA
    else:
        df["xg_model"] = pd.NA

    # set active xg column
    if xg_method == "model" and "xg_model" in df.columns and df["xg_model"].notna().any():
        df["xg"] = df["xg_model"]
        df["xg_source"] = "model"
    else:
        df["xg"] = df["xg_zone"]
        df["xg_source"] = "zone"

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

def start_location_heatmap(df: pd.DataFrame, title: str = "", pitch_mode="rect", pitch_width=64.0):
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    pitch.kdeplot(df["x"], df["y"], ax=ax, fill=True, levels=50)
    ax.set_title((title + "\nStart Locations Heatmap").strip())

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

def pass_map(df: pd.DataFrame, title: str = "", pass_colors: dict | None = None,
             pitch_mode="rect", pitch_width=64.0):
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
        pitch.arrows(dt["x"], dt["y"], dt["x2"], dt["y2"], ax=ax,
                     width=2, alpha=0.85, color=pass_colors.get(t, None))

    ax.set_title((title + "\nPass Map (successful / unsuccessful / key pass / assist)").strip())

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

def shot_map(df: pd.DataFrame, title: str = "", shot_colors: dict | None = None,
             pitch_mode="rect", pitch_width=64.0, show_xg=False):
    shot_colors = shot_colors or {}
    s = df[df["event_type"] == "shot"].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))

    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue
        pitch.scatter(stt["x"], stt["y"], ax=ax, s=90, alpha=0.95, color=shot_colors.get(t, None))

        if show_xg and "xg" in stt.columns:
            for _, r in stt.iterrows():
                try:
                    ax.text(float(r["x"]) + 1.0, float(r["y"]) + 1.0,
                            f'{float(r["xg"]):.2f}', fontsize=9, color="white", weight="bold")
                except Exception:
                    pass

    xg_src = str(df["xg_source"].iloc[0]) if "xg_source" in df.columns and len(df) else ""
    ax.set_title((title + f"\nShot Map (xG: {xg_src})").strip())

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

# ----------------------------
# Report builder (expects prepared df)
# ----------------------------
def build_report_from_prepared_df(
    df_prepared: pd.DataFrame,
    out_dir: str,
    title: str = "Match Report",
    pitch_mode="rect",
    pitch_width=64.0,
    pass_colors: dict | None = None,
    shot_colors: dict | None = None,
    bar_colors: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    df2 = df_prepared.copy()

    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title, bar_colors=bar_colors)),
            ("start_heatmap", start_location_heatmap(df2, title=title, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("pass_map", pass_map(df2, title=title, pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("shot_map", shot_map(df2, title=title, shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width, show_xg=True)),
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
# 4) Shot Detail Card (NO Shot type) + mini goal correct + no clipping
# ----------------------------
def shot_detail_card(
    df_prepared: pd.DataFrame,
    shot_index: int,
    title: str = "Shot Detail",
    pitch_mode="rect",
    pitch_width=64.0,
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
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])

    ax_pitch.set_xlim(-2, 102)
    ax_pitch.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    x, y = float(r["x"]), float(r["y"])
    pitch.scatter([x], [y], ax=ax_pitch, s=520, color=c, edgecolors="white", linewidth=2, zorder=5, clip_on=False)
    pitch.scatter([x], [y], ax=ax_pitch, s=170, color="white", alpha=0.30, zorder=6, clip_on=False)
    ax_pitch.text(x + 1.2, y + 1.2, f"xG {xg_txt}", color="white", fontsize=12, weight="bold", zorder=10)

    # end line + mini goal mapping
    has_end = ("x2" in shots.columns and "y2" in shots.columns and pd.notna(r.get("x2")) and pd.notna(r.get("y2")))
    y_low, y_high = _goal_mouth_bounds(pitch_mode, pitch_width)

    if has_end:
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color="white", alpha=0.9, zorder=4)
        pitch.scatter([x2], [y2], ax=ax_pitch, s=130, color="white", alpha=0.9, zorder=6, clip_on=False)

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

    # info panel (NO shot type)
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
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Football Charts Generator", layout="wide")
st.title("Football Charts Generator (Zone + Model xG)")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
    st.caption("Required columns: outcome, x, y (optional: x2, y2, under_pressure, body_part_name, technique_name, sub_type_name)")

    st.header("Pitch / Transforms")
    pitch_mode = st.selectbox("Pitch Mode", ["rect", "square"], index=0)
    pitch_width = st.slider("Pitch Width (rect)", 50.0, 80.0, 64.0, 1.0) if pitch_mode == "rect" else 100.0
    attack_direction = st.selectbox("Attack Direction", ["ltr", "rtl"], index=0)
    flip_y = st.checkbox("Flip Y", value=False)

    st.header("xG")
    model_file = st.text_input("Model file", value=MODEL_FILE_DEFAULT)

    model_exists = os.path.exists(model_file)
    default_method = "Model" if model_exists else "Zone"
    xg_method_ui = st.radio("xG method", ["Zone", "Model"], index=0 if default_method=="Zone" else 1)
    xg_method = "model" if xg_method_ui == "Model" else "zone"

    st.header("Theme")
    theme_name = st.selectbox("Shot Card Theme", list(THEMES.keys()), index=0)

    st.header("Output")
    out_dir = st.text_input("Output folder", value="out")

# load model if exists
model_pipe = None
if model_exists:
    try:
        model_pipe = joblib.load(model_file)
    except Exception:
        model_pipe = None

colA, colB = st.columns([1.2, 1])

if uploaded is None:
    st.info("Upload a file to start.")
    st.stop()

# Load DF
try:
    df_raw = load_data_any(uploaded)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

# Prepare
df_prepared = prepare_df_for_charts(
    df_raw,
    attack_direction=attack_direction,
    flip_y=flip_y,
    pitch_mode=pitch_mode,
    pitch_width=pitch_width,
    xg_method=xg_method,
    model_pipe=model_pipe,
)

with colA:
    st.subheader("Prepared Data (preview)")
    st.dataframe(df_prepared.head(50), use_container_width=True)

with colB:
    st.subheader("Quick Summary")
    shots = df_prepared[df_prepared["event_type"] == "shot"].copy()
    st.write(f"Shots: **{len(shots)}**")
    if len(shots):
        st.write(f"Total xG ({df_prepared['xg_source'].iloc[0]}): **{float(pd.to_numeric(shots['xg'], errors='coerce').fillna(0).sum()):.2f}**")
        st.write(f"Goals: **{int((shots['outcome']=='goal').sum())}**")

    st.divider()

    st.subheader("Train / Save Model (optional)")
    st.caption("Needs >= 50 shots AND at least 1 goal and 1 non-goal in your dataset.")
    if st.button("Train model from this file and save"):
        try:
            # must have engineered features computed already
            pipe = build_xg_model_pipeline(df_prepared)
            joblib.dump(pipe, model_file)
            st.success(f"Saved model to {model_file}. Reload the app to use it as default.")
        except Exception as e:
            st.warning(str(e))

# Colors
pass_colors = {
    "successful": "#00C2FF",
    "unsuccessful": "#FF4D4D",
    "key pass": "#FFB84D",
    "assist": "#00FF6A",
}
shot_colors = {
    "off target": "#FF8A00",
    "ontarget": "#00C2FF",
    "goal": "#00FF6A",
    "blocked": "#AAAAAA",
}

st.subheader("Charts")
c1, c2 = st.columns(2)
with c1:
    fig1 = outcome_bar(df_prepared, title="Match Report")
    st.pyplot(fig1)
    plt.close(fig1)

with c2:
    fig2 = start_location_heatmap(df_prepared, title="Match Report", pitch_mode=pitch_mode, pitch_width=pitch_width)
    st.pyplot(fig2)
    plt.close(fig2)

c3, c4 = st.columns(2)
with c3:
    fig3 = pass_map(df_prepared, title="Match Report", pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)
    st.pyplot(fig3)
    plt.close(fig3)

with c4:
    fig4 = shot_map(df_prepared, title="Match Report", shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width, show_xg=True)
    st.pyplot(fig4)
    plt.close(fig4)

st.subheader("Shot Detail Card")
shots2 = df_prepared[df_prepared["event_type"] == "shot"].copy().reset_index(drop=True)
if shots2.empty:
    st.info("No shots found to display Shot Card.")
else:
    shots2["label"] = shots2.apply(
        lambda r: f'{r.name+1} | {str(r["outcome"]).upper()} | xG {float(r["xg"]):.2f}',
        axis=1
    )
    selected = st.selectbox("Select a shot", shots2["label"].tolist())
    shot_index = int(selected.split("|")[0].strip()) - 1
    fig_card, _ = shot_detail_card(
        df_prepared,
        shot_index=shot_index,
        title="Shot Detail",
        pitch_mode=pitch_mode,
        pitch_width=pitch_width,
        theme_name=theme_name
    )
    st.pyplot(fig_card)
    plt.close(fig_card)

st.subheader("Export")
if st.button("Build PDF report + PNGs"):
    pdf_path, pngs = build_report_from_prepared_df(
        df_prepared,
        out_dir=out_dir,
        title="Match Report",
        pitch_mode=pitch_mode,
        pitch_width=pitch_width,
        pass_colors=pass_colors,
        shot_colors=shot_colors,
    )
    st.success(f"Saved PDF: {pdf_path}")
    st.write("Saved PNGs:")
    for p in pngs:
        st.write(p)
