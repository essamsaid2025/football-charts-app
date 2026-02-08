# charts.py  (FULL UPDATED) — Model-based xG
# =========================================
# - Fix outcome like "1ontarget"
# - Prepare once (clean + transforms + end-location + xG)
# - xG: computes realistic xG per shot (model-based)
# - Final xg column replaces zone-based approach
# - Adds xg_source column for transparency
# - Charts + Shot Detail Card + Pizza chart
# =========================================

import os
import re
import math
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
    s = re.sub(r"^\d+", "", s).strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    aliases = {
        "on target": "ontarget",
        "ontarget": "ontarget",
        "offtarget": "off target",
        "off target": "off target",
        "block": "blocked",
        "blocked shot": "blocked",
        "blk": "blocked",
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

    for c in ["x", "y", "x2", "y2"]:
        if c not in df.columns and c.upper() in df.columns:
            df.rename(columns={c.upper(): c}, inplace=True)
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

    if "outcome" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "outcome":
                df.rename(columns={c: "outcome"}, inplace=True)
                break

    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()

    df["event_type"] = "pass"
    df.loc[df["outcome"].isin(SHOT_TYPES), "event_type"] = "shot"

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
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
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
# Angle / Distance helpers
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
# ----------------------------
# 2) Realistic xG (replaces zone-based)
# ----------------------------
def realistic_xg(x, y, pitch_mode="rect", pitch_width=64.0, is_penalty=False, body_part=None):
    if is_penalty:
        return 0.78
    angle = _shot_angle_radians(float(x), float(y), pitch_mode, pitch_width)
    dist = _meters_distance_approx(float(x), float(y), pitch_mode, pitch_width)
    b0, b1, b2 = 2.8, -0.11, 1.2
    logit = b0 + (b1 * dist) + (b2 * angle)
    xg = 1 / (1 + np.exp(-logit))
    if body_part is not None and "head" in str(body_part).lower():
        xg *= 0.72
    # tap-ins
    if dist <= 3:
        xg *= 1.25
    # narrow angles
    if angle < 0.25:
        xg *= 0.6
    return float(np.clip(xg, 0.001, 0.95))

def estimate_xg(df: pd.DataFrame, pitch_mode="rect", pitch_width=64.0) -> pd.DataFrame:
    df = df.copy()
    if "xg" not in df.columns:
        df["xg"] = pd.NA
    mask = df["event_type"] == "shot"
    if mask.any():
        df.loc[mask, "xg"] = [
            round(realistic_xg(
                x=row["x"],
                y=row["y"],
                pitch_mode=pitch_mode,
                pitch_width=pitch_width,
                is_penalty=(row.get("shot_type") == "penalty"),
                body_part=row.get("body_part_name")
            ), 3)
            for _, row in df.loc[mask].iterrows()
        ]
    df["xg_source"] = "model"
    return df

# ----------------------------
# (The rest of your code remains the same)
# - fix_shot_end_location
# - build_report_from_prepared_df
# - shot_detail_card
# - pizza_chart
# - outcome_bar
# - pass_map
# - shot_map
# - start_location_heatmap
# ----------------------------
