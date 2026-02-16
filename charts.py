import os
import re
import math
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza

PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]

THEMES = {
    "The Athletic Dark": {
        "bg": "#0E1117",
        "panel": "#111827",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
    },
    "Opta Dark": {
        "bg": "#0E1117",
        "panel": "#141A22",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
    },
    "Sofa Light": {
        "bg": "white",
        "panel": "#F5F7FA",
        "pitch": "#2f6b3a",
        "text": "#111111",
        "muted": "#5A6572",
        "lines": "#DDE3EA",
        "goal": "#444444",
        "pitch_lines": "#FFFFFF",
    },

    # ✅ New theme مشابه للصورة (Grey header + black panels)
    "Box Entry (Grey Header)": {
        "bg": "#E6E6E6",        # header/page background
        "panel": "#0B0B0B",     # black panels
        "pitch": "#0A0A0A",     # dark pitch
        "text": "#FFFFFF",
        "muted": "#C9CDD3",
        "lines": "#2A2A2A",
        "goal": "#EDEDED",
        "pitch_lines": "#EDEDED",
    },
}


def _norm_outcome(s: str) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"^\d+", "", s).strip()
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
        "accurate": "successful",
        "inaccurate": "unsuccessful",

        # touch
        "touch": "touch",
        "ball touch": "touch",
        "receive": "touch",
        "reception": "touch",
        "received": "touch",
    }
    return aliases.get(s, s)


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


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    cols_lower_map = {c.lower(): c for c in df.columns}

    if "outcome" not in cols_lower_map:
        raise ValueError("Missing column: outcome (required).")
    df.rename(columns={cols_lower_map["outcome"]: "outcome"}, inplace=True)

    for want in ["x", "y", "x2", "y2"]:
        if want in cols_lower_map:
            df.rename(columns={cols_lower_map[want]: want}, inplace=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")

    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    else:
        df["event_type"] = "pass"
        df.loc[df["outcome"].isin(SHOT_TYPES), "event_type"] = "shot"

    df_pass = df[df["event_type"] == "pass"].copy()
    df_shot = df[df["event_type"] == "shot"].copy()
    df_touch = df[df["outcome"] == "touch"].copy()

    if not df_pass.empty:
        df_pass = df_pass[df_pass["outcome"].isin(PASS_ORDER)]
    if not df_shot.empty:
        df_shot = df_shot[df_shot["outcome"].isin(SHOT_ORDER)]

    out = pd.concat([df_pass, df_shot, df_touch], ignore_index=True)
    if out.empty:
        out = df.copy()
    return out


def apply_pitch_transforms(
    df: pd.DataFrame,
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0
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


# ✅ تعديل مهم: ناخد لون خطوط الملعب من الثيم
def make_pitch(pitch_mode: str = "rect", pitch_width: float = 64.0, theme: Optional[Dict] = None):
    line_color = (theme or {}).get("pitch_lines", "#E6E6E6")
    if pitch_mode == "square":
        return Pitch(pitch_type="custom", pitch_length=100, pitch_width=100, line_zorder=2, line_color=line_color)
    return Pitch(pitch_type="custom", pitch_length=100, pitch_width=pitch_width, line_zorder=2, line_color=line_color)


def _goal_mouth_bounds(pitch_mode: str = "rect", pitch_width: float = 64.0):
    gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    return gy - goal_mouth / 2.0, gy + goal_mouth / 2.0


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


def _meters_distance_approx(x, y, pitch_mode: str = "rect", pitch_width: float = 64.0) -> float:
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


def zone_based_xg(x, y, pitch_mode: str = "rect", pitch_width: float = 64.0):
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


def estimate_xg_zone(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0) -> pd.DataFrame:
    df = df.copy()
    df["xg_zone"] = pd.NA
    mask = df["event_type"] == "shot"
    if mask.any():
        df.loc[mask, "xg_zone"] = [
            round(zone_based_xg(float(x), float(y), pitch_mode=pitch_mode, pitch_width=pitch_width), 3)
            for x, y in zip(df.loc[mask, "x"], df.loc[mask, "y"])
        ]
    return df


# ====== باقي الملف عندك (model/xg + charts + report + shot card + pizza) ======
# ✅ المهم: استبدال كل (dict | None) و (list[str] | None) و (str | None) بنفس أسلوب Optional/Dict/List
# لأن ده سبب الـ SyntaxError.

# ---------------------------------------------------------------------------
# ملاحظة مهمة:
# في الكود اللي بعتّه باقي أجزاء كتير بتستخدم "| None" كمان (outcome_bar / pass_map / shot_map / report / shot_card / pizza_chart)
# لازم تبدلها كلها بنفس الطريقة.
# ---------------------------------------------------------------------------
