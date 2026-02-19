import os
import re
import math
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from mplsoccer import Pitch, PyPizza


# ✅ Helper: YES only (NO/empty = False)
def _yes_only(s: pd.Series) -> pd.Series:
    """
    YES -> True
    NO / Empty / NaN -> False
    Accepts: yes/no, y/n, true/false, 1/0, arabic نعم/لا
    Anything else -> False (safe default)
    """
    if s is None:
        return pd.Series(dtype=bool)

    x = s.copy()
    x = x.replace("", np.nan)
    x = x.fillna(False)

    if pd.api.types.is_numeric_dtype(x):
        return (pd.to_numeric(x, errors="coerce").fillna(0) == 1)

    xs = x.astype(str).str.strip().str.lower()
    true_vals = {"yes", "y", "true", "t", "1", "نعم"}
    return xs.map(lambda v: True if v in true_vals else False).astype(bool)


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
    "Black Stripe": {
        "bg": "#000000",
        "panel": "#000000",
        "pitch": "#000000",
        "pitch_stripe": "#0A0A0A",
        "text": "#FFFFFF",
        "muted": "#B7B7B7",
        "lines": "#2A2A2A",
        "goal": "#FFFFFF",
        "pitch_lines": "#FFFFFF",
    },
}


# ----------------------------
# Outcome normalization (strong)
# ----------------------------
def _clean_text_basic(s: str) -> str:
    s = str(s)
    s = s.replace("\u00a0", " ").replace("Â", " ")
    s = s.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _norm_outcome(s: Any) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = _clean_text_basic(s)

    # remove leading digits e.g. "1ontarget"
    s = re.sub(r"^\d+", "", s).strip()

    # keep "key pass" intact
    if s.endswith(" pass") and s != "key pass":
        s = s[:-5].strip()

    aliases = {
        # shots
        "on target": "ontarget",
        "shot on target": "ontarget",
        "sot": "ontarget",
        "saved": "ontarget",
        "ontarget": "ontarget",

        "offtarget": "off target",
        "off target": "off target",
        "shot off target": "off target",
        "wide": "off target",
        "miss": "off target",

        "goal": "goal",
        "scored": "goal",

        "block": "blocked",
        "blocked": "blocked",
        "blocked shot": "blocked",
        "blk": "blocked",

        # passes
        "keypass": "key pass",
        "key pass": "key pass",
        "kp": "key pass",

        "assist": "assist",

        # success/fail variants
        "successful": "successful",
        "success": "successful",
        "completed": "successful",
        "complete": "successful",
        "accurate": "successful",
        "true": "successful",
        "yes": "successful",
        "1": "successful",

        "unsuccessful": "unsuccessful",
        "unsuccess": "unsuccessful",
        "failed": "unsuccessful",
        "fail": "unsuccessful",
        "incomplete": "unsuccessful",
        "inaccurate": "unsuccessful",
        "false": "unsuccessful",
        "no": "unsuccessful",
        "0": "unsuccessful",

        # touch
        "touch": "touch",
        "ball touch": "touch",
        "receive": "touch",
        "reception": "touch",
        "received": "touch",
    }
    return aliases.get(s, s)


# ----------------------------
# IO
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
        try:
            return pd.read_csv(path, encoding="latin1", encoding_errors="replace")
        except TypeError:
            return pd.read_csv(path, encoding="latin1")
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use CSV or Excel.")


# ----------------------------
# Validate & Clean
# ----------------------------
def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols_lower_map = {c.lower(): c for c in df.columns}

    if "outcome" not in cols_lower_map:
        raise ValueError("Missing column: outcome (required).")
    df.rename(columns={cols_lower_map["outcome"]: "outcome"}, inplace=True)

    for want in ["x", "y", "x2", "y2"]:
        if want in cols_lower_map:
            df.rename(columns={cols_lower_map[want]: want}, inplace=True)

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: %s. Required: %s" % (missing, REQUIRED))

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


# ----------------------------
# Pitch transforms
# ----------------------------
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


def make_pitch(
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme: Optional[dict] = None,
) -> Pitch:
    theme = theme or {}
    pitch_color = theme.get("pitch", "#1f5f3b")
    line_color = theme.get("pitch_lines", "#E6E6E6")
    stripe_color = theme.get("pitch_stripe", None)
    stripe = True if stripe_color else False

    if pitch_mode == "square":
        return Pitch(
            pitch_type="custom",
            pitch_length=100,
            pitch_width=100,
            line_zorder=2,
            pitch_color=pitch_color,
            line_color=line_color,
            stripe=stripe,
            stripe_color=stripe_color,
        )

    return Pitch(
        pitch_type="custom",
        pitch_length=100,
        pitch_width=pitch_width,
        line_zorder=2,
        pitch_color=pitch_color,
        line_color=line_color,
        stripe=stripe,
        stripe_color=stripe_color,
    )


# ----------------------------
# xG (Zone)
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


def _meters_distance_approx(x: float, y: float, pitch_mode: str = "rect", pitch_width: float = 64.0) -> float:
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


def zone_based_xg(x: float, y: float, pitch_mode: str = "rect", pitch_width: float = 64.0) -> float:
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


# ----------------------------
# xG Model (optional)  (KEEP AS-IS)
# ----------------------------
MODEL_FEATURE_COLS = [
    "x", "y",
    "Assisted", "IndividualPlay", "RegularPlay",
    "LeftFoot", "RightFoot",
    "FromCorner", "FirstTouch", "Head",
    "BigChance", "SetPiece", "Volley", "FastBreak",
    "ThrowinSetPiece", "Penalty", "OneOnOne",
    "KeyPass", "OwnGoal", "OtherBodyPart",
    "shot_distance",
    "period_FirstHalf", "period_SecondHalf",
    "Zone_Back", "Zone_Center", "Zone_Left", "Zone_Right",
]


def build_model_features(df_prepared: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0) -> pd.DataFrame:
    shots = df_prepared[df_prepared["event_type"] == "shot"].copy()

    shots["x"] = pd.to_numeric(shots.get("x"), errors="coerce").fillna(0.0)
    shots["y"] = pd.to_numeric(shots.get("y"), errors="coerce").fillna(0.0)

    if "shot_distance" in shots.columns:
        shots["shot_distance"] = pd.to_numeric(shots["shot_distance"], errors="coerce").fillna(0.0)
    else:
        shots["shot_distance"] = shots.apply(
            lambda r: _meters_distance_approx(float(r["x"]), float(r["y"]), pitch_mode, pitch_width),
            axis=1
        )

    if "period" in shots.columns:
        p = shots["period"].astype(str).str.lower()
        shots["period_FirstHalf"] = p.isin(["1", "firsthalf", "first half", "fh", "1st"]).astype(int)
        shots["period_SecondHalf"] = p.isin(["2", "secondhalf", "second half", "sh", "2nd"]).astype(int)
    else:
        shots["period_FirstHalf"] = 0
        shots["period_SecondHalf"] = 0

    for z in ["Zone_Back", "Zone_Center", "Zone_Left", "Zone_Right"]:
        if z in shots.columns:
            shots[z] = pd.to_numeric(shots[z], errors="coerce").fillna(0).astype(int)

    if not all(z in shots.columns for z in ["Zone_Back", "Zone_Center", "Zone_Left", "Zone_Right"]):
        y_max = pitch_width if pitch_mode == "rect" else 100.0
        left_thr = y_max * 0.33
        right_thr = y_max * 0.67
        shots["Zone_Back"] = (shots["x"] < 50).astype(int)
        shots["Zone_Left"] = (shots["y"] < left_thr).astype(int)
        shots["Zone_Right"] = (shots["y"] > right_thr).astype(int)
        shots["Zone_Center"] = ((shots["y"] >= left_thr) & (shots["y"] <= right_thr)).astype(int)

    flag_cols = [
        "Assisted", "IndividualPlay", "RegularPlay",
        "LeftFoot", "RightFoot",
        "FromCorner", "FirstTouch", "Head",
        "BigChance", "SetPiece", "Volley", "FastBreak",
        "ThrowinSetPiece", "Penalty", "OneOnOne",
        "KeyPass", "OwnGoal", "OtherBodyPart",
    ]
    for c in flag_cols:
        if c in shots.columns:
            shots[c] = pd.to_numeric(shots[c], errors="coerce").fillna(0).astype(int)
        else:
            shots[c] = 0

    for c in MODEL_FEATURE_COLS:
        if c not in shots.columns:
            shots[c] = 0

    return shots[MODEL_FEATURE_COLS].copy()


def estimate_xg_model(
    df: pd.DataFrame,
    model_pipe: Any = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0
) -> pd.DataFrame:
    df = df.copy()
    df["xg_model"] = pd.NA

    if model_pipe is None:
        return df

    mask = df["event_type"] == "shot"
    if not mask.any():
        return df

    try:
        X = build_model_features(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

        model = model_pipe
        feature_cols = None
        if isinstance(model_pipe, dict) and "model" in model_pipe:
            model = model_pipe["model"]
            feature_cols = model_pipe.get("feature_cols")

        if feature_cols:
            for c in feature_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[feature_cols]

        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)

        preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
        df.loc[mask, "xg_model"] = np.round(preds, 3).tolist()
        return df
    except Exception:
        return df


# ----------------------------
# Shot end location fix (Opta-ish)
# ----------------------------
def _goal_mouth_bounds(pitch_mode: str = "rect", pitch_width: float = 64.0) -> Tuple[float, float]:
    y_max = pitch_width if pitch_mode == "rect" else 100.0
    mid = y_max / 2.0
    goal_mouth = y_max * (7.32 / 68.0)
    y_low = mid - goal_mouth / 2.0
    y_high = mid + goal_mouth / 2.0
    return float(y_low), float(y_high)


def fix_shot_end_location(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0) -> pd.DataFrame:
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
# Pass tagging (MANUAL) + collision fix
# ----------------------------
def _pass_success_mask(outcome_series: pd.Series) -> pd.Series:
    s = outcome_series.astype(str).str.lower()
    return s.isin(["successful", "key pass", "assist"])


def _norm_name(x: str) -> str:
    x = str(x).strip().lower()
    x = x.replace("_", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def _find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    col_map = {}
    for col in df.columns:
        key = _norm_name(col)
        if key not in col_map:
            col_map[key] = col

    for cand in cands:
        key = _norm_name(cand)
        if key in col_map:
            return col_map[key]
    return None


def add_pass_tags(
    df_prepared: pd.DataFrame,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0
) -> pd.DataFrame:
    df = df_prepared.copy()

    p_src = df[df.get("event_type", "pass") == "pass"].copy()

    ft_col = _find_col(p_src, ["into_final_third", "into final third", "final third"])
    box_col = _find_col(p_src, ["into_penalty_box", "into penalty box", "into box", "penalty box", "box entry"])
    lb_col = _find_col(p_src, ["line_breaking", "line breaking"])
    prog_col = _find_col(p_src, ["progressive_pass", "progressive pass", "progressive"])
    pack_col = _find_col(p_src, ["packing", "packing_proxy", "packing value"])

    for col in [
        "into_final_third", "into_penalty_box", "line_breaking", "progressive_pass",
        "packing_proxy",
        "is_pass_attempt", "is_pass_successful", "is_pass_unsuccessful"
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    p = df[df["event_type"] == "pass"].copy()
    if p.empty:
        df["is_pass_attempt"] = False
        df["is_pass_successful"] = False
        df["is_pass_unsuccessful"] = False
        df["into_final_third"] = False
        df["into_penalty_box"] = False
        df["line_breaking"] = False
        df["progressive_pass"] = False
        df["packing_proxy"] = 0
        return df

    attempt = p["outcome"].isin(PASS_ORDER)
    success = attempt & _pass_success_mask(p["outcome"])
    unsuccess = attempt & (p["outcome"].astype(str).str.lower() == "unsuccessful")

    into_final_third = attempt & (_yes_only(p[ft_col]) if ft_col and ft_col in p.columns else False)
    into_penalty_box = attempt & (_yes_only(p[box_col]) if box_col and box_col in p.columns else False)
    line_breaking = attempt & (_yes_only(p[lb_col]) if lb_col and lb_col in p.columns else False)
    progressive = attempt & (_yes_only(p[prog_col]) if prog_col and prog_col in p.columns else False)

    if pack_col and pack_col in p.columns:
        packing_proxy = pd.to_numeric(p[pack_col], errors="coerce").fillna(0).astype(int)
    else:
        packing_proxy = pd.Series(0, index=p.index, dtype=int)

    idx = p.index
    df.loc[idx, "is_pass_attempt"] = attempt.values
    df.loc[idx, "is_pass_successful"] = success.values
    df.loc[idx, "is_pass_unsuccessful"] = unsuccess.values
    df.loc[idx, "into_final_third"] = into_final_third.values
    df.loc[idx, "into_penalty_box"] = into_penalty_box.values
    df.loc[idx, "line_breaking"] = line_breaking.values
    df.loc[idx, "progressive_pass"] = progressive.values
    df.loc[idx, "packing_proxy"] = packing_proxy.values

    nonp = df["event_type"] != "pass"
    df.loc[nonp, [
        "is_pass_attempt", "is_pass_successful", "is_pass_unsuccessful",
        "into_final_third", "into_penalty_box", "line_breaking", "progressive_pass"
    ]] = False
    df.loc[nonp, "packing_proxy"] = 0

    return df


# ----------------------------
# Prepare df for charts
# ----------------------------
def prepare_df_for_charts(
    df_raw: pd.DataFrame,
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    xg_method: str = "zone",
    model_pipe: Any = None,
) -> pd.DataFrame:
    df = validate_and_clean(df_raw)
    df = apply_pitch_transforms(df, attack_direction, flip_y, pitch_mode, pitch_width)
    df = fix_shot_end_location(df, pitch_mode=pitch_mode, pitch_width=pitch_width)

    df = estimate_xg_zone(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    df = estimate_xg_model(df, model_pipe=model_pipe, pitch_mode=pitch_mode, pitch_width=pitch_width)

    df["xg"] = pd.to_numeric(df["xg_zone"], errors="coerce")
    df["xg_source"] = "zone"

    xg_method = (xg_method or "zone").strip().lower()
    if xg_method == "model":
        if df["xg_model"].notna().any():
            df["xg"] = pd.to_numeric(df["xg_model"], errors="coerce")
            df["xg_source"] = "model"
        else:
            df["xg"] = pd.to_numeric(df["xg_zone"], errors="coerce")
            df["xg_source"] = "zone (fallback)"

    df = add_pass_tags(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
    return df


# ----------------------------
# Theming helpers
# ----------------------------
def _apply_fig_theme(fig, ax, theme: dict):
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["panel"])


def _draw_pitch(ax, pitch: Pitch, theme: dict):
    pitch.draw(ax=ax)
    ax.set_facecolor(theme["pitch"])


def _add_legend(ax, handles, theme: dict, loc: str = "lower center"):
    if not handles:
        return
    leg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=9
    )
    for t in leg.get_texts():
        t.set_color(theme.get("text", "white"))


# ----------------------------
# Header for PDF charts
# ----------------------------
def add_report_header(
    fig,
    title: str = "",
    subtitle: str = "",
    header_image=None,
    img_side: str = "left",
    img_width_frac: float = 0.10,
    theme_name: str = "The Athletic Dark",
    title_align: str = "center",
    subtitle_align: str = "center",
    title_fontsize: int = 16,
    subtitle_fontsize: int = 11,
    title_color: Optional[str] = None,
    subtitle_color: Optional[str] = None,
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    title_color = title_color or theme["text"]
    subtitle_color = subtitle_color or theme["muted"]

    title = (title or "").strip()
    subtitle = (subtitle or "").strip()

    fig.subplots_adjust(top=0.84)

    def _align_to_x_ha(align: str):
        a = (align or "center").lower().strip()
        if a == "left":
            return 0.08, "left"
        if a == "right":
            return 0.92, "right"
        return 0.50, "center"

    tx, tha = _align_to_x_ha(title_align)
    sx, sha = _align_to_x_ha(subtitle_align)

    if title:
        fig.text(tx, 0.965, title, ha=tha, va="top",
                 color=title_color, fontsize=title_fontsize, weight="bold")
    if subtitle:
        fig.text(sx, 0.935, subtitle, ha=sha, va="top",
                 color=subtitle_color, fontsize=subtitle_fontsize)

    if header_image is None:
        return

    try:
        img = header_image
        if hasattr(img, "convert"):
            img = img.convert("RGBA")
            img_arr = np.asarray(img)
        else:
            img_arr = np.asarray(img)

        img_side = (img_side or "left").lower().strip()
        w = float(max(0.05, min(0.20, img_width_frac)))
        h = w
        y0 = 0.895
        x0 = 0.02 if img_side != "right" else (0.98 - w)

        ax_img = fig.add_axes([x0, y0, w, h], zorder=50)
        ax_img.imshow(img_arr)
        ax_img.axis("off")
        ax_img.set_facecolor("none")
    except Exception:
        return


# ----------------------------
# Charts
# ----------------------------
def outcome_bar(df: pd.DataFrame, bar_colors: Optional[dict] = None, theme_name: str = "The Athletic Dark"):
    bar_colors = bar_colors or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    counts = df["outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_fig_theme(fig, ax, theme)

    labels = counts.index.astype(str).tolist()
    fallback = theme.get("muted", "#A0A7B4")
    colors = [bar_colors.get(k, fallback) for k in labels]

    ax.bar(labels, counts.values, color=colors)
    ax.set_title("Outcome Distribution", color=theme["text"])
    ax.set_ylabel("Count", color=theme["muted"])
    ax.tick_params(axis="x", rotation=25, colors=theme["muted"])
    ax.tick_params(axis="y", colors=theme["muted"])
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])

    handles = [Patch(facecolor=bar_colors.get(k, fallback), label=k) for k in labels[:8]]
    _add_legend(ax, handles, theme, loc="upper center")
    return fig


def start_location_heatmap(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0, theme_name: str = "The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    try:
        pitch.kdeplot(df["x"], df["y"], ax=ax, fill=True, levels=50, alpha=0.7)
    except Exception:
        pitch.scatter(df["x"], df["y"], ax=ax, s=25, alpha=0.6)

    ax.set_title("Start Locations Heatmap", color=theme["text"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    handles = [Patch(facecolor=theme.get("muted", "#A0A7B4"), label="Density / Events")]
    _add_legend(ax, handles, theme, loc="upper center")
    return fig


def touch_map(
    df: pd.DataFrame,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
    dot_color: str = "#34D5FF",
    edge_color: str = "#0B0F14",
    dot_size: int = 220,
    alpha: float = 0.95,
    marker: str = "o",
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)

    d = df.copy()
    if "outcome" in d.columns:
        s = d["outcome"].astype(str).str.strip().str.lower()
        if (s == "touch").any():
            d = d[s == "touch"].copy()

    if "x" not in d.columns or "y" not in d.columns:
        raise ValueError("Touch Map يحتاج أعمدة: x, y")

    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d["y"] = pd.to_numeric(d["y"], errors="coerce")
    d = d.dropna(subset=["x", "y"]).copy()

    fig, ax = plt.subplots(figsize=(12, 7.2))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    pitch.scatter(
        d["x"], d["y"],
        ax=ax,
        s=dot_size,
        marker=marker,
        color=dot_color,
        edgecolors=edge_color,
        linewidth=2,
        alpha=alpha,
        zorder=5
    )

    ax.set_title("Touch Map", color=theme["text"], fontsize=18, weight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    handles = [Line2D([0], [0], marker=marker, color="none", markerfacecolor=dot_color,
                      markeredgecolor=edge_color, markersize=10, label="Touches")]
    _add_legend(ax, handles, theme, loc="upper center")
    return fig


# ----------------------------
# Pass map filters (robust)
# ----------------------------
def _bool_mask(col, index: pd.Index) -> pd.Series:
    if isinstance(col, pd.Series):
        s = col.reindex(index)
        if pd.api.types.is_bool_dtype(s):
            return s.fillna(False)
        s = s.replace("", pd.NA).fillna(False)
        try:
            return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "نعم"])
        except Exception:
            return pd.Series(False, index=index, dtype=bool)
    return pd.Series(False, index=index, dtype=bool)


def _empty_pass_map_figure(pitch_mode: str, pitch_width: float, theme: dict, title: str, msg: str):
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    ax.set_title(title, color=theme["text"])
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color=theme.get("text", "white"), wrap=True)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig


def _filter_passes_for_map(d: pd.DataFrame, pass_view: str = "All passes", result_scope: str = "Attempts (all)", min_packing: int = 1) -> pd.DataFrame:
    dd = d.copy()
    if dd.empty:
        return dd

    view = (pass_view or "All passes").lower().strip()
    idx = dd.index

    if "final third" in view:
        dd = dd[_bool_mask(dd.get("into_final_third", False), idx)].copy()
    elif "penalty box" in view or "penalty" in view or "box" in view:
        dd = dd[_bool_mask(dd.get("into_penalty_box", False), idx)].copy()
    elif "line" in view or "breaking" in view:
        dd = dd[pd.to_numeric(dd.get("packing_proxy", 0), errors="coerce").fillna(0).astype(int) >= int(min_packing)].copy()
    elif "progressive" in view:
        dd = dd[_bool_mask(dd.get("progressive_pass", False), idx)].copy()

    if dd.empty:
        return dd

    scope = (result_scope or "Attempts (all)").lower().strip()
    idx2 = dd.index

    if "successful" in scope:
        dd = dd[_bool_mask(dd.get("is_pass_successful", False), idx2)].copy()
    elif "unsuccessful" in scope or "failed" in scope:
        dd = dd[_bool_mask(dd.get("is_pass_unsuccessful", False), idx2)].copy()
    else:
        if "is_pass_attempt" in dd.columns:
            dd = dd[_bool_mask(dd["is_pass_attempt"], idx2)].copy()

    return dd


def pass_map(
    df: pd.DataFrame,
    pass_colors: Optional[dict] = None,
    pass_markers: Optional[dict] = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
    pass_view: str = "All passes",
    result_scope: str = "Attempts (all)",
    min_packing: int = 1,
):
    pass_colors = pass_colors or {}
    pass_markers = pass_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    d = df[df["event_type"] == "pass"].copy()

    if "x2" in d.columns:
        d["x2"] = pd.to_numeric(d["x2"], errors="coerce")
    else:
        d["x2"] = np.nan
    if "y2" in d.columns:
        d["y2"] = pd.to_numeric(d["y2"], errors="coerce")
    else:
        d["y2"] = np.nan

    d = _filter_passes_for_map(d, pass_view=pass_view, result_scope=result_scope, min_packing=min_packing)

    title = f"Pass Map — {pass_view} — {result_scope}"
    if "line" in (pass_view or "").lower():
        title += f" (min packing {int(min_packing)})"

    if d.empty:
        return _empty_pass_map_figure(
            pitch_mode=pitch_mode,
            pitch_width=pitch_width,
            theme=theme,
            title=title,
            msg="No passes match the selected filters.\nTry changing Pass View / Scope or lowering min packing."
        )

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    has_end = d["x2"].notna() & d["y2"].notna()
    d_end = d[has_end].copy()

    for t in PASS_ORDER:
        dt = d_end[d_end["outcome"] == t]
        if len(dt) == 0:
            continue
        pitch.arrows(
            dt["x"], dt["y"], dt["x2"], dt["y2"],
            ax=ax, width=2, alpha=0.85,
            color=pass_colors.get(t, theme.get("muted", "#A0A7B4"))
        )

    for t in PASS_ORDER:
        dt_all = d[d["outcome"] == t]
        if len(dt_all) == 0:
            continue
        mk = pass_markers.get(t, "o")
        pitch.scatter(
            dt_all["x"], dt_all["y"],
            ax=ax,
            s=90,
            marker=mk,
            color=pass_colors.get(t, theme.get("muted", "#A0A7B4")),
            edgecolors="white",
            linewidth=1.2,
            alpha=0.95,
            zorder=6
        )

    missing_end_n = int((~has_end).sum())
    if missing_end_n > 0:
        title += f"  |  missing x2/y2: {missing_end_n}"

    ax.set_title(title, color=theme["text"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    handles = []
    for t in PASS_ORDER:
        if t in pass_colors:
            mk = pass_markers.get(t, "o")
            handles.append(Line2D([0], [0], marker=mk, color="none",
                                  markerfacecolor=pass_colors.get(t),
                                  markeredgecolor="white",
                                  markersize=8, label=t))
    _add_legend(ax, handles, theme, loc="upper center")
    return fig


def shot_map(
    df: pd.DataFrame,
    shot_colors: Optional[dict] = None,
    shot_markers: Optional[dict] = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    show_xg: bool = False,
    theme_name: str = "The Athletic Dark"
):
    shot_colors = shot_colors or {}
    shot_markers = shot_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    s = df[df["event_type"] == "shot"].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue

        mk = shot_markers.get(t, "o")
        pitch.scatter(
            stt["x"], stt["y"],
            ax=ax,
            s=160,
            marker=mk,
            color=shot_colors.get(t, theme.get("muted", "#A0A7B4")),
            edgecolors="white",
            linewidth=1.6,
            alpha=0.95,
            zorder=5
        )

        if show_xg and "xg" in stt.columns:
            for _, r in stt.iterrows():
                try:
                    ax.text(float(r["x"]) + 1.0, float(r["y"]) + 1.0,
                            f'{float(r["xg"]):.2f}', fontsize=9, color="white", weight="bold")
                except Exception:
                    pass

    xg_src = str(df["xg_source"].iloc[0]) if ("xg_source" in df.columns and len(df)) else ""
    ax.set_title(("Shot Map — xG: %s" % xg_src).strip(), color=theme["text"])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    handles = []
    for t in SHOT_ORDER:
        if t in shot_colors:
            mk = shot_markers.get(t, "o")
            handles.append(Line2D([0], [0], marker=mk, color="none",
                                  markerfacecolor=shot_colors.get(t),
                                  markeredgecolor="white",
                                  markersize=8, label=t))
    _add_legend(ax, handles, theme, loc="upper center")
    return fig


# ----------------------------
# Report builder
# ----------------------------
def build_report_from_prepared_df(
    df_prepared: pd.DataFrame,
    out_dir: str,
    title: str = "Match Report",
    subtitle: str = "",
    header_image=None,
    header_img_side: str = "left",
    header_img_width_frac: float = 0.10,
    title_align: str = "center",
    subtitle_align: str = "center",
    title_fontsize: int = 16,
    subtitle_fontsize: int = 11,
    title_color: Optional[str] = None,
    subtitle_color: Optional[str] = None,
    theme_name: str = "The Athletic Dark",
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    pass_colors: Optional[dict] = None,
    pass_markers: Optional[dict] = None,
    shot_colors: Optional[dict] = None,
    shot_markers: Optional[dict] = None,
    bar_colors: Optional[dict] = None,
    charts_to_include: Optional[List[str]] = None,
    touch_dot_color: str = "#34D5FF",
    touch_dot_edge: str = "#0B0F14",
    touch_dot_size: int = 220,
    touch_alpha: float = 0.95,
    touch_marker: str = "o",
    **kwargs,
):
    pass_view = kwargs.get("pass_view", "All passes")
    pass_result_scope = kwargs.get("pass_result_scope", "Attempts (all)")
    try:
        pass_min_packing = int(kwargs.get("pass_min_packing", 1))
    except Exception:
        pass_min_packing = 1

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    df2 = df_prepared.copy()
    charts_to_include = charts_to_include or ["Outcome Bar", "Start Heatmap", "Touch Map (Scatter)", "Pass Map", "Shot Map"]

    figs = []

    if "Outcome Bar" in charts_to_include:
        figs.append(("outcome_bar", outcome_bar(df2, bar_colors=bar_colors, theme_name=theme_name)))

    if "Start Heatmap" in charts_to_include:
        figs.append(("start_heatmap", start_location_heatmap(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))

    if "Touch Map (Scatter)" in charts_to_include:
        figs.append(("touch_map", touch_map(
            df2,
            pitch_mode=pitch_mode,
            pitch_width=pitch_width,
            theme_name=theme_name,
            dot_color=touch_dot_color,
            edge_color=touch_dot_edge,
            dot_size=touch_dot_size,
            alpha=touch_alpha,
            marker=touch_marker,
        )))

    if "Pass Map" in charts_to_include:
        figs.append(("pass_map", pass_map(
            df2,
            pass_colors=pass_colors,
            pass_markers=pass_markers,
            pitch_mode=pitch_mode,
            pitch_width=pitch_width,
            theme_name=theme_name,
            pass_view=pass_view,
            result_scope=pass_result_scope,
            min_packing=pass_min_packing,
        )))

    if "Shot Map" in charts_to_include:
        figs.append(("shot_map", shot_map(
            df2,
            shot_colors=shot_colors,
            shot_markers=shot_markers,
            pitch_mode=pitch_mode,
            pitch_width=pitch_width,
            show_xg=True,
            theme_name=theme_name
        )))

    pngs = []
    with PdfPages(pdf_path) as pdf:
        for name, fig in figs:
            add_report_header(
                fig,
                title=title,
                subtitle=subtitle,
                header_image=header_image,
                img_side=header_img_side,
                img_width_frac=header_img_width_frac,
                theme_name=theme_name,
                title_align=title_align,
                subtitle_align=subtitle_align,
                title_fontsize=title_fontsize,
                subtitle_fontsize=subtitle_fontsize,
                title_color=title_color,
                subtitle_color=subtitle_color,
            )

            png_path = os.path.join(out_dir, "%s.png" % name)
            fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            pngs.append(png_path)

    return pdf_path, pngs


# =========================================================
# ✅ Pizza Chart (PRO) — center image + footer + Opta-ish look
# =========================================================
def pizza_chart(
    df_pizza: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    slice_colors: Optional[List[str]] = None,
    show_values_legend: bool = True,
    center_image=None,
    center_img_scale: float = 0.22,  # 0.18-0.28 مناسب
    footer_text: str = "",           # مثال: "Data:fbref | By: Essam Mohamed"
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

    bg = "#2B2B2B"  # قريب من الصورة اللي بعتهالك
    ring = "#0B0F14"
    muted = "#E6E6E6"

    pizza = PyPizza(
        params=params,
        background_color=bg,
        straight_line_color=ring,
        straight_line_lw=2.0,
        last_circle_color=muted,
        last_circle_lw=2.3,
        other_circle_ls="--",
        other_circle_lw=1.6,
        other_circle_color="#3A3F46",
    )

    try:
        fig, ax = pizza.make_pizza(
            values,
            figsize=(10, 10),
            blank_alpha=0.22,
            slice_colors=slice_colors,
            kwargs_slices=dict(edgecolor=ring, linewidth=1.8),
            kwargs_params=dict(color="white", fontsize=15, fontweight="bold"),
            kwargs_values=dict(
                color="white",
                fontsize=13,
                fontweight="bold",
                bbox=dict(
                    edgecolor=ring,
                    facecolor="#1E2126",
                    boxstyle="round,pad=0.25",
                    linewidth=1.2,
                ),
            ),
        )
    except TypeError:
        fig, ax = pizza.make_pizza(
            values,
            figsize=(10, 10),
            blank_alpha=0.22,
            value_bck_colors=slice_colors,
            kwargs_slices=dict(edgecolor=ring, linewidth=1.8),
            kwargs_params=dict(color="white", fontsize=15, fontweight="bold"),
            kwargs_values=dict(color="white", fontsize=13, fontweight="bold"),
        )

    fig.patch.set_facecolor(bg)

    # titles
    fig.text(0.5, 0.975, (title or "").strip(), ha="center", va="top",
             color="white", fontsize=24, fontweight="bold", family="serif")
    fig.text(0.5, 0.945, (subtitle or "").strip(), ha="center", va="top",
             color="#D0D4DA", fontsize=16, family="serif")

    # footer
    if footer_text:
        fig.text(0.98, 0.03, footer_text, ha="right", va="bottom",
                 color="white", fontsize=12)

    # ✅ center image
    if center_image is not None:
        try:
            img = center_image
            if hasattr(img, "convert"):
                img = img.convert("RGBA")
            img_arr = np.asarray(img)

            s = float(center_img_scale)
            s = max(0.12, min(0.32, s))

            ax_img = fig.add_axes([0.5 - s / 2.0, 0.5 - s / 2.0, s, s], zorder=50)
            ax_img.imshow(img_arr)
            ax_img.axis("off")
            ax_img.set_facecolor("none")

            # circular crop
            circ = plt.Circle((0.5, 0.5), 0.5, transform=ax_img.transAxes,
                              facecolor="none", edgecolor="none")
            ax_img.add_patch(circ)
            ax_img.set_clip_path(circ)

            # ring border
            ring1 = plt.Circle((0.5, 0.5), 0.5, transform=ax_img.transAxes,
                               fill=False, edgecolor=ring, linewidth=6.0, alpha=1.0)

            ax_img.add_patch(ring1)
           
        except Exception:
            pass

    # optional legend block
    if show_values_legend:
        lines = [f"{m}: {v}   (pct {p:.1f})" for m, v, p in zip(params, value_text, values)]
        fig.text(
            0.02, 0.02, "\n".join(lines),
            ha="left", va="bottom",
            color="white", fontsize=10, family="monospace"
        )

    # cleanup spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


# ----------------------------
# Shot Detail Card
# ----------------------------
def shot_detail_card(
    df_prepared: pd.DataFrame,
    shot_index: int,
    title: str = "Shot Detail",
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    shot_colors: Optional[dict] = None,
    shot_markers: Optional[dict] = None,
    theme_name: str = "The Athletic Dark",
):
    shot_colors = shot_colors or {
        "off target": "#FF8A00",
        "ontarget": "#00C2FF",
        "goal": "#00FF6A",
        "blocked": "#AAAAAA",
    }
    shot_markers = shot_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    shots = df_prepared[df_prepared["event_type"] == "shot"].copy().reset_index(drop=True)
    if shots.empty:
        raise ValueError("No shots found.")
    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError("Shot index out of range.")

    r = shots.iloc[shot_index]

    xg_txt = "NA"
    try:
        xg_txt = "%.2f" % float(r.get("xg"))
    except Exception:
        pass

    xg_src = str(r.get("xg_source", "")).strip()

    outcome = str(r.get("outcome", "")).lower()
    display_outcome = "On target" if outcome == "ontarget" else outcome.title()
    c = shot_colors.get(outcome, "#00C2FF")
    mk = shot_markers.get(outcome, "o")

    fig = plt.figure(figsize=(12, 6), facecolor=theme["bg"])
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[0.25, 1.0], wspace=0.08, hspace=0.05)

    ax_goal = fig.add_subplot(gs[0, 0])
    ax_pitch = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])

    ax_goal.set_facecolor(theme["panel"])
    ax_goal.set_xlim(0, 100)
    ax_goal.set_ylim(0, 30)
    ax_goal.axis("off")
    ax_goal.plot([25, 75], [5, 5], lw=2, color=theme["goal"])
    ax_goal.plot([25, 25], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([75, 75], [5, 22], lw=2, color=theme["goal"])
    ax_goal.plot([25, 75], [22, 22], lw=2, color=theme["goal"])

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor(theme["pitch"])
    ax_pitch.set_xlim(-2, 102)
    ax_pitch.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)

    x, y = float(r["x"]), float(r["y"])
    pitch.scatter([x], [y], ax=ax_pitch, s=520, marker=mk, color=c, edgecolors="white", linewidth=2, zorder=5, clip_on=False)
    pitch.scatter([x], [y], ax=ax_pitch, s=190, marker=mk, color="white", alpha=0.25, zorder=6, clip_on=False)
    ax_pitch.text(x + 1.2, y + 1.2, "xG %s" % xg_txt, color="white", fontsize=12, weight="bold", zorder=10)

    has_end = ("x2" in shots.columns and "y2" in shots.columns and pd.notna(r.get("x2")) and pd.notna(r.get("y2")))
    y_low, y_high = _goal_mouth_bounds(pitch_mode, pitch_width)

    if has_end:
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color="white", alpha=0.9, zorder=4)
        pitch.scatter([x2], [y2], ax=ax_pitch, s=140, marker="o", color="white", alpha=0.9, zorder=6, clip_on=False)

        def map_to_mini_goal(y_val: float) -> float:
            y_val = float(y_val)
            y_clamped = max(y_low, min(y_high, y_val))
            t = (y_clamped - y_low) / (y_high - y_low + 1e-9)
            return 25 + t * 50

        gx = map_to_mini_goal(y2)
        ax_goal.scatter([gx], [12], s=240, marker=mk, color=c, edgecolors="white", linewidth=2, zorder=5)
    else:
        gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
        ax_pitch.plot([x, 100], [y, gy], linestyle=":", linewidth=3, color="white", alpha=0.6, zorder=4)

    ax_info.set_facecolor(theme["panel"])
    ax_info.axis("off")

    ax_info.text(0.02, 0.94, title, color=theme["text"], fontsize=18, weight="bold", transform=ax_info.transAxes)
    if xg_src:
        ax_info.text(0.02, 0.89, "xG source: %s" % xg_src, color=theme["muted"], fontsize=12, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.80, "xG", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.72, xg_txt, color=theme["text"], fontsize=26, weight="bold", transform=ax_info.transAxes)
    ax_info.plot([0.02, 0.98], [0.67, 0.67], color=theme["lines"], lw=2, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.55, "Outcome", color=theme["muted"], fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.47, display_outcome, color=theme["text"], fontsize=26, weight="bold", transform=ax_info.transAxes)

    return fig, shots
