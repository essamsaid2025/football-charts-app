import os
import re
import math
from typing import Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from mplsoccer import Pitch, PyPizza

# =========================================================
# THEMES
# =========================================================
THEMES = {
    "The Athletic Dark": {
        "bg": "#0E1117",
        "panel": "#111827",
        "panel_2": "#0F172A",
        "pitch": "#1f5f3b",
        "text": "white",
        "muted": "#A0A7B4",
        "lines": "#2A3240",
        "goal": "#E6E6E6",
        "pitch_lines": "#E6E6E6",
        "accent": "#38BDF8",
        "accent_2": "#22C55E",
        "danger": "#EF4444",
        "warning": "#F59E0B",
        "success": "#22C55E",
        "legend_bg": "#111827",
        "legend_border": "#334155",
        "legend_text": "#F3F4F6",
    },
    "Opta Analyst Light": {
        "bg": "#ECECEC",
        "panel": "#F5F5F5",
        "panel_2": "#E9E9E9",
        "pitch": "#ECECEC",
        "pitch_stripe": None,
        "text": "#201C2B",
        "muted": "#7A7584",
        "lines": "#A7A7A7",
        "goal": "#8F8F8F",
        "pitch_lines": "#9F9F9F",
        "accent": "#6D28D9",
        "accent_2": "#8B5CF6",
        "danger": "#D64045",
        "warning": "#B0B0B0",
        "success": "#22A06B",
        "legend_bg": "#F5F5F5",
        "legend_border": "#B8B8B8",
        "legend_text": "#201C2B",
    },
    "Black Stripe": {
        "bg": "#000000",
        "panel": "#000000",
        "panel_2": "#050505",
        "pitch": "#000000",
        "pitch_stripe": "#0A0A0A",
        "text": "#FFFFFF",
        "muted": "#B7B7B7",
        "lines": "#2A2A2A",
        "goal": "#FFFFFF",
        "pitch_lines": "#FFFFFF",
        "accent": "#38BDF8",
        "accent_2": "#22C55E",
        "danger": "#EF4444",
        "warning": "#D1D5DB",
        "success": "#22C55E",
        "legend_bg": "#000000",
        "legend_border": "#333333",
        "legend_text": "#FFFFFF",
    },
}

PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]

DEF_ACTION_COLS = [
    "interception",
    "tackle",
    "recovery",
    "aerial_duel",
    "ground_duel",
    "clearance",
]

# =========================================================
# HELPERS
# =========================================================
def _yes_only(s: pd.Series) -> pd.Series:
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
    s = re.sub(r"^\d+", "", s).strip()

    if s.endswith(" pass") and s != "key pass":
        s = s[:-5].strip()

    aliases = {
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

        "keypass": "key pass",
        "key pass": "key pass",
        "kp": "key pass",

        "assist": "assist",

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

        "touch": "touch",
        "ball touch": "touch",
        "receive": "receive",
        "reception": "receive",
        "received": "receive",

        "carry": "carry",
        "dribble": "carry",
    }
    return aliases.get(s, s)


def _standardize_defensive_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}

    for c in out.columns:
        c0 = str(c).strip().lower()

        if c0 == "x":
            rename_map[c] = "x"
        elif c0 == "y":
            rename_map[c] = "y"
        elif c0 in ["interception", "interceptions"]:
            rename_map[c] = "interception"
        elif c0 in ["tackle", "tackles"]:
            rename_map[c] = "tackle"
        elif c0 in ["recovery", "recoveries"]:
            rename_map[c] = "recovery"
        elif c0 in ["aerial duel", "aerial duels", "aerial_duel", "aerial_duels"]:
            rename_map[c] = "aerial_duel"
        elif c0 in ["ground duel", "ground duels", "ground_duel", "ground_duels"]:
            rename_map[c] = "ground_duel"
        elif c0 in ["clearance", "clearances"]:
            rename_map[c] = "clearance"
        elif c0 == "outcome":
            rename_map[c] = "outcome"

    if rename_map:
        out = out.rename(columns=rename_map)

    return out


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


def _is_no_marker(marker) -> bool:
    return marker is None or str(marker).strip().lower() in {"none", "no marker", "null", ""}


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
        frameon=True,
        fontsize=9,
        fancybox=True,
        borderpad=0.6,
    )
    frame = leg.get_frame()
    frame.set_facecolor(theme.get("legend_bg", theme.get("panel", "white")))
    frame.set_edgecolor(theme.get("legend_border", theme.get("lines", "#CCCCCC")))
    frame.set_alpha(0.96)
    for t in leg.get_texts():
        t.set_color(theme.get("legend_text", theme.get("text", "black")))


# =========================================================
# IO
# =========================================================
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


# =========================================================
# VALIDATE / CLEAN
# =========================================================
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
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")

    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()
    df = _standardize_defensive_columns(df)

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    else:
        df["event_type"] = "other"
        df.loc[df["outcome"].isin(PASS_ORDER), "event_type"] = "pass"
        df.loc[df["outcome"].isin(SHOT_TYPES), "event_type"] = "shot"
        df.loc[df["outcome"] == "touch", "event_type"] = "touch"
        df.loc[df["outcome"] == "receive", "event_type"] = "receive"
        df.loc[df["outcome"] == "carry", "event_type"] = "carry"

    available_def_cols = [c for c in DEF_ACTION_COLS if c in df.columns]
    if available_def_cols:
        def_mask = pd.Series(False, index=df.index)
        for c in available_def_cols:
            def_mask = def_mask | _yes_only(df[c])
        df.loc[def_mask, "event_type"] = "defensive"

    return df


# =========================================================
# PITCH / TRANSFORMS
# =========================================================
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


# =========================================================
# xG
# =========================================================
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
        ("0-6", "big"): 0.55, ("0-6", "mid"): 0.45, ("0-6", "small"): 0.32,
        ("6-12", "big"): 0.32, ("6-12", "mid"): 0.22, ("6-12", "small"): 0.12,
        ("12-18", "big"): 0.18, ("12-18", "mid"): 0.10, ("12-18", "small"): 0.05,
        ("18-25", "big"): 0.08, ("18-25", "mid"): 0.05, ("18-25", "small"): 0.03,
        ("25+", "big"): 0.04, ("25+", "mid"): 0.025, ("25+", "small"): 0.015,
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

    for c in ["Assisted", "IndividualPlay", "RegularPlay", "LeftFoot", "RightFoot", "Head", "BigChance", "SetPiece", "Volley", "FastBreak", "Penalty", "OneOnOne", "KeyPass", "OtherBodyPart"]:
        if c not in shots.columns:
            shots[c] = 0

    model_cols = [c for c in shots.columns if c in [
        "x", "y", "Assisted", "IndividualPlay", "RegularPlay", "LeftFoot", "RightFoot", "Head", "BigChance", "SetPiece", "Volley", "FastBreak", "Penalty", "OneOnOne", "KeyPass", "OtherBodyPart", "shot_distance"
    ]]
    return shots[model_cols].copy()


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
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
        preds = np.clip(np.asarray(preds, dtype=float), 0.0, 1.0)
        df.loc[mask, "xg_model"] = np.round(preds, 3).tolist()
        return df
    except Exception:
        return df


# =========================================================
# SHOT END FIX
# =========================================================
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


# =========================================================
# EXTRA TAGS
# =========================================================
def _pass_success_mask(outcome_series: pd.Series) -> pd.Series:
    s = outcome_series.astype(str).str.lower()
    return s.isin(["successful", "key pass", "assist"])


def add_pass_tags(df_prepared: pd.DataFrame) -> pd.DataFrame:
    df = df_prepared.copy()

    ft_col = _find_col(df, ["into_final_third", "into final third", "final third"])
    box_col = _find_col(df, ["into_penalty_box", "into penalty box", "into box", "penalty box", "box entry"])
    lb_col = _find_col(df, ["line_breaking", "line breaking"])
    prog_col = _find_col(df, ["progressive_pass", "progressive pass", "progressive"])
    pack_col = _find_col(df, ["packing", "packing_proxy", "packing value"])
    prog_carry_col = _find_col(df, ["progressive_carry", "progressive carry"])
    receive_col = _find_col(df, ["receive", "received", "reception"])
    carry_col = _find_col(df, ["carry", "dribble"])

    for col in [
        "into_final_third", "into_penalty_box", "line_breaking", "progressive_pass",
        "packing_proxy", "progressive_carry", "is_receive", "is_carry",
        "is_pass_attempt", "is_pass_successful", "is_pass_unsuccessful"
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    p = df[df["event_type"] == "pass"].copy()
    if not p.empty:
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

    if prog_carry_col and prog_carry_col in df.columns:
        df["progressive_carry"] = _yes_only(df[prog_carry_col])
    else:
        df["progressive_carry"] = df["event_type"].eq("carry")

    if receive_col and receive_col in df.columns:
        df["is_receive"] = _yes_only(df[receive_col])
    else:
        df["is_receive"] = df["event_type"].eq("receive")

    if carry_col and carry_col in df.columns:
        df["is_carry"] = _yes_only(df[carry_col])
    else:
        df["is_carry"] = df["event_type"].eq("carry")

    return df


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

    df = add_pass_tags(df)
    return df


# =========================================================
# CORE CHARTS
# =========================================================
def outcome_bar(df: pd.DataFrame, theme_name: str = "The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    counts = df["outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    _apply_fig_theme(fig, ax, theme)
    ax.bar(counts.index.astype(str).tolist(), counts.values, color=theme.get("accent", "#6D28D9"))
    ax.set_title("Outcome Distribution", color=theme["text"])
    ax.tick_params(axis="x", rotation=25, colors=theme["muted"])
    ax.tick_params(axis="y", colors=theme["muted"])
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])
    return fig


def start_location_heatmap(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0, theme_name: str = "The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)
    try:
        pitch.kdeplot(df["x"], df["y"], ax=ax, fill=True, levels=50, alpha=0.7)
    except Exception:
        pitch.scatter(df["x"], df["y"], ax=ax, s=25, alpha=0.6, color=theme.get("accent", "#6D28D9"))
    ax.set_title("Start Locations Heatmap", color=theme["text"])
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
    d = d[(d["event_type"] == "touch") | (d["outcome"] == "touch")].copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)
    if not d.empty and not _is_no_marker(marker):
        pitch.scatter(d["x"], d["y"], ax=ax, s=dot_size, marker=marker, color=dot_color, edgecolors=edge_color, linewidth=1.5, alpha=alpha, zorder=5)
    ax.set_title("Touch Map", color=theme["text"])
    return fig


def _filter_passes_for_map(d: pd.DataFrame, pass_view: str = "All passes", result_scope: str = "Attempts (all)", min_packing: int = 1) -> pd.DataFrame:
    dd = d.copy()
    if dd.empty:
        return dd
    view = (pass_view or "All passes").lower().strip()
    idx = dd.index

    if "final third" in view:
        dd = dd[_bool_mask(dd.get("into_final_third", False), idx)].copy()
    elif "penalty box" in view or "box" in view:
        dd = dd[_bool_mask(dd.get("into_penalty_box", False), idx)].copy()
    elif "line" in view:
        dd = dd[pd.to_numeric(dd.get("packing_proxy", 0), errors="coerce").fillna(0).astype(int) >= int(min_packing)].copy()
    elif "progressive" in view:
        dd = dd[_bool_mask(dd.get("progressive_pass", False), idx)].copy()

    if dd.empty:
        return dd

    scope = (result_scope or "Attempts (all)").lower().strip()
    idx2 = dd.index

    if "successful" in scope:
        dd = dd[_bool_mask(dd.get("is_pass_successful", False), idx2)].copy()
    elif "unsuccessful" in scope:
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
    if "x2" not in d.columns:
        d["x2"] = np.nan
    if "y2" not in d.columns:
        d["y2"] = np.nan

    d = _filter_passes_for_map(d, pass_view=pass_view, result_scope=result_scope, min_packing=min_packing)

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    for t in PASS_ORDER:
        dt = d[d["outcome"] == t]
        if len(dt) == 0:
            continue
        color = pass_colors.get(t, theme.get("accent", "#6D28D9"))
        pitch.arrows(dt["x"], dt["y"], dt["x2"], dt["y2"], ax=ax, width=2, alpha=0.85, color=color)
        mk = pass_markers.get(t, "o")
        if not _is_no_marker(mk):
            pitch.scatter(dt["x"], dt["y"], ax=ax, s=70, marker=mk, color=color, edgecolors="white", linewidth=1.0, alpha=0.95, zorder=6)

    ax.set_title(f"Pass Map — {pass_view}", color=theme["text"])
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
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue
        mk = shot_markers.get(t, "o")
        color = shot_colors.get(t, theme.get("accent", "#6D28D9"))
        if not _is_no_marker(mk):
            pitch.scatter(stt["x"], stt["y"], ax=ax, s=160, marker=mk, color=color, edgecolors="white", linewidth=1.6, alpha=0.95, zorder=5)
        if show_xg and "xg" in stt.columns:
            for _, r in stt.iterrows():
                ax.text(float(r["x"]) + 1.0, float(r["y"]) + 1.0, f'{float(r["xg"]):.2f}', fontsize=8, color=theme["text"], weight="bold")

    xg_src = str(df["xg_source"].iloc[0]) if ("xg_source" in df.columns and len(df)) else ""
    ax.set_title(("Shot Map — xG: %s" % xg_src).strip(), color=theme["text"])
    return fig


def defensive_actions_map(
    df: pd.DataFrame,
    def_colors: Optional[dict] = None,
    def_markers: Optional[dict] = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
):
    def_colors = def_colors or {}
    def_markers = def_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    d = df[df["event_type"] == "defensive"].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    legend_handles = []
    for act in DEF_ACTION_COLS:
        if act not in d.columns:
            continue
        subset = d[_yes_only(d[act])].copy()
        if subset.empty:
            continue

        color = def_colors.get(act, theme.get("accent", "#6D28D9"))
        marker = def_markers.get(act, "o")
        label = act.replace("_", " ").title()

        if not _is_no_marker(marker):
            pitch.scatter(subset["x"], subset["y"], ax=ax, s=130, marker=marker, color=color, edgecolors="white", linewidth=1.4, alpha=0.95, zorder=6)
            legend_handles.append(Line2D([0], [0], marker=marker, color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label))
    ax.set_title("Defensive Actions Map", color=theme["text"])
    _add_legend(ax, legend_handles, theme, loc="upper center")
    return fig


# =========================================================
# NEW SCOUTING CHARTS
# =========================================================
def progressive_actions_chart(df: pd.DataFrame, theme_name: str = "The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _apply_fig_theme(fig, ax, theme)

    prog_pass = int(_bool_mask(df.get("progressive_pass", False), df.index).sum())
    prog_carry = int(_bool_mask(df.get("progressive_carry", False), df.index).sum())
    key_pass = int((df["outcome"] == "key pass").sum())
    assists = int((df["outcome"] == "assist").sum())
    box_entries = int(_bool_mask(df.get("into_penalty_box", False), df.index).sum())
    final_third = int(_bool_mask(df.get("into_final_third", False), df.index).sum())

    labels = ["Prog Pass", "Prog Carry", "Key Pass", "Assist", "Box Entry", "Final 3rd"]
    values = [prog_pass, prog_carry, key_pass, assists, box_entries, final_third]
    colors = [
        theme.get("accent", "#6D28D9"),
        theme.get("accent_2", "#8B5CF6"),
        theme.get("success", "#22A06B"),
        theme.get("danger", "#D64045"),
        theme.get("warning", "#B0B0B0"),
        theme.get("muted", "#7A7584"),
    ]

    ax.bar(labels, values, color=colors)
    ax.set_title("Progressive Actions", color=theme["text"], fontsize=16, weight="bold")
    ax.tick_params(axis="x", rotation=20, colors=theme["muted"])
    ax.tick_params(axis="y", colors=theme["muted"])
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])
    return fig


def passing_direction_chart(df: pd.DataFrame, theme_name: str = "The Athletic Dark"):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    d = df[df["event_type"] == "pass"].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        _apply_fig_theme(fig, ax, theme)
        ax.text(0.5, 0.5, "No passes found", ha="center", va="center", color=theme["text"])
        ax.set_axis_off()
        return fig

    d["dx"] = pd.to_numeric(d["x2"], errors="coerce") - pd.to_numeric(d["x"], errors="coerce")
    d["dy"] = pd.to_numeric(d["y2"], errors="coerce") - pd.to_numeric(d["y"], errors="coerce")

    def classify(row):
        dx, dy = row["dx"], row["dy"]
        if pd.isna(dx) or pd.isna(dy):
            return "Unknown"
        if abs(dx) >= abs(dy):
            return "Forward" if dx > 0 else "Backward"
        return "Lateral"

    d["direction"] = d.apply(classify, axis=1)
    counts = d["direction"].value_counts().reindex(["Forward", "Backward", "Lateral", "Unknown"]).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _apply_fig_theme(fig, ax, theme)
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=[theme.get("accent"), theme.get("danger"), theme.get("warning"), theme.get("muted")])
    ax.set_title("Passing Direction", color=theme["text"], fontsize=16, weight="bold")
    ax.tick_params(colors=theme["muted"])
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])
    return fig


def carry_map(
    df: pd.DataFrame,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    d = df[(df["event_type"] == "carry") | (_bool_mask(df.get("is_carry", False), df.index))].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    if d.empty:
        ax.set_title("Carry Map — No carries found", color=theme["text"])
        return fig

    if "x2" not in d.columns:
        d["x2"] = d["x"]
    if "y2" not in d.columns:
        d["y2"] = d["y"]

    prog_mask = _bool_mask(d.get("progressive_carry", False), d.index)
    d_prog = d[prog_mask].copy()
    d_all = d[~prog_mask].copy()

    if not d_all.empty:
        pitch.arrows(d_all["x"], d_all["y"], d_all["x2"], d_all["y2"], ax=ax, width=2, alpha=0.55, color=theme.get("warning", "#B0B0B0"))
    if not d_prog.empty:
        pitch.arrows(d_prog["x"], d_prog["y"], d_prog["x2"], d_prog["y2"], ax=ax, width=2.5, alpha=0.9, color=theme.get("accent", "#6D28D9"))

    handles = [
        Line2D([0], [0], color=theme.get("warning", "#B0B0B0"), lw=3, label="Carry"),
        Line2D([0], [0], color=theme.get("accent", "#6D28D9"), lw=3, label="Progressive Carry"),
    ]
    _add_legend(ax, handles, theme, loc="upper center")
    ax.set_title("Carry Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def receive_map(
    df: pd.DataFrame,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    d = df[(df["event_type"] == "receive") | (_bool_mask(df.get("is_receive", False), df.index))].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    if d.empty:
        ax.set_title("Receive Map — No receives found", color=theme["text"])
        return fig

    pitch.scatter(d["x"], d["y"], ax=ax, s=150, marker="o", color=theme.get("accent", "#6D28D9"), edgecolors="white", linewidth=1.5, alpha=0.9)
    ax.set_title("Receive Map", color=theme["text"], fontsize=16, weight="bold")
    return fig


def zone_heatmap(
    df: pd.DataFrame,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
    event_type: str = "all",
    title: str = "Zone Heatmap",
):
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    d = df.copy()
    if event_type != "all":
        d = d[d["event_type"] == event_type].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(8.6, 11.0))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    x_edges = np.array([0, 20, 40, 60, 80, 100])
    y_edges = np.linspace(0, y_max, 5)

    counts, _, _ = np.histogram2d(pd.to_numeric(d["x"], errors="coerce"), pd.to_numeric(d["y"], errors="coerce"), bins=[x_edges, y_edges])
    counts = counts.T

    cmap = LinearSegmentedColormap.from_list(
        "zone_heat",
        [theme.get("panel_2", "#E9E9E9"), theme.get("accent_2", "#8B5CF6"), theme.get("accent", "#6D28D9")]
    )
    vmax = max(1.0, float(np.nanmax(counts)))
    norm = Normalize(vmin=0, vmax=vmax)

    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            x0 = x_edges[xi]
            y0 = y_edges[yi]
            w = x_edges[xi + 1] - x_edges[xi]
            h = y_edges[yi + 1] - y_edges[yi]
            val = counts[yi, xi]
            rect = Rectangle((x0, y0), w, h, facecolor=cmap(norm(val)), edgecolor=theme["pitch_lines"], linewidth=1.5, alpha=0.85, zorder=1)
            ax.add_patch(rect)
            if val > 0:
                ax.text(x0 + w/2, y0 + h/2, str(int(val)), ha="center", va="center", color=theme["text"], fontsize=11, weight="bold")

    pitch.draw(ax=ax)
    ax.set_title(title, color=theme["text"], fontsize=18, weight="bold")
    return fig


def shot_spot_and_direction_map(
    df: pd.DataFrame,
    title: str = "Shot Spot + Direction Map",
    shot_colors: Optional[dict] = None,
    shot_markers: Optional[dict] = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
    plot_all: bool = True,
    shot_index: Optional[int] = None,
):
    shot_colors = shot_colors or {}
    shot_markers = shot_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    shots = df[df["event_type"] == "shot"].copy().reset_index(drop=True)
    if shots.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        _apply_fig_theme(fig, ax, theme)
        ax.text(0.5, 0.5, "No shots found", ha="center", va="center", color=theme["text"])
        ax.set_axis_off()
        return fig

    if not plot_all and shot_index is not None and 0 <= shot_index < len(shots):
        shots = shots.iloc[[shot_index]].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    for _, r in shots.iterrows():
        outc = str(r.get("outcome", "")).lower()
        color = shot_colors.get(outc, theme.get("accent", "#6D28D9"))
        marker = shot_markers.get(outc, "o")

        if not _is_no_marker(marker):
            pitch.scatter([r["x"]], [r["y"]], ax=ax, s=180, marker=marker, color=color, edgecolors="white", linewidth=1.5, zorder=6)

        if pd.notna(r.get("x2")) and pd.notna(r.get("y2")):
            ax.plot([r["x"], r["x2"]], [r["y"], r["y2"]], linestyle="--", linewidth=2.2, color=color, alpha=0.9, zorder=5)

    ax.set_title(title, color=theme["text"], fontsize=16, weight="bold")
    return fig


def player_comparison_dashboard(
    df: pd.DataFrame,
    player_col: str,
    player_1: str,
    player_2: str,
    metrics: list[str],
    theme_name: str = "The Athletic Dark",
    title: str = "Player Comparison",
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    row1 = df.loc[df[player_col] == player_1]
    row2 = df.loc[df[player_col] == player_2]

    if row1.empty or row2.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        _apply_fig_theme(fig, ax, theme)
        ax.text(0.5, 0.5, "One or both players not found", ha="center", va="center", color=theme["text"])
        ax.set_axis_off()
        return fig

    vals1 = []
    vals2 = []
    used_metrics = []
    for m in metrics:
        v1 = pd.to_numeric(row1.iloc[0][m], errors="coerce")
        v2 = pd.to_numeric(row2.iloc[0][m], errors="coerce")
        if pd.isna(v1) and pd.isna(v2):
            continue
        vals1.append(0 if pd.isna(v1) else float(v1))
        vals2.append(0 if pd.isna(v2) else float(v2))
        used_metrics.append(m)

    y = np.arange(len(used_metrics))
    fig, ax = plt.subplots(figsize=(9, max(5, len(used_metrics) * 0.55)))
    _apply_fig_theme(fig, ax, theme)

    ax.barh(y + 0.18, vals1, height=0.34, color=theme.get("accent", "#6D28D9"), label=player_1)
    ax.barh(y - 0.18, vals2, height=0.34, color=theme.get("accent_2", "#8B5CF6"), label=player_2)

    ax.set_yticks(y)
    ax.set_yticklabels(used_metrics, color=theme["text"])
    ax.tick_params(axis="x", colors=theme["muted"])
    ax.invert_yaxis()
    ax.set_title(title, color=theme["text"], fontsize=18, weight="bold")
    for spine in ax.spines.values():
        spine.set_color(theme["lines"])
    _add_legend(ax, [], theme)
    ax.legend(facecolor=theme.get("legend_bg", theme["panel"]), edgecolor=theme.get("legend_border", theme["lines"]), labelcolor=theme.get("legend_text", theme["text"]))
    return fig


# =========================================================
# REPORT / PIZZA / DETAIL CARD
# =========================================================
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
        fig.text(tx, 0.965, title, ha=tha, va="top", color=title_color, fontsize=title_fontsize, weight="bold")
    if subtitle:
        fig.text(sx, 0.935, subtitle, ha=sha, va="top", color=subtitle_color, fontsize=subtitle_fontsize)

    if header_image is None:
        return

    try:
        img = header_image.convert("RGBA") if hasattr(header_image, "convert") else header_image
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
    def_colors: Optional[dict] = None,
    def_markers: Optional[dict] = None,
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
    pass_min_packing = int(kwargs.get("pass_min_packing", 1))

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    df2 = df_prepared.copy()

    charts_to_include = charts_to_include or [
        "Outcome Bar", "Start Heatmap", "Touch Map (Scatter)", "Pass Map", "Shot Map", "Defensive Actions Map"
    ]

    figs = []

    if "Outcome Bar" in charts_to_include:
        figs.append(("outcome_bar", outcome_bar(df2, theme_name=theme_name)))
    if "Start Heatmap" in charts_to_include:
        figs.append(("start_heatmap", start_location_heatmap(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))
    if "Touch Map (Scatter)" in charts_to_include:
        figs.append(("touch_map", touch_map(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, dot_color=touch_dot_color, edge_color=touch_dot_edge, dot_size=touch_dot_size, alpha=touch_alpha, marker=touch_marker)))
    if "Pass Map" in charts_to_include:
        figs.append(("pass_map", pass_map(df2, pass_colors=pass_colors, pass_markers=pass_markers, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, pass_view=pass_view, result_scope=pass_result_scope, min_packing=pass_min_packing)))
    if "Shot Map" in charts_to_include:
        figs.append(("shot_map", shot_map(df2, shot_colors=shot_colors, shot_markers=shot_markers, pitch_mode=pitch_mode, pitch_width=pitch_width, show_xg=True, theme_name=theme_name)))
    if "Defensive Actions Map" in charts_to_include:
        figs.append(("defensive_actions_map", defensive_actions_map(df2, def_colors=def_colors, def_markers=def_markers, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))
    if "Progressive Actions" in charts_to_include:
        figs.append(("progressive_actions", progressive_actions_chart(df2, theme_name=theme_name)))
    if "Passing Direction" in charts_to_include:
        figs.append(("passing_direction", passing_direction_chart(df2, theme_name=theme_name)))
    if "Carry Map" in charts_to_include:
        figs.append(("carry_map", carry_map(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))
    if "Receive Map" in charts_to_include:
        figs.append(("receive_map", receive_map(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))
    if "Zone Heatmap" in charts_to_include:
        figs.append(("zone_heatmap", zone_heatmap(df2, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name, event_type="all", title="Zone Heatmap — All Events")))
    if "Shot Spot + Direction" in charts_to_include:
        figs.append(("shot_spot_direction", shot_spot_and_direction_map(df2, shot_colors=shot_colors, shot_markers=shot_markers, pitch_mode=pitch_mode, pitch_width=pitch_width, theme_name=theme_name)))

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
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight", pad_inches=0.25)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            pngs.append(png_path)

    return pdf_path, pngs


def pizza_chart(
    df_pizza: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    slice_colors: Optional[List[str]] = None,
    show_values_legend: bool = True,
    center_image=None,
    center_img_scale: float = 0.22,
    footer_text: str = "",
    theme_name: str = "The Athletic Dark",
):
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])
    dfp = df_pizza.copy()
    dfp.columns = [c.strip().lower() for c in dfp.columns]
    required = {"metric", "value", "percentile"}
    if not required.issubset(set(dfp.columns)):
        raise ValueError("Pizza input needs metric, value, percentile")

    params = dfp["metric"].astype(str).tolist()
    values = pd.to_numeric(dfp["percentile"], errors="coerce").fillna(0).tolist()
    value_text = dfp["value"].astype(str).tolist()

    if slice_colors is None or len(slice_colors) != len(values):
        slice_colors = [theme.get("accent", "#6D28D9")] * len(values)

    bg = theme.get("panel", "#F5F5F5")
    ring = theme.get("lines", "#A7A7A7")
    muted = theme.get("text", "#201C2B")

    pizza = PyPizza(
        params=params,
        background_color=bg,
        straight_line_color=ring,
        straight_line_lw=2.0,
        last_circle_color=muted,
        last_circle_lw=2.3,
        other_circle_ls="--",
        other_circle_lw=1.6,
        other_circle_color=theme.get("muted", "#7A7584"),
    )

    fig, ax = pizza.make_pizza(
        values,
        figsize=(10, 10),
        blank_alpha=0.22,
        slice_colors=slice_colors,
        kwargs_slices=dict(edgecolor=ring, linewidth=1.8),
        kwargs_params=dict(color=theme["text"], fontsize=14, fontweight="bold"),
        kwargs_values=dict(
            color=theme["text"],
            fontsize=12,
            fontweight="bold",
            bbox=dict(edgecolor=ring, facecolor=theme.get("panel_2", "#E9E9E9"), boxstyle="round,pad=0.25", linewidth=1.2),
        ),
    )

    fig.patch.set_facecolor(bg)
    fig.text(0.5, 0.975, (title or "").strip(), ha="center", va="top", color=theme["text"], fontsize=24, fontweight="bold")
    fig.text(0.5, 0.945, (subtitle or "").strip(), ha="center", va="top", color=theme.get("muted", "#7A7584"), fontsize=15)

    if footer_text:
        fig.text(0.98, 0.03, footer_text, ha="right", va="bottom", color=theme["text"], fontsize=12)

    if center_image is not None:
        try:
            img = center_image.convert("RGBA") if hasattr(center_image, "convert") else center_image
            img_arr = np.asarray(img)
            s = float(center_img_scale)
            s = max(0.12, min(0.32, s))
            ax_img = fig.add_axes([0.5 - s / 2.0, 0.5 - s / 2.0, s, s], zorder=50)
            ax_img.imshow(img_arr)
            ax_img.axis("off")
            ax_img.set_facecolor("none")
        except Exception:
            pass

    if show_values_legend:
        lines = [f"{m}: {v}   (pct {p:.1f})" for m, v, p in zip(params, value_text, values)]
        fig.text(0.02, 0.02, "\n".join(lines), ha="left", va="bottom", color=theme["text"], fontsize=9, family="monospace")
    return fig


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
    shot_colors = shot_colors or {}
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
    c = shot_colors.get(outcome, theme.get("accent", "#6D28D9"))
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

    x, y = float(r["x"]), float(r["y"])
    if not _is_no_marker(mk):
        pitch.scatter([x], [y], ax=ax_pitch, s=520, marker=mk, color=c, edgecolors="white", linewidth=2, zorder=5, clip_on=False)
        pitch.scatter([x], [y], ax=ax_pitch, s=190, marker=mk, color="white", alpha=0.25, zorder=6, clip_on=False)

    ax_pitch.text(x + 1.2, y + 1.2, "xG %s" % xg_txt, color=theme["text"], fontsize=12, weight="bold", zorder=10)

    if pd.notna(r.get("x2")) and pd.notna(r.get("y2")):
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color=theme["text"], alpha=0.9, zorder=4)

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


def defensive_regains_map(
    df: pd.DataFrame,
    title: str = "Ball Regains Map",
    def_colors: Optional[dict] = None,
    def_markers: Optional[dict] = None,
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    theme_name: str = "The Athletic Dark",
    marker_size: int = 110,
    zone_alpha: float = 0.78,
    show_zone_values: bool = False,
):
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    def_colors = def_colors or {}
    def_markers = def_markers or {}
    theme = THEMES.get(theme_name, THEMES["The Athletic Dark"])

    d = df.copy()
    d = _standardize_defensive_columns(d)
    for c in ["x", "y"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["x", "y"]).copy()

    available_def_cols = [c for c in DEF_ACTION_COLS if c in d.columns]
    if available_def_cols:
        mask = pd.Series(False, index=d.index)
        for c in available_def_cols:
            mask = mask | _yes_only(d[c])
        d = d[mask].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width, theme=theme)
    fig, ax = plt.subplots(figsize=(8.6, 11.4))
    fig.patch.set_facecolor(theme["bg"])
    _draw_pitch(ax, pitch, theme)

    y_max = pitch_width if pitch_mode == "rect" else 100.0
    x_edges = np.array([0, 20, 40, 60, 80, 100])
    y_edges = np.linspace(0, y_max, 5)

    counts, _, _ = np.histogram2d(d["x"], d["y"], bins=[x_edges, y_edges])
    counts = counts.T

    cmap = LinearSegmentedColormap.from_list(
        "regains_map",
        [theme.get("panel_2", "#E9E9E9"), theme.get("accent_2", "#8B5CF6"), theme.get("danger", "#D64045")]
    )
    vmax = max(1.0, float(np.nanmax(counts)))
    norm = Normalize(vmin=0, vmax=vmax)

    for yi in range(len(y_edges) - 1):
        for xi in range(len(x_edges) - 1):
            x0 = x_edges[xi]
            y0 = y_edges[yi]
            w = x_edges[xi + 1] - x_edges[xi]
            h = y_edges[yi + 1] - y_edges[yi]
            val = counts[yi, xi]

            rect = Rectangle((x0, y0), w, h, facecolor=cmap(norm(val)), edgecolor=theme["pitch_lines"], linewidth=1.6, alpha=zone_alpha, zorder=1)
            ax.add_patch(rect)

            if show_zone_values and val > 0:
                ax.text(x0 + w/2.0, y0 + h/2.0, str(int(val)), ha="center", va="center", color=theme["text"], fontsize=11, weight="bold", zorder=2)

    pitch.draw(ax=ax)

    action_order = ["tackle", "interception", "recovery", "aerial_duel", "ground_duel", "clearance"]
    counts_by_action = {}
    for act in action_order:
        if act not in d.columns:
            continue
        subset = d[_yes_only(d[act])].copy()
        if subset.empty:
            continue
        counts_by_action[act] = len(subset)
        marker = def_markers.get(act, "o")
        if _is_no_marker(marker):
            continue
        pitch.scatter(subset["x"], subset["y"], ax=ax, s=marker_size, marker=marker, color="white", edgecolors=def_colors.get(act, theme.get("accent", "#6D28D9")), linewidth=1.8, alpha=0.98, zorder=5)

    ax.set_title(title, color=theme["text"], fontsize=24, weight="bold", pad=18)
    return fig
