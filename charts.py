# charts_no_xg.py — بدون xG + صورة في التقرير + Legend + dot map للـstart location
# =============================================================================
import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# ----------------------------
# Constants
# ----------------------------
PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]

# ----------------------------
# Theme System
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
}

# ----------------------------
# Outcome Normalization
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

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED}")

    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

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
# End location helper
# ----------------------------
def _goal_mouth_bounds(pitch_mode="rect", pitch_width=64.0):
    gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    return gy - goal_mouth / 2.0, gy + goal_mouth / 2.0

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
        x2, y2 = r.get("x2"), r.get("y2")

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

        df.at[i, "x2"] = x2
        df.at[i, "y2"] = y2

    return df

# ----------------------------
# Prepare DF
# ----------------------------
def prepare_df_for_charts(
    df_raw: pd.DataFrame,
    attack_direction="ltr",
    flip_y=False,
    pitch_mode="rect",
    pitch_width=64.0,
) -> pd.DataFrame:
    df = validate_and_clean(df_raw)
    df = apply_pitch_transforms(df, attack_direction, flip_y, pitch_mode, pitch_width)
    df = fix_shot_end_location(df, pitch_mode=pitch_mode, pitch_width=pitch_width)
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
    # Legend
    for k, v in bar_colors.items():
        ax.bar(0, 0, color=v, label=k)
    ax.legend(loc="upper right")
    return fig

def start_location_heatmap(df: pd.DataFrame, title: str = "", pitch_mode="rect", pitch_width=64.0):
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    s = df[df["event_type"] == "shot"].copy()
    ax.scatter(s["x"], s["y"], s=50, color="red", alpha=0.7)
    ax.set_title((title + "\nStart Locations (Dot Map)").strip())
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

def pass_map(df: pd.DataFrame, title: str = "", pass_colors: dict | None = None,
             pitch_mode="rect", pitch_width=64.0):
    pass_colors = pass_colors or {}
    d = df[df["event_type"] == "pass"].copy()
    d = d.dropna(subset=["x2", "y2"])
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    for t in PASS_ORDER:
        dt = d[d["outcome"] == t]
        if len(dt) == 0:
            continue
        pitch.arrows(dt["x"], dt["y"], dt["x2"], dt["y2"], ax=ax,
                     width=2, alpha=0.85, color=pass_colors.get(t, None))
    ax.set_title((title + "\nPass Map").strip())
    # Legend
    for k, v in pass_colors.items():
        ax.plot([], [], color=v, label=k)
    ax.legend(loc="upper right")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

def shot_map(df: pd.DataFrame, title: str = "", shot_colors: dict | None = None,
             pitch_mode="rect", pitch_width=64.0):
    shot_colors = shot_colors or {}
    s = df[df["event_type"] == "shot"].copy()
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))
    for t in SHOT_ORDER:
        stt = s[s["outcome"] == t]
        if len(stt) == 0:
            continue
        pitch.scatter(stt["x"], stt["y"], ax=ax, s=90, alpha=0.95, color=shot_colors.get(t, None))
    ax.set_title((title + "\nShot Map").strip())
    # Legend
    for k, v in shot_colors.items():
        ax.scatter([], [], color=v, label=k)
    ax.legend(loc="upper right")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, pitch_width + 2 if pitch_mode == "rect" else 102)
    return fig

# ----------------------------
# Report builder مع صورة
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
    title_image_path: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    df2 = df_prepared.copy()
    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title, bar_colors=bar_colors)),
            ("start_heatmap", start_location_heatmap(df2, title=title, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("pass_map", pass_map(df2, title=title, pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
            ("shot_map", shot_map(df2, title=title, shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width)),
        ]
        pngs = []
        for name, fig in figs:
            if title_image_path and os.path.exists(title_image_path):
                img = plt.imread(title_image_path)
                imagebox = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(imagebox, (0.5, 0.95), frameon=False, xycoords='figure fraction')
                fig.add_artist(ab)
            png_path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(png_path, dpi=220, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pngs.append(png_path)
    return pdf_path, pngs
