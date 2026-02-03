import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mplsoccer import Pitch

PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]
SHOT_ORDER = ["off target", "ontarget", "goal"]

SHOT_TYPES = set(SHOT_ORDER)
REQUIRED = ["outcome", "x", "y"]  # x2,y2 optional (required only for pass arrows)

def _norm_outcome(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    aliases = {
        "on target": "ontarget",
        "on-target": "ontarget",
        "on_target": "ontarget",
        "offtarget": "off target",
        "off-target": "off target",
        "off_target": "off target",
        "keypass": "key pass",
        "key-pass": "key pass",
        "key_pass": "key pass",
        "unsucssesful": "unsuccessful",
        "unsuccessfull": "unsuccessful",
        "un-successful": "unsuccessful",
        "un successful": "unsuccessful",
    }
    return aliases.get(s, s)

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Use CSV or Excel.")

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Required columns are: {REQUIRED} (x2,y2 needed for pass arrows).")

    for c in ["x", "y", "x2", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["outcome"] = df["outcome"].apply(_norm_outcome)
    df = df.dropna(subset=["x", "y"]).copy()

    df["event_type"] = "pass"
    df.loc[df["outcome"].isin(SHOT_TYPES), "event_type"] = "shot"
    return df

def apply_pitch_transforms(
    df: pd.DataFrame,
    attack_direction: str = "ltr",
    flip_y: bool = False,
    pitch_mode: str = "rect",  # "rect" or "square"
    pitch_width: float = 64.0, # used only for rect mode
) -> pd.DataFrame:
    """
    Your tag tool coordinates assumed 0-100 for both axes.

    - attack_direction: 'ltr' or 'rtl' (flip X)
    - flip_y: if your Y origin is bottom, set True (flip Y)
    - pitch_mode:
        * "square": keep 0-100 on both axes -> square pitch
        * "rect": scale Y from 0-100 -> 0-pitch_width to make the pitch rectangular
    """
    df = df.copy()

    # Flip Y (in the original 0-100 space) if needed
    if flip_y:
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    # Flip X for right-to-left attack (in 0-100 space)
    if attack_direction == "rtl":
        for c in ["x", "x2"]:
            if c in df.columns:
                df[c] = 100 - df[c]

    # Make pitch rectangular by scaling Y to pitch_width
    if pitch_mode == "rect":
        scale = pitch_width / 100.0
        for c in ["y", "y2"]:
            if c in df.columns:
                df[c] = df[c] * scale

    return df

def make_pitch(pitch_mode: str = "rect", pitch_width: float = 64.0):
    """
    Custom pitch that matches your coordinate system:
    - square: 100x100 (will look square)
    - rect: 100 x pitch_width (looks like a real pitch)
    """
    if pitch_mode == "square":
        return Pitch(pitch_type="custom", pitch_length=100, pitch_width=100, line_zorder=2)

    return Pitch(pitch_type="custom", pitch_length=100, pitch_width=pitch_width, line_zorder=2)

# ----------------------------
# Charts
# ----------------------------
def outcome_bar(df: pd.DataFrame, title: str = ""):
    counts = df["outcome"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index.astype(str), counts.values)
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

def pass_map(df: pd.DataFrame, title: str = "", pass_colors: dict | None = None, pitch_mode: str = "rect", pitch_width: float = 64.0):
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
        pitch.arrows(dt["x"], dt["y"], dt["x2"], dt["y2"],
                     ax=ax, width=2, alpha=0.85,
                     color=pass_colors.get(t, None))

    ax.set_title((title + "\nPass Map (successful / unsuccessful / key pass / assist)").strip())
    return fig

def shot_map(df: pd.DataFrame, title: str = "", shot_colors: dict | None = None, pitch_mode: str = "rect", pitch_width: float = 64.0):
    shot_colors = shot_colors or {}
    s = df[df["event_type"] == "shot"].copy()

    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    fig, ax = pitch.draw(figsize=(7.6, 4.8))

    for t in SHOT_ORDER:
        st = s[s["outcome"] == t]
        if len(st) == 0:
            continue
        pitch.scatter(st["x"], st["y"], ax=ax, s=90, alpha=0.95, color=shot_colors.get(t, None))

    ax.set_title((title + "\nShot Map (off target / on target / goal)").strip())
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
):
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")

    df2 = apply_pitch_transforms(df, attack_direction=attack_direction, flip_y=flip_y, pitch_mode=pitch_mode, pitch_width=pitch_width)

    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title)),
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
