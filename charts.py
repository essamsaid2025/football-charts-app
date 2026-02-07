import os
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch, PyPizza

PASS_ORDER = ["unsuccessful", "successful", "key pass", "assist"]

# Shots: add blocked
SHOT_ORDER = ["off target", "ontarget", "goal", "blocked"]
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
        "block": "blocked",
        "blocked shot": "blocked",
        "blk": "blocked",
    }
    return aliases.get(s, s)


def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        # safer for arabic/windows exports
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
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}. Required columns are: {REQUIRED} (x2,y2 needed for pass arrows)."
        )

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
    pitch_mode: str = "rect",   # "rect" or "square"
    pitch_width: float = 64.0,  # used only for rect mode
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
# xG (location-based, no training)
# ----------------------------
def _shot_angle_and_distance(x: float, y: float, pitch_mode: str, pitch_width: float):
    """
    Coordinates assumed in the SAME space as the pitch drawn:
      - x in [0,100]
      - y in [0,pitch_width] for rect, or [0,100] for square

    Goal is at x=100, y=midline.
    """
    goal_x = 100.0
    goal_y = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0

    # convert "goal mouth" width to your y-units
    # real goal width 7.32m vs pitch width 68m => ratio 0.10765
    # so goal mouth in your units:
    goal_mouth = (pitch_width * 0.10765) if pitch_mode == "rect" else (100.0 * 0.10765)
    left_post_y = goal_y - goal_mouth / 2.0
    right_post_y = goal_y + goal_mouth / 2.0

    dx = goal_x - x
    dy = goal_y - y
    distance = math.sqrt(dx * dx + dy * dy) + 1e-9

    # angle between lines to posts
    a = math.atan2(right_post_y - y, goal_x - x)
    b = math.atan2(left_post_y - y, goal_x - x)
    angle = abs(a - b)
    if angle > math.pi:
        angle = 2 * math.pi - angle

    return angle, distance


def estimate_xg_from_location(x: float, y: float, pitch_mode: str = "rect", pitch_width: float = 64.0) -> float:
    """
    Simple logistic model using distance + angle.
    Produces a reasonable 0-1 xG from shot location ONLY.
    """
    angle, distance = _shot_angle_and_distance(x, y, pitch_mode, pitch_width)

    # logistic: tuned for "nice-looking" outputs (approx)
    # increase xG for wider angle, decrease with distance
    z = -3.2 + (3.0 * angle) - (0.075 * distance)
    xg = 1.0 / (1.0 + math.exp(-z))

    # clamp
    if xg < 0.0:
        xg = 0.0
    if xg > 0.99:
        xg = 0.99
    return float(xg)


def estimate_xg_for_shots(df: pd.DataFrame, pitch_mode: str = "rect", pitch_width: float = 64.0) -> pd.DataFrame:
    """
    Adds/overwrites column 'xg' for shot rows only.
    """
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


def start_location_heatmap(
    df: pd.DataFrame, title: str = "", pitch_mode: str = "rect", pitch_width: float = 64.0
):
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

    # estimate xG if needed
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
                    ax.text(float(r["x"]) + 1.0, float(r["y"]) + 1.0, f'{float(r["xg"]):.2f}',
                            fontsize=9, color="white", weight="bold")
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

    with PdfPages(pdf_path) as pdf:
        figs = [
            ("outcome_bar", outcome_bar(df2, title=title, bar_colors=bar_colors)),
            (
                "start_heatmap",
                start_location_heatmap(df2, title=title, pitch_mode=pitch_mode, pitch_width=pitch_width),
            ),
            (
                "pass_map",
                pass_map(df2, title=title, pass_colors=pass_colors, pitch_mode=pitch_mode, pitch_width=pitch_width),
            ),
            (
                "shot_map",
                shot_map(df2, title=title, shot_colors=shot_colors, pitch_mode=pitch_mode, pitch_width=pitch_width),
            ),
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
# Shot detail card (like your screenshot)
# ----------------------------
def shot_detail_card(
    df: pd.DataFrame,
    shot_index: int,
    title: str = "",
    pitch_mode: str = "rect",
    pitch_width: float = 64.0,
    shot_colors: dict | None = None,
):
    """
    Card:
    - Left: mini goal + pitch
    - Right: xG + Outcome + Shot type (we use outcome as shot type: on/off/goal/blocked)
    - Dotted line to end location if x2,y2 exist for that shot
    - xG is estimated from location if not present
    """
    shot_colors = shot_colors or {
        "off target": "#FF8A00",
        "ontarget": "#00C2FF",
        "goal": "#00FF6A",
        "blocked": "#AAAAAA",
    }

    shots = df[df["event_type"] == "shot"].copy().reset_index(drop=True)
    if shots.empty:
        raise ValueError("No shots found in the file (event_type == shot).")

    if shot_index < 0 or shot_index >= len(shots):
        raise ValueError("Shot index out of range.")

    r = shots.iloc[shot_index]

    # ensure xg exists (estimate if missing)
    if "xg" not in shots.columns or pd.isna(r.get("xg")):
        xg = estimate_xg_from_location(float(r["x"]), float(r["y"]), pitch_mode=pitch_mode, pitch_width=pitch_width)
        xg_txt = f"{xg:.2f}"
    else:
        try:
            xg_txt = f"{float(r.get('xg')):.2f}"
        except Exception:
            xg_txt = str(r.get("xg"))

    outcome = str(r.get("outcome", "")).lower()
    display_outcome = outcome.title() if outcome != "ontarget" else "On target"
    c = shot_colors.get(outcome, "#00C2FF")

    # Figure layout
    fig = plt.figure(figsize=(12, 6))
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
    ax_goal.set_facecolor("#222222")
    ax_goal.set_xlim(0, 100)
    ax_goal.set_ylim(0, 30)
    ax_goal.axis("off")

    # simple goal frame
    ax_goal.plot([25, 75], [5, 5], lw=2, color="#bbbbbb")
    ax_goal.plot([25, 25], [5, 22], lw=2, color="#bbbbbb")
    ax_goal.plot([75, 75], [5, 22], lw=2, color="#bbbbbb")
    ax_goal.plot([25, 75], [22, 22], lw=2, color="#bbbbbb")

    # ---------- pitch ----------
    pitch = make_pitch(pitch_mode=pitch_mode, pitch_width=pitch_width)
    pitch.draw(ax=ax_pitch)
    ax_pitch.set_facecolor("#2f6b3a")

    x, y = float(r["x"]), float(r["y"])

    # ball marker
    pitch.scatter([x], [y], ax=ax_pitch, s=500, color=c, edgecolors="white", linewidth=2, zorder=5)
    pitch.scatter([x], [y], ax=ax_pitch, s=150, color="white", alpha=0.35, zorder=6)

    # xG label
    ax_pitch.text(x + 1.2, y + 1.2, f"xG {xg_txt}", color="white", fontsize=12, weight="bold")

    # Dotted line to end if available
    has_end = ("x2" in shots.columns and "y2" in shots.columns and pd.notna(r.get("x2")) and pd.notna(r.get("y2")))
    if has_end:
        x2, y2 = float(r["x2"]), float(r["y2"])
        ax_pitch.plot([x, x2], [y, y2], linestyle=":", linewidth=3, color="white", alpha=0.9, zorder=4)
        pitch.scatter([x2], [y2], ax=ax_pitch, s=120, color="white", alpha=0.9, zorder=6)

        # mini-goal indicator (simple)
        ax_goal.plot([50, 50], [22, 10], linestyle=":", linewidth=3, color="white", alpha=0.9)
        ax_goal.scatter([70], [12], s=220, color=c, edgecolors="white", linewidth=2)
    else:
        # fallback to goal center
        gy = (pitch_width / 2.0) if pitch_mode == "rect" else 50.0
        ax_pitch.plot([x, 100], [y, gy], linestyle=":", linewidth=3, color="white", alpha=0.6, zorder=4)

    # ---------- info panel ----------
    ax_info.set_facecolor("#1f1f1f")
    ax_info.axis("off")

    if title:
        ax_info.text(0.02, 0.96, title, color="white", fontsize=18, weight="bold", transform=ax_info.transAxes)

    ax_info.text(0.02, 0.83, "xG", color="#aaaaaa", fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.75, xg_txt, color="white", fontsize=22, weight="bold", transform=ax_info.transAxes)

    ax_info.plot([0.02, 0.98], [0.70, 0.70], color="#333333", lw=2, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.60, "Outcome", color="#aaaaaa", fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.53, display_outcome, color="white", fontsize=22, weight="bold", transform=ax_info.transAxes)

    ax_info.plot([0.02, 0.98], [0.48, 0.48], color="#333333", lw=2, transform=ax_info.transAxes)

    ax_info.text(0.02, 0.38, "Shot type", color="#aaaaaa", fontsize=14, transform=ax_info.transAxes)
    ax_info.text(0.02, 0.31, display_outcome, color="white", fontsize=22, weight="bold", transform=ax_info.transAxes)

    return fig, shots


# ----------------------------
# Pizza chart (for player comparison)
# ----------------------------
def pizza_chart(
    df_pizza: pd.DataFrame,
    title: str = "",
    subtitle: str = "",
    slice_colors: list | None = None,
    show_values_legend: bool = True,
):
    """
    df_pizza columns:
      - metric
      - value
      - percentile (0-100)
    """
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
