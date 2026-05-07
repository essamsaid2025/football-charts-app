import io
import math
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# POSITION / ROLE TEMPLATES
# =========================================================
ROLE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "Goalkeeper": {
        "Core": ["Save %", "Goals prevented", "Clean sheets", "Goals conceded /90"],
        "Distribution": ["Pass accuracy %", "Long pass accuracy %", "Launches completed %"],
        "Sweeper": ["Defensive actions outside box /90", "Claims /90"],
    },
    "Centre Back": {
        "Defending": ["Aerial duels won %", "Defensive duels won %", "Interceptions /90", "Clearances /90", "Blocks /90"],
        "Build Up": ["Progressive passes /90", "Passes into final third /90", "Pass accuracy %", "Long pass accuracy %"],
        "Risk": ["Fouls /90", "Errors leading to shot /90"],
    },
    "Full Back / Wing Back": {
        "Attacking Width": ["Progressive carries /90", "Crosses /90", "Accurate crosses %", "Touches in box /90"],
        "Progression": ["Progressive passes /90", "Passes into final third /90", "Key passes /90"],
        "Defending": ["Defensive duels won %", "Tackles /90", "Interceptions /90", "Recoveries /90"],
    },
    "Defensive Midfielder": {
        "Ball Winning": ["Interceptions /90", "Tackles /90", "Recoveries /90", "Defensive duels won %"],
        "Security": ["Pass accuracy %", "Turnovers /90", "Dispossessed /90"],
        "Progression": ["Progressive passes /90", "Passes into final third /90", "Line-breaking passes /90"],
    },
    "Central Midfielder": {
        "Progression": ["Progressive passes /90", "Progressive carries /90", "Passes into final third /90"],
        "Creation": ["Key passes /90", "xA /90", "Shot assists /90"],
        "Work Rate": ["Recoveries /90", "Pressures /90", "Defensive duels won %"],
    },
    "Attacking Midfielder": {
        "Creation": ["xA /90", "Key passes /90", "Shot assists /90", "Through balls /90"],
        "Final Third": ["Touches in box /90", "Passes into penalty area /90", "Dribbles completed /90"],
        "Goal Threat": ["xG /90", "Shots /90", "Shots on target %"],
    },
    "Winger": {
        "1v1": ["Dribbles attempted /90", "Dribbles completed /90", "Dribble success %"],
        "Progression": ["Progressive carries /90", "Carries into penalty area /90", "Touches in box /90"],
        "Creation": ["xA /90", "Key passes /90", "Crosses /90", "Accurate crosses %"],
        "Goal Threat": ["xG /90", "Shots /90", "Shots on target %"],
    },
    "Striker": {
        "Goal Threat": ["xG /90", "Goals /90", "Shots /90", "Shots on target %", "Conversion %"],
        "Box Presence": ["Touches in box /90", "Aerial duels won %", "Non-penalty xG /90"],
        "Link Play": ["xA /90", "Key passes /90", "Pass accuracy %"],
    },
}

POSITION_TO_ROLE = {
    "gk": "Goalkeeper", "goalkeeper": "Goalkeeper",
    "cb": "Centre Back", "rcb": "Centre Back", "lcb": "Centre Back", "centre back": "Centre Back", "center back": "Centre Back",
    "rb": "Full Back / Wing Back", "lb": "Full Back / Wing Back", "rwb": "Full Back / Wing Back", "lwb": "Full Back / Wing Back", "full back": "Full Back / Wing Back",
    "dm": "Defensive Midfielder", "cdm": "Defensive Midfielder", "defensive midfielder": "Defensive Midfielder",
    "cm": "Central Midfielder", "central midfielder": "Central Midfielder", "mc": "Central Midfielder",
    "am": "Attacking Midfielder", "cam": "Attacking Midfielder", "attacking midfielder": "Attacking Midfielder",
    "rw": "Winger", "lw": "Winger", "winger": "Winger", "wide midfielder": "Winger",
    "st": "Striker", "cf": "Striker", "striker": "Striker", "centre forward": "Striker", "center forward": "Striker",
}

ALIASES = {
    "Save %": ["save %", "save percentage", "saves %"],
    "Goals prevented": ["goals prevented", "psxg +/-", "post-shot xg +/-", "prevented goals"],
    "Clean sheets": ["clean sheets", "cs"],
    "Goals conceded /90": ["goals conceded /90", "conceded /90", "ga /90"],
    "Defensive actions outside box /90": ["def actions outside box /90", "defensive actions outside box /90", "sweeper actions /90"],
    "Claims /90": ["claims /90", "crosses claimed /90"],
    "Aerial duels won %": ["aerial duels won %", "aerial win %", "aerial won %", "aerial duels %"],
    "Defensive duels won %": ["defensive duels won %", "def duel win %", "defensive duels %"],
    "Interceptions /90": ["interceptions /90", "interceptions per 90", "interceptions", "int /90"],
    "Clearances /90": ["clearances /90", "clearances per 90", "clearances"],
    "Blocks /90": ["blocks /90", "blocks per 90", "blocks"],
    "Tackles /90": ["tackles /90", "tackles per 90", "tackles"],
    "Recoveries /90": ["recoveries /90", "ball recoveries /90", "recoveries", "ball recoveries"],
    "Pressures /90": ["pressures /90", "pressures per 90", "pressures"],
    "Fouls /90": ["fouls /90", "fouls per 90", "fouls"],
    "Errors leading to shot /90": ["errors leading to shot /90", "errors /90", "mistakes /90"],
    "Pass accuracy %": ["pass accuracy %", "passing accuracy %", "accurate passes %", "pass %"],
    "Long pass accuracy %": ["long pass accuracy %", "long passes accurate %", "accurate long balls %"],
    "Launches completed %": ["launches completed %", "launched passes completed %"],
    "Progressive passes /90": ["progressive passes /90", "progressive passes per 90", "progressive passes", "prog passes /90"],
    "Passes into final third /90": ["passes into final third /90", "final third passes /90", "passes to final third /90"],
    "Line-breaking passes /90": ["line-breaking passes /90", "line breaking passes /90", "linebreak passes /90"],
    "Progressive carries /90": ["progressive carries /90", "progressive carries per 90", "progressive carries", "prog carries /90"],
    "Crosses /90": ["crosses /90", "crosses per 90", "crosses"],
    "Accurate crosses %": ["accurate crosses %", "cross accuracy %", "crosses accuracy %"],
    "Key passes /90": ["key passes /90", "key passes per 90", "key passes"],
    "Shot assists /90": ["shot assists /90", "shot assists", "assists to shots /90"],
    "Through balls /90": ["through balls /90", "through passes /90"],
    "xA /90": ["xa /90", "expected assists /90", "xA per 90"],
    "xG /90": ["xg /90", "expected goals /90", "xG per 90"],
    "Non-penalty xG /90": ["non-penalty xg /90", "npxg /90", "np xg /90"],
    "Goals /90": ["goals /90", "goals per 90", "g /90"],
    "Shots /90": ["shots /90", "shots per 90", "shots"],
    "Shots on target %": ["shots on target %", "shot on target %", "sot %", "shots on target percentage"],
    "Conversion %": ["conversion %", "goal conversion %", "shot conversion %"],
    "Touches in box /90": ["touches in box /90", "touches in penalty area /90", "box touches /90"],
    "Dribbles attempted /90": ["dribbles attempted /90", "dribbles /90", "take ons /90", "1v1 attempted /90"],
    "Dribbles completed /90": ["dribbles completed /90", "successful dribbles /90", "successful take ons /90"],
    "Dribble success %": ["dribble success %", "successful dribbles %", "take on success %"],
    "Carries into penalty area /90": ["carries into penalty area /90", "carries into box /90", "box carries /90"],
    "Passes into penalty area /90": ["passes into penalty area /90", "passes into box /90", "box passes /90"],
    "Turnovers /90": ["turnovers /90", "turnovers per 90", "turnovers"],
    "Dispossessed /90": ["dispossessed /90", "dispossessed per 90", "dispossessed"],
}

NEGATIVE_METRIC_WORDS = ["conceded", "fouls", "errors", "mistakes", "turnovers", "dispossessed", "cards"]


def clean_col_name(x: str) -> str:
    x = str(x).strip().lower().replace("_", " ").replace("-", " ")
    x = re.sub(r"\s+", " ", x)
    return x


def load_player_data(file) -> pd.DataFrame:
    name = getattr(file, "name", str(file))
    if str(name).lower().endswith(".csv"):
        encs = ["utf-8", "utf-8-sig", "cp1256", "cp1252", "latin1", "utf-16"]
        for enc in encs:
            try:
                return pd.read_csv(file, encoding=enc)
            except Exception:
                try:
                    file.seek(0)
                except Exception:
                    pass
        return pd.read_csv(file, encoding="latin1", encoding_errors="replace")
    return pd.read_excel(file)


def guess_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lookup = {clean_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = clean_col_name(cand)
        if key in lookup:
            return lookup[key]
    for cand in candidates:
        key = clean_col_name(cand)
        for k, v in lookup.items():
            if key in k or k in key:
                return v
    return None


def standard_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "player": guess_column(df, ["player", "player name", "name", "short name"]),
        "team": guess_column(df, ["team", "club", "squad"]),
        "position": guess_column(df, ["position", "pos", "role"]),
        "age": guess_column(df, ["age"]),
        "minutes": guess_column(df, ["minutes", "mins", "minutes played", "playing time"]),
        "market_value": guess_column(df, ["market value", "value", "price", "transfer value"]),
    }


def numeric_metrics(df: pd.DataFrame, exclude: List[Optional[str]]) -> List[str]:
    ex = {c for c in exclude if c}
    out = []
    for c in df.columns:
        if c in ex:
            continue
        s = pd.to_numeric(df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
        if s.notna().sum() >= max(2, int(len(df) * 0.15)):
            out.append(c)
    return out


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    return out


def flatten_template(role: str) -> List[str]:
    groups = ROLE_TEMPLATES.get(role, {})
    metrics = []
    for vals in groups.values():
        metrics.extend(vals)
    return list(dict.fromkeys(metrics))


def match_template_metrics(df: pd.DataFrame, role: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    template = flatten_template(role)
    lookup = {clean_col_name(c): c for c in df.columns}
    matched, missing, mapping = [], [], {}
    for metric in template:
        candidates = [metric] + ALIASES.get(metric, [])
        found = None
        for cand in candidates:
            key = clean_col_name(cand)
            if key in lookup:
                found = lookup[key]
                break
        if not found:
            for cand in candidates:
                key = clean_col_name(cand)
                for k, v in lookup.items():
                    if key == k or key in k or k in key:
                        found = v
                        break
                if found:
                    break
        if found:
            matched.append(found)
            mapping[metric] = found
        else:
            missing.append(metric)
    return list(dict.fromkeys(matched)), missing, mapping


def infer_role_from_position(pos: str) -> Optional[str]:
    p = clean_col_name(pos)
    if p in POSITION_TO_ROLE:
        return POSITION_TO_ROLE[p]
    for k, v in POSITION_TO_ROLE.items():
        if k in p:
            return v
    return None


def percentile_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    pct = x.rank(pct=True) * 100
    if not higher_is_better:
        pct = 100 - pct
    return pct


def is_negative_metric(metric: str) -> bool:
    m = clean_col_name(metric)
    return any(w in m for w in NEGATIVE_METRIC_WORDS)


def add_percentiles_and_score(df: pd.DataFrame, metrics: List[str], group_col: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    metrics = [m for m in metrics if m in out.columns]
    if not metrics:
        out["Scouting Score"] = np.nan
        return out
    pct_cols = []
    for m in metrics:
        pct_col = f"pct__{m}"
        higher = not is_negative_metric(m)
        if group_col and group_col in out.columns:
            out[pct_col] = out.groupby(group_col, dropna=False)[m].transform(lambda x: percentile_series(x, higher))
        else:
            out[pct_col] = percentile_series(out[m], higher)
        pct_cols.append(pct_col)
    out["Scouting Score"] = out[pct_cols].mean(axis=1).round(1)
    return out


def player_profile(df_scored: pd.DataFrame, player_col: str, player_name: str, metrics: List[str], top_n: int = 5) -> Dict[str, object]:
    row = df_scored[df_scored[player_col].astype(str) == str(player_name)]
    if row.empty:
        return {"score": np.nan, "strengths": [], "weaknesses": [], "summary": "Player not found."}
    r = row.iloc[0]
    pct_pairs = []
    for m in metrics:
        pc = f"pct__{m}"
        if pc in df_scored.columns and pd.notna(r.get(pc)):
            pct_pairs.append((m, float(r[pc])))
    pct_pairs_sorted = sorted(pct_pairs, key=lambda x: x[1], reverse=True)
    strengths = pct_pairs_sorted[:top_n]
    weaknesses = sorted(pct_pairs, key=lambda x: x[1])[:top_n]
    score = float(r.get("Scouting Score", np.nan)) if pd.notna(r.get("Scouting Score", np.nan)) else np.nan
    if pd.isna(score):
        label = "Not enough data"
    elif score >= 80:
        label = "Highly Recommended"
    elif score >= 65:
        label = "Good Option"
    elif score >= 50:
        label = "Average / Watch More"
    else:
        label = "Risky"
    return {"score": score, "label": label, "strengths": strengths, "weaknesses": weaknesses}


def profile_text(player_name: str, profile: Dict[str, object]) -> str:
    score = profile.get("score", np.nan)
    label = profile.get("label", "")
    strengths = profile.get("strengths", [])
    weaknesses = profile.get("weaknesses", [])
    lines = []
    score_txt = "NA" if pd.isna(score) else f"{score:.1f}/100"
    lines.append(f"{player_name} profile: {label} — Scouting Score {score_txt}.")
    if strengths:
        st = ", ".join([f"{m} ({p:.0f}th pct)" for m, p in strengths[:3]])
        lines.append(f"Main strengths: {st}.")
    if weaknesses:
        wk = ", ".join([f"{m} ({p:.0f}th pct)" for m, p in weaknesses[:3]])
        lines.append(f"Main concerns: {wk}.")
    return "\n".join(lines)


def similar_players(df_scored: pd.DataFrame, player_col: str, player_name: str, metrics: List[str], top_n: int = 5) -> pd.DataFrame:
    d = df_scored.copy()
    metrics = [m for m in metrics if m in d.columns]
    if not metrics:
        return pd.DataFrame()
    idx = d[d[player_col].astype(str) == str(player_name)].index
    if len(idx) == 0:
        return pd.DataFrame()
    x = d[metrics].apply(pd.to_numeric, errors="coerce")
    z = (x - x.mean()) / x.std(ddof=0).replace(0, np.nan)
    target = z.loc[idx[0]]
    dist = ((z - target) ** 2).mean(axis=1).pow(0.5)
    d["Similarity Distance"] = dist
    d = d[d[player_col].astype(str) != str(player_name)].copy()
    return d.sort_values("Similarity Distance").head(top_n)


def auto_dataset_insights(df_scored: pd.DataFrame, player_col: str, metrics: List[str], team_col: Optional[str] = None, age_col: Optional[str] = None, minutes_col: Optional[str] = None) -> List[str]:
    insights = []
    if "Scouting Score" in df_scored.columns and player_col in df_scored.columns:
        top = df_scored.sort_values("Scouting Score", ascending=False).head(5)
        names = ", ".join([f"{r[player_col]} ({r['Scouting Score']:.1f})" for _, r in top.iterrows() if pd.notna(r.get("Scouting Score"))])
        if names:
            insights.append(f"Top scouting scores: {names}.")
    if age_col and age_col in df_scored.columns and "Scouting Score" in df_scored.columns:
        u23 = df_scored[pd.to_numeric(df_scored[age_col], errors="coerce") <= 23].sort_values("Scouting Score", ascending=False).head(5)
        names = ", ".join([f"{r[player_col]} ({r['Scouting Score']:.1f})" for _, r in u23.iterrows() if pd.notna(r.get("Scouting Score"))])
        if names:
            insights.append(f"Best U23 options: {names}.")
    if minutes_col and minutes_col in df_scored.columns and "Scouting Score" in df_scored.columns:
        low_sample = df_scored[(pd.to_numeric(df_scored[minutes_col], errors="coerce") < 700) & (df_scored["Scouting Score"] >= 70)]
        if len(low_sample):
            insights.append(f"Red flag: {len(low_sample)} high-score players have low minutes sample (<700), so watch more games before final recommendation.")
    for m in metrics[:8]:
        pc = f"pct__{m}"
        if pc in df_scored.columns:
            elite = df_scored[df_scored[pc] >= 85]
            if len(elite):
                insights.append(f"{len(elite)} players are elite (85th percentile+) in {m}.")
    return insights[:8]


def comparison_chart(df_scored: pd.DataFrame, player_col: str, p1: str, p2: str, metrics: List[str], use_percentiles: bool = True):
    d = df_scored.copy()
    rows = d[d[player_col].astype(str).isin([str(p1), str(p2)])]
    if len(rows) < 2 or not metrics:
        raise ValueError("Select two players and at least one metric.")
    vals = []
    labels = []
    for p in [p1, p2]:
        r = rows[rows[player_col].astype(str) == str(p)].iloc[0]
        row_vals = []
        for m in metrics:
            col = f"pct__{m}" if use_percentiles and f"pct__{m}" in d.columns else m
            row_vals.append(pd.to_numeric(r.get(col), errors="coerce"))
        vals.append(row_vals)
        labels.append(p)
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(metrics) * 1.1), 5.2))
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#111827")
    ax.bar(x - width/2, vals[0], width, label=labels[0])
    ax.bar(x + width/2, vals[1], width, label=labels[1])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right", color="#A0A7B4")
    ax.tick_params(axis="y", colors="#A0A7B4")
    ax.set_title("Player Comparison" + (" — Percentiles" if use_percentiles else ""), color="white", weight="bold")
    ax.legend(frameon=False, labelcolor="white")
    for spine in ax.spines.values():
        spine.set_color("#2A3240")
    fig.tight_layout()
    return fig


def radar_chart(df_scored: pd.DataFrame, player_col: str, p1: str, p2: str, metrics: List[str]):
    metrics = metrics[:10]
    rows = df_scored[df_scored[player_col].astype(str).isin([str(p1), str(p2)])]
    if len(rows) < 2 or len(metrics) < 3:
        raise ValueError("Radar needs two players and at least 3 metrics.")
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(7, 7), facecolor="#0E1117")
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor("#111827")
    for p in [p1, p2]:
        r = rows[rows[player_col].astype(str) == str(p)].iloc[0]
        values = []
        for m in metrics:
            col = f"pct__{m}"
            values.append(float(r.get(col, 0)) if pd.notna(r.get(col, np.nan)) else 0)
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=p)
        ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, color="white", fontsize=9)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#A0A7B4")
    ax.set_ylim(0, 100)
    ax.set_title("Role Radar — Percentiles", color="white", weight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False, labelcolor="white")
    return fig


def make_template_csv() -> bytes:
    rows = [
        {
            "Player": "Player A", "Team": "Team 1", "Position": "LW", "Age": 21, "Minutes": 1250, "Market Value": 500000,
            "xG /90": 0.32, "xA /90": 0.21, "Shots /90": 2.4, "Shots on target %": 38,
            "Key passes /90": 1.6, "Progressive carries /90": 4.1, "Dribbles completed /90": 2.2,
            "Dribble success %": 58, "Crosses /90": 3.5, "Accurate crosses %": 29,
            "Touches in box /90": 5.7, "Progressive passes /90": 3.0, "Interceptions /90": 0.9,
            "Tackles /90": 1.2, "Recoveries /90": 4.8, "Defensive duels won %": 51,
            "Aerial duels won %": 35, "Pass accuracy %": 79, "Turnovers /90": 2.4,
        },
        {
            "Player": "Player B", "Team": "Team 2", "Position": "RW", "Age": 24, "Minutes": 1800, "Market Value": 800000,
            "xG /90": 0.25, "xA /90": 0.30, "Shots /90": 1.9, "Shots on target %": 42,
            "Key passes /90": 2.1, "Progressive carries /90": 3.6, "Dribbles completed /90": 1.7,
            "Dribble success %": 61, "Crosses /90": 4.2, "Accurate crosses %": 34,
            "Touches in box /90": 4.9, "Progressive passes /90": 3.5, "Interceptions /90": 1.1,
            "Tackles /90": 1.6, "Recoveries /90": 5.2, "Defensive duels won %": 55,
            "Aerial duels won %": 41, "Pass accuracy %": 82, "Turnovers /90": 1.9,
        },
    ]
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")

# =========================================================
# RECOMMENDATION TEXT GENERATOR
# =========================================================
def _safe_cell(row, col, default=""):
    try:
        if col and col in row.index and pd.notna(row.get(col)):
            return row.get(col)
    except Exception:
        pass
    return default


def _metric_phrase(metrics_pairs, max_items=3):
    if not metrics_pairs:
        return "not enough standout metrics"
    return ", ".join([f"{m} ({p:.0f}th percentile)" for m, p in metrics_pairs[:max_items]])


def recommendation_text(
    df_scored: pd.DataFrame,
    player_col: str,
    player_name: str,
    metrics: List[str],
    role: str = "Player",
    team_col: Optional[str] = None,
    position_col: Optional[str] = None,
    age_col: Optional[str] = None,
    minutes_col: Optional[str] = None,
) -> str:
    """Generate a scout-style recommendation based on score, percentiles, role and sample size."""
    row = df_scored[df_scored[player_col].astype(str) == str(player_name)]
    if row.empty:
        return "**Recommendation:** Player not found."

    r = row.iloc[0]
    prof = player_profile(df_scored, player_col, player_name, metrics, top_n=5)
    score = prof.get("score", np.nan)
    label = prof.get("label", "Not enough data")
    strengths = prof.get("strengths", [])
    weaknesses = prof.get("weaknesses", [])

    team = _safe_cell(r, team_col, "")
    pos = _safe_cell(r, position_col, role)
    age = _safe_cell(r, age_col, "")
    mins = _safe_cell(r, minutes_col, "")

    score_txt = "NA" if pd.isna(score) else f"{float(score):.1f}/100"
    strength_txt = _metric_phrase(strengths, 3)
    weakness_txt = _metric_phrase(weaknesses, 3)

    # Context labels
    age_note = ""
    try:
        age_f = float(age)
        if age_f <= 21:
            age_note = "He also has a strong age profile, so the upside/potential angle is positive."
        elif age_f <= 24:
            age_note = "He is still in a good development window with room to improve."
        elif age_f >= 30:
            age_note = "Age should be considered carefully if the target is long-term squad building."
    except Exception:
        pass

    sample_note = ""
    try:
        mins_f = float(mins)
        if mins_f < 450:
            sample_note = "The sample size is low, so this profile should be validated with more minutes/video before making a final decision."
        elif mins_f < 900:
            sample_note = "The minutes sample is moderate, so the numbers are useful but still need video confirmation."
        else:
            sample_note = "The minutes sample is solid enough to treat the statistical profile with confidence."
    except Exception:
        pass

    # Role-specific summary flavour
    role_l = str(role or pos).lower()
    if "wing" in role_l:
        fit_line = "He looks most relevant as a wide player who can impact progression, 1v1 situations and chance creation."
    elif "striker" in role_l or "forward" in role_l:
        fit_line = "He should be judged mainly on box presence, shot volume, xG output and link-play value."
    elif "midfielder" in role_l and "defensive" in role_l:
        fit_line = "He profiles as a midfield screening option where ball-winning, security and progression are the key checks."
    elif "midfielder" in role_l:
        fit_line = "He profiles as a midfield option where progression, creation and work-rate balance are the key checks."
    elif "back" in role_l:
        fit_line = "He should be assessed on defensive reliability, duel strength and build-up contribution."
    else:
        fit_line = "His suitability should be assessed against the role requirements and the team's tactical needs."

    if pd.isna(score):
        final = "Watch More"
    elif score >= 80:
        final = "Highly Recommended"
    elif score >= 65:
        final = "Recommended / Good Option"
    elif score >= 50:
        final = "Watch More"
    else:
        final = "Not Recommended Yet / Risky"

    header_bits = []
    if team:
        header_bits.append(f"Team: {team}")
    if pos:
        header_bits.append(f"Position: {pos}")
    if age != "":
        header_bits.append(f"Age: {age}")
    if mins != "":
        header_bits.append(f"Minutes: {mins}")
    header = " • ".join(header_bits)

    return f"""
**Recommendation:** {final}  
**Role Fit:** {role}  
**Scouting Score:** {score_txt}  
{header if header else ''}

**Summary:**  
{player_name} profiles as **{label}** for the selected role. {fit_line} His strongest indicators are {strength_txt}.

**Strengths:**  
{strength_txt}.

**Weaknesses / Checks:**  
{weakness_txt}.

**Scout Note:**  
{sample_note} {age_note}

**Final Decision:**  
{final}. Use this as a data-led recommendation, then confirm the key strengths and weaknesses with video scouting.
""".strip()
