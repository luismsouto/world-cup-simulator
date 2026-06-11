import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

# Calibrated standard deviation for the Gaussian draw probability
# Fitted to reproduce Table VI from Xiong et al. (2016):
# "Mathematical Model of Ranking Accuracy and Popularity Promotion"
# σ = 1 / (√2π × 0.28) ≈ 1.426, where 0.28 is the peak draw
# probability at dr=0, consistent with ~28% draw rate in
# international football between evenly matched teams.
SIGMA = 1.426

# ─────────────────────────────────────────────
# STEP 1: Load Elo ratings from CSV
# ─────────────────────────────────────────────

def load_elo_ratings(csv_path):
    """
    Load Elo ratings from a CSV file with two columns:
        - country    : team name (must match names used elsewhere)
        - elo_points : Elo rating (higher = stronger)

    Returns a dict: { "Brazil": 1991, "France": 2063, ... }
    """
    df = pd.read_csv(csv_path)

    required = {"country", "elo_points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Elo ratings CSV is missing columns: {missing}")

    return dict(zip(df["country"], df["elo_points"]))


# ─────────────────────────────────────────────
# STEP 2: Compute draw probability (Eq. 5)
# ─────────────────────────────────────────────

def _p_draw(dr, sigma=SIGMA):
    """
    Gaussian draw probability as per Equation (5) in:
    Xiong et al. (2016), ICSAI 2016.

    P(draw) = 1/(√2π × σ) × exp(-(dr/200)² / (2σ²))

    Peaks at ~28% when dr=0, decays naturally as |dr| grows.

    Parameters
    ----------
    dr    : float — Elo difference (team_a - team_b)
    sigma : float — standard deviation parameter (default 1.426)
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
        -((dr / 200) ** 2) / (2 * sigma ** 2)
    )


# ─────────────────────────────────────────────
# STEP 3: Compute win/draw/lose probabilities
# ─────────────────────────────────────────────

def elo_to_probs(elo_a, elo_b, max_win=0.9, min_lose=0.033, min_draw=0.067):
    """
    Convert Elo ratings to win/draw/lose probabilities using
    Xiong et al. (2016) Equations (4)-(7), with realistic
    probability bounds applied to prevent extreme values from
    producing unrealistic Dixon-Coles lambda parameters.

    Bounds (defaults):
        max_win  = 0.85  — no team wins more than 85% of the time
        min_lose = 0.05  — weakest team retains at least 5% win chance
        min_draw = 0.10  — draw always at least 10% likely
    """
    dr = elo_a - elo_b

    p_draw  = _p_draw(dr)
    p_win_a = 1 / (1 + 10 ** (-dr / 400)) - p_draw / 2
    p_win_b = 1 / (1 + 10 ** (+dr / 400)) - p_draw / 2

    # Clamp negatives from extreme Elo gaps
    p_win_a = max(p_win_a, 0.0)
    p_win_b = max(p_win_b, 0.0)

    # ── Apply realistic football bounds ───────────────────────────
    # Identify stronger and weaker team
    if p_win_a >= p_win_b:
        p_win_a = min(p_win_a, max_win)
        p_win_b = max(p_win_b, min_lose)
    else:
        p_win_b = min(p_win_b, max_win)
        p_win_a = max(p_win_a, min_lose)

    p_draw = max(p_draw, min_draw)

    # Renormalize to ensure sum = 1
    total   = p_win_a + p_draw + p_win_b
    p_win_a /= total
    p_draw  /= total
    p_win_b /= total

    return p_win_a, p_draw, p_win_b


# ─────────────────────────────────────────────
# STEP 4: Convert probabilities to fair odds
# ─────────────────────────────────────────────

def probs_to_fair_odds(p_win_a, p_draw, p_win_b, max_odds=999.99):
    """
    Convert win/draw/lose probabilities to fair decimal odds
    (no bookmaker margin applied).

    If a probability is zero or effectively zero, the odd is
    capped at max_odds (default 999.99) rather than returning
    None or raising a division by zero error.
    """
    def _to_odds(p):
        return 1 / p if p > 0 else max_odds

    return {
        "home_odds" : _to_odds(p_win_a),
        "draw_odds" : _to_odds(p_draw),
        "away_odds" : _to_odds(p_win_b),
    }


# ─────────────────────────────────────────────
# STEP 5: Full pipeline for a single matchup
# ─────────────────────────────────────────────

def elo_match_params(team_a, team_b, elo_ratings):
    """
    Full pipeline: given two team names and the Elo ratings dict,
    compute probabilities, fair odds, and fitted Dixon-Coles
    lambda parameters ready for match simulation.

    Parameters
    ----------
    team_a      : str  — name of team A
    team_b      : str  — name of team B
    elo_ratings : dict — { team_name: elo_points }

    Returns
    -------
    dict with keys:
        team_a, team_b,
        elo_a, elo_b, dr,
        p_win_a, p_draw, p_win_b,
        home_odds, draw_odds, away_odds,
        lam_home, lam_away, rho
    """
    from models.predict_poisson_DC import fit_lambdas

    if team_a not in elo_ratings:
        raise ValueError(f"Team '{team_a}' not found in Elo ratings")
    if team_b not in elo_ratings:
        raise ValueError(f"Team '{team_b}' not found in Elo ratings")

    elo_a = elo_ratings[team_a]
    elo_b = elo_ratings[team_b]
    dr    = elo_a - elo_b

    p_win_a, p_draw, p_win_b = elo_to_probs(elo_a, elo_b)
    odds = probs_to_fair_odds(p_win_a, p_draw, p_win_b)

    fair_probs = {
        "home" : p_win_a,
        "draw" : p_draw,
        "away" : p_win_b
    }
    lam_home, lam_away, rho = fit_lambdas(fair_probs)

    return {
        "team_a"     : team_a,
        "team_b"     : team_b,
        "elo_a"      : elo_a,
        "elo_b"      : elo_b,
        "dr"         : dr,
        "p_win_a"    : round(p_win_a, 4),
        "p_draw"     : round(p_draw,  4),
        "p_win_b"    : round(p_win_b, 4),
        **odds,
        "lam_home"   : lam_home,
        "lam_away"   : lam_away,
        "rho"        : rho
    }


# ─────────────────────────────────────────────
# STEP 6: Diagnostic printer
# ─────────────────────────────────────────────

def print_matchup(team_a, team_b, elo_ratings):
    """
    Print a formatted summary of the Elo-derived probabilities
    and Dixon-Coles parameters for a given matchup.
    """
    p = elo_match_params(team_a, team_b, elo_ratings)

    print(f"\n{'='*55}")
    print(f"  {team_a} vs {team_b}")
    print(f"{'='*55}")
    print(f"  Elo ratings  : {p['elo_a']} vs {p['elo_b']}  (dr = {p['dr']:+d})")
    print(f"\n  Probabilities (Xiong et al. Eq. 5-7):")
    print(f"    {team_a:<20} : {p['p_win_a']*100:5.1f}%")
    print(f"    Draw        {'':<8} : {p['p_draw']*100:5.1f}%")
    print(f"    {team_b:<20} : {p['p_win_b']*100:5.1f}%")
    print(f"\n  Fair Odds:")
    print(f"    {team_a:<20} : {p['home_odds']}")
    print(f"    Draw        {'':<8} : {p['draw_odds']}")
    print(f"    {team_b:<20} : {p['away_odds']}")
    print(f"\n  Dixon-Coles Parameters:")
    print(f"    λ home ({team_a:<16}) : {p['lam_home']:.3f}")
    print(f"    λ away ({team_b:<16}) : {p['lam_away']:.3f}")
    print(f"    ρ (rho)              : {p['rho']:.4f}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    elo_ratings = load_elo_ratings("elo_rankings.csv")

    # Example matchups
    matchups = [
        ("Spain",  "Argentina"),
        ("Brazil", "Morocco"),
        ("USA",    "Panama"),
        ("Spain",  "Qatar"),
    ]

    for team_a, team_b in matchups:
        print_matchup(team_a, team_b, elo_ratings)
