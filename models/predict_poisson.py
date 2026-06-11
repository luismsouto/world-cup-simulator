import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import pandas as pd

# ─────────────────────────────────────────────
# STEP 1: Convert odds to fair probabilities
# ─────────────────────────────────────────────

def odds_to_fair_probs(home_odds, draw_odds, away_odds):
    """
    Convert decimal odds to fair probabilities.
    """
    raw = np.array([1/home_odds, 1/draw_odds, 1/away_odds])
    fair = raw / raw.sum()
    return {"home": fair[0], "draw": fair[1], "away": fair[2]}

# ─────────────────────────────────────────────
# STEP 2: Compute match outcome probs from λ
# ─────────────────────────────────────────────

def outcome_probs_from_lambda(lam_home, lam_away, max_goals = 10):
    """
    Given expected goals for each team, compute P(home win), P(draw), P(away win)
    using independent Poisson distributions.
    """
    home_goals = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_goals = poisson.pmf(np.arange(max_goals + 1), lam_away)
    score_matrix = np.outer(home_goals, away_goals)

    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, 1).sum()

    return p_home, p_draw, p_away

# ─────────────────────────────────────────────
# STEP 3: Optimize λ to match implied probs
# ─────────────────────────────────────────────

def fit_lambdas(fair_probs, max_goals=10, initial_guess=(1.5, 1.2)):
    """
    Find (lam_home, lam_away) whose Poisson outcome probabilities
    best match the implied fair probabilities from the odds.
    """
    target = np.array([fair_probs["home"], fair_probs["draw"], fair_probs["away"]])

    def loss(params):
        lam_h, lam_a = params
        if lam_h <= 0 or lam_a <= 0:
            return 1e6
        computed = np.array(outcome_probs_from_lambda(lam_h, lam_a, max_goals))
        return np.sum((computed - target) ** 2)

    result = minimize(
        loss,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000}
    )

    return result.x[0], result.x[1]

# ─────────────────────────────────────────────
# STEP 4: Build score probability matrix
# ─────────────────────────────────────────────

def build_score_matrix(lam_home, lam_away, max_goals=8):
    """
    Returns a DataFrame where entry [i, j] = P(home scores i, away scores j).
    """
    home_goals = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_goals = poisson.pmf(np.arange(max_goals + 1), lam_away)
    matrix = np.outer(home_goals, away_goals)

    df = pd.DataFrame(
        matrix,
        index=[f"Home {i}" for i in range(max_goals + 1)],
        columns=[f"Away {j}" for j in range(max_goals + 1)]
    )
    return df

# ─────────────────────────────────────────────
# STEP 5: Get top N most likely scorelines
# ─────────────────────────────────────────────

def top_scorelines(lam_home, lam_away, max_goals=8, top_n=10):
    """
    Returns the top N most likely scorelines as a sorted DataFrame.
    """
    df = build_score_matrix(lam_home, lam_away, max_goals)
    rows = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            rows.append({
                "Score": f"{i} - {j}",
                "Probability": df.iloc[i, j],
                "Percentage": f"{df.iloc[i, j]*100:.2f}%"
            })

    result = pd.DataFrame(rows).sort_values("Probability", ascending=False).head(top_n)
    return result.reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP 6: Analyze a single match
# ─────────────────────────────────────────────

def analyze_match(home_team, away_team, home_odds, draw_odds, away_odds,
                  date="", time="", max_goals=8, top_n=10, verbose=True):
    """
    Full pipeline for a single match:
    odds -> fair probs -> lambdas -> scoreline distribution.
    """
    fair = odds_to_fair_probs(home_odds, draw_odds, away_odds)
    lam_h, lam_a = fit_lambdas(fair, max_goals=max_goals)
    p_h, p_d, p_a = outcome_probs_from_lambda(lam_h, lam_a, max_goals)
    top = top_scorelines(lam_h, lam_a, max_goals, top_n)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {date} {time}  |  {home_team} vs {away_team}")
        print(f"{'='*60}")
        print(f"  Odds:  Home {home_odds} | Draw {draw_odds} | Away {away_odds}")

        print(f"\n  Fair Probabilities:")
        print(f"    Home Win : {fair['home']*100:.1f}%")
        print(f"    Draw     : {fair['draw']*100:.1f}%")
        print(f"    Away Win : {fair['away']*100:.1f}%")

        print(f"\n  Expected Goals (fitted):")
        print(f"    λ Home ({home_team:}) : {lam_h:.3f}")
        print(f"    λ Away ({away_team:}) : {lam_a:.3f}")

        print(f"\n  Verification (fitted vs implied):")
        print(f"    Home  : fitted {p_h*100:.1f}% | implied {fair['home']*100:.1f}%")
        print(f"    Draw  : fitted {p_d*100:.1f}% | implied {fair['draw']*100:.1f}%")
        print(f"    Away  : fitted {p_a*100:.1f}% | implied {fair['away']*100:.1f}%")

        print(f"\n  Top {top_n} Most Likely Scorelines:")
        print(top.to_string(index=True))

    return {
        "lam_home": lam_h,
        "lam_away": lam_a,
        "top_scores": top
    }

# ─────────────────────────────────────────────
# STEP 7: Load CSV and analyze all matches
# ─────────────────────────────────────────────

def load_matches(csv_path):
    """
    Load match odds from CSV file.
    Expected columns: date, time, home_team, away_team,
                      home_odds, draw_odds, away_odds
    """
    df = pd.read_csv(csv_path)
    required = {"date", "time", "home_team", "away_team",
                "home_odds", "draw_odds", "away_odds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def analyze_all_matches(csv_path, max_goals=8, top_n=5, verbose=True):
    """
    Load all matches from CSV and run the full Poisson analysis on each.
    Returns a summary DataFrame with the most likely score per match.
    """
    df = load_matches(csv_path)
    summary_rows = []

    for _, m in df.iterrows():
        result = analyze_match(
            home_team  = m["home_team"],
            away_team  = m["away_team"],
            home_odds  = m["home_odds"],
            draw_odds  = m["draw_odds"],
            away_odds  = m["away_odds"],
            date       = m["date"],
            time       = m["time"],
            max_goals  = max_goals,
            top_n      = top_n,
            verbose    = verbose
        )

        top = result["top_scores"]
        summary_rows.append({
            "Date"          : m["date"],
            "Time"          : m["time"],
            "Home"          : m["home_team"],
            "Away"          : m["away_team"],
            "λ Home"        : round(result["lam_home"], 3),
            "λ Away"        : round(result["lam_away"], 3),
            "Top Score"     : top.iloc[0]["Score"],
            "Probability"   : top.iloc[0]["Percentage"],
            "2nd Score"     : top.iloc[1]["Score"],
            "3rd Score"     : top.iloc[2]["Score"],
        })

    summary = pd.DataFrame(summary_rows)

    print("\n\n" + "="*60)
    print("  SUMMARY — Most Likely Scores per Match")
    print("="*60)
    print(summary.to_string(index=False))

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # Option A: analyze all matches from CSV
    summary = analyze_all_matches(
        csv_path  = "odds_input.csv",
        max_goals = 8,
        top_n     = 5,
        verbose   = True   # set to False for summary-only output
    )

    # Optionally save summary to CSV
    summary.to_csv("results.csv", index=False)
    print("\n  Results saved to results.csv")

    # ─────────────────────────────────────────
    # Option B: analyze a single match manually
    # ─────────────────────────────────────────
    # analyze_match(
    #     home_team = "France",
    #     away_team = "Brazil",
    #     home_odds = 2.50,
    #     draw_odds = 3.20,
    #     away_odds = 2.80,
    #     date      = "2026-06-20",
    #     time      = "20:00",
    #     top_n     = 10
    # )
