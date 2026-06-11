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

def dixon_coles_tau(home_goals, away_goals, lam_home, lam_away, rho):
    """
    Correction factor (tau) applied only to low-scoring cells:
    (0,0), (1,0), (0,1), (1,1).

    - rho = 0   → no correction (pure Poisson)
    - rho < 0   → boosts 0-0 and 1-1, reduces 1-0 and 0-1 (typical)
    - rho range : usually between -0.2 and 0.0 for football
    """
    if home_goals == 0 and away_goals == 0:
        tau = 1 - lam_home * lam_away * rho
    elif home_goals == 1 and away_goals == 0:
        tau = 1 + lam_away * rho
    elif home_goals == 0 and away_goals == 1:
        tau = 1 + lam_home * rho
    elif home_goals == 1 and away_goals == 1:
        tau = 1 - rho
    else:
        return 1.0

    return max(tau, 0.0)  # ── clamp to prevent negative probabilities



def outcome_probs_from_lambda(lam_home, lam_away, rho=0.0, max_goals=None):
    """
    Compute P(home win), P(draw), P(away win) using Poisson distributions
    with optional Dixon-Coles correction (rho).
    """
    if max_goals is None:
        max_lam = max(lam_home, lam_away)
        max_goals = max(8, int(poisson.ppf(0.99999, max_lam)) + 1)

    home_pmf = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lam_away)

    score_mat = np.outer(home_pmf, away_pmf)

    # Apply Dixon-Coles tau correction to low-score cells
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            score_mat[i, j] *= dixon_coles_tau(i, j, lam_home, lam_away, rho)

    # ── Renormalize after Dixon-Coles correction ──────────────────
    score_mat /= score_mat.sum()

    p_home = np.tril(score_mat, -1).sum()
    p_draw = np.trace(score_mat)
    p_away = np.triu(score_mat, 1).sum()

    return p_home, p_draw, p_away

# ─────────────────────────────────────────────
# STEP 3: Optimize λ to match implied probs
# ─────────────────────────────────────────────

def fit_lambdas(fair_probs, max_goals=None, initial_guess=(1.5, 1.2, -0.1)):
    """
    Find (lam_home, lam_away, rho) whose Poisson + Dixon-Coles outcome
    probabilities best match the implied fair probabilities from the odds.

    rho is constrained to [-0.5, 0.0] — negative values are the
    meaningful range for football (boosts draws and low scores).
    """
    target = np.array([fair_probs["home"], fair_probs["draw"], fair_probs["away"]])

    def loss(params):
        lam_h, lam_a, rho = params
        if lam_h <= 0 or lam_a <= 0:
            return 1e6
        if rho < -0.5 or rho > 0.0:
            return 1e6  # keep rho in valid range
        computed = np.array(outcome_probs_from_lambda(lam_h, lam_a, rho, max_goals))
        return np.sum((computed - target) ** 2)

    result = minimize(
        loss,
        x0=initial_guess,
        method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 10000}
    )

    lam_home, lam_away, rho = result.x
    return lam_home, lam_away, rho

# ─────────────────────────────────────────────
# STEP 4: Build score probability matrix
# ─────────────────────────────────────────────

def build_score_matrix(lam_home, lam_away, rho=0.0, max_goals=None):
    """
    Returns a DataFrame where entry [i, j] = P(home scores i, away scores j),
    with Dixon-Coles correction applied to low-scoring cells.
    """
    if max_goals is None:
        max_lam = max(lam_home, lam_away)
        max_goals = max(8, int(poisson.ppf(0.99999, max_lam)) + 1)

    home_pmf = poisson.pmf(np.arange(max_goals + 1), lam_home)
    away_pmf = poisson.pmf(np.arange(max_goals + 1), lam_away)

    matrix = np.outer(home_pmf, away_pmf)

    # Apply Dixon-Coles correction
    for i in range(min(2, max_goals + 1)):
        for j in range(min(2, max_goals + 1)):
            matrix[i, j] *= dixon_coles_tau(i, j, lam_home, lam_away, rho)

    # Renormalize so the full matrix still sums to 1
    matrix /= matrix.sum()

    df = pd.DataFrame(
        matrix,
        index   = [f"Home {i}" for i in range(max_goals + 1)],
        columns = [f"Away {j}" for j in range(max_goals + 1)]
    )
    return df

# ─────────────────────────────────────────────
# STEP 5: Get top N most likely scorelines
# ─────────────────────────────────────────────

def top_scorelines(lam_home, lam_away, rho=0.0, max_goals=None, top_n=10):
    """
    Returns the top N most likely scorelines as a sorted DataFrame.
    """
    df = build_score_matrix(lam_home, lam_away, rho, max_goals)
    rows = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            rows.append({
                "Score"       : f"{i} - {j}",
                "Probability" : df.iloc[i, j],
                "Percentage"  : f"{df.iloc[i, j]*100:.2f}%"
            })

    result = pd.DataFrame(rows).sort_values("Probability", ascending=False).head(top_n)
    return result.reset_index(drop=True)

# ─────────────────────────────────────────────
# STEP 6: Analyze a single match
# ─────────────────────────────────────────────

def analyze_match(home_team, away_team, home_odds, draw_odds, away_odds,
                  date="", time="", max_goals=None, top_n=10, verbose=True):
    """
    Full pipeline for a single match:
    odds -> fair probs -> lambdas + rho -> Dixon-Coles scoreline distribution.
    """
    fair = odds_to_fair_probs(home_odds, draw_odds, away_odds)
    lam_h, lam_a, rho = fit_lambdas(fair, max_goals=max_goals)
    p_h, p_d, p_a = outcome_probs_from_lambda(lam_h, lam_a, rho, max_goals)
    top = top_scorelines(lam_h, lam_a, rho, max_goals, top_n)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {date} {time}  |  {home_team} vs {away_team}")
        print(f"{'='*60}")
        print(f"  Odds : Home {home_odds} | Draw {draw_odds} | Away {away_odds}")

        print(f"\n  Fair Probabilities (vig removed):")
        print(f"    Home Win : {fair['home']*100:.1f}%")
        print(f"    Draw     : {fair['draw']*100:.1f}%")
        print(f"    Away Win : {fair['away']*100:.1f}%")

        print(f"\n  Fitted Parameters (Dixon-Coles):")
        print(f"    λ Home ({home_team}) : {lam_h:.3f}")
        print(f"    λ Away ({away_team}) : {lam_a:.3f}")
        print(f"    ρ (rho)              : {rho:.4f}")

        print(f"\n  Verification (fitted vs implied):")
        print(f"    Home  : fitted {p_h*100:.1f}% | implied {fair['home']*100:.1f}%")
        print(f"    Draw  : fitted {p_d*100:.1f}% | implied {fair['draw']*100:.1f}%")
        print(f"    Away  : fitted {p_a*100:.1f}% | implied {fair['away']*100:.1f}%")

        print(f"\n  Top {top_n} Most Likely Scorelines:")
        print(top.to_string(index=True))

    return {
        "lam_home"   : lam_h,
        "lam_away"   : lam_a,
        "rho"        : rho,
        "top_scores" : top
    }

# ─────────────────────────────────────────────
# STEP 7: Load CSV and analyze all matches
# ─────────────────────────────────────────────

def load_matches(csv_path):
    """
    Load match odds from CSV file.
    Expected columns: match_id, group, date, time,
                      home_team, away_team,
                      home_odds, draw_odds, away_odds
    """
    df = pd.read_csv(csv_path)
    required = {"date", "time", "home_team", "away_team",
                "home_odds", "draw_odds", "away_odds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def analyze_all_matches(csv_path, max_goals=None, top_n=5, verbose=True):
    """
    Load all matches from CSV and run the full Dixon-Coles analysis on each.
    Returns a summary DataFrame with the most likely score per match.
    """
    df = load_matches(csv_path)
    summary_rows = []

    for _, m in df.iterrows():
        result = analyze_match(
            home_team = m["home_team"],
            away_team = m["away_team"],
            home_odds = m["home_odds"],
            draw_odds = m["draw_odds"],
            away_odds = m["away_odds"],
            date      = m.get("date", ""),
            time      = m.get("time", ""),
            max_goals = max_goals,
            top_n     = top_n,
            verbose   = verbose
        )

        top = result["top_scores"]
        summary_rows.append({
            "Match ID"    : m.get("match_id", ""),
            "Group"       : m.get("group", ""),
            "Date"        : m.get("date", ""),
            "Time"        : m.get("time", ""),
            "Home"        : m["home_team"],
            "Away"        : m["away_team"],
            "λ Home"      : round(result["lam_home"], 3),
            "λ Away"      : round(result["lam_away"], 3),
            "ρ"           : round(result["rho"], 4),
            "Top Score"   : top.iloc[0]["Score"],
            "Probability" : top.iloc[0]["Percentage"],
            "2nd Score"   : top.iloc[1]["Score"],
            "3rd Score"   : top.iloc[2]["Score"],
        })

    summary = pd.DataFrame(summary_rows)

    print("\n\n" + "="*60)
    print("  SUMMARY — Most Likely Scores per Match (Dixon-Coles)")
    print("="*60)
    print(summary.to_string(index=False))

    return summary


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
# if __name__ == "__main__":
#
#     # Option A: analyze all matches from CSV
#     summary = analyze_all_matches(
#         csv_path  = "odds_input.csv",
#         max_goals = None,   # dynamic — adapts per match
#         top_n     = 5,
#         verbose   = True    # set to False for summary-only output
#     )
#
#     # Save results
#     summary.to_csv("results_dc_correction.csv", index=False)
#     print("\n  Results saved to results.csv")
#
#     # ─────────────────────────────────────────
#     # Option B: single match
#     # ─────────────────────────────────────────
#     # analyze_match(
#     #     home_team = "France",
#     #     away_team = "Brazil",
#     #     home_odds = 2.50,
#     #     draw_odds = 3.20,
#     #     away_odds = 2.80,
#     #     top_n     = 10
#     # )
