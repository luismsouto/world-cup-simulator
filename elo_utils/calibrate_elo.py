import numpy as np
import pandas as pd
from scipy.optimize import minimize
from models.predict_poisson_DC import load_matches, odds_to_fair_probs
from elo_utils import load_elo_ratings, _p_draw

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CSV_PATH        = "../input_data/odds_input.csv"
ELO_RATINGS_CSV = "input_data/elo_rankings.csv"
OUTPUT_CSV      = "input_data/elo_rankings_calibrated.csv"

# Regularisation strength:
#   Higher λ → smaller adjustments, stay closer to original Elo
#   Lower  λ → larger adjustments, fit bookmaker odds more aggressively
#
# With 72 matches and ~3 per team, λ=0.01 is a sensible default:
# it allows meaningful corrections (up to ~150-200 pts) without
# overfitting to the small sample.
LAMBDA_REG = 0.000003 #0.00001

# Maximum allowed adjustment per team in Elo points.
# Prevents the optimiser from producing nonsensical ratings.
MAX_DELTA  = 400


# ─────────────────────────────────────────────
# STEP 1: Raw (unbounded) Elo probabilities
# ─────────────────────────────────────────────

def _elo_to_probs_raw(elo_a, elo_b):
    """
    Compute win/draw/lose probabilities without applying the
    realistic football bounds (max_win, min_lose, min_draw).

    Used inside the optimiser so the gradient signal is always
    informative, even for extreme Elo gaps where the bounds
    would otherwise clip the probabilities and hide the error.

    Parameters
    ----------
    elo_a : float
    elo_b : float

    Returns
    -------
    p_win_a, p_draw, p_win_b : floats (may not sum to exactly 1
                                before clamping negatives)
    """
    dr      = elo_a - elo_b
    p_draw  = _p_draw(dr)
    p_win_a = 1 / (1 + 10 ** (-dr / 400)) - p_draw / 2
    p_win_b = 1 / (1 + 10 ** (+dr / 400)) - p_draw / 2

    # Clamp negatives only — do NOT apply football bounds
    p_win_a = max(p_win_a, 0.0)
    p_win_b = max(p_win_b, 0.0)

    # Renormalise
    total   = p_win_a + p_draw + p_win_b
    p_win_a /= total
    p_draw  /= total
    p_win_b /= total

    return p_win_a, p_draw, p_win_b


# ─────────────────────────────────────────────
# STEP 2: Build optimisation inputs
# ─────────────────────────────────────────────

def build_optimisation_inputs(csv_path, elo_ratings):
    """
    Load all matches and extract bookmaker fair probabilities
    and original Elo ratings for each team pair.

    Parameters
    ----------
    csv_path    : str
    elo_ratings : dict { team: elo_points }

    Returns
    -------
    matches : list of dicts with keys:
        home_team, away_team,
        p_home_bk, p_draw_bk, p_away_bk,
        elo_home_orig, elo_away_orig
    teams : list of str — unique ordered team list
    team_index : dict { team: index } — for delta vector indexing
    skipped : list of str — matches skipped due to missing Elo
    """
    df      = load_matches(csv_path)
    matches = []
    skipped = []

    for _, m in df.iterrows():
        home = m["home_team"]
        away = m["away_team"]

        if home not in elo_ratings or away not in elo_ratings:
            skipped.append(f"{home} vs {away}")
            continue

        fair = odds_to_fair_probs(
            m["home_odds"], m["draw_odds"], m["away_odds"]
        )

        matches.append({
            "home_team"     : home,
            "away_team"     : away,
            "p_home_bk"     : fair["home"],
            "p_draw_bk"     : fair["draw"],
            "p_away_bk"     : fair["away"],
            "elo_home_orig" : elo_ratings[home],
            "elo_away_orig" : elo_ratings[away],
        })

    # Build ordered team list and index
    teams = sorted({m["home_team"] for m in matches} |
                   {m["away_team"] for m in matches})
    team_index = {team: i for i, team in enumerate(teams)}

    return matches, teams, team_index, skipped


# ─────────────────────────────────────────────
# STEP 3: Loss function
# ─────────────────────────────────────────────

def build_loss_fn(matches, teams, team_index, elo_ratings, lambda_reg):
    """
    Build and return the loss function for the optimiser.

    The loss has two components:
        1. Prediction loss: MSE between Elo-derived and bookmaker
           fair probabilities, summed across all matches
        2. Regularisation: penalises large deviations from the
           original Elo ratings, scaled by lambda_reg

    Loss = Σ_matches [ (P_win_elo - P_win_bk)² +
                       (P_draw_elo - P_draw_bk)² +
                       (P_away_elo - P_away_bk)² ]
         + λ * Σ_teams delta_i²

    Parameters
    ----------
    matches     : list of match dicts
    teams       : list of str
    team_index  : dict { team: index }
    elo_ratings : dict { team: elo_points }
    lambda_reg  : float

    Returns
    -------
    loss_fn : callable(deltas) -> float
    """
    def loss_fn(deltas):
        prediction_loss = 0.0

        for m in matches:
            i_home = team_index[m["home_team"]]
            i_away = team_index[m["away_team"]]

            elo_home = m["elo_home_orig"] + deltas[i_home]
            elo_away = m["elo_away_orig"] + deltas[i_away]

            p_win, p_draw, p_lose = _elo_to_probs_raw(elo_home, elo_away)

            prediction_loss += (p_win  - m["p_home_bk"]) ** 2
            prediction_loss += (p_draw - m["p_draw_bk"]) ** 2
            prediction_loss += (p_lose - m["p_away_bk"]) ** 2

        reg_loss = lambda_reg * np.sum(deltas ** 2)

        return prediction_loss + reg_loss

    return loss_fn


# ─────────────────────────────────────────────
# STEP 4: Run calibration
# ─────────────────────────────────────────────

def calibrate(csv_path=CSV_PATH,
              elo_ratings_csv=ELO_RATINGS_CSV,
              output_csv=OUTPUT_CSV,
              lambda_reg=LAMBDA_REG,
              max_delta=MAX_DELTA):
    """
    Run the full Elo calibration pipeline:
        1. Load matches and original Elo ratings
        2. Optimise per-team delta offsets
        3. Print diagnostics
        4. Save calibrated ratings to CSV

    Parameters
    ----------
    csv_path        : str — path to odds_input.csv
    elo_ratings_csv : str — path to elo_ratings.csv
    output_csv      : str — path to save calibrated ratings
    lambda_reg      : float — regularisation strength (default 0.01)
    max_delta       : float — max allowed Elo adjustment per team

    Returns
    -------
    calibrated_ratings : dict { team: calibrated_elo }
    """
    print("\n" + "=" * 65)
    print("  ELO CALIBRATION")
    print("=" * 65)
    print(f"  λ (regularisation) : {lambda_reg}")
    print(f"  Max delta          : ±{max_delta} pts")

    # ── Load data ─────────────────────────────────────────────────
    elo_ratings = load_elo_ratings(elo_ratings_csv)
    matches, teams, team_index, skipped = build_optimisation_inputs(
        csv_path, elo_ratings
    )

    print(f"  Matches loaded     : {len(matches)}")
    print(f"  Teams              : {len(teams)}")

    if skipped:
        print(f"\n  ⚠️  Skipped matches (missing Elo rating):")
        for s in skipped:
            print(f"       - {s}")

    # ── Build and run optimiser ───────────────────────────────────
    loss_fn = build_loss_fn(
        matches, teams, team_index, elo_ratings, lambda_reg
    )

    n_teams  = len(teams)
    x0       = np.zeros(n_teams)
    bounds   = [(-max_delta, max_delta)] * n_teams

    print(f"\n  Running optimisation ({n_teams} parameters)...")

    result = minimize(
        loss_fn,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 50000,  # was 10000
            "ftol": 1e-15,  # was 1e-12
            "gtol": 1e-10,  # was 1e-8
        }
    )

    if not result.success:
        print(f"\n  ⚠️  Optimiser warning: {result.message}")
    else:
        print(f"  ✅ Converged in {result.nit} iterations")
        print(f"  Final loss: {result.fun:.6f}")

    deltas = result.x

    # ── Build calibrated ratings ──────────────────────────────────
    calibrated_ratings = {
        team: round(elo_ratings[team] + deltas[team_index[team]], 1)
        for team in teams
    }

    # ── Print diagnostics ─────────────────────────────────────────
    _print_diagnostics(
        matches, teams, team_index, elo_ratings,
        calibrated_ratings, deltas, lambda_reg
    )

    # ── Save to CSV ───────────────────────────────────────────────
    rows = [
        {
            "country"              : team,
            #"elo_points"           : elo_ratings[team],
            #"elo_points_calibrated": calibrated_ratings[team],
            "elo_points": calibrated_ratings[team]
            #delta"                : round(deltas[team_index[team]], 1),
        }
        for team in sorted(teams)
    ]
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"\n  Calibrated ratings saved to {output_csv}")

    return calibrated_ratings


# ─────────────────────────────────────────────
# STEP 5: Diagnostics
# ─────────────────────────────────────────────

def _print_diagnostics(matches, teams, team_index, elo_ratings,
                        calibrated_ratings, deltas, lambda_reg):
    """
    Print a before/after comparison of Elo probabilities vs
    bookmaker probabilities, plus per-team adjustment summary.
    """

    # ── Per-match comparison ──────────────────────────────────────
    print(f"\n  {'─'*100}")
    print(f"  MATCH-LEVEL DIAGNOSTICS")
    print(f"  {'─'*100}")
    print(
        f"  {'Home':<22} {'Away':<22} "
        f"{'BK H':>6} {'BK D':>6} {'BK A':>6}  "
        f"{'Old H':>6} {'Old D':>6} {'Old A':>6}  "
        f"{'New H':>6} {'New D':>6} {'New A':>6}"
    )
    print(
        f"  {'-'*22} {'-'*22} "
        f"{'-'*6} {'-'*6} {'-'*6}  "
        f"{'-'*6} {'-'*6} {'-'*6}  "
        f"{'-'*6} {'-'*6} {'-'*6}"
    )

    old_mse = 0.0
    new_mse = 0.0

    for m in matches:
        # Old probabilities
        p_old_h, p_old_d, p_old_a = _elo_to_probs_raw(
            m["elo_home_orig"], m["elo_away_orig"]
        )

        # New probabilities
        p_new_h, p_new_d, p_new_a = _elo_to_probs_raw(
            calibrated_ratings[m["home_team"]],
            calibrated_ratings[m["away_team"]]
        )

        old_mse += (p_old_h - m["p_home_bk"]) ** 2
        old_mse += (p_old_d - m["p_draw_bk"]) ** 2
        old_mse += (p_old_a - m["p_away_bk"]) ** 2

        new_mse += (p_new_h - m["p_home_bk"]) ** 2
        new_mse += (p_new_d - m["p_draw_bk"]) ** 2
        new_mse += (p_new_a - m["p_away_bk"]) ** 2

        print(
            f"  {m['home_team']:<22} {m['away_team']:<22} "
            f"{m['p_home_bk']*100:>5.1f}% "
            f"{m['p_draw_bk']*100:>5.1f}% "
            f"{m['p_away_bk']*100:>5.1f}%  "
            f"{p_old_h*100:>5.1f}% "
            f"{p_old_d*100:>5.1f}% "
            f"{p_old_a*100:>5.1f}%  "
            f"{p_new_h*100:>5.1f}% "
            f"{p_new_d*100:>5.1f}% "
            f"{p_new_a*100:>5.1f}%"
        )

    n = len(matches)
    print(f"\n  MSE before calibration : {old_mse/n:.6f}")
    print(f"  MSE after  calibration : {new_mse/n:.6f}")
    print(f"  Improvement            : {(1 - new_mse/old_mse)*100:.1f}%")

    # ── Per-team adjustments ──────────────────────────────────────
    print(f"\n  {'─'*65}")
    print(f"  PER-TEAM ELO ADJUSTMENTS  (λ={lambda_reg})")
    print(f"  {'─'*65}")
    print(f"  {'Team':<22} {'Original':>9} {'Calibrated':>11} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*9} {'-'*11} {'-'*8}")

    sorted_teams = sorted(teams, key=lambda t: deltas[team_index[t]],
                          reverse=True)
    for team in sorted_teams:
        orig  = elo_ratings[team]
        calib = calibrated_ratings[team]
        delta = deltas[team_index[team]]
        bar   = "▲" if delta > 0 else "▼" if delta < 0 else " "
        print(
            f"  {team:<22} {orig:>9.0f} {calib:>11.1f} "
            f"{delta:>+7.1f} {bar}"
        )


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    calibrate(
        csv_path        = CSV_PATH,
        elo_ratings_csv = ELO_RATINGS_CSV,
        output_csv      = OUTPUT_CSV,
        lambda_reg      = LAMBDA_REG,
        max_delta       = MAX_DELTA,
    )
