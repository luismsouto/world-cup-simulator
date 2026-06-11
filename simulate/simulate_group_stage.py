import numpy as np
import pandas as pd
from models.predict_poisson_DC import load_matches, fit_lambdas, odds_to_fair_probs, build_score_matrix

# ─────────────────────────────────────────────
# STEP 1: Pre-fit all match parameters from odds
# ─────────────────────────────────────────────

def prepare_match_params(csv_path):
    """
    Load all matches and pre-fit lambda/rho parameters for each.
    Returns a dict keyed by match_id with all parameters ready for simulation.
    """
    df = load_matches(csv_path)
    match_params = {}

    print("Fitting Dixon-Coles parameters for all matches...")
    for _, m in df.iterrows():
        fair = odds_to_fair_probs(m["home_odds"], m["draw_odds"], m["away_odds"])
        lam_h, lam_a, rho = fit_lambdas(fair)
        match_params[m["match_id"]] = {
            "match_id"  : m["match_id"],
            "group"     : m["group"],
            "date"      : m["date"],
            "time"      : m["time"],
            "home_team" : m["home_team"],
            "away_team" : m["away_team"],
            "lam_home"  : lam_h,
            "lam_away"  : lam_a,
            "rho"       : rho
        }
        print(f"  ✅ {m['home_team']} vs {m['away_team']} "
              f"| λh={lam_h:.3f} λa={lam_a:.3f} ρ={rho:.4f}")

    return match_params

# ─────────────────────────────────────────────
# STEP 2: Simulate a single match
# ─────────────────────────────────────────────

def simulate_match(lam_home, lam_away, rho):
    """
    Simulate a single match by sampling from the Dixon-Coles
    score probability matrix.
    Returns (home_goals, away_goals).
    """
    score_matrix = build_score_matrix(lam_home, lam_away, rho)
    matrix_values = score_matrix.values

    # Flatten matrix into 1D probability distribution and sample
    probs = matrix_values.flatten()
    probs = np.clip(probs, 0.0, None)  # ── guard against any residual negatives
    probs /= probs.sum()

    idx = np.random.choice(len(probs), p=probs)

    home_goals, away_goals = np.unravel_index(idx, matrix_values.shape)
    return home_goals, away_goals
# ─────────────────────────────────────────────
# STEP 3: Simulate a full group
# ─────────────────────────────────────────────
def simulate_group(group_matches, match_params, fifa_rankings):
    """
    Simulate all matches in a single group and return final standings
    using the official FIFA tiebreaker rules.

    Tiebreaker cascade (applied within clusters of teams tied on points):
      Step 1a: H2H points
      Step 1b: H2H goal difference
      Step 1c: H2H goals scored
      Step 2a: Overall goal difference
      Step 2b: Overall goals scored
      ----(yellow/red cards) -> ignored
      Step 3:  FIFA ranking (lower number = better)
    """

    # ── Collect all teams in this group ──────────────────────────
    teams = list({m["home_team"] for m in group_matches} |
                 {m["away_team"] for m in group_matches})

    # ── Initialize overall records ────────────────────────────────
    records = {t: {
        "team"   : t,
        "played" : 0,
        "wins"   : 0,
        "draws"  : 0,
        "losses" : 0,
        "gf"     : 0,   # goals for
        "ga"     : 0,   # goals against
        "gd"     : 0,   # goal difference
        "points" : 0
    } for t in teams}

    # ── Initialize head-to-head records ──────────────────────────
    # h2h[team][opponent] = {"gf": 0, "ga": 0, "points": 0}
    h2h = {t: {opp: {"gf": 0, "ga": 0, "points": 0}
               for opp in teams if opp != t}
           for t in teams}

    match_results = []

    # ── Simulate all matches and update records ───────────────────
    for m in group_matches:
        p  = match_params[m["match_id"]]
        hg, ag = simulate_match(p["lam_home"], p["lam_away"], p["rho"])

        home = m["home_team"]
        away = m["away_team"]

        # Overall records
        records[home]["played"] += 1
        records[away]["played"] += 1
        records[home]["gf"]     += hg
        records[home]["ga"]     += ag
        records[away]["gf"]     += ag
        records[away]["ga"]     += hg

        # Head-to-head records
        h2h[home][away]["gf"] += hg
        h2h[home][away]["ga"] += ag
        h2h[away][home]["gf"] += ag
        h2h[away][home]["ga"] += hg

        if hg > ag:       # home win
            records[home]["wins"]     += 1
            records[home]["points"]   += 3
            records[away]["losses"]   += 1
            h2h[home][away]["points"] += 3
        elif hg == ag:    # draw
            records[home]["draws"]    += 1
            records[home]["points"]   += 1
            records[away]["draws"]    += 1
            records[away]["points"]   += 1
            h2h[home][away]["points"] += 1
            h2h[away][home]["points"] += 1
        else:             # away win
            records[away]["wins"]     += 1
            records[away]["points"]   += 3
            records[home]["losses"]   += 1
            h2h[away][home]["points"] += 3

        match_results.append({
            "home_team"  : home,
            "away_team"  : away,
            "home_goals" : hg,
            "away_goals" : ag,
            "result"     : f"{hg} - {ag}"
        })

    # Overall goal difference
    for t in teams:
        records[t]["gd"] = records[t]["gf"] - records[t]["ga"]


    # ── Tiebreaker helpers ────────────────────────────────────────

    def h2h_key(team, opponents):
        """
        Step 1: H2H points, GD and GF against a specific
        set of opponents only.
        """
        pts = sum(h2h[team][opp]["points"] for opp in opponents)
        gf  = sum(h2h[team][opp]["gf"]     for opp in opponents)
        ga  = sum(h2h[team][opp]["ga"]     for opp in opponents)
        return (pts, gf - ga, gf)

    def overall_key(team):
        """
        Step 2: Overall GD and GF across all group matches.
        """
        return (records[team]["gd"], records[team]["gf"])

    def fifa_key(team):
        """
        Step 3: FIFA ranking — lower rank number is better,
        so we negate it so that higher values sort first.
        """
        return -fifa_rankings.get(team, 9999)


    # ── Sorting logic ─────────────────────────────────────────────

    def sort_cluster(cluster):
        if len(cluster) <= 1:
            return cluster

        opponents = cluster  # H2H only among the tied teams

        # ── Step 1: H2H criteria ──────────────────────────────────
        after_step1 = sorted(
            cluster,
            key=lambda t: h2h_key(t, [opp for opp in opponents if opp != t]),
            reverse=True
        )

        resolved = []
        i = 0
        while i < len(after_step1):
            j = i + 1
            while j < len(after_step1) and (
                    h2h_key(after_step1[j], [opp for opp in opponents if opp != after_step1[j]]) ==
                    h2h_key(after_step1[i], [opp for opp in opponents if opp != after_step1[i]])
            ):
                j += 1

            sub_cluster = after_step1[i:j]

            if len(sub_cluster) == len(cluster):
                # ── H2H made no progress at all: move to Step 2 ──────
                after_step2 = sorted(sub_cluster, key=overall_key, reverse=True)

                k = 0
                while k < len(after_step2):
                    l = k + 1
                    while l < len(after_step2) and (
                            overall_key(after_step2[l]) == overall_key(after_step2[k])
                    ):
                        l += 1

                    sub_sub_cluster = after_step2[k:l]

                    if len(sub_sub_cluster) == 1:
                        resolved.extend(sub_sub_cluster)
                    else:
                        # ── Step 3: FIFA ranking ──────────────────────
                        resolved.extend(
                            sorted(sub_sub_cluster, key=fifa_key, reverse=True)
                        )

                    k = l

            elif len(sub_cluster) == 1:
                # Fully resolved by H2H
                resolved.extend(sub_cluster)
            else:
                # ── H2H made progress but sub-cluster still tied:   ──
                # ── restart the full cascade for this smaller group ──
                resolved.extend(sort_cluster(sub_cluster))

            i = j

        return resolved

    # ── Build final standings ─────────────────────────────────────

    # First sort all teams by overall points
    by_points = sorted(teams, key=lambda t: records[t]["points"], reverse=True)

    # Identify clusters tied on points and resolve each one
    final_order = []
    i = 0
    while i < len(by_points):
        j = i + 1
        while j < len(by_points) and (
            records[by_points[j]]["points"] == records[by_points[i]]["points"]
        ):
            j += 1

        cluster = by_points[i:j]
        final_order.extend(sort_cluster(cluster))
        i = j

    # Build standings DataFrame
    standings = pd.DataFrame([
        {"rank": rank, **records[team]}
        for rank, team in enumerate(final_order, start=1)
    ])

    return standings, match_results

