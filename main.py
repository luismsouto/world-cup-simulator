import numpy as np
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
from simulate.simulate_group_stage import prepare_match_params, simulate_group
from simulate.simulate_knockout_phase import simulate_knockout_phase
from simulate.third_place_combinations import load_third_place_combinations
from elo_utils.elo_utils import load_elo_ratings

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CSV_PATH          = "input_data/odds_input.csv"
FIFA_RANKINGS_CSV = "input_data/fifa_rankings.csv"
ELO_RATINGS_CSV   = "input_data/elo_rankings_calibrated.csv"
COMBINATIONS_CSV  = "input_data/third_place_match_combinations.csv"
N_SIMULATIONS     = 1
RANDOM_SEED       = 30
MODE              = "single"   # "single" or "multi"
OUTPUT_DIR        = "output_data"

# ─────────────────────────────────────────────
# STEP 0: Load static data
# ─────────────────────────────────────────────

def load_fifa_rankings(csv_path):
    """
    Load FIFA rankings from a CSV file with two columns:
        - country : team name (must match team names in odds_input.csv)
        - rank    : FIFA ranking position (lower = better)

    Returns a dict: { "Brazil": 4, "France": 2, ... }
    """
    df = pd.read_csv(csv_path)
    required = {"country", "rank"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FIFA rankings CSV is missing columns: {missing}")
    return dict(zip(df["country"], df["rank"]))


def load_group_structure(match_params):
    """
    Derive group structure from pre-fitted match parameters.
    Returns a dict: { group_name: [list of match dicts] }
    """
    groups = defaultdict(list)
    for match_id, p in match_params.items():
        groups[p["group"]].append({
            "match_id"  : p["match_id"],
            "home_team" : p["home_team"],
            "away_team" : p["away_team"],
            "group"     : p["group"],
        })
    return dict(groups)


# ─────────────────────────────────────────────
# STEP 1: Identify best 8 third-place teams
# ─────────────────────────────────────────────

def get_best_third_places(third_place_records, fifa_rankings, n=8):
    """
    Given a list of 3rd-place team records (one per group),
    return a dict of { team: group } for the n best third places.

    Ranking criteria (in order):
        1. Points
        2. Goal difference
        3. Goals scored
        4. FIFA ranking (lower number = better, so we negate)
    """
    ranked = sorted(
        third_place_records,
        key=lambda r: (
            r["points"],
            r["gd"],
            r["gf"],
            -fifa_rankings.get(r["team"], 9999)
        ),
        reverse=True
    )
    return {r["team"]: r["group"] for r in ranked[:n]}


# ─────────────────────────────────────────────
# STEP 2: Build group stage results dict
# ─────────────────────────────────────────────

def build_group_stage_results(all_standings):
    """
    Convert per-group standings into a flat slot -> team dict
    for use by the knockout phase builder.

    Parameters
    ----------
    all_standings : dict { group: standings_DataFrame }

    Returns
    -------
    dict e.g.:
        {
            "1A": "Mexico",
            "2A": "South Korea",
            "3A": "Czech Republic",
            "4A": "South Africa",
            "1B": "Switzerland",
            ...
        }
    """
    results = {}
    for group, standings in all_standings.items():
        for _, row in standings.iterrows():
            results[f"{row['rank']}{group}"] = row["team"]
    return results


# ─────────────────────────────────────────────
# STEP 3: Single simulation
# ─────────────────────────────────────────────

def run_single_simulation(match_params, groups, fifa_rankings,
                           elo_ratings, combinations):
    """
    Run one full tournament simulation and print detailed output:
        - Group stage match results
        - Group stage standings
        - Best 8 third-place teams
        - Full knockout phase match-by-match results
    """
    print("\n" + "=" * 65)
    print("  SINGLE SIMULATION — FULL TOURNAMENT")
    print("=" * 65)

    all_standings       = {}
    third_place_records = []

    # ── Group stage ───────────────────────────────────────────────
    for group, matches in sorted(groups.items()):
        standings, match_results = simulate_group(
            matches, match_params, fifa_rankings
        )
        all_standings[group] = standings

        # Print match results
        print(f"\n  ── Group {group} Matches ──")
        for r in match_results:
            print(f"    {r['home_team']:<22} {r['result']:^7} {r['away_team']}")

        # Print standings
        print(f"\n  Group {group} Standings:")
        print(f"  {'#':<3} {'Team':<22} {'P':>3} {'W':>3} {'D':>3} "
              f"{'L':>3} {'GF':>4} {'GA':>4} {'GD':>4} {'Pts':>4}")
        print(f"  {'-'*3} {'-'*22} {'-'*3} {'-'*3} {'-'*3} "
              f"{'-'*3} {'-'*4} {'-'*4} {'-'*4} {'-'*4}")
        for _, row in standings.iterrows():
            print(
                f"  {row['rank']:<3} {row['team']:<22} "
                f"{row['played']:>3} {row['wins']:>3} {row['draws']:>3} "
                f"{row['losses']:>3} {row['gf']:>4} {row['ga']:>4} "
                f"{row['gd']:>4} {row['points']:>4}"
            )

        # Collect 3rd place record
        third = standings[standings["rank"] == 3].iloc[0]
        third_place_records.append({
            "team"   : third["team"],
            "points" : third["points"],
            "gd"     : third["gd"],
            "gf"     : third["gf"],
            "group"  : group
        })

    # ── Best 8 third-place teams ──────────────────────────────────
    best_thirds = get_best_third_places(
        third_place_records, fifa_rankings, n=8
    )

    print(f"\n  ── Best 8 Third-Place Teams ──")
    print(f"  {'Team':<22} {'Group':<8} {'Pts':>4} {'GD':>4} {'GF':>4}")
    print(f"  {'-'*22} {'-'*8} {'-'*4} {'-'*4} {'-'*4}")
    for r in sorted(
        third_place_records,
        key=lambda x: (x["points"], x["gd"], x["gf"]),
        reverse=True
    ):
        qualified = "✅" if r["team"] in best_thirds else "❌"
        print(f"  {r['team']:<22} {r['group']:<8} "
              f"{r['points']:>4} {r['gd']:>4} {r['gf']:>4}  {qualified}")

    # ── Build group stage results and qualifying groups ───────────
    group_stage_results = build_group_stage_results(all_standings)
    qualifying_groups   = list(best_thirds.values())

    # ── Knockout phase ────────────────────────────────────────────
    bracket, champion = simulate_knockout_phase(
        group_stage_results = group_stage_results,
        qualifying_groups   = qualifying_groups,
        combinations        = combinations,
        elo_ratings         = elo_ratings,
        match_params        = match_params
    )

    # ── Print knockout results round by round ─────────────────────
    round_labels = [
        ("Round of 32",       range(73, 89)),
        ("Round of 16",       range(89, 97)),
        ("Quarterfinals",     range(97, 101)),
        ("Semifinals",        [101, 102]),
        ("Third Place Match", [104]),
        ("Final",             [103]),
    ]

    for round_name, match_nos in round_labels:
        print(f"\n  ── {round_name} ──")
        for match_no in match_nos:
            if match_no not in bracket:
                continue
            m = bracket[match_no]

            if m["penalties"]:
                # Show aggregate score, mark penalty winner with (p)
                total_a = m["goals_a_90"] + m["goals_a_et"]
                total_b = m["goals_b_90"] + m["goals_b_et"]
                if m["penalty_winner"] == m["team_a"]:
                    score = f"{total_a}(p)-{total_b}"
                else:
                    score = f"{total_a}-{total_b}(p)"

            elif m["goals_a_et"] is not None:
                # Show aggregate score, mark with (et)
                total_a = m["goals_a_90"] + m["goals_a_et"]
                total_b = m["goals_b_90"] + m["goals_b_et"]
                if total_a > total_b:
                    score = f"{total_a}(et)-{total_b}"
                else:
                    score = f"{total_a}-{total_b}(et)"

            else:
                score = f"{m['goals_a_90']}-{m['goals_b_90']}"

            source = f"[{m.get('params_source', 'elo')[0].upper()}]"
            print(f"    {m['team_a']:<22} {score:^20} {m['team_b']:<22} "
                  f"→ {m['winner']}  {source}")

    print(f"\n  {'='*65}")
    print(f"  🏆  CHAMPION: {champion}")
    print(f"  {'='*65}\n")

    return champion


# ─────────────────────────────────────────────
# STEP 4: Multi simulation
# ─────────────────────────────────────────────

def run_multi_simulation(match_params, groups, fifa_rankings,
                          elo_ratings, combinations, n_simulations):
    """
    Run N full tournament simulations and accumulate counts for
    every team at every stage.

    Tracked stages per team:
        Group stage  : 1st, 2nd, 3rd_q, 3rd_out, 4th
        Knockout     : r32, r16, qf, sf, 3rd_place, final, champion
    """
    # All teams
    all_teams = {
        team
        for matches in groups.values()
        for m in matches
        for team in (m["home_team"], m["away_team"])
    }

    # Stage counters
    stages = [
        "1st", "2nd", "3rd_q", "3rd_out", "4th",
        "r32", "r16", "qf", "sf", "3rd_place", "final", "champion"
    ]
    counts = {team: defaultdict(int) for team in all_teams}

    # Map match numbers to stage labels
    match_to_stage = {}
    for m in range(73, 89):  match_to_stage[m] = "r32"
    for m in range(89, 97):  match_to_stage[m] = "r16"
    for m in range(97, 101): match_to_stage[m] = "qf"
    for m in [101, 102]:     match_to_stage[m] = "sf"
    match_to_stage[103] = "final"

    print(f"\nRunning {n_simulations:,} simulations...")

    for sim in range(n_simulations):
        if (sim + 1) % 1 == 0:
            print(f"  Simulation {sim + 1:,} / {n_simulations:,}")

        all_standings       = {}
        third_place_records = []

        # ── Group stage ───────────────────────────────────────────
        for group, matches in groups.items():
            standings, _ = simulate_group(matches, match_params, fifa_rankings)
            all_standings[group] = standings

            for _, row in standings.iterrows():
                rank = row["rank"]
                if rank == 1:   counts[row["team"]]["1st"] += 1
                elif rank == 2: counts[row["team"]]["2nd"] += 1
                elif rank == 4: counts[row["team"]]["4th"] += 1

            third = standings[standings["rank"] == 3].iloc[0]
            third_place_records.append({
                "team"   : third["team"],
                "points" : third["points"],
                "gd"     : third["gd"],
                "gf"     : third["gf"],
                "group"  : group
            })

        # ── Best 8 third places ───────────────────────────────────
        best_thirds = get_best_third_places(
            third_place_records, fifa_rankings, n=8
        )
        for record in third_place_records:
            if record["team"] in best_thirds:
                counts[record["team"]]["3rd_q"]   += 1
            else:
                counts[record["team"]]["3rd_out"] += 1

        # ── Knockout phase ────────────────────────────────────────
        group_stage_results = build_group_stage_results(all_standings)
        qualifying_groups   = list(best_thirds.values())

        bracket, champion = simulate_knockout_phase(
            group_stage_results = group_stage_results,
            qualifying_groups   = qualifying_groups,
            combinations        = combinations,
            elo_ratings         = elo_ratings,
            match_params        = match_params
        )

        # Count knockout stage appearances
        for match_no, stage in match_to_stage.items():
            if match_no not in bracket:
                continue
            m = bracket[match_no]
            counts[m["team_a"]][stage] += 1
            counts[m["team_b"]][stage] += 1

        # Champion and tournament 3rd place
        counts[champion]["champion"] += 1
        if 104 in bracket:
            counts[bracket[104]["winner"]]["3rd_place"] += 1

    return counts


# ─────────────────────────────────────────────
# STEP 5: Build multi-sim summary
# ─────────────────────────────────────────────

def build_multi_summary(counts, groups, n_simulations):
    """
    Build a summary DataFrame with one row per team showing
    probability of reaching each stage of the tournament.
    """
    team_to_group = {
        team: group
        for group, matches in groups.items()
        for m in matches
        for team in (m["home_team"], m["away_team"])
    }

    rows = []
    for team, c in counts.items():
        n = n_simulations
        rows.append({
            "Group"      : team_to_group.get(team, "?"),
            "Team"       : team,
            "1st %"      : round(c["1st"]       / n * 100, 1),
            "2nd %"      : round(c["2nd"]       / n * 100, 1),
            "3rd(Q) %"   : round(c["3rd_q"]     / n * 100, 1),
            "3rd(out) %" : round(c["3rd_out"]   / n * 100, 1),
            "4th %"      : round(c["4th"]       / n * 100, 1),
            "Qualify %"  : round((c["1st"] + c["2nd"] + c["3rd_q"]) / n * 100, 1),
            "R32 %"      : round(c["r32"]       / n * 100, 1),
            "R16 %"      : round(c["r16"]       / n * 100, 1),
            "QF %"       : round(c["qf"]        / n * 100, 1),
            "SF %"       : round(c["sf"]        / n * 100, 1),
            "3rd Place %" : round(c["3rd_place"] / n * 100, 1),
            "Final %"    : round(c["final"]     / n * 100, 1),
            "Champion %" : round(c["champion"]  / n * 100, 1),
        })

    summary = (
        pd.DataFrame(rows)
        .sort_values(["Group", "Qualify %"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return summary


# ─────────────────────────────────────────────
# STEP 6: Print multi-sim summary
# ─────────────────────────────────────────────

def print_multi_summary(summary):
    print("\n" + "=" * 125)
    print("  TOURNAMENT SIMULATION RESULTS")
    print("=" * 125)

    # ── Group stage section ───────────────────────────────────────
    print(f"\n  {'─'*125}")
    print(f"  GROUP STAGE")
    print(f"  {'─'*125}")
    print(
        f"  {'Team':<22} {'Group':<7} {'1st':>6} {'2nd':>6} "
        f"{'3rd(Q)':>8} {'3rd(out)':>10} {'4th':>6} {'Qualify':>8}"
    )
    print(
        f"  {'-'*22} {'-'*7} {'-'*6} {'-'*6} "
        f"{'-'*8} {'-'*10} {'-'*6} {'-'*8}"
    )
    for _, row in summary.sort_values(
        ["Group", "Qualify %"], ascending=[True, False]
    ).iterrows():
        print(
            f"  {row['Team']:<22} {row['Group']:<7} "
            f"{row['1st %']:>5.1f}% "
            f"{row['2nd %']:>5.1f}% "
            f"{row['3rd(Q) %']:>7.1f}% "
            f"{row['3rd(out) %']:>9.1f}% "
            f"{row['4th %']:>5.1f}% "
            f"{row['Qualify %']:>7.1f}%"
        )

    # ── Knockout stage section ────────────────────────────────────
    print(f"\n  {'─'*125}")
    print(f"  KNOCKOUT STAGE")
    print(f"  {'─'*125}")

    ko_summary = (
        summary[summary["Qualify %"] > 0]
        .sort_values("Champion %", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"  {'Team':<22} {'Group':<7} {'R32':>6} {'R16':>6} "
        f"{'QF':>6} {'SF':>6} {'3rd':>6} {'Final':>7} {'Champion':>9}"
    )
    print(
        f"  {'-'*22} {'-'*7} {'-'*6} {'-'*6} "
        f"{'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*9}"
    )
    for _, row in ko_summary.iterrows():
        print(
            f"  {row['Team']:<22} {row['Group']:<7} "
            f"{row['R32 %']:>5.1f}% "
            f"{row['R16 %']:>5.1f}% "
            f"{row['QF %']:>5.1f}% "
            f"{row['SF %']:>5.1f}% "
            f"{row['3rd Place %']:>5.1f}% "
            f"{row['Final %']:>6.1f}% "
            f"{row['Champion %']:>8.1f}%"
        )

    print("\n" + "=" * 125)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    # ── Load static data ──────────────────────────────────────────
    fifa_rankings = load_fifa_rankings(FIFA_RANKINGS_CSV)
    print(f"  Loaded FIFA rankings for {len(fifa_rankings)} teams")

    elo_ratings = load_elo_ratings(ELO_RATINGS_CSV)
    print(f"  Loaded Elo ratings for {len(elo_ratings)} teams")

    combinations = load_third_place_combinations(COMBINATIONS_CSV)
    print(f"  Loaded {len(combinations)} third-place bracket combinations")

    # ── Fit Dixon-Coles parameters once ───────────────────────────
    match_params = prepare_match_params(CSV_PATH)

    # ── Derive group structure ────────────────────────────────────
    groups = load_group_structure(match_params)

    # ── Warn about unranked teams ─────────────────────────────────
    all_teams = {
        team
        for matches in groups.values()
        for m in matches
        for team in (m["home_team"], m["away_team"])
    }
    unranked_fifa = all_teams - set(fifa_rankings.keys())
    unranked_elo  = all_teams - set(elo_ratings.keys())

    if unranked_fifa:
        print(f"\n  ⚠️  Teams missing from FIFA rankings (default rank 9999):")
        for t in sorted(unranked_fifa):
            print(f"       - {t}")

    if unranked_elo:
        print(f"\n  ⚠️  Teams missing from Elo ratings (knockout sims may be affected):")
        for t in sorted(unranked_elo):
            print(f"       - {t}")

    # ── Run simulation ────────────────────────────────────────────
    if MODE == "single":
        run_single_simulation(
            match_params  = match_params,
            groups        = groups,
            fifa_rankings = fifa_rankings,
            elo_ratings   = elo_ratings,
            combinations  = combinations,
        )

    else:
        counts = run_multi_simulation(
            match_params   = match_params,
            groups         = groups,
            fifa_rankings  = fifa_rankings,
            elo_ratings    = elo_ratings,
            combinations   = combinations,
            n_simulations  = N_SIMULATIONS,
        )

        summary = build_multi_summary(counts, groups, N_SIMULATIONS)
        print_multi_summary(summary)

        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"simulation_results_{timestamp}.csv")

        summary.to_csv(output_path, index=False)
        print(f"\n  Results saved to {output_path}\n")