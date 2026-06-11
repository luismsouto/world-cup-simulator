import numpy as np
from simulate.simulate_group_stage import simulate_match
from elo_utils.elo_utils import elo_match_params
from simulate.third_place_combinations import resolve_third_place_slots

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

ELO_RATINGS_CSV  = "elo_rankings.csv"
COMBINATIONS_CSV = "third_place_match_combinations.csv"

# Penalty shootout win probability for team A (50/50)
PENALTY_WIN_PROB = 0.5

# Extra time lambda scaling factor:
# 30 extra time minutes ≈ 30/90 = 1/3 of a full match
EXTRA_TIME_SCALE = 30.0 / 90.0

# ─────────────────────────────────────────────
# FIXED BRACKET PROGRESSION
# ─────────────────────────────────────────────

# Maps each match number to its two participant slots.
# Round of 32 participants involving third-place teams are
# placeholders ("3_1A" etc.) resolved at runtime.
# R16 onward use "W{match_no}" / "L{match_no}" resolved
# after each round is simulated.
#
# Third-place slot notation:
#   "3_1A" = third-place team assigned to face the winner of group A
#
# Fixed slot notation:
#   "1A"  = winner of group A
#   "2A"  = runner-up of group A

BRACKET_TEMPLATE = {
    # ── Round of 32 ──────────────────────────────────────────────
    73:  ("2A",  "2B"),
    74:  ("1E",  "3_1E"),
    75:  ("1F",  "2C"),
    76:  ("1C",  "2F"),
    77:  ("1I",  "3_1I"),
    78:  ("2E",  "2I"),
    79:  ("1A",  "3_1A"),
    80:  ("1L",  "3_1L"),
    81:  ("1D",  "3_1D"),
    82:  ("1G",  "3_1G"),
    83:  ("2K",  "2L"),
    84:  ("1H",  "2J"),
    85:  ("1B",  "3_1B"),
    86:  ("1J",  "2H"),
    87:  ("1K",  "3_1K"),
    88:  ("2D",  "2G"),

    # ── Round of 16 ──────────────────────────────────────────────
    89:  ("W74", "W77"),
    90:  ("W73", "W75"),
    91:  ("W76", "W78"),
    92:  ("W79", "W80"),
    93:  ("W83", "W84"),
    94:  ("W81", "W82"),
    95:  ("W86", "W88"),
    96:  ("W85", "W87"),

    # ── Quarterfinals ────────────────────────────────────────────
    97:  ("W89", "W90"),
    98:  ("W93", "W94"),
    99:  ("W91", "W92"),
    100: ("W95", "W96"),

    # ── Semifinals ───────────────────────────────────────────────
    101: ("W97", "W98"),
    102: ("W99", "W100"),

    # ── Final ────────────────────────────────────────────────────
    103: ("W101", "W102"),

    # ── Third place match ─────────────────────────────────────────
    104: ("L101", "L102"),
}

# Match number ranges per round
ROUNDS = {
    "Round of 32"   : list(range(73, 89)),
    "Round of 16"   : list(range(89, 97)),
    "Quarterfinals" : list(range(97, 101)),
    "Semifinals"    : [101, 102],
    "Final"         : [103],
    "Third Place"   : [104],
}


# ─────────────────────────────────────────────
# STEP 1: Build Round of 32
# ─────────────────────────────────────────────

def build_round_of_32(group_stage_results, qualifying_groups, combinations):
    """
    Resolve all 16 Round of 32 matchups using group stage results
    and the third-place combination lookup table.

    Parameters
    ----------
    group_stage_results : dict
        Maps slot strings to actual team names, e.g.:
        {
            "1A": "Mexico",
            "2A": "South Korea",
            "3A": "Czech Republic",
            "1B": "Switzerland",
            ...
        }
    qualifying_groups : iterable of str
        The 8 group letters whose third-place teams qualified,
        e.g. ["E", "F", "G", "H", "I", "J", "K", "L"]
    combinations : dict
        Loaded from load_third_place_combinations()

    Returns
    -------
    dict mapping match_number to:
        {
            "team_a" : str,
            "team_b" : str,
            "winner" : None,
            "loser"  : None,
        }
    """
    # Resolve third-place slot assignments
    # e.g. {"1A": "E", "1B": "J", ...}
    slots = resolve_third_place_slots(qualifying_groups, combinations)

    bracket = {}
    for match_no in ROUNDS["Round of 32"]:
        p1, p2 = BRACKET_TEMPLATE[match_no]

        team_a = _resolve_group_stage_participant(p1, slots, group_stage_results)
        team_b = _resolve_group_stage_participant(p2, slots, group_stage_results)

        bracket[match_no] = {
            "team_a" : team_a,
            "team_b" : team_b,
            "winner" : None,
            "loser"  : None,
        }

    return bracket


def _resolve_group_stage_participant(participant, slots, group_stage_results):
    """
    Resolve a Round of 32 participant string to an actual team name.

    Handles:
        "1A"    → group stage winner of group A
        "2B"    → group stage runner-up of group B
        "3_1A"  → third-place team assigned to face 1A,
                   whose group letter is looked up from slots
    """
    if participant.startswith("3_"):
        winner_slot  = participant[2:]          # e.g. "1A"
        group_letter = slots[winner_slot]       # e.g. "E"
        return group_stage_results[f"3{group_letter}"]
    else:
        return group_stage_results[participant]


# ─────────────────────────────────────────────
# STEP 2: Source Dixon-Coles parameters
# ─────────────────────────────────────────────

def _get_match_params(team_a, team_b, elo_ratings, match_params):
    """
    Return (lam_home, lam_away, rho) for a knockout match.

    Priority:
        1. Pre-fitted bookmaker params (exact order)    — team_a was home
        2. Pre-fitted bookmaker params (reversed order) — team_b was home,
           so lam_home and lam_away are swapped
        3. Elo-derived params as fallback

    Parameters
    ----------
    team_a       : str
    team_b       : str
    elo_ratings  : dict
    match_params : dict or None
        { match_id: { "home_team": ..., "away_team": ...,
                      "lam_home": ..., "lam_away": ..., "rho": ... } }

    Returns
    -------
    lam_home : float
    lam_away : float
    rho      : float
    source   : str — "bookmaker" or "elo" for logging purposes
    """
    if match_params:
        for p in match_params.values():
            if p["home_team"] == team_a and p["away_team"] == team_b:
                # Exact match — team_a was home in the group stage
                return p["lam_home"], p["lam_away"], p["rho"], "bookmaker"

            if p["home_team"] == team_b and p["away_team"] == team_a:
                # Reversed — team_b was home, so swap lambdas
                return p["lam_away"], p["lam_home"], p["rho"], "bookmaker"

    # Fallback: derive from Elo ratings
    params = elo_match_params(team_a, team_b, elo_ratings)
    return params["lam_home"], params["lam_away"], params["rho"], "elo"


# ─────────────────────────────────────────────
# STEP 3: Simulate a single knockout match
# ─────────────────────────────────────────────

def simulate_knockout_match(team_a, team_b, elo_ratings, match_params=None):
    """
    Simulate a single knockout match between two teams.

    Uses bookmaker-derived Dixon-Coles parameters if the two teams
    met in the group stage, otherwise falls back to Elo ratings.

    Simulation flow:
        1. Simulate 90 minutes
        2. If drawn, simulate 30 minutes of extra time
           (lambdas scaled by 30/90)
        3. If still drawn, decide via penalty shootout (50/50)

    Parameters
    ----------
    team_a       : str  — name of team A
    team_b       : str  — name of team B
    elo_ratings  : dict — { team_name: elo_points }
    match_params : dict or None
        Pre-fitted match parameters from prepare_match_params().
        If None, Elo ratings are always used.

    Returns
    -------
    winner : str
    loser  : str
    result : dict with keys:
        team_a, team_b,
        goals_a_90, goals_b_90,
        goals_a_et, goals_b_et,
        penalties, penalty_winner,
        winner, loser,
        params_source
    """
    # ── Source Dixon-Coles parameters ─────────────────────────────
    lam_home, lam_away, rho, source = _get_match_params(
        team_a, team_b, elo_ratings, match_params
    )

    # ── Simulate 90 minutes ───────────────────────────────────────
    goals_a_90, goals_b_90 = simulate_match(lam_home, lam_away, rho)

    goals_a_et     = None
    goals_b_et     = None
    penalties      = False
    penalty_winner = None

    if goals_a_90 != goals_b_90:
        # Decided in 90 minutes
        winner = team_a if goals_a_90 > goals_b_90 else team_b
        loser  = team_b if goals_a_90 > goals_b_90 else team_a

    else:
        # ── Extra time (30 minutes, scaled lambdas) ───────────────
        lam_home_et = lam_home * EXTRA_TIME_SCALE
        lam_away_et = lam_away * EXTRA_TIME_SCALE

        goals_a_et, goals_b_et = simulate_match(lam_home_et, lam_away_et, rho)

        total_a = goals_a_90 + goals_a_et
        total_b = goals_b_90 + goals_b_et

        if total_a != total_b:
            # Decided in extra time
            winner = team_a if total_a > total_b else team_b
            loser  = team_b if total_a > total_b else team_a

        else:
            # ── Penalty shootout ──────────────────────────────────
            penalties = True
            if np.random.random() < PENALTY_WIN_PROB:
                winner = team_a
                loser  = team_b
            else:
                winner = team_b
                loser  = team_a
            penalty_winner = winner

    result = {
        "team_a"         : team_a,
        "team_b"         : team_b,
        "goals_a_90"     : goals_a_90,
        "goals_b_90"     : goals_b_90,
        "goals_a_et"     : goals_a_et,
        "goals_b_et"     : goals_b_et,
        "penalties"      : penalties,
        "penalty_winner" : penalty_winner,
        "winner"         : winner,
        "loser"          : loser,
        "params_source"  : source,
    }

    return winner, loser, result


# ─────────────────────────────────────────────
# STEP 4: Resolve bracket participants
# ─────────────────────────────────────────────

def _resolve_bracket_participant(participant, bracket):
    """
    Resolve a participant string using already-simulated match results.

    Handles:
        "W74"  → winner of match 74
        "L101" → loser of match 101
    """
    if participant.startswith("W"):
        match_no = int(participant[1:])
        return bracket[match_no]["winner"]
    elif participant.startswith("L"):
        match_no = int(participant[1:])
        return bracket[match_no]["loser"]
    else:
        raise ValueError(f"Unexpected participant format: '{participant}'")


# ─────────────────────────────────────────────
# STEP 5: Simulate the full knockout phase
# ─────────────────────────────────────────────

def simulate_knockout_phase(group_stage_results, qualifying_groups,
                            combinations, elo_ratings, match_params=None):
    """
    Simulate the full knockout phase from Round of 32 to Final.

    Parameters
    ----------
    group_stage_results : dict
        Maps slot strings to team names:
        {"1A": "Mexico", "2A": "South Korea", "3E": "Scotland", ...}
    qualifying_groups : iterable of str
        The 8 group letters whose third-place teams qualified.
    combinations : dict
        Loaded from load_third_place_combinations().
    elo_ratings : dict
        Loaded from load_elo_ratings().
    match_params : dict or None
        Pre-fitted group stage match parameters from prepare_match_params().
        When provided, bookmaker-derived lambdas are used for any knockout
        rematch of a group stage fixture. Defaults to None.

    Returns
    -------
    bracket  : dict mapping match_number to completed match dict
    champion : str — tournament winner
    """
    # ── Build Round of 32 ────────────────────────────────────────
    bracket = build_round_of_32(
        group_stage_results, qualifying_groups, combinations
    )

    # ── Simulate round by round ───────────────────────────────────
    round_order = (
        ROUNDS["Round of 32"]   +
        ROUNDS["Round of 16"]   +
        ROUNDS["Quarterfinals"] +
        ROUNDS["Semifinals"]    +
        ROUNDS["Final"]         +
        ROUNDS["Third Place"]
    )

    for match_no in round_order:
        if match_no not in bracket:
            # Initialise R16 onward matches from template
            p1, p2 = BRACKET_TEMPLATE[match_no]
            team_a = _resolve_bracket_participant(p1, bracket)
            team_b = _resolve_bracket_participant(p2, bracket)

            bracket[match_no] = {
                "team_a" : team_a,
                "team_b" : team_b,
                "winner" : None,
                "loser"  : None,
            }

        team_a = bracket[match_no]["team_a"]
        team_b = bracket[match_no]["team_b"]

        winner, loser, result = simulate_knockout_match(
            team_a, team_b, elo_ratings, match_params
        )

        bracket[match_no].update(result)

    champion = bracket[103]["winner"]
    return bracket, champion
