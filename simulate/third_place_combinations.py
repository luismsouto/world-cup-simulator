import pandas as pd

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

COMBINATIONS_CSV = "third_place_match_combinations.csv"

# ─────────────────────────────────────────────
# STEP 1: Load the combinations table
# ─────────────────────────────────────────────

def load_third_place_combinations(csv_path=COMBINATIONS_CSV):
    """
    Load the 495 third-place bracket combinations from CSV.

    Expected CSV columns:
        combo_id  : int, row number (1-495)
        groups    : str, hyphen-separated qualifying groups e.g. "E-F-G-H-I-J-K-L"
        slot_1A   : str, third-place team that faces 1A e.g. "3E"
        slot_1B   : str, third-place team that faces 1B
        slot_1D   : str, third-place team that faces 1D
        slot_1E   : str, third-place team that faces 1E
        slot_1G   : str, third-place team that faces 1G
        slot_1I   : str, third-place team that faces 1I
        slot_1K   : str, third-place team that faces 1K
        slot_1L   : str, third-place team that faces 1L

    Returns a dict keyed by frozenset of 8 group letters:
        {
            frozenset({"E","F","G","H","I","J","K","L"}): {
                "1A": "3E",
                "1B": "3J",
                "1D": "3I",
                "1E": "3F",
                "1G": "3H",
                "1I": "3G",
                "1K": "3L",
                "1L": "3K"
            },
            ...
        }
    """
    df = pd.read_csv(csv_path)

    required = {
        "combo_id", "groups",
        "slot_1A", "slot_1B", "slot_1D", "slot_1E",
        "slot_1G", "slot_1I", "slot_1K", "slot_1L"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Combinations CSV is missing columns: {missing}")

    if len(df) != 495:
        raise ValueError(f"Expected 495 combinations, found {len(df)}")

    combinations = {}
    for _, row in df.iterrows():
        # Parse the hyphen-separated group letters into a frozenset key
        groups = frozenset(row["groups"].split("-"))

        if len(groups) != 8:
            raise ValueError(
                f"Combo {row['combo_id']}: expected 8 groups, "
                f"got {len(groups)} from '{row['groups']}'"
            )

        combinations[groups] = {
            "1A": row["slot_1A"],
            "1B": row["slot_1B"],
            "1D": row["slot_1D"],
            "1E": row["slot_1E"],
            "1G": row["slot_1G"],
            "1I": row["slot_1I"],
            "1K": row["slot_1K"],
            "1L": row["slot_1L"],
        }

    return combinations


# ─────────────────────────────────────────────
# STEP 2: Resolve third-place assignments
# ─────────────────────────────────────────────

def resolve_third_place_slots(qualifying_groups, combinations):
    """
    Given the 8 groups whose third-place teams qualified,
    look up the bracket assignments from the combinations table.

    Parameters
    ----------
    qualifying_groups : iterable of str
        The 8 group letters whose third-place teams qualified,
        e.g. ["E", "F", "G", "H", "I", "J", "K", "L"]
    combinations : dict
        The loaded combinations table from load_third_place_combinations()

    Returns
    -------
    dict mapping group winner slot to third-place group letter:
        {
            "1A": "E",   # 1A plays the third-place team from group E
            "1B": "J",
            "1D": "I",
            ...
        }

    Raises
    ------
    KeyError if the combination is not found in the table.
    """
    key = frozenset(qualifying_groups)

    if key not in combinations:
        raise KeyError(
            f"No bracket combination found for groups: "
            f"{sorted(qualifying_groups)}. "
            f"Check that exactly 8 valid group letters were provided."
        )

    raw = combinations[key]

    # Strip the leading "3" so callers get just the group letter
    return {
        winner_slot: third_place_code.lstrip("3")
        for winner_slot, third_place_code in raw.items()
    }


# ─────────────────────────────────────────────
# STEP 3: Diagnostic printer
# ─────────────────────────────────────────────

def print_bracket_assignments(qualifying_groups, combinations):
    """
    Print the Round of 32 third-place assignments for a given
    set of qualifying groups in a readable format.
    """
    slots = resolve_third_place_slots(qualifying_groups, combinations)

    print(f"\n  Third-place qualifying groups : "
          f"{', '.join(sorted(qualifying_groups))}")
    print(f"\n  Round of 32 third-place assignments:")
    print(f"  {'Winner':<8} vs {'3rd Place From'}")
    print(f"  {'-'*8}    {'-'*14}")
    for winner_slot, group_letter in sorted(slots.items()):
        print(f"  {winner_slot:<8} vs 3{group_letter}")
