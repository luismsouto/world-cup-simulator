import pandas as pd
from models.predict_poisson_DC import load_matches, odds_to_fair_probs
from elo_utils.elo_utils import load_elo_ratings, elo_to_probs

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CSV_PATH        = "odds_input.csv"
ELO_RATINGS_CSV = "elo_rankings.csv"

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    df           = load_matches(CSV_PATH)
    elo_ratings  = load_elo_ratings(ELO_RATINGS_CSV)

    rows = []

    for _, m in df.iterrows():
        home = m["home_team"]
        away = m["away_team"]

        # ── Bookmaker implied probabilities ───────────────────────
        fair = odds_to_fair_probs(m["home_odds"], m["draw_odds"], m["away_odds"])
        p_home_bk = fair["home"]
        p_draw_bk = fair["draw"]
        p_away_bk = fair["away"]

        # ── Elo derived probabilities ─────────────────────────────
        elo_a = elo_ratings.get(home)
        elo_b = elo_ratings.get(away)

        if elo_a is None or elo_b is None:
            p_home_elo = p_draw_elo = p_away_elo = None
            dr = None
        else:
            p_home_elo, p_draw_elo, p_away_elo = elo_to_probs(elo_a, elo_b)
            dr = elo_a - elo_b

        rows.append({
            "Group"        : m.get("group", ""),
            "Home"         : home,
            "Away"         : away,
            "Elo Home"     : elo_a,
            "Elo Away"     : elo_b,
            "dr"           : dr,
            # Bookmaker
            "BK Home %"    : round(p_home_bk * 100, 1),
            "BK Draw %"    : round(p_draw_bk * 100, 1),
            "BK Away %"    : round(p_away_bk * 100, 1),
            # Elo
            "Elo Home %"   : round(p_home_elo * 100, 1) if p_home_elo is not None else None,
            "Elo Draw %"   : round(p_draw_elo * 100, 1) if p_draw_elo is not None else None,
            "Elo Away %"   : round(p_away_elo * 100, 1) if p_away_elo is not None else None,
            # Differences (Elo minus Bookmaker)
            "Δ Home %"     : round((p_home_elo - p_home_bk) * 100, 1) if p_home_elo is not None else None,
            "Δ Draw %"     : round((p_draw_elo - p_draw_bk) * 100, 1) if p_draw_elo is not None else None,
            "Δ Away %"     : round((p_away_elo - p_away_bk) * 100, 1) if p_away_elo is not None else None,
        })

    results = pd.DataFrame(rows)

    # ── Print ─────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  BOOKMAKER vs ELO PROBABILITY COMPARISON — GROUP STAGE MATCHES")
    print("=" * 110)
    print(
        f"  {'Grp':<5} {'Home':<22} {'Away':<22} "
        f"{'BK H':>6} {'BK D':>6} {'BK A':>6}  "
        f"{'Elo H':>6} {'Elo D':>6} {'Elo A':>6}  "
        f"{'Δ H':>6} {'Δ D':>6} {'Δ A':>6}"
    )
    print(f"  {'-'*5} {'-'*22} {'-'*22} "
          f"{'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}")

    for _, row in results.iterrows():
        elo_h = f"{row['Elo Home %']:>5.1f}%" if row["Elo Home %"] is not None else "   N/A"
        elo_d = f"{row['Elo Draw %']:>5.1f}%" if row["Elo Draw %"] is not None else "   N/A"
        elo_a = f"{row['Elo Away %']:>5.1f}%" if row["Elo Away %"] is not None else "   N/A"
        d_h   = f"{row['Δ Home %']:>+5.1f}%" if row["Δ Home %"]   is not None else "   N/A"
        d_d   = f"{row['Δ Draw %']:>+5.1f}%" if row["Δ Draw %"]   is not None else "   N/A"
        d_a   = f"{row['Δ Away %']:>+5.1f}%" if row["Δ Away %"]   is not None else "   N/A"

        print(
            f"  {str(row['Group']):<5} {row['Home']:<22} {row['Away']:<22} "
            f"{row['BK Home %']:>5.1f}% {row['BK Draw %']:>5.1f}% {row['BK Away %']:>5.1f}%  "
            f"{elo_h} {elo_d} {elo_a}  "
            f"{d_h} {d_d} {d_a}"
        )

    print("=" * 110)

    # ── Summary statistics ────────────────────────────────────────
    valid = results.dropna(subset=["Δ Home %"])
    print(f"\n  SUMMARY OF DIFFERENCES (Elo minus Bookmaker)")
    print(f"  {'':30} {'Δ Home':>8} {'Δ Draw':>8} {'Δ Away':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Mean':30} {valid['Δ Home %'].mean():>+7.1f}% {valid['Δ Draw %'].mean():>+7.1f}% {valid['Δ Away %'].mean():>+7.1f}%")
    print(f"  {'Std Dev':30} {valid['Δ Home %'].std():>7.1f}%  {valid['Δ Draw %'].std():>7.1f}%  {valid['Δ Away %'].std():>7.1f}% ")
    print(f"  {'Mean Absolute Difference':30} {valid['Δ Home %'].abs().mean():>7.1f}%  {valid['Δ Draw %'].abs().mean():>7.1f}%  {valid['Δ Away %'].abs().mean():>7.1f}% ")
    print(f"  {'Max Overestimate (Elo>BK)':30} {valid['Δ Home %'].max():>+7.1f}%  {valid['Δ Draw %'].max():>+7.1f}%  {valid['Δ Away %'].max():>+7.1f}% ")
    print(f"  {'Max Underestimate (Elo<BK)':30} {valid['Δ Home %'].min():>+7.1f}%  {valid['Δ Draw %'].min():>+7.1f}%  {valid['Δ Away %'].min():>+7.1f}% ")

    # ── Save ──────────────────────────────────────────────────────
    results.to_csv("elo_vs_bookmaker.csv", index=False)
    print(f"\n  Results saved to elo_vs_bookmaker.csv\n")
