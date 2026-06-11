# 🌍 FIFA World Cup Simulator (WCS)

A full Monte Carlo simulation engine for the 2026 FIFA World Cup, built in Python. The games are simulated using the 
[Dixon-Coles](https://grokipedia.com/page/DixonColes_model) model, and to estimate the parameters we fit the model
against win/draw/loss probabilities that are either derived from **(a)** bookmaker odds (for group stage matches) 
or **(b)** Elo ratings (for the knockout phase matches). It supports both single and multi run modes.

## 1. Methodology

### Goal Scoring Model: Dixon-Coles

Each match is modelled as a pair of independent Poisson random variables representing
goals scored by the home and away teams. The expected goals $\lambda$ for each team
are estimated using a Dixon-Coles framework. The probability of a scoreline $(i, j)$ is given by:

$$P(X = i,\ Y = j) = \tau(\lambda_h,\ \lambda_a,\ i,\ j,\ \rho) \cdot \frac{e^{-\lambda_h} \lambda_h^i}{i!} \cdot \frac{e^{-\lambda_a} \lambda_a^j}{j!}$$

where $\lambda_h$ and $\lambda_a$ are the expected goals for the home and away team
respectively, and $\tau$ is the Dixon-Coles low-score correction factor:

$$
\tau(\lambda_h, \lambda_a, i, j, \rho) =
\begin{cases}
1 - \lambda_h \lambda_a \rho & \text{if } i=0,\ j=0 \\
1 + \lambda_h \rho           & \text{if } i=0,\ j=1 \\
1 + \lambda_a \rho           & \text{if } i=1,\ j=0 \\
1 - \rho                     & \text{if } i=1,\ j=1 \\
1                            & \text{otherwise}
\end{cases}
$$

The correction parameter $\rho$ adjusts for the empirically observed over-frequency of
0-0, 1-0, 0-1, and 1-1 scorelines relative to the independent Poisson prediction.

### Deriving Win/Draw/Loss probabilities

For each match we require three outcome probabilities $(p_W,\ p_D,\ p_L)$ — the probability
of a home win, draw, and away win respectively. These are sourced differently depending
on the phase of the tournament.

#### a) Bookmaker Odds

Raw bookmaker odds $o_W,\ o_D,\ o_L$ imply probabilities that sum to more than 1 due to
the bookmaker's margin (overround). We remove this by normalising:

$$p_i = \frac{1/o_i}{\sum_{j \in \{W, D, L\}} 1/o_j}$$

#### b) Elo Ratings

Elo ratings only yield a *head-to-head win probability* for each team. The probability
that team $A$ beats team $B$ is:

$$P(A\ \text{beats}\ B) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

Since Elo provides no native draw probability, we allocate a fixed draw share $p_D$ and
distribute the remainder proportionally:

$$p_W = (1 - p_D) \cdot P(A\ \text{beats}\ B), \qquad p_L = (1 - p_D) \cdot P(B\ \text{beats}\ A)$$


### Fitting Parameters Against Bookmaker Odds

For each match, bookmaker odds are converted to implied probabilities $(p_H,\ p_D,\ p_A)$
for home win, draw, and away win respectively, after normalising to remove the overround:

$$p_i = \frac{1/o_i}{\sum_j 1/o_j}$$

where $o_i$ is the bookmaker odd for outcome $i$.

The Dixon-Coles model produces a full scoreline distribution $P(X=i,\ Y=j)$, from which
match outcome probabilities can be recovered by marginalisation:

$$\hat{p}H = \sum{i > j} P(X=i,\ Y=j), \qquad \hat{p}D = \sum{i=j} P(X=i,\ Y=j), \qquad \hat{p}A = \sum{i < j} P(X=i,\ Y=j)$$

The model parameters $\boldsymbol{\theta} = (\alpha_i,\ \beta_i,\ \gamma,\ \rho)$ are then estimated
by minimising the cross-entropy loss between the bookmaker-implied and model-implied outcome
probabilities across all group stage matches:

$$\mathcal{L}(\boldsymbol{\theta}) = -\sum_{\text{matches}} \left[ p_H \log \hat{p}_H + p_D \log \hat{p}_D + p_A \log \hat{p}_A \right]$$

### 2. Elo Rating Model — Knockout Phase

Once the group stage is resolved, the knockout phase uses **Elo ratings** to determine
win probabilities in each bilateral match (no draw possible after 90 minutes in knockout).

The expected win probability for team $A$ against team $B$ is:

$$P(A\ \text{beats}\ B) = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

where $R_A$ and $R_B$ are the Elo ratings of teams $A$ and $B$ respectively.

Knockout matches can be resolved through:
- **Full time (90 min)**
- **Extra time (ET)**
- **Penalty shootout (PSO)**

Each is simulated sequentially with appropriately scaled goal expectations.

---

### 3. Elo Calibration

Raw Elo ratings are calibrated to align with bookmaker-implied win probabilities for
the group stage matches. The calibration scales the Elo difference to minimise the
gap between $P_{\text{Elo}}$ and $P_{\text{bookmaker}}$, improving the accuracy of
knockout phase predictions.

The calibrated ratings are stored in `elo_rankings_calibrated.csv`. Diagnostic output
comparing raw Elo vs. bookmaker probabilities is written to `elo_vs_bookmaker.csv`.

---

### 4. Group Stage Tiebreakers

When teams finish level on points, the following criteria are applied **in order**:

1. Points
2. Goal difference
3. Goals scored
4. FIFA ranking (lower rank number = better)

This mirrors official FIFA World Cup regulations and uses `fifa_rankings.csv` as the
final tiebreaker input.

---

### 5. Best Third-Place Selection

In a 48-team World Cup with 12 groups of 4, the top two teams from each group
qualify automatically. The **best 8 of the 12 third-place teams** also advance to
the Round of 32.

Third-place teams are ranked by the same criteria as the group stage tiebreaker:

$$\text{Sort key} = (\text{Points},\ \text{GD},\ \text{GF},\ -\text{FIFA rank})$$

The bracket slot assigned to each qualifying third-place team depends on which groups
they came from. Valid assignments are pre-computed and stored in
`third_place_match_combinations.csv`.

---

## ⚙️ Configuration

All key parameters are defined at the top of `main.py`:

| Parameter          | Default                              | Description                               |
|--------------------|--------------------------------------|-------------------------------------------|
| `CSV_PATH`         | `input_data/odds_input.csv`          | Bookmaker odds for group stage matches    |
| `FIFA_RANKINGS_CSV`| `input_data/fifa_rankings.csv`       | FIFA rankings for tiebreaking             |
| `ELO_RATINGS_CSV`  | `input_data/elo_rankings_calibrated.csv` | Calibrated Elo ratings               |
| `COMBINATIONS_CSV` | `input_data/third_place_match_combinations.csv` | Third-place bracket rules      |
| `N_SIMULATIONS`    | `1`                                  | Number of Monte Carlo iterations          |
| `RANDOM_SEED`      | `30`                                 | NumPy random seed for reproducibility    |
| `MODE`             | `"multi"`                            | `"single"` for verbose or `"multi"` for probabilistic |
| `OUTPUT_DIR`       | `output_data`                        | Directory for CSV output                  |


## 1. Project Structure

**Main folder**
- `main.py` Entry point - orchestrates the full pipeline

**elo_utils**
- `calibrate_elo.py` Calibrates raw Elo ratings against bookmaker odds
- 

│ ├── elo_utils.py # Loads and exposes Elo ratings
│ └── elo_vs_bookmaker.py # Diagnostic comparison of Elo vs. bookmaker probabilities

├── models/
│ ├── init.py
│ ├── predict_poisson.py # Basic independent Poisson goal model
│ └── predict_poisson_DC.py # Dixon-Coles corrected Poisson model
│
├── simulate/
│ ├── init.py
│ ├── simulate_group_stage.py # Group stage simulation logic
│ ├── simulate_knockout_phase.py # Knockout bracket simulation logic
│ └── third_place_combinations.py # Handles best third-place team bracket rules
│
├── input_data/
│ ├── odds_input.csv # Bookmaker odds for all group stage matches
│ ├── fifa_rankings.csv # Official FIFA rankings (used as tiebreaker)
│ ├── elo_rankings.csv # Raw Elo ratings for all teams
│ ├── elo_rankings_calibrated.csv # Elo ratings after bookmaker calibration
│ └── third_place_match_combinations.csv # Valid R32 bracket assignments for best 3rds
│
└── output_data/
├── elo_vs_bookmaker.csv # Diagnostic output from calibration
├── simulation_results.csv # Latest simulation results
└── simulation_results_.csv # Timestamped simulation snapshots

---

## Future improvements
- Allow for the use of Elo ratings as the basis to simulate the entire tournament (not just knockout phase)