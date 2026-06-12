# 🌍 FIFA World Cup Simulator (WCS)

A full Monte Carlo simulation engine for the 2026 FIFA World Cup, built in Python. The games are simulated using the 
[Dixon-Coles](https://grokipedia.com/page/DixonColes_model) model, with parameters fitted to match implied 
win/loss/draw probabilities derived from either **(a)** bookmaker fair odds (for group stage matches) or **(b)** 
calibrated Elo rankings (for knockout phase matches where no odds are available). The simulation covers the full 
tournament from group stage through to the final, including FIFA tiebreaker rules, best 8 third-place qualification, 
and the complete 32-team knockout bracket. Both single-run and multi-run modes are available.

## 1. Match Model

### Dixon-Coles framework

Each match is modelled as a pair of independent Poisson random variables representing goals scored by 
the home and away teams. The expected goals $\lambda$ for each team are estimated within a Dixon-Coles 
framework. With this model, the probability of a scoreline $(i, j)$ is given by:

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

The correction parameter $\rho$ is typically negative in football, reflecting two
empirically observed patterns relative to the independent Poisson prediction:

- *Low-scoring draws are more frequent than expected*: the probability of 0-0 and 1-1 scorelines
  is boosted (since normally $\rho < 0$)
- *One-goal wins are less frequent than expected*: the probability of 1-0 and 0-1
  scorelines is reduced

### Deriving Win/Draw/Loss probabilities

For each match we require three outcome probabilities $(p_W,\ p_D,\ p_L)$: the probability
of a home win, draw, and away win respectively. These are sourced differently depending
on the phase of the tournament.

#### a) Bookmaker Odds (Group Stage)

For all 72 group stage matches, win/draw/loss probabilities are derived directly from pre-tournament bookmaker decimal 
odds. Raw bookmaker odds embed a profit margin, so they are first converted to fair probabilities by
normalising the implied probabilities to sum to 1:

$$p_i^{\text{fair}} = \frac{1/o_i}{\sum_{k} 1/o_k}$$

where $o_i$ are the decimal odds for each outcome $i \in \{\text{home},\ \text{draw},\ \text{away}\}$.

#### b) Calibrated Elo Ratings (Knockout Phase)
For knockout phase matches, no bookmaker odds are available in advance since the
matchups depend on the group stage results. Instead, win/draw/loss probabilities
are derived from calibrated Elo ratings using the method of [Xiong et al. (2016)](https://www.researchgate.net/publication/312617531_Mathematical_Model_of_Ranking_Accuracy_and_Popularity_Promotion).
Raw Elo ratings from [eloratings.net](https://www.eloratings.net/) are calibrated to adjust the 
win/draw/loss probabilities against the 72 group stage bookmaker odds, which serve as a more accurate 
and up-to-date signal of team quality.

For each of the 72 group stage matches, we can compute both:
- $p_i^{\text{bk}}$: the bookmaker fair probabilities
- $p_i^{\text{elo}}(\mathbf{r})$: the Elo-derived probabilities, which
  depend on the team ratings $\mathbf{r}$

We introduce a per-team offset $\delta_k$ for each team $k$, such that the calibrated
rating is $\tilde{r}_k = r_k + \delta_k$. The offsets are found by finding the minimum of 
function $J$, which expresses a regularized least squares problem:

$$ \mathcal{J}(\boldsymbol{\delta}) =
\sum_{\text{matches}} \sum_i \left(p_i^{\text{elo}} - p_i^{\text{bk}}\right)^2 + \lambda \sum_k \delta_k^2 $$

The regularisation term $\lambda \sum_k \delta_k^2$ penalises large deviations from
the original Elo ratings to avoid overfitting. The regularisation strength parameter
$\lambda$ controls the trade-off between fitting the bookmaker data and staying
anchored to the prior ratings.

The optimization is solved numerically using L-BFGS with symmetric bounds
$\delta_k \in [-400, +400]$. After testing, we set $\lambda = 3 \times 10^{-6}$, 
which yields a 92.3% reduction in mean squared error relative to the uncalibrated ratings,
while producing adjustments that are directionally consistent with current team strength. 
The largest corrections are shown below:

| Team | Original Elo | Calibrated Elo | $\delta$ |
|---|---|---|---|
| Ghana | 1510 | 1675 | +165 |
| USA | 1726 | 1845 | +119 |
| South Africa | 1518 | 1611 | +93 |
| Ecuador | 1938 | 1833 | -105 |
| Spain | 2157 | 2087 | -70 |
| Japan | 1906 | 1838 | -69 |

The win expectancy for the home team is given by the standard Elo formula, where 
$dr = \tilde{r}_h - \tilde{r}_a$ is the difference in calibrated Elo ratings between the
home and away team:

$$W_h = \frac{1}{1 + 10^{-dr/400}}$$

We follow Xiong et al. (2016) to model the draw probability as a Gaussian function of the 
rating difference, to ensure draws are most likely when the two teams are evenly matched, 
i.e. $dr = 0$:

$$P(\text{draw}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(dr/200)^2}{2\sigma^2}\right)$$

The parameter $\sigma$ is calibrated by matching the peak draw probability at $dr = 0$ 
to the empirically observed draw rate in international football (~28%), giving
$\sigma \approx 1.426$. The win and loss probabilities are then obtained by subtracting 
half the draw probability from each team's base win expectancy, capped and floored at 
90% and 3.3%, respectively, to ensure plausible values for $\lambda_h$ and $\lambda_a$.

$$P(\text{home win}) = \min\left(0.9,\ W_h - \frac{1}{2} P(\text{draw})\right)$$

$$P(\text{away win}) = \max\left(0.033,\ (1 - W_h) - \frac{1}{2} P(\text{draw})\right)$$

All three probabilities are subsequently re-normalized to sum to 1.

### Parameter Fitting

Given the three outcome probabilities $(p_W, p_D, p_L)$ from either source above,
the Dixon-Coles parameters $(\lambda_h, \lambda_a, \rho)$ are obtained by solving the
following least squares problem:

$$(\hat{\lambda}_h,\ \hat{\lambda}_a,\ \hat{\rho}) = \underset{\lambda_h,\ \lambda_a,\ \rho}{\arg\min}
\sum_{i \in \{W, D, L\}} \left( P_i^{\text{DC}}(\lambda_h, \lambda_a, \rho) - p_i \right)^2$$

where $P_W^{\text{DC}}$, $P_D^{\text{DC}}$ and $P_L^{\text{DC}}$ are the home win, draw
and away win probabilities implied by the Dixon-Coles score matrix, obtained by summing
over the relevant scorelines:

$$P_W^{\text{DC}} = \sum_{i > j} P(X=i, Y=j), \quad
P_D^{\text{DC}} = \sum_{i = j} P(X=i, Y=j), \quad
P_L^{\text{DC}} = \sum_{i < j} P(X=i, Y=j)$$

The optimization is solved numerically using the Nelder-Mead algorithm, with the
following constraints:

$$\lambda_h > 0, \quad \lambda_a > 0, \quad \rho \in [-0.5,\ 0]$$

The constraint $\rho \leq 0$ reflects the empirical finding that low-scoring draws are
over-represented in football relative to the independent Poisson prediction.

## 2. Simulation

### Group Stage

The group stage consists of 12 groups of 4 teams each, with every team playing the
other three teams in their group once. Each match is simulated by sampling a scoreline 
$(i, j)$ from the Dixon-Coles score probability matrix, using the fitted parameters
$(\hat{\lambda}_h, \hat{\lambda}_a, \hat{\rho})$ derived from bookmaker odds. After all 
group matches are simulated, teams are ranked within each group by points.

#### Standings and Tiebreakers

Ties in points are resolved using the official FIFA tiebreaker cascade, excluding 
the conduct score criterion (yellow/red cards):

1. Head-to-head points among tied teams
2. Head-to-head goal difference among tied teams
3. Head-to-head goals scored among tied teams
4. Overall goal difference
5. Overall goals scored
6. FIFA world ranking

#### Third-Place Qualification

The 8 best third-placed teams across all 12 groups qualify for the knockout phase.
Third-placed teams are ranked using the same criteria as above (points, goal
difference, goals scored, FIFA ranking), applied across groups.

The specific Round of 32 matchups involving third-placed teams depend on which 8
groups they come from. The 495 possible combinations are defined in Annex C of the
FIFA tournament regulations, and are encoded as a lookup table as an input file 
`third_place_match_combinations.csv`, in the `input_data` folder.

---

### Knockout Phase

The knockout phase follows a standard single-elimination bracket from the Round of
32 through to the Final, with a separate third-place match between the two losing
semi-finalists.

#### Match Simulation

Each knockout match is simulated in three stages:

1. *90 minutes*: a scoreline is sampled from the Dixon-Coles score matrix using
   parameters derived from either bookmaker odds (if the two teams met in the group
   stage) or calibrated Elo ratings (otherwise)

2. *Extra time*: if the score is level after 90 minutes, an additional 30 minutes
   are simulated by sampling from the Dixon-Coles matrix with scaled expected goals
   $\lambda_h^{\text{et}} = \lambda_h \times \frac{30}{90}$ and
   $\lambda_a^{\text{et}} = \lambda_a \times \frac{30}{90}$

3. *Penalty shootout*: if the score remains level after extra time, the winner is
   decided by a penalty shootout modelled as a fair coin flip

#### Bracket Resolution

The bracket is resolved dynamically round by round. The Round of 32 matchups are
fully determined once the group stage results are known, by combining:

- The group winners and runners-up from each group
- The third-place bracket assignment from the 495-combination lookup table

From the Round of 16 onward, each match is initialised only once both participants
have been determined by the results of the preceding round.

---

### Monte Carlo Aggregation

The full tournament — group stage and knockout phase — is simulated $N$ times
independently. Across all simulations, the following probabilities are estimated for
each team by counting the fraction of simulations in which the team achieves each outcome:

| Metric | Description |
|---|---|
| 1st / 2nd / 3rd / 4th | Group stage finishing position |
| 3rd (qualified) | Finished 3rd and qualified as one of the best 8 third-placed teams |
| Qualify % | Finished 1st, 2nd, or qualified as best 3rd |
| R32 / R16 / QF / SF | Reached the respective knockout round |
| 3rd Place | Won the third-place match |
| Final | Reached the final |
| Champion | Won the tournament |

The Dixon-Coles parameters $(\hat{\lambda}_h, \hat{\lambda}_a, \hat{\rho})$ are fitted
once per match before the simulation loop and reused across all $N$ simulations,
ensuring the computational cost of parameter fitting is incurred only once regardless
of the number of simulations.

---

## Future improvements
- Allow for the use of Elo ratings as the basis to simulate the entire tournament (not just knockout phase)