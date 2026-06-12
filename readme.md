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

#### Tiebreakers

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

### Knockout Phase

The knockout phase follows a standard single-elimination bracket from the Round of
32 through to the Final, with a separate third-place match between the two losing
semi-finalists.

#### Match Simulation

Each knockout match is simulated in three stages:

1. **90 minutes**: a scoreline is sampled from the Dixon-Coles score matrix using
   parameters derived from either bookmaker odds (if the two teams met in the group
   stage) or calibrated Elo ratings (otherwise)

2. **Extra time**: if the score is level after 90 minutes, an additional 30 minutes
   are simulated by sampling from the Dixon-Coles matrix with scaled expected goals
   $\lambda_h^{\text{et}} = \lambda_h \times \frac{30}{90}$ and
   $\lambda_a^{\text{et}} = \lambda_a \times \frac{30}{90}$

3. **Penalty shootout**: if the score remains level after extra time, each team has a 50% 
chance of winning the penalty shootout

#### Bracket Resolution

The bracket is resolved dynamically round by round. The Round of 32 matchups are
fully determined once the group stage results are known. From the Round of 16 onward, 
each match is initialised only once both participants have been determined by the results 
of the preceding round.

### Monte Carlo Aggregation

The full tournament — group stage and knockout phase — is simulated $N$ times
independently. Across all simulations, the following probabilities are estimated for
each team by counting the fraction of simulations in which the team achieves each outcome:

| Metric                | Description                                                            |
|-----------------------|------------------------------------------------------------------------|
| 1st / 2nd / 3rd / 4th | Group stage finishing position                                         |
| 3rd (qualified)       | Finished 3rd and qualified as one of the best 8 third-placed teams     |
| 3rd (not qualified)   | Finished 3rd and not qualified as one of the best 8 third-placed teams |
| Qualify %             | Finished 1st, 2nd, or qualified as a best 3rd                          |
| R32 / R16 / QF / SF   | Reached the respective knockout round                                  |
| 3rd Place             | Won the third-place match                                              |
| Final                 | Reached the final                                                      |
| Champion              | Won the tournament                                                     |

The Dixon-Coles parameters $(\hat{\lambda}_h, \hat{\lambda}_a, \hat{\rho})$ are fitted
once per match before the simulation loop and reused across all $N$ simulations,
ensuring the computational cost of parameter fitting is incurred only once regardless
of the number of simulations.

## 3. Results

The simulation was run $N = 1{,}000$ times with a fixed random seed for
reproducibility. The results below report the probability of each group stage finishing 
position, along with the probability of reaching each stage of the tournament. Teams are
ranked by (i) highest probability of winning the tournament and (ii) highest probability 
of reaching the knockout stage.

| Group | Team | 1st % | 2nd % | 3rd(Q) % | 3rd(out) % | 4th % | Qualify % | R32 % | R16 % | QF % | SF % | 3rd Place % | Final % | Champion % |
|-------|------|-------|-------|----------|------------|-------|-----------|-------|-------|------|------|-------------|---------|------------|
| H | Spain | 77.3 | 19.6 | 2.2 | 0.2 | 0.7 | 99.1 | 99.1 | 73.1 | 55.4 | 44 | 7.8 | 32.8 | 22.4 |
| I | France | 63.6 | 25.9 | 8.5 | 0.7 | 1.3 | 98 | 98 | 82.4 | 59.5 | 44.8 | 12 | 26.3 | 17.3 |
| J | Argentina | 68 | 22.1 | 8.1 | 0.7 | 1.1 | 98.2 | 98.2 | 69.3 | 55.3 | 40.4 | 10.9 | 25.4 | 13.8 |
| K | Portugal | 58.8 | 30.5 | 6.6 | 1.4 | 2.7 | 95.9 | 95.9 | 72.3 | 50.6 | 29.2 | 6.9 | 18.5 | 10.4 |
| L | England | 66.5 | 22.4 | 7.7 | 1.1 | 2.3 | 96.6 | 96.6 | 66.1 | 46 | 27.5 | 7.1 | 15.9 | 8.2 |
| C | Brazil | 75.5 | 19.4 | 4 | 0.3 | 0.8 | 98.9 | 98.9 | 73 | 47.6 | 27.1 | 7.7 | 14 | 6.5 |
| E | Germany | 69 | 21.6 | 7.9 | 0.4 | 1.1 | 98.5 | 98.5 | 67.8 | 32.8 | 22 | 6.7 | 8.4 | 4.2 |
| I | Norway | 25.7 | 40.4 | 22.9 | 6.5 | 4.5 | 89 | 89 | 57.2 | 31.8 | 15.9 | 3.6 | 8.4 | 3.5 |
| K | Colombia | 34.2 | 44.5 | 11.3 | 4.2 | 5.8 | 90 | 90 | 57.2 | 31.4 | 17.7 | 5.7 | 6.9 | 3 |
| F | Netherlands | 50.6 | 28.6 | 11.1 | 2 | 7.7 | 90.3 | 90.3 | 48.9 | 32.7 | 14.5 | 4 | 6.3 | 2.3 |
| G | Belgium | 64.7 | 23.4 | 7.2 | 1.1 | 3.6 | 95.3 | 95.3 | 65.9 | 37.3 | 13 | 3.6 | 5.6 | 1.6 |
| H | Uruguay | 19.2 | 59.2 | 11.1 | 4 | 6.5 | 89.5 | 89.5 | 36 | 19.5 | 8.2 | 2 | 2.8 | 1.1 |
| C | Morocco | 16.8 | 47.8 | 22.8 | 5.9 | 6.7 | 87.4 | 87.4 | 42.7 | 23 | 8.5 | 2 | 2.8 | 0.8 |
| I | Senegal | 10 | 29.7 | 33 | 12.7 | 14.6 | 72.7 | 72.7 | 37.9 | 17.2 | 7.5 | 2.3 | 2.5 | 0.8 |
| D | Turkey | 32.2 | 31 | 17.7 | 4.3 | 14.8 | 80.9 | 80.9 | 49.8 | 20.2 | 7 | 1.9 | 2.5 | 0.7 |
| F | Japan | 28.4 | 29.8 | 20.9 | 6.3 | 14.6 | 79.1 | 79.1 | 27.9 | 13.8 | 5.1 | 1.2 | 1.8 | 0.6 |
| B | Switzerland | 56.6 | 28.4 | 9.5 | 2.4 | 3.1 | 94.5 | 94.5 | 57.9 | 21.7 | 6.5 | 1.3 | 2.4 | 0.5 |
| E | Ecuador | 18.5 | 40.5 | 28 | 7.5 | 5.5 | 87 | 87 | 39.9 | 15.4 | 6.4 | 1.7 | 2.1 | 0.5 |
| L | Croatia | 23.9 | 45.9 | 13.8 | 6.2 | 10.2 | 83.6 | 83.6 | 39.7 | 17.1 | 7.1 | 2.1 | 2 | 0.4 |
| A | Mexico | 50.3 | 29 | 12.1 | 3 | 5.6 | 91.4 | 91.4 | 54.9 | 20.5 | 7.4 | 2 | 1.8 | 0.3 |
| J | Austria | 19.7 | 42.7 | 21.2 | 7.1 | 9.3 | 83.6 | 83.6 | 35.5 | 18.2 | 8.2 | 1.6 | 2.4 | 0.2 |
| D | USA | 38.9 | 28.5 | 14.2 | 4.7 | 13.7 | 81.6 | 81.6 | 50.6 | 20.1 | 6.9 | 1.7 | 1.8 | 0.2 |
| C | Scotland | 7.4 | 29 | 35 | 15.6 | 13 | 71.4 | 71.4 | 27.3 | 9.3 | 1.8 | 0.4 | 0.8 | 0.2 |
| B | Canada | 30.8 | 40.9 | 16.5 | 5.3 | 6.5 | 88.2 | 88.2 | 42.1 | 15 | 3.9 | 0.9 | 0.7 | 0.1 |
| J | Algeria | 10.9 | 28 | 28.6 | 13.2 | 19.3 | 67.5 | 67.5 | 23.3 | 9.1 | 2.4 | 0.4 | 1 | 0.1 |
| A | Czech Republic | 19.4 | 28.2 | 19.7 | 9.1 | 23.6 | 67.3 | 67.3 | 31.5 | 10 | 2.3 | 0.3 | 0.5 | 0.1 |
| D | Australia | 9.9 | 18.2 | 18.4 | 8.9 | 44.6 | 46.5 | 46.5 | 18.5 | 5.5 | 1.7 | 0.1 | 0.5 | 0.1 |
| K | Uzbekistan | 3.2 | 11.1 | 20.3 | 18.2 | 47.2 | 34.6 | 34.6 | 8.3 | 2.7 | 0.5 | 0.1 | 0.1 | 0.1 |
| E | Ivory Coast | 12 | 35 | 33.3 | 11 | 8.7 | 80.3 | 80.3 | 27.8 | 7.5 | 2.1 | 0.3 | 0.9 | 0 |
| G | Egypt | 19 | 35.9 | 19.9 | 8.9 | 16.3 | 74.8 | 74.8 | 29.3 | 8.4 | 1.8 | 0.3 | 0.5 | 0 |
| A | South Korea | 22.5 | 29.3 | 18.7 | 7.1 | 22.4 | 70.5 | 70.5 | 30.3 | 8.4 | 1.8 | 0.3 | 0.3 | 0 |
| B | Bosnia and Herzegovina | 10.3 | 24.7 | 32.1 | 12 | 20.9 | 67.1 | 67.1 | 17.5 | 1.9 | 0.1 | 0.1 | 0 | 0 |
| D | Paraguay | 19 | 22.3 | 24.3 | 7.5 | 26.9 | 65.6 | 65.6 | 32.6 | 11.4 | 3.2 | 0.5 | 0.7 | 0 |
| G | Iran | 12.3 | 27.8 | 24.5 | 12.2 | 23.2 | 64.6 | 64.6 | 20.2 | 4 | 0.5 | 0 | 0.1 | 0 |
| F | Sweden | 14.4 | 25.3 | 23.3 | 10 | 27 | 63 | 63 | 15.9 | 5.5 | 1.4 | 0.3 | 0.3 | 0 |
| L | Ghana | 6.3 | 20.7 | 26.7 | 13.5 | 32.8 | 53.7 | 53.7 | 10.2 | 2.4 | 0.2 | 0 | 0 | 0 |
| A | South Africa | 7.8 | 13.5 | 18.5 | 11.8 | 48.4 | 39.8 | 39.8 | 9.8 | 1.7 | 0.2 | 0 | 0 | 0 |
| F | Tunisia | 6.6 | 16.3 | 16.6 | 9.8 | 50.7 | 39.5 | 39.5 | 6 | 1.4 | 0.1 | 0 | 0 | 0 |
| K | DR Congo | 3.8 | 13.9 | 18 | 20 | 44.3 | 35.7 | 35.7 | 9.7 | 2.8 | 0.3 | 0 | 0.2 | 0 |
| H | Cape Verde | 1.7 | 10 | 18.8 | 22.9 | 46.6 | 30.5 | 30.5 | 6.2 | 1.3 | 0.1 | 0 | 0 | 0 |
| H | Saudi Arabia | 1.8 | 11.2 | 17.2 | 23.6 | 46.2 | 30.2 | 30.2 | 4.7 | 0.4 | 0 | 0 | 0 | 0 |
| L | Panama | 3.3 | 11 | 15.6 | 15.4 | 54.7 | 29.9 | 29.9 | 4.4 | 1 | 0.2 | 0 | 0 | 0 |
| G | New Zealand | 4 | 12.9 | 12.2 | 14 | 56.9 | 29.1 | 29.1 | 6.8 | 1.3 | 0.2 | 0 | 0 | 0 |
| B | Qatar | 2.3 | 6 | 12.3 | 9.9 | 69.5 | 20.6 | 20.6 | 2.4 | 0.1 | 0 | 0 | 0 | 0 |
| J | Jordan | 1.4 | 7.2 | 9.5 | 11.6 | 70.3 | 18.1 | 18.1 | 3.9 | 1 | 0.2 | 0.2 | 0 | 0 |
| I | Iraq | 0.7 | 4 | 7.3 | 8.4 | 79.6 | 12 | 12 | 2.8 | 0.4 | 0 | 0 | 0 | 0 |
| C | Haiti | 0.3 | 3.8 | 6.3 | 10.1 | 79.5 | 10.4 | 10.4 | 1.4 | 0.3 | 0.1 | 0 | 0 | 0 |
| E | Curacao | 0.5 | 2.9 | 4.6 | 7.3 | 84.7 | 8 | 8 | 1.1 | 0.1 | 0 | 0 | 0 | 0 |


### Key Findings

- **Spain (22.4%) and France (17.3%)** are the two most likely champions, together
  accounting for nearly 40% of all simulated tournament wins. Both teams combine a
  very high group stage qualification probability (>99%) with strong knockout
  progression rates.
- **Argentina (13.8%) and Portugal (10.4%)** are the next most likely winners,
  forming a clear second tier alongside England (8.2%) and Brazil (6.5%).
- **Germany (4.2%)** qualifies in 98.5% of simulations but faces a tougher
  bracket path, limiting their champion probability relative to their group
  stage dominance.
- **Ghana (53.7% qualification)** reflects the Elo calibration correction: after
  adjusting upward by +165 Elo points, Ghana is modelled as a competitive
  mid-table qualifier rather than a heavy underdog.

## 4. Future improvements
- Allow for the use of Elo ratings as the basis to simulate the entire tournament
(not just knockout phase)