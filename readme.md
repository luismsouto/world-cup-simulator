# 🌍 FIFA World Cup Simulator (WCS)

A full Monte Carlo simulation engine for the 2026 FIFA World Cup, built in Python. The games are simulated using the 
[Dixon-Coles](https://grokipedia.com/page/DixonColes_model) model, with parameters fitted to match implied 
win/loss/draw probabilities derived from either **(a)** bookmaker fair odds (for group stage matches) or **(b)** 
calibrated Elo rankings (for knockout phase matches where no odds are available). The simulation covers the full 
tournament from group stage through to the final, including FIFA tiebreaker rules, best 8 third-place qualification, 
and the complete 32-team knockout bracket. Both single-run and multi-run modes are available.

## 1. Methodology

### Goal Scoring Model: Dixon-Coles

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
- $p_i^{\text{bk}}$: the bookmaker fair probabilities (Section 2a)
- $p_i^{\text{elo}}(\mathbf{r})$: the Elo-derived probabilities (Section 2b), which
  depend on the team ratings $\mathbf{r}$

We introduce a per-team offset $\delta_k$ for each team $k$, such that the calibrated
rating is $\tilde{r}_k = r_k + \delta_k$. The offsets are found by finding the minimum of 
function $J$, which expresses a regularized least squares problem:

$$ \mathcal{J}(\boldsymbol{\delta}) =
\sum_{\text{matches}} \sum_i \left(p_i^{\text{elo}} - p_i^{\text{bk}}\right)^2 + \lambda \sum_k \delta_k^2 $$

with respect to $\boldsymbol{\delta}$.
The regularisation term $\lambda \sum_k \delta_k^2$ penalises large deviations from
the original Elo ratings, which is important given the limited training data
(each team appears in only 3 group stage matches). The regularisation strength
$\lambda$ controls the trade-off between fitting the bookmaker data and staying
anchored to the prior ratings.

The optimisation is solved using L-BFGS-B with symmetric bounds
$\delta_k \in [-400, +400]$ for all teams. After testing a range of values, we set
$\lambda = 3 \times 10^{-6}$, which yields a *92.3% reduction in mean squared error*
relative to the uncalibrated ratings while producing adjustments that are directionally
consistent with current team strength. The largest corrections are shown below:

| Team | Original Elo | Calibrated Elo | $\delta$ |
|---|---|---|---|
| Ghana | 1510 | 1675 | +165 |
| USA | 1726 | 1845 | +119 |
| South Africa | 1518 | 1611 | +93 |
| Ecuador | 1938 | 1833 | -105 |
| Spain | 2157 | 2087 | -70 |
| Japan | 1906 | 1838 | -69 |
| Argentina | 2114 | 2053 | -61 |

The win expectancy for the home team is given by the standard Elo formula:

$$W_h = \frac{1}{1 + 10^{-dr/400}}$$

where $dr = r_h - r_a$ is the difference in Elo ratings between the home and away team.

Since the original Elo system was designed for chess — a game with rare draws — a
separate draw probability function is required for football. Following Xiong et al.
(2016), the draw probability is modelled as a Gaussian function of the rating
difference:

$$P(\text{draw}) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\!\left(-\frac{(dr/200)^2}{2\sigma^2}\right)$$



This formulation ensures that draws are most likely when the two teams are evenly
matched ($dr = 0$) and decay naturally as the rating gap grows. The parameter
$\sigma$ is calibrated by matching the peak draw probability at $dr = 0$ to the
empirically observed draw rate in international football (~28%), giving
$\sigma \approx 1.426$.

The win and loss probabilities are then obtained by subtracting half the draw
probability from each team's base win expectancy (Xiong et al., 2016, Equations 6--7):

$$P(\text{home win}) = W_h - \frac{1}{2} P(\text{draw})$$

$$P(\text{away win}) = (1 - W_h) - \frac{1}{2} P(\text{draw})$$

To prevent extreme Elo gaps from producing implausible probabilities — and to ensure
numerical stability in the Dixon-Coles parameter fitting — the three probabilities
are bounded to realistic football ranges before being passed to the optimiser:

$$P(\text{win}) \leq 0.90, \quad P(\text{loss}) \geq 0.033, \quad P(\text{draw}) \geq 0.067$$

---

## Future improvements
- Allow for the use of Elo ratings as the basis to simulate the entire tournament (not just knockout phase)