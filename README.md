# NFL Defensive Personnel Optimizer

**A Bayesian Hierarchical Model for Prescriptive Defensive Playcalling**

*Asa Arnold — University of Notre Dame — Applied AI, Data & Modeling*

---

## Overview

NFL defenses have historically matched personnel based on player type — adding an extra defensive back for every receiver added to the field. Offensive play callers have begun to exploit this convention by deploying heavy packages (12, 13, and 22 personnel) and passing at unexpectedly high rates. The 2025 Los Angeles Rams exemplify this trend, running 13 personnel (1 RB, 3 TE, 1 WR) at a 30%+ rate — more than double the next highest team — while generating roughly 0.5 EPA per pass from that formation.

This project uses a **Bayesian Hierarchical Model** to prescribe optimal defensive personnel packages (Base, Nickel, or Dime) for defensive coordinators on a play-by-play basis. Rather than predicting what will happen, the model recommends what the defense *should* do to minimize expected points allowed given the current game situation.

A companion **XGBoost baseline model** was developed first to establish performance benchmarks and identify structural limitations, including the dime-pass confounding issue that motivated several architectural decisions in the Bayesian model.

---

## The Research Question

> Given offensive personnel, down, distance, score differential, clock, and team identities — which defensive package (Base, Nickel, or Dime) minimizes Expected Points Added (EPA) allowed on this play?

---

## Personnel Notation

**Offensive packages** are described by two digits:
- First digit = number of running backs
- Second digit = number of tight ends
- Remaining skill positions are wide receivers

| Package | Composition | Typical Use |
|---------|-------------|-------------|
| 11 | 1 RB, 1 TE, 3 WR | Most common modern passing formation |
| 12 | 1 RB, 2 TE, 2 WR | Balanced, run or pass |
| 13 | 1 RB, 3 TE, 1 WR | Heavy formation — the Rams' signature package |
| 21 | 2 RB, 1 TE, 2 WR | Moderate weight |
| 22 | 2 RB, 2 TE, 1 WR | Very heavy, typically run-oriented |

**Defensive packages** are defined by total defensive backs on the field:

| Package | Defensive Backs | Traditional Use |
|---------|-----------------|-----------------|
| Base | 4 | Heavy offensive personnel |
| Nickel | 5 | Most common modern default |
| Dime | 6 | Obvious passing situations |

---

## Model Architecture

The model uses a **three-stage pipeline** to avoid selection bias in EPA estimation.

### Why Three Models?

A single EPA regression cannot be used prescriptively because Dime packages disproportionately face pass plays, and passes have higher EPA than runs. A naive single-model approach makes Dime appear ineffective — not because it defends passes poorly, but because it faces more of them. This is the **Dime selection bias problem**.

### The Three Stages

**Stage 1 — Pass Probability**
- Family: Bernoulli with logit link
- Outcome: P(pass | situation, defensive_package)
- Trained on: All training plays

**Stage 2a — Pass EPA**
- Family: Student-t regression
- Outcome: E[EPA | pass plays, situation, defensive_package]
- Trained on: Pass plays only

**Stage 2b — Run EPA**
- Family: Student-t regression
- Outcome: E[EPA | run plays, situation, defensive_package]
- Trained on: Run plays only

### The Marginalisation Formula

```
E[EPA | package] = P(pass | package) × E[EPA | pass, package]
                 + P(run  | package) × E[EPA | run,  package]
```

This is computed across all 6,000 posterior draws. The package with the lowest marginal EPA is recommended. The **probability that each package is best** is the fraction of draws where it produces the minimum EPA.

---

## The Dime-Pass Confounding Problem

The most significant methodological challenge in this project is confounding between defensive package selection and play type. Obvious passing situations (3rd and long, trailing late) cause both the offense to pass *and* the defense to show Dime. In the historical data this creates a strong correlation between Dime and passing — but the causal arrow runs from game situation to both decisions, not from Dime to passing.

The XGBoost baseline exposed this concretely: on 2nd and 2 in neutral situations, the model predicted higher pass probability when the defense showed Dime — the opposite of what any coach would expect. A rational offense seeing a light box (6 DBs) would immediately run.

**The v4 fix** adds situational interaction terms to Stage 1:

```r
defense_personnel_group:down
defense_personnel_group:ydstogo
defense_personnel_group:score_differential
```

These allow the model to learn that Dime in a 3rd-and-15 situation (where passing was almost certain regardless of package) is a different signal than Dime in a 2nd-and-3. The situational context absorbs some of the confounded correlation, though the limitation cannot be fully eliminated with observational data alone.

---

## Data

### Source
`nflreadr::load_participation(seasons = 2023:2025, include_pbp = TRUE)`

The data operates at the play level and includes EPA, personnel strings, field position, team identities, score, and clock — everything needed to model and recommend defensive personnel decisions.

### Data Window
- **Training**: 2023 season + 2024 season + 2025 Weeks 1–9
- **Test**: 2025 Weeks 10–18

The week-based 2025 split ensures all teams have approximately 8–9 games of current-season data in training before the test period begins. This gives team-season random effects real 2025 observations to anchor to rather than falling back entirely on historical priors.

### Recency Weights

```r
recency_weight = case_when(
  season == 2025 ~ 1.0,
  season == 2024 ~ 0.8,
  season == 2023 ~ 0.6
)
```

Weights are passed directly to the brms likelihood via `| weights(recency_weight)` in the formula, downweighting older seasons without discarding their contribution to fixed effects.

### Filters Applied
- Remove plays with missing personnel strings, EPA, or situation variables
- Keep `play_type %in% c("run", "pass")` only
- Remove QB kneels and QB spikes
- **Garbage time filter**: Remove Q4 plays where `|score_differential| >= 24` and `game_seconds_remaining < 900`

### Personnel Parsing

Personnel strings are extracted from natural language format via regex:

```r
rb_count = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*RB)"))
te_count = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*TE)"))
offense_personnel_group = paste0(rb_count, te_count)

db_count = cb_count + fs_count + ss_count
# 4 DBs → Base | 5 DBs → Nickel | 6 DBs → Dime
```

---

## Bayesian Model Details

### Why Bayesian?

1. **Full uncertainty quantification** — every recommendation includes probability of being optimal, not just a point estimate
2. **Partial pooling** — sparse matchup cells (e.g., Dime vs 13 personnel with only 4 observations) are automatically shrunk toward the team-season mean rather than producing noisy raw estimates
3. **Hierarchical structure** — matches how NFL data is naturally organized (plays nested in teams, teams nested in seasons)
4. **Priors encode domain knowledge** — coefficient priors reflect realistic NFL effect sizes

### Random Effects Structure

| Random Effect | Football Meaning |
|--------------|-----------------|
| `(1 \| season)` | Each season has its own EPA baseline — NFL evolves year to year |
| `(1 \| posteam)` | Each offense has a baseline efficiency |
| `(1 \| defteam)` | Each defense has a baseline EPA-allowed profile |
| `(1 \| off_team_season)` | Team strength changes annually |
| `(1 \| def_team_season)` | Same for defense |
| `(1 \| off_personnel_team_season)` | Team EPA by personnel group with shrinkage (Stages 2a/2b only) |
| `(1 \| matchup_cell)` | **Key innovation** — replaces raw matchup_epa means with partial pooling |

### The matchup_cell Innovation

In earlier versions, `matchup_epa` was a raw group mean joined as a fixed predictor. A cell like "MIN_2025 vs 13 personnel in Dime" might have only 4 observations in a given season. The raw mean of 4 plays is noise treated as certainty.

The v3/v4 solution replaces this with `(1 | matchup_cell)` where:
```
matchup_cell = def_team_season : offense_personnel_group : defense_personnel_group
```

Sparse cells are heavily shrunk toward the team-season mean. Well-observed cells (Nickel vs 11 personnel) are trusted. Shrinkage scales automatically with evidence — no manual threshold required.

### Priors

```r
# Stage 1 (Bernoulli-logit)
prior(student_t(3, 0, 1.5), class = "Intercept")  # baseline ~50% pass
prior(normal(0, 0.75),      class = "b")           # modest log-odds effects
prior(student_t(3, 0, 0.5), class = "sd")          # random effect SDs

# Stage 2a/2b (Student-t EPA)
prior(student_t(3, 0, 1),   class = "Intercept")  # baseline EPA near 0
prior(normal(0, 0.5),       class = "b")           # <0.5 EPA per SD shift
prior(student_t(3, 0, 0.5), class = "sd")          # team/matchup variation
prior(student_t(3, 0, 1),   class = "sigma")       # residual noise
prior(gamma(2, 0.1),        class = "nu")          # tail heaviness (mean=20)
```

The Student-t likelihood for EPA models handles heavy tails — pick-sixes and long touchdowns swing EPA by ~7 points. A Gaussian model would be distorted by these extreme plays.

### MCMC Configuration

```r
brm(
  backend = "cmdstanr",
  chains  = 4,
  cores   = 4,
  iter    = 3000,
  warmup  = 1500,
  seed    = 42,
  control = list(adapt_delta = 0.95, max_treedepth = 12)
)
```

4 chains × 1,500 post-warmup samples = **6,000 posterior draws** for all summaries.

---

## Model Performance

All metrics evaluated on 2025 Weeks 10–18 test set.

### Comparison: XGBoost Baseline vs Bayesian Hierarchical Model

| Metric | XGBoost | Bayesian | Change |
|--------|---------|----------|--------|
| Play type accuracy | 69.49% | 76.39% | +6.9 pp |
| AUC-ROC | 0.7687 | 0.8252 | +7.4% |
| Brier score | N/A | 0.1678 | — |
| Brier skill score | N/A | 31.8% better than naive | — |
| Pass EPA RMSE | 1.6152 | 1.5550 | -3.7% |
| Pass EPA CRPS | N/A | 0.8163 | — |
| Run EPA RMSE | 0.9934 | 0.7664 | -22.8% |
| Run EPA CRPS | N/A | 0.3689 | — |

**Notes:**
- Brier score and CRPS require probabilistic forecasts — XGBoost cannot produce these metrics
- Run EPA improvement (+22.8%) is substantially larger than pass EPA (+3.7%) because run plays have lower variance and personnel has a more direct effect on run stopping
- Play type accuracy improvement (+6.9 pp over a 57% naive baseline) is significant because prediction errors compound through the three-stage pipeline

---

## Recommendation Output

For any input game situation, the model returns:

- **Recommended package** (Base, Nickel, or Dime)
- **Confidence level**: Strong Recommend / Recommend / Light Recommend
- **Expected EPA** for the recommended package with 80% and 95% credible intervals
- **P(pass)** — estimated pass probability if the defense shows that package
- **EPA edge** over the second-best package
- **Full posterior distributions** for all three packages

### Confidence Thresholds

| Level | Condition | Interpretation |
|-------|-----------|----------------|
| **Strong Recommend** ★★★ | `prob_best >= 0.75` AND `edge_lower80 > 0` | Confident it is best AND margin is reliably positive |
| **Recommend** ★★ | `prob_best >= 0.55` OR `expected_edge > 0.02` | Clear preference but meaningful uncertainty |
| **Light Recommend** ★ | All other cases | Distributions overlap substantially |

The two-condition requirement for Strong Recommend prevents false confidence: a model can be 75% likely to recommend Nickel while the EPA edge over Base still has a negative lower bound, meaning Base might actually be better in meaningful fraction of draws.

---

## Known Limitations

**Dime-pass confounding** — The model cannot fully distinguish "Dime causes passing" from "passing situations cause Dime" using observational data alone. The v4 interaction terms reduce but do not eliminate this bias. A complete solution would require instrumental variable estimation or a randomized experiment, neither of which is practically available.

**EPA underestimation** — The model's expected EPA outputs are lower than raw team averages suggest (e.g., -0.327 EPA for the Rams in 13 personnel vs their actual +0.2 average). This reflects the model's conservative posterior mean across all game contexts rather than Rams-specific performance in favorable situations.

**Play-level independence** — Each play is evaluated in isolation. The model has no memory of in-game context — it cannot account for play-action setups, within-game adjustments, or how many times a coordinator has already used a particular package.

**Personnel granularity** — The model knows the offense is in 13 personnel but not which specific tight ends are on the field. The difference between Tyler Higbee and a practice squad replacement is invisible to the model.

**Temporal stationarity** — Teams change year over year. The seasonal weighting and off_team_season random effect partially address this, but coaching staff turnover and roster changes mid-season remain invisible to the model.

---

## Future Work

**Nash Equilibrium Pass Probability** — Replace the observational P(pass | package) with a game-theoretic estimate. Use E[EPA | pass, package] and E[EPA | run, package] from the model to compute the mixed-strategy Nash equilibrium for the offense, eliminating the confounding issue at its source.

**Win Probability as Alternative Outcome** — EPA optimizes scoring efficiency but does not capture clock management goals. In late-game lead-protection scenarios, a model targeting win probability or success rate may better reflect what coaches are actually optimizing.

**Sequential Bayesian Updating** — Use the posterior from Week 11 as the prior for Week 12 updating. The Bayesian framework supports this natively — it is the most theoretically appealing improvement and currently unused.

**Causal Inference Framework** — Implement a DAG-based identification strategy using `dagitty` (R) or `DoWhy` (Python) to formally identify the causal effect of defensive package choice on EPA, exploiting natural experiments such as unexpected pregame injuries.

**Coverage Shell Recommendations** — Extend beyond Base/Nickel/Dime to recommend specific coverage shells (Cover 2, Cover 3, Cover 4, Cover 1 Robber) within each package. The three-stage architecture generalizes directly.

---

## Repository Structure

```
nfl-personnel-optimizer/
│
├── README.md
│
├── data/
│   └── model_data.rds                         # Processed train/test data (generated locally)
│
├── models/
│   ├── fit_pass_resaved.rds                   # Stage 1: P(pass) fitted model
│   ├── fit_epa_pass_resaved.rds               # Stage 2a: Pass EPA fitted model
│   └── fit_epa_run_resaved.rds                # Stage 2b: Run EPA fitted model
│
├── training/
│   ├── local_bayes_files_personnel.R          # Local data prep script (generates model_data.rds)
│   ├── stage1_pass_probability_v4.R           # CRC training script — Stage 1
│   ├── stage2a_epa_pass_v4.R                  # CRC training script — Stage 2a
│   └── stage2b_epa_run_v4.R                   # CRC training script — Stage 2b
│
├── evaluation/
│   ├── model_evaluation_and_recommendations_complete.qmd   # Full evaluation + recommendations
│   └── model_evaluation_and_recommendations_complete.html  # Rendered HTML output
│
├── baseline/
│   └── personnel_xgboost_baseline.qmd         # XGBoost baseline model
│
└── app/
    └── app.R                                  # Shiny recommendation application
```

> **Note on model files:** The fitted `.rds` model files are large (24–80 MB each). If they exceed GitHub's file size limit, use [Git Large File Storage (LFS)](https://git-lfs.github.com/) or host them externally and link from this README.

---

## Installation and Setup

### Prerequisites

```r
install.packages(c(
  "nflreadr", "dplyr", "stringr", "tidyr", "forcats",
  "brms", "cmdstanr", "tidybayes", "posterior",
  "bayesplot", "ggplot2", "patchwork",
  "scoringRules", "pROC", "shiny", "jsonlite"
))
```

### Install CmdStan (required for brms backend)

```r
cmdstanr::install_cmdstan()
cmdstanr::cmdstan_version()  # verify installation
```

**Platform-specific prerequisites:**
- **Mac**: `xcode-select --install` in Terminal
- **Windows**: Install [RTools](https://cran.r-project.org/bin/windows/Rtools/) for your R version
- **Linux**: `sudo apt install build-essential`

### Load Pre-Trained Models

```r
fit_pass     <- readRDS("models/fit_pass_resaved.rds")
fit_epa_pass <- readRDS("models/fit_epa_pass_resaved.rds")
fit_epa_run  <- readRDS("models/fit_epa_run_resaved.rds")
data_list    <- readRDS("data/model_data.rds")
```

### Run the Shiny App

```r
shiny::runApp("app/app.R")
```

---

## Retraining from Scratch

The models were trained on the Notre Dame Center for Research Computing (CRC) due to the substantial compute requirements (~15 hours per model stage on a 4-core cluster node).

### Step 1: Generate processed data locally

```r
source("training/local_bayes_files_personnel.R")
# Produces: data/model_data.rds
```

### Step 2: Upload to compute cluster and submit jobs

```bash
# Upload files
scp data/model_data.rds user@cluster:~/project/
scp training/stage1_pass_probability_v4.R user@cluster:~/project/
scp training/stage2a_epa_pass_v4.R user@cluster:~/project/
scp training/stage2b_epa_run_v4.R user@cluster:~/project/

# Submit all three jobs simultaneously (they are independent)
qsub run_stage1.txt
qsub run_stage2a.txt
qsub run_stage2b.txt
```

All three training jobs are completely independent and can run in parallel on separate nodes.

### Estimated Training Times (4-core CRC node)

| Stage | Data | Estimated Time |
|-------|------|----------------|
| Stage 1: P(pass) | Full training set | 15–20 hours |
| Stage 2a: Pass EPA | Pass plays only (~60%) | 12–16 hours |
| Stage 2b: Run EPA | Run plays only (~40%) | 8–12 hours |

---

## Example Output

**Scenario:** LA Rams (13 personnel) vs MIN Vikings — 1st & 10 — MIN 30 — Q4 — +7 score — 6:35 remaining

```
╔══════════════════════════════════════════════════════╗
║         DEFENSIVE PERSONNEL RECOMMENDATION          ║
╚══════════════════════════════════════════════════════╝

  PACKAGE:    Dime
  CONFIDENCE: ★★ Recommend

  ── EPA Distribution (recommended package) ──────────
  Expected EPA allowed:  -0.327
  80% credible interval: [-0.519, -0.135]

  ── Edge over second-best ────────────────────────────
  Expected edge:         +0.028 EPA
  Probability best:      52.7%

  ── All packages ─────────────────────────────────────
  Package   P(pass)   EPA|pass   EPA|run   E[EPA]    P(best)
  Dime      39.5%     -0.032     -0.482    -0.327    52.7%
  Nickel    17.7%     -0.014     -0.401    -0.298    27.3%
  Base      16.7%      0.001     -0.392    -0.290    20.0%
```

---

## References

Baldwin, Ben. "4th Down Research." *nfl4th*. https://www.nfl4th.com/articles/4th-down-research.html

"Offensive Personnel Packages Common in the NFL." *Inside the 49*, 2 Apr. 2025. https://insidethe49.com/z/offensive-personnel-packages-nfl/

"Offensive Personnel Tendencies — 13 Personnel." *SumerSports*. https://sumersports.com/teams/offensive/personnel-tendency/?personnel=13

"Offensive Team Statistics." *SumerSports*. https://sumersports.com/teams/offensive/

Olsen, Greg. Tweet. *X (formerly Twitter)*, 2024. https://x.com/gregolsen88/status/1984741411657298239

Pitts, Jordan, and Brian Evans. "Defensive Coordinator and Head Coach Effects on In-Game Decision Making." *Journal of Quantitative Analysis in Sports*, De Gruyter. https://www.degruyterbrill.com/document/doi/10.2202/1559-0410.1255/html

"Personnel Packages Explained." *YouTube*. https://www.youtube.com/watch?v=8By4b6cTVFU

---

## Author

**Asa Arnold**
University of Notre Dame
Applied AI, Data & Modeling — 2026

*Built with assistance from Claude (Anthropic) — Sonnet 4.6*
