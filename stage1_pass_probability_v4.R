library(brms)
library(cmdstanr)
library(dplyr)

data_list <- readRDS("model_data.rds")
train_df  <- data_list$train_df

cat("Training rows:", nrow(train_df), "\n")
cat("2025 rows in training:", sum(as.integer(as.character(train_df$season)) == 2025), "\n")

pass_formula <- bf(
  is_pass | weights(recency_weight) ~
    down +
    s(ydstogo, k = 5) +
    s(yardline_100, k = 5) +
    s(score_differential, k = 5) +
    s(game_seconds_remaining, k = 5) +
    s(half_seconds_remaining, k = 5) +
    qtr +
    clock_stop +
    offense_personnel_group +
    defense_personnel_group +
    
    # Personnel matchup fixed effect
    offense_personnel_group:defense_personnel_group +
    
    # Dime confounding fix:
    # These interactions allow the model to learn that the relationship
    # between defensive package and pass probability is moderated by
    # game situation. Without these, the model cannot distinguish
    # "dime causes passing" from "passing situations cause dime."
    defense_personnel_group:down +
    defense_personnel_group:ydstogo +
    defense_personnel_group:score_differential +
    
    off_rate_run + off_rate_pass +
    def_rate_Base + def_rate_Nickel + def_rate_Dime +
    
    # Random effects
    # off_personnel_team_season intentionally omitted — see notes above
    (1 | season) +
    (1 | posteam) +
    (1 | defteam) +
    (1 | off_team_season) +
    (1 | def_team_season)
)

pass_priors <- c(
  prior(student_t(3, 0, 1.5), class = "Intercept"),
  prior(normal(0, 0.75),      class = "b"),
  prior(student_t(3, 0, 0.5), class = "sd")
)

fit_pass <- brm(
  formula  = pass_formula,
  data     = train_df,
  family   = bernoulli(link = "logit"),
  prior    = pass_priors,
  backend  = "cmdstanr",
  chains   = 4,
  cores    = 4,
  iter     = 3000,
  warmup   = 1500,
  seed     = 42,
  control  = list(adapt_delta = 0.95, max_treedepth = 12),
  file     = "brms_v4_stage1_pass_probability"
)

summary(fit_pass)

saveRDS(fit_pass, "fit_pass_v4.rds")
cat("Stage 1 complete. Saved to fit_pass_v4.rds\n")