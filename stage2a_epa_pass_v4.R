library(brms)
library(cmdstanr)
library(dplyr)

data_list  <- readRDS("model_data.rds")

head(data_list)

train_df   <- data_list$train_df
train_pass <- train_df %>% filter(play_type == "pass")

cat("Pass play training rows:", nrow(train_pass), "\n")
cat("matchup_cell levels in pass data:",
    nlevels(droplevels(train_pass$matchup_cell)), "\n")

epa_pass_formula <- bf(
  epa | weights(recency_weight) ~
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
    offense_personnel_group:defense_personnel_group +
    def_rate_Base + def_rate_Nickel + def_rate_Dime +
    off_epa_by_personnel + off_success_by_personnel +
    def_epa_allowed_Base + def_epa_allowed_Nickel + def_epa_allowed_Dime +
    (1 | season) +
    (1 | posteam) +
    (1 | defteam) +
    (1 | off_team_season) +
    (1 | def_team_season) +
    (1 | off_personnel_team_season) +
    (1 | matchup_cell)
)

epa_priors <- c(
  prior(student_t(3, 0, 1),   class = "Intercept"),
  prior(normal(0, 0.5),       class = "b"),
  prior(student_t(3, 0, 0.5), class = "sd"),
  prior(student_t(3, 0, 1),   class = "sigma"),
  prior(gamma(2, 0.1),        class = "nu")
)

fit_epa_pass <- brm(
  formula  = epa_pass_formula,
  data     = train_pass,
  family   = student(),
  prior    = epa_priors,
  backend  = "cmdstanr",
  chains   = 4,
  cores    = 4,
  iter     = 3000,
  warmup   = 1500,
  seed     = 42,
  control  = list(adapt_delta = 0.95, max_treedepth = 12),
  file     = "brms_v4_stage2a_epa_pass"
)

summary(fit_epa_pass)

saveRDS(fit_epa_pass, "fit_epa_pass_v4.rds")
cat("Stage 2a complete. Saved to fit_epa_pass_v4.rds\n")