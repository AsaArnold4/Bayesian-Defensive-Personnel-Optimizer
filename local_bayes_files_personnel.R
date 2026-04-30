library(nflreadr)
library(dplyr)
library(stringr)
library(tidyr)
library(forcats)

seasons <- 2022:2025

participation_df <- nflreadr::load_participation(
  seasons     = seasons,
  include_pbp = TRUE
)

# Confirm week column is present
cat("Columns available for splitting:\n")
cat(intersect(c("week", "season", "game_date"), names(participation_df)), "\n")
cat("Week range in 2025 data:",
    range(participation_df$week[participation_df$season == 2025], na.rm = TRUE), "\n")

personnel_df <- participation_df %>%
  filter(
    !is.na(offense_personnel),
    !is.na(defense_personnel),
    !is.na(epa),
    play_type %in% c("run", "pass")
  ) %>%
  mutate(
    rb_count = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*RB)")),
    te_count = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*TE)")),
    offense_personnel_group = ifelse(
      !is.na(rb_count) & !is.na(te_count),
      paste0(rb_count, te_count),
      NA_character_
    ),
    cb_count = coalesce(as.integer(str_extract(defense_personnel, "\\d+(?=\\s*CB)")), 0L),
    fs_count = coalesce(as.integer(str_extract(defense_personnel, "\\d+(?=\\s*FS)")), 0L),
    ss_count = coalesce(as.integer(str_extract(defense_personnel, "\\d+(?=\\s*SS)")), 0L),
    db_count = cb_count + fs_count + ss_count,
    defense_personnel_group = case_when(
      db_count == 4 ~ "Base",
      db_count == 5 ~ "Nickel",
      db_count == 6 ~ "Dime",
      TRUE          ~ NA_character_
    ),
    is_pass = as.integer(play_type == "pass")
  ) %>%
  filter(
    !is.na(offense_personnel_group),
    !is.na(defense_personnel_group),
    down %in% 1:4,
    !is.na(ydstogo), !is.na(yardline_100), !is.na(score_differential),
    !is.na(qtr), !is.na(game_seconds_remaining), !is.na(half_seconds_remaining),
    !is.na(posteam), !is.na(defteam), !is.na(week),
    !qb_kneel, !qb_spike
  ) %>%
  mutate(
    offense_personnel_group = ifelse(
      offense_personnel_group %in% c("11", "12", "13", "21", "22"),
      offense_personnel_group, "Other"
    ),
    offense_personnel_group = factor(offense_personnel_group,
                                     levels = c("11", "12", "13", "21", "22", "Other")),
    defense_personnel_group = factor(defense_personnel_group,
                                     levels = c("Base", "Nickel", "Dime")),
    posteam         = factor(posteam),
    defteam         = factor(defteam),
    season          = factor(season),
    qtr             = factor(qtr),
    season_int      = as.integer(as.character(season)),
    off_team_season = factor(paste0(posteam, "_", season)),
    def_team_season = factor(paste0(defteam, "_", season))
  ) %>%
  filter(!(qtr == 4 & abs(score_differential) >= 24 & game_seconds_remaining < 900))

cat("Total plays after filtering:", nrow(personnel_df), "\n")
cat("2025 plays by week:\n")
print(table(personnel_df$week[personnel_df$season_int == 2025]))

personnel_df <- personnel_df %>%
  mutate(
    clock_stop = as.integer(
      incomplete_pass == 1 | out_of_bounds == 1 |
        first_down == 1 | touchdown == 1 | timeout == 1
    )
  )
head(personnel_df)

personnel_df <- personnel_df %>%
  mutate(
    matchup_season = ifelse(season_int >= 2022,
                            as.character(season), "pre2022"),
    matchup_cell = factor(paste(
      paste0(defteam, "_", matchup_season),
      offense_personnel_group,
      defense_personnel_group,
      sep = ":"
    )),
    off_personnel_team_season = factor(paste(
      off_team_season, offense_personnel_group, sep = ":"
    ))
  )

cat("matchup_cell levels:", nlevels(personnel_df$matchup_cell), "\n")
cat("off_personnel_team_season levels:",
    nlevels(personnel_df$off_personnel_team_season), "\n")

tail(personnel_df)

def_pkg_rates <- personnel_df %>%
  count(def_team_season, defense_personnel_group, name = "n") %>%
  group_by(def_team_season) %>%
  mutate(rate = n / sum(n)) %>%
  select(def_team_season, defense_personnel_group, rate) %>%
  pivot_wider(names_from = defense_personnel_group, values_from = rate,
              names_prefix = "def_rate_", values_fill = 0)

off_split <- personnel_df %>%
  count(off_team_season, offense_personnel_group, play_type, name = "n") %>%
  group_by(off_team_season, offense_personnel_group) %>%
  mutate(rate = n / sum(n)) %>%
  ungroup() %>%
  pivot_wider(names_from = play_type, values_from = rate,
              names_prefix = "off_rate_", values_fill = 0)

off_strength <- personnel_df %>%
  group_by(off_team_season, offense_personnel_group) %>%
  summarise(
    off_epa_by_personnel     = mean(epa, na.rm = TRUE),
    off_success_by_personnel = mean(epa > 0, na.rm = TRUE),
    .groups = "drop"
  )

def_strength <- personnel_df %>%
  group_by(def_team_season, defense_personnel_group) %>%
  summarise(
    def_epa_allowed_by_package = mean(epa, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = defense_personnel_group,
              values_from = def_epa_allowed_by_package,
              names_prefix = "def_epa_allowed_")

league_epa_mean <- mean(personnel_df$epa, na.rm = TRUE)
league_success  <- mean(personnel_df$epa > 0, na.rm = TRUE)

model_df <- personnel_df %>%
  left_join(def_pkg_rates, by = "def_team_season",
            relationship = "many-to-many") %>%
  left_join(off_split,     by = c("off_team_season", "offense_personnel_group"),
            relationship = "many-to-many") %>%
  left_join(off_strength,  by = c("off_team_season", "offense_personnel_group"),
            relationship = "many-to-many") %>%
  left_join(def_strength,  by = "def_team_season",
            relationship = "many-to-many") %>%
  mutate(
    across(starts_with("def_rate_"),        ~ coalesce(.x, 1/3)),
    off_rate_run             = coalesce(off_rate_run, 0.5),
    off_rate_pass            = coalesce(off_rate_pass, 0.5),
    off_epa_by_personnel     = coalesce(off_epa_by_personnel, league_epa_mean),
    off_success_by_personnel = coalesce(off_success_by_personnel, league_success),
    across(starts_with("def_epa_allowed_"), ~ coalesce(.x, league_epa_mean))
  )

model_df <- model_df %>%
  mutate(
    recency_weight = case_when(
      season_int == 2025 ~ 1.0,   # both train (Wk 1-9) and test (Wk 10-18)
      season_int == 2024 ~ 0.8,
      season_int == 2023 ~ 0.6,
      season_int == 2022 ~ 0.4,
      season_int == 2021 ~ 0.2,
      TRUE               ~ 0.2
    )
  )

train_df <- model_df %>%
  filter(season_int < 2025 | (season_int == 2025 & week <= 9))

test_df <- model_df %>%
  filter(season_int == 2025 & week >= 10)

cat("Training plays:", nrow(train_df), "\n")
cat("  of which 2025 Weeks 1-9:", sum(train_df$season_int == 2025), "\n")
cat("Test plays (2025 Weeks 10-18):", nrow(test_df), "\n")
cat("Test week range:", range(test_df$week), "\n")

scale_vars <- c(
  "down", "ydstogo", "yardline_100", "score_differential",
  "game_seconds_remaining", "half_seconds_remaining", "clock_stop",
  "def_rate_Base", "def_rate_Nickel", "def_rate_Dime",
  "off_rate_run", "off_rate_pass",
  "off_epa_by_personnel", "off_success_by_personnel",
  "def_epa_allowed_Base", "def_epa_allowed_Nickel", "def_epa_allowed_Dime"
)

train_means <- sapply(train_df[scale_vars], mean, na.rm = TRUE)
train_sds   <- sapply(train_df[scale_vars], sd,   na.rm = TRUE)

scale_apply <- function(df, means, sds, vars) {
  out <- df
  for (v in vars) {
    out[[v]] <- (out[[v]] - means[[v]]) / ifelse(sds[[v]] == 0, 1, sds[[v]])
  }
  out
}

train_df <- scale_apply(train_df, train_means, train_sds, scale_vars)
test_df  <- scale_apply(test_df,  train_means, train_sds, scale_vars)

write.csv(train_df, "C:/Users/asaar/Downloads/pbp_train.csv", row.names = FALSE)
write.csv(test_df,  "C:/Users/asaar/Downloads/pbp_test.csv",  row.names = FALSE)

cat("\nDone. Check Downloads for pbp_train.csv and pbp_test.csv\n")
cat("Train size:", round(file.size("C:/Users/asaar/Downloads/pbp_train.csv") / 1e6, 1), "MB\n")
cat("Test size:",  round(file.size("C:/Users/asaar/Downloads/pbp_test.csv")  / 1e6, 1), "MB\n")


