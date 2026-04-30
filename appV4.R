# ============================================================
# NFL Defensive Personnel Optimizer — Shiny App v3
# ============================================================
# Setup:
#   1. Place app.R in any folder you like
#   2. Update the four file paths in Section 1 below
#   3. In RStudio: shiny::runApp("path/to/app.R")
# ============================================================

library(shiny)
library(brms)
library(dplyr)
library(tibble)
library(tidyr)
library(ggplot2)
library(jsonlite)

# ============================================================
# 1. Load models and data — UPDATE THESE PATHS
# ============================================================
fit_pass     <- readRDS("fit_pass_resaved.rds")
fit_epa_pass <- readRDS("fit_epa_pass_resaved.rds")
fit_epa_run  <- readRDS("fit_epa_run_resaved.rds")
data_list    <- readRDS("model_data.rds")

train_df    <- data_list$train_df
train_means <- data_list$train_means
train_sds   <- data_list$train_sds
scale_vars  <- data_list$scale_vars

# ── Build team tendency lookup tables from 2025 data ───────────────────────
# Used internally — no manual user input required.

team_tendencies <- train_df %>%
  filter(as.character(season) == "2025") %>%
  group_by(posteam, offense_personnel_group) %>%
  summarise(
    off_rate_run             = mean(play_type == "run",  na.rm = TRUE),
    off_rate_pass            = mean(play_type == "pass", na.rm = TRUE),
    off_epa_by_personnel     = mean(epa,                 na.rm = TRUE),
    off_success_by_personnel = mean(epa > 0,             na.rm = TRUE),
    .groups = "drop"
  )

def_tendencies <- train_df %>%
  filter(as.character(season) == "2025") %>%
  group_by(defteam) %>%
  summarise(
    def_rate_Base          = mean(defense_personnel_group == "Base",   na.rm = TRUE),
    def_rate_Nickel        = mean(defense_personnel_group == "Nickel", na.rm = TRUE),
    def_rate_Dime          = mean(defense_personnel_group == "Dime",   na.rm = TRUE),
    def_epa_allowed_Base   = mean(epa[defense_personnel_group == "Base"],   na.rm = TRUE),
    def_epa_allowed_Nickel = mean(epa[defense_personnel_group == "Nickel"], na.rm = TRUE),
    def_epa_allowed_Dime   = mean(epa[defense_personnel_group == "Dime"],   na.rm = TRUE),
    .groups = "drop"
  )

# League-average fallbacks used when a team/personnel combo has no 2025 data
league_off <- train_df %>%
  filter(as.character(season) == "2025") %>%
  group_by(offense_personnel_group) %>%
  summarise(
    off_rate_run             = mean(play_type == "run",  na.rm = TRUE),
    off_rate_pass            = mean(play_type == "pass", na.rm = TRUE),
    off_epa_by_personnel     = mean(epa,                 na.rm = TRUE),
    off_success_by_personnel = mean(epa > 0,             na.rm = TRUE),
    .groups = "drop"
  )

league_def <- tibble(
  def_rate_Base          = 0.33,
  def_rate_Nickel        = 0.50,
  def_rate_Dime          = 0.17,
  def_epa_allowed_Base   = 0.00,
  def_epa_allowed_Nickel = 0.00,
  def_epa_allowed_Dime   = 0.00
)

# Lookup helpers — fall back to league average if the team/personnel has no data
get_off_tendencies <- function(posteam_code, personnel) {
  row <- team_tendencies %>%
    filter(posteam == posteam_code,
           offense_personnel_group == personnel)
  if (nrow(row) == 1) return(row)
  league_off %>% filter(offense_personnel_group == personnel)
}

get_def_tendencies <- function(defteam_code) {
  row <- def_tendencies %>% filter(defteam == defteam_code)
  if (nrow(row) == 1) return(row)
  league_def
}

# ============================================================
# 2. NFL team reference table
# ============================================================
nfl_teams <- tibble(
  code = c("ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE",
           "DAL","DEN","DET","GB", "HOU","IND","JAX","KC",
           "LA", "LAC","LV", "MIA","MIN","NE", "NO", "NYG",
           "NYJ","PHI","PIT","SEA","SF", "TB", "TEN","WAS"),
  name = c(
    "Arizona Cardinals","Atlanta Falcons","Baltimore Ravens",
    "Buffalo Bills","Carolina Panthers","Chicago Bears",
    "Cincinnati Bengals","Cleveland Browns","Dallas Cowboys",
    "Denver Broncos","Detroit Lions","Green Bay Packers",
    "Houston Texans","Indianapolis Colts","Jacksonville Jaguars",
    "Kansas City Chiefs","Los Angeles Rams","Los Angeles Chargers",
    "Las Vegas Raiders","Miami Dolphins","Minnesota Vikings",
    "New England Patriots","New Orleans Saints","New York Giants",
    "New York Jets","Philadelphia Eagles","Pittsburgh Steelers",
    "Seattle Seahawks","San Francisco 49ers","Tampa Bay Buccaneers",
    "Tennessee Titans","Washington Commanders"
  ),
  slug = c(
    "ari","atl","bal","buf","car","chi","cin","cle",
    "dal","den","det","gb", "hou","ind","jax","kc",
    "lar","lac","lv", "mia","min","ne", "no", "nyg",
    "nyj","phi","pit","sea","sf", "tb", "ten","wsh"
  )
) %>%
  mutate(logo = paste0(
    "https://a.espncdn.com/i/teamlogos/nfl/500/", slug, ".png"
  ))

logo_map <- setNames(nfl_teams$logo, nfl_teams$code)

# ============================================================
# 3. Engine helper functions
# ============================================================

scale_apply <- function(df, means, sds, vars) {
  out <- df
  for (v in vars) {
    out[[v]] <- (out[[v]] - means[[v]]) / ifelse(sds[[v]] == 0, 1, sds[[v]])
  }
  out
}

classify_strength <- function(prob_best, edge_lower80, expected_edge) {
  if      (prob_best >= 0.75 && edge_lower80 > 0)     "Strong Recommend"
  else if (prob_best >= 0.55 || expected_edge > 0.02) "Recommend"
  else                                                 "Light Recommend"
}

prepare_scenario <- function(
    season, posteam, defteam, down, ydstogo, yardline_100,
    score_differential, qtr, game_seconds_remaining,
    half_seconds_remaining, offense_personnel_group,
    clock_stop               = 0,
    def_rate_Base            = 0.33,
    def_rate_Nickel          = 0.50,
    def_rate_Dime            = 0.17,
    off_rate_run             = 0.50,
    off_rate_pass            = 0.50,
    off_epa_by_personnel     = 0.00,
    off_success_by_personnel = 0.45,
    def_epa_allowed_Base     = 0.00,
    def_epa_allowed_Nickel   = 0.00,
    def_epa_allowed_Dime     = 0.00
) {
  packages       <- c("Base", "Nickel", "Dime")
  season_int     <- as.integer(season)
  matchup_season <- ifelse(season_int >= 2022, as.character(season), "pre2022")

  base_row <- tibble(
    season                   = factor(season,  levels = levels(train_df$season)),
    posteam                  = factor(posteam, levels = levels(train_df$posteam)),
    defteam                  = factor(defteam, levels = levels(train_df$defteam)),
    off_team_season          = factor(paste0(posteam, "_", season),
                                      levels = levels(train_df$off_team_season)),
    def_team_season          = factor(paste0(defteam, "_", season),
                                      levels = levels(train_df$def_team_season)),
    down                     = down,
    ydstogo                  = ydstogo,
    yardline_100             = yardline_100,
    score_differential       = score_differential,
    qtr                      = factor(qtr, levels = levels(train_df$qtr)),
    game_seconds_remaining   = game_seconds_remaining,
    half_seconds_remaining   = half_seconds_remaining,
    clock_stop               = clock_stop,
    offense_personnel_group  = factor(offense_personnel_group,
                                      levels = levels(train_df$offense_personnel_group)),
    def_rate_Base            = def_rate_Base,
    def_rate_Nickel          = def_rate_Nickel,
    def_rate_Dime            = def_rate_Dime,
    off_rate_run             = off_rate_run,
    off_rate_pass            = off_rate_pass,
    off_epa_by_personnel     = off_epa_by_personnel,
    off_success_by_personnel = off_success_by_personnel,
    def_epa_allowed_Base     = def_epa_allowed_Base,
    def_epa_allowed_Nickel   = def_epa_allowed_Nickel,
    def_epa_allowed_Dime     = def_epa_allowed_Dime
  )

  scenario_df <- bind_rows(lapply(packages, function(pkg) {
    base_row %>% mutate(
      defense_personnel_group = factor(pkg, levels = c("Base","Nickel","Dime")),
      matchup_cell = factor(
        paste(paste0(defteam, "_", matchup_season),
              offense_personnel_group, pkg, sep = ":"),
        levels = levels(train_df$matchup_cell)
      ),
      off_personnel_team_season = factor(
        paste(paste0(posteam, "_", season),
              offense_personnel_group, sep = ":"),
        levels = levels(train_df$off_personnel_team_season)
      )
    )
  }))

  scale_apply(scenario_df, train_means, train_sds, scale_vars)
}

recommend_personnel <- function(scenario_df) {
  packages <- as.character(scenario_df$defense_personnel_group)

  p_pass_draws   <- posterior_epred(fit_pass,     newdata = scenario_df,
                                    allow_new_levels = TRUE)
  epa_pass_draws <- posterior_epred(fit_epa_pass, newdata = scenario_df,
                                    allow_new_levels = TRUE)
  epa_run_draws  <- posterior_epred(fit_epa_run,  newdata = scenario_df,
                                    allow_new_levels = TRUE)

  marginal <- p_pass_draws * epa_pass_draws + (1 - p_pass_draws) * epa_run_draws
  colnames(marginal)       <- packages
  colnames(p_pass_draws)   <- packages
  colnames(epa_pass_draws) <- packages
  colnames(epa_run_draws)  <- packages

  summary_tbl <- tibble(
    package       = packages,
    p_pass_mean   = colMeans(p_pass_draws),
    epa_pass_mean = colMeans(epa_pass_draws),
    epa_run_mean  = colMeans(epa_run_draws),
    epa_mean      = colMeans(marginal),
    epa_sd        = apply(marginal, 2, sd),
    epa_lower80   = apply(marginal, 2, quantile, 0.10),
    epa_median    = apply(marginal, 2, quantile, 0.50),
    epa_upper80   = apply(marginal, 2, quantile, 0.90),
    epa_lower95   = apply(marginal, 2, quantile, 0.025),
    epa_upper95   = apply(marginal, 2, quantile, 0.975)
  )

  best_idx  <- apply(marginal, 1, which.min)
  prob_best <- tibble(
    package   = packages,
    prob_best = sapply(seq_along(packages), function(i) mean(best_idx == i))
  )

  summary_tbl <- summary_tbl %>%
    left_join(prob_best, by = "package") %>%
    arrange(epa_mean)

  best_pkg   <- summary_tbl$package[1]
  second_pkg <- summary_tbl$package[2]
  edge_draws <- marginal[, second_pkg] - marginal[, best_pkg]

  list(
    summary          = summary_tbl,
    recommendation   = best_pkg,
    probability_best = summary_tbl$prob_best[1],
    expected_edge    = mean(edge_draws),
    edge_lower80     = quantile(edge_draws, 0.10),
    edge_upper80     = quantile(edge_draws, 0.90),
    draws            = marginal
  )
}

# ============================================================
# 4. Constants
# ============================================================
pkg_colors <- c(Base = "#1F4E79", Nickel = "#C55A11", Dime = "#375623")

# ============================================================
# 5. UI
# ============================================================
ui <- fluidPage(
  title = "NFL Personnel Optimizer",

  tags$head(tags$style(HTML("
    body { background:#F0F4F8; font-family:'Segoe UI',Arial,sans-serif; margin:0; }

    .app-header {
      background:#1F3864; color:white;
      padding:20px 32px; margin-bottom:22px;
    }
    .app-header h2 { margin:0; font-size:22px; font-weight:700; }
    .app-header p  { margin:5px 0 0; font-size:13px; opacity:.75; }

    .panel-card {
      background:white; border-radius:8px;
      border:1px solid #D6E4F0;
      padding:18px 20px; margin-bottom:14px;
      box-shadow:0 1px 3px rgba(0,0,0,.05);
    }
    .panel-title {
      font-size:11px; font-weight:700; color:#1F4E79;
      text-transform:uppercase; letter-spacing:1px;
      border-bottom:2px solid #BDD7EE;
      padding-bottom:7px; margin-bottom:14px;
    }

    .field-note { font-size:11px; color:#8899AA; margin:-2px 0 8px; display:block; }

    .team-wrap { position:relative; margin-bottom:14px; }
    .team-logo {
      position:absolute; left:10px; top:50%;
      transform:translateY(-50%);
      width:28px; height:28px; object-fit:contain;
      z-index:5; pointer-events:none;
    }
    .team-wrap select {
      width:100%; padding:8px 10px 8px 46px;
      height:42px; font-size:13px;
      border:1.5px solid #BDD7EE; border-radius:6px;
      background:white; cursor:pointer;
      -webkit-appearance:none; appearance:none;
    }
    .team-wrap select:focus { outline:none; border-color:#2E75B6; }

    .rec-box {
      border-radius:10px; padding:20px 24px;
      margin-bottom:14px; text-align:center;
    }
    .rec-pkg  { font-size:46px; font-weight:800; line-height:1; }
    .rec-conf { font-size:15px; font-weight:600; margin-top:6px; }
    .rec-prob { font-size:13px; margin-top:3px; opacity:.8; }

    .strong    { background:#E2EFDA; border:2px solid #375623; color:#375623; }
    .recommend { background:#FFF2CC; border:2px solid #BF8F00; color:#7F5800; }
    .light-rec { background:#DEEAF9; border:2px solid #2E75B6; color:#1F4E79; }

    .metric-row  { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:4px; }
    .metric-card {
      flex:1; min-width:110px;
      background:#F8FBFF; border:1px solid #D6E4F0;
      border-radius:7px; padding:10px 12px; text-align:center;
    }
    .metric-label { font-size:10px; color:#6B7C99; font-weight:700;
                    text-transform:uppercase; letter-spacing:.5px; }
    .metric-val   { font-size:19px; font-weight:700; color:#1F3864; margin-top:2px; }
    .metric-sub   { font-size:10px; color:#8899AA; margin-top:1px; }

    .pkg-row {
      display:flex; align-items:center;
      justify-content:space-between;
      padding:10px 14px; border-radius:7px;
      margin-bottom:8px;
    }
    .pkg-name  { font-weight:700; font-size:15px; }
    .pkg-epa   { font-size:14px; font-weight:600; }
    .pkg-stats { font-size:11px; color:#6B7C99; }
    .pkg-badge {
      font-size:10px; font-weight:700; color:white;
      padding:3px 9px; border-radius:20px; margin-left:8px;
    }

    .run-btn {
      width:100%; padding:13px; font-size:15px; font-weight:700;
      background:#1F3864; color:white; border:none;
      border-radius:7px; cursor:pointer; letter-spacing:.4px;
      transition:background .2s;
    }
    .run-btn:hover { background:#2E75B6; }

    .placeholder { text-align:center; padding:60px 20px; color:#8899AA; }
    .placeholder-icon { font-size:52px; margin-bottom:12px; }

    .form-control {
      border:1.5px solid #BDD7EE !important;
      border-radius:6px !important; font-size:13px !important;
    }
    .form-control:focus {
      border-color:#2E75B6 !important; box-shadow:none !important;
    }
    label { font-size:12px; font-weight:600; color:#1F4E79; }
  "))),

  tags$div(class = "app-header",
    tags$h2("\U0001F3C8  NFL Defensive Personnel Optimizer"),
    tags$p("Bayesian Hierarchical Model  \u2014  2025 Season  \u2014  Built by Asa Arnold")
  ),

  fluidRow(

    # ── LEFT: inputs ──────────────────────────────────────────
    column(4,

      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "Matchup"),

        tags$label("Offense", `for` = "posteam"),
        tags$div(class = "team-wrap",
          tags$img(id = "off_logo", class = "team-logo",
                   src = logo_map[["LA"]]),
          tags$select(id = "posteam",
            lapply(seq_len(nrow(nfl_teams)), function(i)
              tags$option(value = nfl_teams$code[i],
                          selected = if (nfl_teams$code[i] == "LA") "" else NULL,
                          nfl_teams$name[i])
            )
          )
        ),

        tags$label("Defense", `for` = "defteam"),
        tags$div(class = "team-wrap",
          tags$img(id = "def_logo", class = "team-logo",
                   src = logo_map[["MIN"]]),
          tags$select(id = "defteam",
            lapply(seq_len(nrow(nfl_teams)), function(i)
              tags$option(value = nfl_teams$code[i],
                          selected = if (nfl_teams$code[i] == "MIN") "" else NULL,
                          nfl_teams$name[i])
            )
          )
        ),

        # Sync native selects to Shiny and update logos
        tags$script(HTML(sprintf("
          var LOGOS = %s;
          function bindTeam(selId, imgId, shinId) {
            var sel = document.getElementById(selId);
            var img = document.getElementById(imgId);
            if (!sel || !img) return;
            Shiny.setInputValue(shinId, sel.value, {priority:'event'});
            sel.addEventListener('change', function() {
              img.src = LOGOS[this.value] || '';
              Shiny.setInputValue(shinId, this.value, {priority:'event'});
            });
          }
          $(document).on('shiny:connected', function() {
            bindTeam('posteam','off_logo','posteam');
            bindTeam('defteam','def_logo','defteam');
          });
        ", toJSON(logo_map))))
      ),

      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "Game Situation"),

        fluidRow(
          column(6,
            selectInput("down", "Down",
                        choices  = c("1st"=1,"2nd"=2,"3rd"=3,"4th"=4),
                        selected = 1)
          ),
          column(6,
            numericInput("ydstogo", "Distance (yds)",
                         value = 10, min = 1, max = 30)
          )
        ),

        fluidRow(
          column(6,
            numericInput("yardline_100", "Yards to End Zone",
                         value = 30, min = 1, max = 99)
          ),
          column(6,
            selectInput("qtr", "Quarter",
                        choices  = c("Q1"=1,"Q2"=2,"Q3"=3,"Q4"=4,"OT"=5),
                        selected = 4)
          )
        ),

        numericInput("score_differential",
                     "Score Differential (offense POV)",
                     value = 7, min = -50, max = 50),
        tags$span(class = "field-note",
                  "Positive = offense leading. E.g. +7 = offense up by 7."),

        fluidRow(
          column(6,
            numericInput("game_secs", "Game Secs Remaining",
                         value = 395, min = 0, max = 3600)
          ),
          column(6,
            numericInput("half_secs", "Half Secs Remaining",
                         value = 395, min = 0, max = 1800)
          )
        )
      ),

      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "Offensive Personnel"),
        selectInput("off_personnel", "Personnel Group",
                    choices = c(
                      "11  \u2014 1 RB 1 TE 3 WR" = "11",
                      "12  \u2014 1 RB 2 TE 2 WR" = "12",
                      "13  \u2014 1 RB 3 TE 1 WR" = "13",
                      "21  \u2014 2 RB 1 TE 2 WR" = "21",
                      "22  \u2014 2 RB 2 TE 1 WR" = "22",
                      "Other"                      = "Other"
                    ),
                    selected = "13")
      ),

      tags$button(
        id    = "run_btn",
        class = "run-btn action-button",
        type  = "button",
        "\u25B6  Get Recommendation"
      )
    ),

    # ── RIGHT: results ────────────────────────────────────────
    column(8,
      uiOutput("result_ui")
    )
  )
)

# ============================================================
# 6. Server
# ============================================================
server <- function(input, output, session) {

  result <- eventReactive(input$run_btn, {
    req(input$posteam, input$defteam)

    withProgress(message = "Running Bayesian model\u2026", value = 0, {
      incProgress(0.15, detail = "Building scenario")

      # Look up all team tendencies directly from 2025 data
      off_t <- get_off_tendencies(input$posteam, input$off_personnel)
      def_t <- get_def_tendencies(input$defteam)

      sc <- prepare_scenario(
        season                   = "2025",
        posteam                  = input$posteam,
        defteam                  = input$defteam,
        down                     = as.integer(input$down),
        ydstogo                  = input$ydstogo,
        yardline_100             = input$yardline_100,
        score_differential       = input$score_differential,
        qtr                      = as.character(input$qtr),
        game_seconds_remaining   = input$game_secs,
        half_seconds_remaining   = input$half_secs,
        offense_personnel_group  = input$off_personnel,
        off_rate_run             = off_t$off_rate_run,
        off_rate_pass            = off_t$off_rate_pass,
        off_epa_by_personnel     = off_t$off_epa_by_personnel,
        off_success_by_personnel = off_t$off_success_by_personnel,
        def_rate_Base            = def_t$def_rate_Base,
        def_rate_Nickel          = def_t$def_rate_Nickel,
        def_rate_Dime            = def_t$def_rate_Dime,
        def_epa_allowed_Base     = def_t$def_epa_allowed_Base,
        def_epa_allowed_Nickel   = def_t$def_epa_allowed_Nickel,
        def_epa_allowed_Dime     = def_t$def_epa_allowed_Dime
      )

      incProgress(0.60, detail = "Sampling from posterior (30-60 sec)")
      res <- recommend_personnel(sc)

      incProgress(0.25, detail = "Preparing output")
      res
    })
  })

  output$result_ui <- renderUI({

    if (input$run_btn == 0) {
      return(tags$div(class = "panel-card placeholder",
        tags$div(class = "placeholder-icon", "\U0001F3C8"),
        tags$p("Set up the situation on the left"),
        tags$p(tags$strong("then click Get Recommendation"))
      ))
    }

    res <- result()

    strength  <- classify_strength(res$probability_best,
                                   res$edge_lower80,
                                   res$expected_edge)
    box_class <- switch(strength,
      "Strong Recommend" = "strong",
      "Recommend"        = "recommend",
      "Light Recommend"  = "light-rec"
    )
    stars <- switch(strength,
      "Strong Recommend" = "\u2605\u2605\u2605",
      "Recommend"        = "\u2605\u2605",
      "Light Recommend"  = "\u2605"
    )

    best_row   <- res$summary[1, ]
    off_name   <- nfl_teams$name[nfl_teams$code == input$posteam]
    def_name   <- nfl_teams$name[nfl_teams$code == input$defteam]
    off_logo   <- logo_map[[input$posteam]]
    def_logo   <- logo_map[[input$defteam]]
    down_label <- c("1"="1st","2"="2nd","3"="3rd","4"="4th")[input$down]

    tagList(

      # Matchup header
      tags$div(class = "panel-card",
        tags$div(style = "display:flex;align-items:center;justify-content:center;gap:28px;",
          tags$div(style = "text-align:center;",
            tags$img(src = off_logo, height = "56px",
                     style = "object-fit:contain;display:block;margin:0 auto;"),
            tags$p(off_name,
                   style = "font-size:12px;color:#6B7C99;margin:6px 0 0;font-weight:600;")
          ),
          tags$div(style = "font-size:20px;font-weight:700;color:#BDD7EE;", "vs"),
          tags$div(style = "text-align:center;",
            tags$img(src = def_logo, height = "56px",
                     style = "object-fit:contain;display:block;margin:0 auto;"),
            tags$p(def_name,
                   style = "font-size:12px;color:#6B7C99;margin:6px 0 0;font-weight:600;")
          )
        ),
        tags$hr(style = "border:none;border-top:1px solid #E8EFF7;margin:12px 0;"),
        tags$p(
          style = "text-align:center;font-size:13px;color:#6B7C99;margin:0;",
          sprintf(
            "%s & %d  \u2022  %d yds to end zone  \u2022  Q%s  \u2022  %s%d score  \u2022  %s personnel",
            down_label, input$ydstogo, input$yardline_100, input$qtr,
            ifelse(input$score_differential >= 0, "+", ""),
            input$score_differential, input$off_personnel
          )
        ),
        tags$p(
          style = "text-align:center;font-size:12px;color:#8899AA;margin:4px 0 0;",
          sprintf("%d:%02d remaining in game",
                  input$game_secs %/% 60, input$game_secs %% 60)
        )
      ),

      # Recommendation badge
      tags$div(class = paste("rec-box", box_class),
        tags$div(class = "rec-pkg", res$recommendation),
        tags$div(class = "rec-conf", paste(stars, strength)),
        tags$div(class = "rec-prob",
          sprintf("%.1f%% probability this is the optimal package",
                  res$probability_best * 100))
      ),

      # Four metric cards
      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "Key Numbers"),
        tags$div(class = "metric-row",

          tags$div(class = "metric-card",
            tags$div(class = "metric-label", "Expected EPA"),
            tags$div(class = "metric-val",
                     style = paste0("color:",
                       ifelse(best_row$epa_mean < 0, "#375623", "#A32D2D")),
                     sprintf("%+.3f", best_row$epa_mean)),
            tags$div(class = "metric-sub", "lower = better for D")
          ),

          tags$div(class = "metric-card",
            tags$div(class = "metric-label", "80% Interval"),
            tags$div(class = "metric-val", style = "font-size:14px;",
                     sprintf("[%+.3f, %+.3f]",
                             best_row$epa_lower80, best_row$epa_upper80)),
            tags$div(class = "metric-sub", "posterior uncertainty")
          ),

          tags$div(class = "metric-card",
            tags$div(class = "metric-label", "P(Pass) if shown"),
            tags$div(class = "metric-val",
                     sprintf("%.1f%%", best_row$p_pass_mean * 100)),
            tags$div(class = "metric-sub", "offense reaction estimate")
          ),

          tags$div(class = "metric-card",
            tags$div(class = "metric-label", "Edge vs #2"),
            tags$div(class = "metric-val",
                     style = paste0("color:",
                       ifelse(res$expected_edge > 0, "#375623", "#A32D2D")),
                     sprintf("%+.3f", res$expected_edge)),
            tags$div(class = "metric-sub", "EPA over second-best")
          )
        )
      ),

      # All packages comparison
      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "All Package Comparison"),

        lapply(seq_len(nrow(res$summary)), function(i) {
          row     <- res$summary[i, ]
          pkg     <- row$package
          is_best <- (i == 1)

          tags$div(
            class = "pkg-row",
            style = paste0(
              "background:", if (is_best) "#F5FAFF" else "white", ";",
              "border:", if (is_best)
                paste0("2px solid ", pkg_colors[[pkg]])
              else "1px solid #E8EEF5", ";"
            ),

            tags$div(style = "display:flex;align-items:center;",
              tags$span(class = "pkg-name",
                        style = paste0("color:", pkg_colors[[pkg]], ";"),
                        pkg),
              if (is_best)
                tags$span(class = "pkg-badge",
                          style = paste0("background:", pkg_colors[[pkg]], ";"),
                          "RECOMMENDED")
            ),

            tags$div(style = "text-align:center;",
              tags$div(class = "pkg-epa",
                       style = paste0("color:",
                         ifelse(row$epa_mean < 0, "#375623", "#A32D2D")),
                       sprintf("%+.3f EPA", row$epa_mean)),
              tags$div(class = "pkg-stats",
                       sprintf("[%+.3f, %+.3f]",
                               row$epa_lower80, row$epa_upper80))
            ),

            tags$div(style = "text-align:right;",
              tags$div(style = "font-size:13px;font-weight:600;color:#1F3864;",
                       sprintf("%.1f%% pass", row$p_pass_mean * 100)),
              tags$div(class = "pkg-stats",
                       sprintf("P(best): %.1f%%", row$prob_best * 100))
            )
          )
        })
      ),

      # Posterior density plot
      tags$div(class = "panel-card",
        tags$div(class = "panel-title", "EPA Posterior Distributions"),
        tags$span(class = "field-note",
          "Highlighted = recommended package. Dashed lines = posterior means."),

        renderPlot({
          draws_df <- as.data.frame(res$draws) %>%
            pivot_longer(everything(),
                         names_to  = "package",
                         values_to = "epa") %>%
            mutate(
              package = factor(package, levels = c("Base","Nickel","Dime")),
              is_best = (package == res$recommendation)
            )

          ggplot(draws_df, aes(x = epa, fill = package, alpha = is_best)) +
            geom_density(color = NA) +
            geom_vline(
              data     = res$summary,
              aes(xintercept = epa_mean,
                  color = factor(package, levels = c("Base","Nickel","Dime"))),
              linetype = "dashed", linewidth = 1.0
            ) +
            scale_fill_manual(values = pkg_colors, name = NULL) +
            scale_color_manual(values = pkg_colors, name = NULL) +
            scale_alpha_manual(values = c("TRUE"=0.55,"FALSE"=0.18),
                               guide  = "none") +
            labs(
              x = "Expected Points Added  (negative = better for defense)",
              y = "Posterior density"
            ) +
            theme_bw(base_size = 13) +
            theme(
              legend.position  = "top",
              legend.key.size  = unit(0.5, "cm"),
              panel.grid.minor = element_blank(),
              plot.background  = element_rect(fill = "white", color = NA),
              panel.background = element_rect(fill = "white")
            )
        }, height = 270, bg = "white")
      )
    )
  })
}

# ============================================================
# 7. Launch
# ============================================================
shinyApp(ui, server)
