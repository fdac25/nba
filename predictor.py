import joblib
import numpy as np
import pandas as pd

df = pd.read_csv(
    "data_collection/scraped_data/combined_current_data/2025.csv", index_col=0
)

# Preprocessing

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values(by=["SEASON", "TEAM", "DATE"]).reset_index(drop=True)

for cat_col in ["HOME/AWAY", "WIN/LOSS"]:
    if cat_col in df.columns:
        df[cat_col] = df[cat_col].astype("category")

if "HOME/AWAY" in df:
    df["IS_HOME"] = (
        df["HOME/AWAY"]
        .astype(str)
        .str.upper()
        .map({"HOME": 1, "AWAY": 0})
        .fillna(0)
        .astype(int)
    )
else:
    df["IS_HOME"] = 0

if "WIN/LOSS" in df:
    df["WIN"] = (
        df["WIN/LOSS"]
        .astype(str)
        .str.upper()
        .map({"WIN": 1, "LOSS": 0})
        .fillna(0)
        .astype(int)
    )

# Takes scraped data and gets ELO and rolling averages for each team
ROLL_WINDOW = 5
ELO_K = 20.0
ELO_H = 65.0
ELO_BASE = 1500.0
SEASONAL_RESET = True


# implement elo ratings
def expected_score(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def add_elo_features(
    frame: pd.DataFrame, base_elo=1500.0, k=20.0, h=65.0, seasonal_reset=True
) -> pd.DataFrame:
    frame = frame.copy()
    frame["team_elo_pre"] = np.nan
    frame["opp_elo_pre"] = np.nan

    if seasonal_reset:
        season_iter = frame.groupby("SEASON", sort=False)
    else:
        season_iter = [("ALL", frame)]

    for _, sdf in season_iter:
        work = sdf.copy()
        # Pair the two rows for the same game using a robust key
        tmin = np.minimum(work["TEAM"].values, work["TEAM_OPP"].values)
        tmax = np.maximum(work["TEAM"].values, work["TEAM_OPP"].values)
        work["_game_key"] = list(zip(work["DATE"].values, tmin, tmax))

        elo = {}

        # Iterate games in chronological order
        for _, g in work.sort_values("DATE").groupby("_game_key", sort=False):
            if len(g) != 2:
                continue

            i, j = g.index[0], g.index[1]
            team_i, team_j = frame.loc[i, "TEAM"], frame.loc[j, "TEAM"]

            R_i = elo.get(team_i, base_elo)
            R_j = elo.get(team_j, base_elo)

            # Expected scores with home advantage applied only to expectation
            R_i_eff = R_i + (h if frame.loc[i, "IS_HOME"] == 1 else 0.0)
            R_j_eff = R_j + (h if frame.loc[j, "IS_HOME"] == 1 else 0.0)

            E_i = expected_score(R_i_eff, R_j_eff)

            # Store PRE-game Elos (no leakage)
            frame.at[i, "team_elo_pre"] = R_i
            frame.at[i, "opp_elo_pre"] = R_j
            frame.at[j, "team_elo_pre"] = R_j
            frame.at[j, "opp_elo_pre"] = R_i

            # Update ratings using actual results
            S_i = float(frame.loc[i, "WIN"])
            S_j = float(frame.loc[j, "WIN"])

            elo[team_i] = R_i + k * (S_i - E_i)
            elo[team_j] = R_j + k * (S_j - (1.0 - E_i))

    frame["elo_diff"] = frame["team_elo_pre"] - frame["opp_elo_pre"]
    return frame


df = add_elo_features(
    df, base_elo=ELO_BASE, k=ELO_K, h=ELO_H, seasonal_reset=SEASONAL_RESET
)

home_stats = [
    "PTS",
    "FG",
    "FGA",
    "FG_PCT",
    "FG3",
    "FG3A",
    "FG3_PCT",
    "FT",
    "FTA",
    "FT_PCT",
    "ORB",
    "DRB",
    "TRB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
]

away_stats = [
    "PTS_OPP",
    "FG_OPP",
    "FGA_OPP",
    "FG_PCT_OPP",
    "FG3_OPP",
    "FG3A_OPP",
    "FG3_PCT_OPP",
    "FT_OPP",
    "FTA_OPP",
    "FT_PCT_OPP",
    "ORB_OPP",
    "DRB_OPP",
    "TRB_OPP",
    "AST_OPP",
    "STL_OPP",
    "BLK_OPP",
    "TOV_OPP",
    "PF_OPP",
]

roll_cols = home_stats + away_stats


def add_rolling_features(frame: pd.DataFrame, cols, window=5):
    out = frame.copy()
    g = out.groupby(["SEASON", "TEAM"], group_keys=False)
    for c in cols:
        out[f"{c}_roll{window}"] = (
            g[c]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=window).mean())
            .values
        )
    return out


df = add_rolling_features(df, roll_cols, window=ROLL_WINDOW)

needed = [f"{c}_roll{ROLL_WINDOW}" for c in roll_cols] + [
    "team_elo_pre",
    "opp_elo_pre",
    "elo_diff",
]

df_model = df.dropna(subset=needed).copy()


# --- Function to build features for a new matchup ---
def create_matchup_features(df, home_team, away_team):
    """
    Creates a single-row feature DataFrame ready for model prediction.
    Only includes the selected columns you listed.
    """

    feature_cols = [c + "_roll5" for c in roll_cols] + ["IS_HOME", "elo_diff"]

    # Get latest available rolling averages for each team
    home_row = df[(df["TEAM"] == home_team)].sort_values("DATE").tail(1)
    away_row = df[(df["TEAM"] == away_team)].sort_values("DATE").tail(1)

    if home_row.empty or away_row.empty:
        raise ValueError(
            f"Not enough data for one or both teams: {home_team}, {away_team}"
        )

    features = {}

    # Copy all rolling stats from the *home team*
    for col in feature_cols:
        if col in ["IS_HOME", "elo_diff"]:
            continue
        features[col] = home_row[col].values[0]

    # Add IS_HOME = 1 (since the home team is the one we’re predicting from)
    features["IS_HOME"] = 1

    # Compute elo_diff using current pre-game elos
    home_elo = home_row["team_elo_pre"].values[0]
    away_elo = away_row["team_elo_pre"].values[0]
    features["elo_diff"] = home_elo - away_elo

    # Make sure the columns are in the right order
    X_pred = pd.DataFrame([features])[feature_cols]

    return X_pred


model = joblib.load("ml_models/ridge.pkl")


def predict_winner(home_team: str, away_team: str) -> bool:
    X_pred = create_matchup_features(df_model, home_team, away_team)

    home_win = model.predict(X_pred.values)

    return home_win
