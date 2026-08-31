import io
import re
import requests
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
START_YEAR = 2016
END_YEAR = 2025
SEASONS = list(range(START_YEAR, END_YEAR + 1))

# Official nflverse curated repositories for historical games and injuries
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
INJURY_URL_TEMPLATE = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv"

NAME_TO_ABBR = {
    "arizona cardinals": "ARI", "cardinals": "ARI", "ari": "ARI", "az": "ARI",
    "atlanta falcons": "ATL", "falcons": "ATL", "atl": "ATL",
    "baltimore ravens": "BAL", "ravens": "BAL", "bal": "BAL",
    "buffalo bills": "BUF", "bills": "BUF", "buf": "BUF",
    "carolina panthers": "CAR", "panthers": "CAR", "car": "CAR",
    "chicago bears": "CHI", "bears": "CHI", "chi": "CHI",
    "cincinnati bengals": "CIN", "bengals": "CIN", "cin": "CIN",
    "cleveland browns": "CLE", "browns": "CLE", "cle": "CLE",
    "dallas cowboys": "DAL", "cowboys": "DAL", "dal": "DAL",
    "denver broncos": "DEN", "broncos": "DEN", "den": "DEN",
    "detroit lions": "DET", "lions": "DET", "det": "DET",
    "green bay packers": "GB", "packers": "GB", "gb": "GB",
    "houston texans": "HOU", "texans": "HOU", "hou": "HOU",
    "indianapolis colts": "IND", "colts": "IND", "ind": "IND",
    "jacksonville jaguars": "JAX", "jaguars": "JAX", "jax": "JAX",
    "kansas city chiefs": "KC", "chiefs": "KC", "kc": "KC",
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV", "oak": "LV", "oakland raiders": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC", "sd": "LAC", "san diego chargers": "LAC",
    "los angeles rams": "LAR", "rams": "LAR", "lar": "LAR", "la": "LAR", "st. louis rams": "LAR",
    "miami dolphins": "MIA", "dolphins": "MIA", "mia": "MIA",
    "minnesota vikings": "MIN", "vikings": "MIN", "min": "MIN",
    "new england patriots": "NE", "patriots": "NE", "ne": "NE",
    "new orleans saints": "NO", "saints": "NO", "no": "NO",
    "new york giants": "NYG", "giants": "NYG", "nyg": "NYG",
    "new york jets": "NYJ", "jets": "NYJ", "nyj": "NYJ",
    "philadelphia eagles": "PHI", "eagles": "PHI", "phi": "PHI",
    "pittsburgh steelers": "PIT", "steelers": "PIT", "pit": "PIT",
    "san francisco 49ers": "SF", "49ers": "SF", "sf": "SF",
    "seattle seahawks": "SEA", "seahawks": "SEA", "sea": "SEA",
    "tampa bay buccaneers": "TB", "buccaneers": "TB", "tb": "TB",
    "tennessee titans": "TEN", "titans": "TEN", "ten": "TEN",
    "washington commanders": "WAS", "commanders": "WAS", "was": "WAS",
    "washington football team": "WAS", "washington redskins": "WAS"
}

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def fetch_historical_games():
    print(f"Fetching games dataset across seasons {START_YEAR}-{END_YEAR}...")
    res = requests.get(GAMES_URL, timeout=30)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch games database. HTTP {res.status_code}")
    
    df = pd.read_csv(io.StringIO(res.text), low_memory=False)
    df = df[(df["season"].isin(SEASONS)) & (df["game_type"] == "REG")].copy()
    
    df["home_team"] = df["home_team"].apply(team_to_abbr)
    df["away_team"] = df["away_team"].apply(team_to_abbr)
    
    # Calculate winner and point differential
    df["point_differential"] = df["home_score"] - df["away_score"]
    df["winner"] = np.where(df["home_score"] > df["away_score"], df["home_team"],
                            np.where(df["home_score"] < df["away_score"], df["away_team"], "TIE"))
    
    # Standardize spread: standard negative betting notation for favorites
    df["home_closing_spread"] = -df["spread_line"]
    df["away_closing_spread"] = df["spread_line"]
    
    return df

def fetch_historical_injuries():
    print("Scraping weekly injury report summaries...")
    injury_records = []
    
    for year in SEASONS:
        url = INJURY_URL_TEMPLATE.format(year=year)
        try:
            res = requests.get(url, timeout=20)
            if res.status_code == 200:
                inj_df = pd.read_csv(io.StringIO(res.text), low_memory=False)
                # Filter down to notable reportable game statuses
                if "report_status" in inj_df.columns:
                    inj_df = inj_df[inj_df["report_status"].isin(["Out", "Doubtful", "Questionable"])].copy()
                    inj_df["team"] = inj_df["team"].apply(team_to_abbr)
                    
                    # Aggregate injury summary per team-week
                    grouped = inj_df.groupby(["season", "week", "team"]).apply(
                        lambda g: "; ".join(f"{r.get('full_name', r.get('player', 'Player'))} ({r.get('position', 'POS')}: {r.get('report_status')})" 
                                            for _, r in g.iterrows())
                    ).reset_index(name="key_injuries_reported")
                    
                    injury_records.append(grouped)
        except Exception as e:
            print(f"Notice: Could not parse injury data for {year}: {e}")
            
    if injury_records:
        return pd.concat(injury_records, ignore_index=True)
    return pd.DataFrame(columns=["season", "week", "team", "key_injuries_reported"])

def build_consolidated_dataset():
    games_df = fetch_historical_games()
    injuries_df = fetch_historical_injuries()
    
    # Merge Home injuries
    if not injuries_df.empty:
        games_df = games_df.merge(
            injuries_df, 
            left_on=["season", "week", "home_team"], 
            right_on=["season", "week", "team"], 
            how="left"
        ).rename(columns={"key_injuries_reported": "home_injuries"}).drop(columns=["team"], errors="ignore")
        
        # Merge Away injuries
        games_df = games_df.merge(
            injuries_df, 
            left_on=["season", "week", "away_team"], 
            right_on=["season", "week", "team"], 
            how="left"
        ).rename(columns={"key_injuries_reported": "away_injuries"}).drop(columns=["team"], errors="ignore")
    else:
        games_df["home_injuries"] = "None reported"
        games_df["away_injuries"] = "None reported"

    # Select and organize primary columns for backtesting & training
    export_columns = [
        "season", "week", "game_id", "gameday", "weekday", "gametime",
        "home_team", "away_team", "home_score", "away_score", "winner", "point_differential",
        "home_closing_spread", "away_closing_spread", "total_line",
        "home_moneyline", "away_moneyline", "home_rest", "away_rest",
        "temp", "wind", "roof", "surface",
        "home_injuries", "away_injuries"
    ]
    
    available_cols = [c for c in export_columns if c in games_df.columns]
    final_df = games_df[available_cols].sort_values(by=["season", "week", "gameday"]).reset_index(drop=True)
    
    output_filename = "nfl_10yr_historical_data_2016_2025.csv"
    final_df.to_csv(output_filename, index=False)
    print(f"\n Dataset successfully generated: '{output_filename}' ({len(final_df)} regular season games).")

if __name__ == "__main__":
    build_consolidated_dataset()
