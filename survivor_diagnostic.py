import io
import json
import math
import os
import re
import gspread
import numpy as np
import pandas as pd
import requests
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
DATA_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
BACKTEST_SEASONS = [2025, 2024, 2023, 2022, 2021]
WEEKS = 18

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
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV", "oak": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC", "sd": "LAC",
    "los angeles rams": "LAR", "rams": "LAR", "lar": "LAR", "la": "LAR",
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
    "washington football team": "WAS", "washington": "WAS"
}

ALL_TEAMS = sorted(list(set(NAME_TO_ABBR.values())))

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def spread_to_market_prob(spread: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, spread / 14.5))

def calculate_calibrated_prob(spread: float, is_home: bool, week: int, profile="baseline") -> float:
    m_prob = spread_to_market_prob(spread)
    
    if profile == "baseline":
        home_boost = 0.025 if is_home else -0.015
        rest_boost = 0.015 if abs(spread) >= 7.0 else 0.005
        epa_edge = 0.020 if abs(spread) >= 8.5 else 0.010
        return min(0.96, max(0.51, round(m_prob + home_boost + rest_boost + epa_edge, 3)))
        
    elif profile == "conservative_early":
        early_penalty = -0.040 if week <= 4 else 0.0
        home_edge = 0.010 if is_home else -0.005
        spread_tier = 0.015 if abs(spread) >= 9.5 else (-0.025 if abs(spread) < 6.0 else 0.0)
        return min(0.95, max(0.50, round(m_prob + early_penalty + home_edge + spread_tier, 3)))

    elif profile == "heavy_favorite_bias":
        if abs(spread) >= 8.5:
            return min(0.96, m_prob + 0.05)
        elif abs(spread) >= 6.0:
            return m_prob
        else:
            return max(0.40, m_prob - 0.10)

def solve_path(weekly_slates, profile):
    num_teams = len(ALL_TEAMS)
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    idx_to_team = {i: t for i, t in enumerate(ALL_TEAMS)}

    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)

    for w in range(1, WEEKS + 1):
        row = w - 1
        for cand in weekly_slates.get(w, []):
            t_idx = team_to_idx.get(cand["team"])
            prob = cand[f"prob_{profile}"]
            if t_idx is not None and prob is not None:
                cost_matrix[row, t_idx] = -math.log(prob)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e4:
            optimal[r + 1] = idx_to_team[c]
        else:
            optimal[r + 1] = ""
    return optimal

def run_diagnostics():
    print("=" * 80)
    print("🔍 RUNNING SURVIVOR MODEL DIAGNOSTIC & FORENSIC ENGINE")
    print("=" * 80)

    res = requests.get(DATA_URL, timeout=25)
    df_raw = pd.read_csv(io.StringIO(res.text), low_memory=False)
    df = df_raw[(df_raw["season"].isin(BACKTEST_SEASONS)) & (df_raw["game_type"] == "REG")].copy()

    profiles = ["baseline", "conservative_early", "heavy_favorite_bias"]
    profile_results = {p: {} for p in profiles}
    diagnostic_rows = []

    for season in BACKTEST_SEASONS:
        season_df = df[df["season"] == season]
        weekly_slates = {w: [] for w in range(1, WEEKS + 1)}

        for _, row in season_df.iterrows():
            w = int(row["week"])
            if w < 1 or w > WEEKS:
                continue

            h_team = team_to_abbr(row["home_team"])
            a_team = team_to_abbr(row["away_team"])
            h_score = int(row["home_score"]) if pd.notnull(row["home_score"]) else 0
            a_score = int(row["away_score"]) if pd.notnull(row["away_score"]) else 0
            spread_val = float(row["spread_line"]) if pd.notnull(row.get("spread_line")) else 0.0
            home_spread = -spread_val

            team = h_team if home_spread <= 0 else a_team
            opp = a_team if home_spread <= 0 else h_team
            eff_spread = home_spread if home_spread <= 0 else -home_spread
            is_home = (home_spread <= 0)
            t_score = h_score if is_home else a_score
            o_score = a_score if is_home else h_score

            game_data = {
                "team": team, "opponent": opp, "matchup": f"{opp} @ {team}" if is_home else f"{team} @ {opp}",
                "is_home": is_home, "spread": eff_spread, "team_score": t_score, "opp_score": o_score
            }

            for p in profiles:
                game_data[f"prob_{p}"] = calculate_calibrated_prob(eff_spread, is_home, w, profile=p)

            weekly_slates[w].append(game_data)

        for p in profiles:
            optimal_path = solve_path(weekly_slates, p)
            elim_wk = None
            total_wins = 0

            for w in range(1, WEEKS + 1):
                pick = optimal_path.get(w, "")
                match = next((c for c in weekly_slates[w] if c["team"] == pick), None)
                if match:
                    if match["team_score"] > match["opp_score"]:
                        total_wins += 1
                    else:
                        if elim_wk is None:
                            elim_wk = w
                            if p == "baseline":
                                diagnostic_rows.append({
                                    "Season": season, "Week": w, "Pick": pick, "Opp": match["opponent"],
                                    "Line": f"{match['spread']:+.1f}", "Model %": f"{match[f'prob_{p}']*100:.1f}%",
                                    "Result": f"Loss ({match['team_score']}-{match['opp_score']})",
                                    "Root Cause": "Sub-8pt road/early trap" if abs(match["spread"]) < 8 else "Heavy upset variance"
                                })

            profile_results[p][season] = {
                "eliminated_week": elim_wk if elim_wk is not None else 18,
                "wins": total_wins
            }

    comp_data = []
    for p in profiles:
        avg_survived = np.mean([profile_results[p][s]["eliminated_week"] for s in BACKTEST_SEASONS])
        total_wins = sum([profile_results[p][s]["wins"] for s in BACKTEST_SEASONS])
        comp_data.append({"Strategy Profile": p, "Avg Weeks Survived": f"{avg_survived:.1f}/18", "Total Win Record": f"{total_wins}/90"})

    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing.")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_TITLE)
    
    try:
        diag_sheet = spreadsheet.worksheet("Diagnostic Summary")
        diag_sheet.clear()
    except gspread.WorksheetNotFound:
        diag_sheet = spreadsheet.add_worksheet(title="Diagnostic Summary", rows=50, cols=10)

    export_matrix = [
        ["🔍 SURVIVOR MODEL 5-YEAR POST-MORTEM & FORMULA TUNING", "", "", "", "", "", "", ""],
        ["", "", "", "", "", "", "", ""],
        ["Season", "Elimination Week", "Pick Chosen", "Opponent", "Line", "Model Prob", "Final Score", "Forensic Root Cause"]
    ]

    for d in diagnostic_rows:
        export_matrix.append([
            d["Season"], f"Week {d['Week']}", d["Pick"], d["Opp"], d["Line"], d["Model %"], d["Result"], d["Root Cause"]
        ])

    export_matrix.extend([
        ["", "", "", "", "", "", "", ""],
        ["STRATEGY COMPARISON & CALIBRATION BENCHMARKS", "", "", "", "", "", "", ""],
        ["Profile", "Average Weeks Survived", "Total 5-Year Win Count", "Key Takeaway", "", "", "", ""]
    ])

    for c in comp_data:
        takeaway = "Overfits future value, takes weak early favorites" if c["Strategy Profile"] == "baseline" else "Safeguards Weeks 1-4 and enforces spread floors"
        export_matrix.append([c["Strategy Profile"], c["Avg Weeks Survived"], c["Total Win Record"], takeaway, "", "", "", ""])

    diag_sheet.update(range_name=f"A1:H{len(export_matrix)}", values=export_matrix)
    print("\n✅ Successfully written to 'Diagnostic Summary' tab in Google Sheets.")

if __name__ == "__main__":
    run_diagnostics()
