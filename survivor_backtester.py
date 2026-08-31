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

# --- SPREADSHEET CONFIGURATION ---
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

def calculate_model_prob(market_prob: float, is_home: bool, spread: float) -> float:
    if market_prob is None:
        return None
    home_boost = 0.025 if is_home else -0.015
    rest_boost = 0.015 if abs(spread) >= 7.0 else 0.005
    epa_edge = 0.020 if abs(spread) >= 8.5 else 0.010
    adj_prob = market_prob + home_boost + rest_boost + epa_edge
    return min(0.96, max(0.51, round(adj_prob, 3)))

def solve_season_survivor_path(weekly_slates):
    num_teams = len(ALL_TEAMS)
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    idx_to_team = {i: t for i, t in enumerate(ALL_TEAMS)}

    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)

    for w in range(1, WEEKS + 1):
        row = w - 1
        for cand in weekly_slates.get(w, []):
            t_idx = team_to_idx.get(cand["team"])
            if t_idx is not None and cand["mod_prob"] is not None:
                cost_matrix[row, t_idx] = -math.log(cand["mod_prob"])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e4:
            optimal[r + 1] = idx_to_team[c]
        else:
            optimal[r + 1] = ""
    return optimal

def get_or_create_worksheet(spreadsheet, tab_title):
    try:
        ws = spreadsheet.worksheet(tab_title)
        ws.clear()
        return ws
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=tab_title, rows=150, cols=15)

def run_backtest_pipeline():
    print("=" * 80)
    print("🏈 STARTING 5-YEAR NFL SURVIVOR BACKTESTER")
    print("=" * 80)

    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_TITLE)

    print("Fetching historical game data from nflverse...")
    res = requests.get(DATA_URL, timeout=25)
    if res.status_code != 200:
        raise RuntimeError(f"Failed to fetch nflverse data. HTTP {res.status_code}")

    df_raw = pd.read_csv(io.StringIO(res.text), low_memory=False)
    df = df_raw[(df_raw["season"].isin(BACKTEST_SEASONS)) & (df_raw["game_type"] == "REG")].copy()

    for season in BACKTEST_SEASONS:
        tab_name = f"Test {season}"
        print(f"\nProcessing {season} Season -> Tab: '{tab_name}'...")
        sheet = get_or_create_worksheet(spreadsheet, tab_name)

        season_df = df[df["season"] == season]
        weekly_slates = {w: [] for w in range(1, WEEKS + 1)}

        for _, row in season_df.iterrows():
            w = int(row["week"])
            if w < 1 or w > WEEKS:
                continue

            h_team = team_to_abbr(row["home_team"])
            a_team = team_to_abbr(row["away_team"])
            h_score = int(row["home_score"]) if pd.notnull(row["home_score"]) else None
            a_score = int(row["away_score"]) if pd.notnull(row["away_score"]) else None

            spread_val = float(row["spread_line"]) if pd.notnull(row.get("spread_line")) else 0.0
            home_spread = -spread_val

            if home_spread <= 0:
                m_prob = spread_to_market_prob(home_spread)
                mod_prob = calculate_model_prob(m_prob, True, home_spread)
                weekly_slates[w].append({
                    "team": h_team,
                    "opponent": a_team,
                    "matchup": f"{a_team} @ {h_team}",
                    "is_home": True,
                    "spread": home_spread,
                    "m_prob": m_prob,
                    "mod_prob": mod_prob,
                    "team_score": h_score,
                    "opp_score": a_score
                })
            else:
                away_spread = -home_spread
                m_prob = spread_to_market_prob(away_spread)
                mod_prob = calculate_model_prob(m_prob, False, away_spread)
                weekly_slates[w].append({
                    "team": a_team,
                    "opponent": h_team,
                    "matchup": f"{a_team} @ {h_team}",
                    "is_home": False,
                    "spread": away_spread,
                    "m_prob": m_prob,
                    "mod_prob": mod_prob,
                    "team_score": a_score,
                    "opp_score": h_score
                })

        for w in range(1, WEEKS + 1):
            weekly_slates[w].sort(
                key=lambda x: (x["mod_prob"] is not None, x["mod_prob"] if x["mod_prob"] is not None else 0),
                reverse=True
            )

        optimal_path = solve_season_survivor_path(weekly_slates)

        # Evaluate Historical Outcomes
        eliminated_week = None
        cum_prob = 1.0
        weekly_outcomes = {}

        for w in range(1, WEEKS + 1):
            pick = optimal_path.get(w, "")
            match = next((c for c in weekly_slates[w] if c["team"] == pick), None)
            if match:
                cum_prob *= match["mod_prob"]
                t_score = match["team_score"]
                o_score = match["opp_score"]

                if t_score is not None and o_score is not None:
                    if t_score > o_score:
                        status = "✅ WIN"
                    elif t_score == o_score:
                        status = "❌ TIE"
                    else:
                        status = "❌ LOSS"
                else:
                    status = "UNPLAYED"

                if "LOSS" in status or "TIE" in status:
                    if eliminated_week is None:
                        eliminated_week = w

                weekly_outcomes[w] = {
                    "pick_display": f"{pick} ({status} {t_score}-{o_score})",
                    "status": status
                }
            else:
                weekly_outcomes[w] = {"pick_display": pick, "status": ""}

        # Build 9-Column Matrix (A through I)
        total_grid_rows = 1 + (WEEKS * 6)
        headers = [
            "Week", "Recommended Pick", "|", "Historical Outcome",
            "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %"
        ]
        matrix = [["" for _ in range(9)] for _ in range(total_grid_rows + 2)]
        matrix[0] = headers

        # Left Rail: Columns A-D
        for w in range(1, WEEKS + 1):
            r_idx = w
            matrix[r_idx][0] = f"Week {w}"
            matrix[r_idx][1] = optimal_path.get(w, "")
            matrix[r_idx][2] = ""
            matrix[r_idx][3] = weekly_outcomes[w]["pick_display"]

        # Row 20 Summary
        survived_text = "🏆 SURVIVED 18-0" if eliminated_week is None else f"❌ OUT WK {eliminated_week}"
        matrix[19][0] = f"🏆 Season Result ({survived_text})"
        matrix[19][1] = f"{cum_prob * 100:.2f}% Model Proj"

        yellow_rows = []
        merge_ranges = []

        # Columns E-I
        for w in range(1, WEEKS + 1):
            rec_team = optimal_path.get(w, "")
            cands = weekly_slates.get(w, [])
            block_start_row = 1 + (w - 1) * 6 + 1

            matrix[block_start_row - 1][4] = f"Top candidates for Week {w}"
            merge_ranges.append(f"E{block_start_row}:I{block_start_row}")

            for i in range(5):
                cand_row_num = block_start_row + 1 + i
                if i < len(cands):
                    cand = cands[i]
                    is_rec = (cand["team"] == rec_team and rec_team != "")
                    if is_rec:
                        yellow_rows.append(cand_row_num)

                    team_display = f"**{cand['team']}**" if cand.get("is_home", False) else cand["team"]
                    spread_display = f"{cand['spread']:+.1f}" if cand["spread"] is not None else ""
                    m_prob_display = f"{cand['m_prob'] * 100:.1f}%" if cand["m_prob"] is not None else ""
                    mod_prob_display = f"{cand['mod_prob'] * 100:.1f}%" if cand["mod_prob"] is not None else ""

                    matrix[cand_row_num - 1][4] = team_display
                    matrix[cand_row_num - 1][5] = cand.get("matchup", "")
                    matrix[cand_row_num - 1][6] = spread_display
                    matrix[cand_row_num - 1][7] = m_prob_display
                    matrix[cand_row_num - 1][8] = mod_prob_display

        # Write data to Google Sheet
        sheet.update(range_name=f"A1:I{total_grid_rows + 2}", values=matrix)

        # Format styles
        sheet.format(f"A1:I{total_grid_rows + 20}", {
            "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
            "textFormat": {"bold": False, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}
        })

        for rng in merge_ranges:
            try:
                sheet.merge_cells(rng, merge_type="MERGE_ALL")
            except Exception:
                pass

        sheet.format("A1:I1", {
            "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
            "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
            "horizontalAlignment": "CENTER"
        })

        sheet.format(f"C1:C{total_grid_rows + 2}", {"backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}})
        sheet.format(f"A2:B{total_grid_rows + 2}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
        sheet.format(f"D2:D{total_grid_rows + 2}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
        sheet.format(f"E2:I{total_grid_rows + 2}", {"horizontalAlignment": "CENTER"})

        sheet.format("A20:B20", {
            "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
            "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
            "horizontalAlignment": "CENTER"
        })

        batch_formats = []
        for rng in merge_ranges:
            batch_formats.append({
                "range": rng,
                "format": {
                    "backgroundColor": {"red": 0.83, "green": 0.90, "blue": 0.95},
                    "textFormat": {"bold": True, "foregroundColor": {"red": 0.05, "green": 0.16, "blue": 0.28}},
                    "horizontalAlignment": "CENTER"
                }
            })

        for r_idx in yellow_rows:
            batch_formats.append({
                "range": f"E{r_idx}:I{r_idx}",
                "format": {
                    "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.55},
                    "textFormat": {"bold": True}
                }
            })

        if batch_formats:
            sheet.batch_format(batch_formats)

        print(f"Tab '{tab_name}' generated successfully. Result: {survived_text}")

    print("\n" + "=" * 80)
    print("✅ 5-YEAR BACKTEST COMPLETE: ALL HISTORICAL TABS CREATED IN GOOGLE SHEETS")
    print("=" * 80)

if __name__ == "__main__":
    run_backtest_pipeline()
