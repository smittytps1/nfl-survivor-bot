import io
import json
import math
import os
import re
import time
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

def calculate_model_prob(market_prob: float, is_home: bool, spread: float, week: int) -> float:
    """
    Calibrated Model Win Probability:
    - Weeks 1-4 early uncertainty discount (-3.5%).
    - De-amplified home edge (+1.0% home, -0.5% away) avoiding double-counting.
    - Tiered safety adjustment (+2.0% for >=9.5 pt favorites, -3.0% penalty for <6.5 pt games).
    """
    if market_prob is None:
        return None
    early_discount = -0.035 if week <= 4 else 0.0
    home_edge = 0.010 if is_home else -0.005
    heavy_fav_boost = 0.020 if abs(spread) >= 9.5 else (-0.030 if abs(spread) < 6.5 else 0.0)
    
    adj_prob = market_prob + early_discount + home_edge + heavy_fav_boost
    return min(0.96, max(0.50, round(adj_prob, 3)))

def solve_season_survivor_path(weekly_slates):
    num_teams = len(ALL_TEAMS)
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    idx_to_team = {i: t for i, t in enumerate(ALL_TEAMS)}

    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)

    for w in range(1, WEEKS + 1):
        row = w - 1
        # Dynamic Future-Value Decay Multiplier
        if w <= 6:
            fv_weight = 0.20
        elif w <= 12:
            fv_weight = 0.60
        else:
            fv_weight = 1.00

        for cand in weekly_slates.get(w, []):
            t_idx = team_to_idx.get(cand["team"])
            if t_idx is not None and cand["mod_prob"] is not None:
                # Enforce strict safety floor for early weeks
                if w <= 10 and cand["spread"] is not None and abs(cand["spread"]) < 6.5:
                    cost_matrix[row, t_idx] = -math.log(max(0.40, cand["mod_prob"] - 0.25)) * (1.0 + fv_weight)
                else:
                    cost_matrix[row, t_idx] = -math.log(cand["mod_prob"]) * (1.0 + fv_weight)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal = {}
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < 1e4:
            optimal[r + 1] = idx_to_team[c]
        else:
            optimal[r + 1] = ""
    return optimal

def get_spreadsheet_with_retries(client, title, max_retries=4):
    for attempt in range(max_retries):
        try:
            return client.open(title)
        except Exception as e:
            print(f"Connection notice (Attempt {attempt+1}/{max_retries}): {e}. Retrying in 3s...")
            time.sleep(3)
    return client.open(title)

def get_or_create_worksheet(spreadsheet, tab_title):
    try:
        ws = spreadsheet.worksheet(tab_title)
        ws.clear()
        return ws
    except gspread.WorksheetNotFound:
        return spreadsheet.add_worksheet(title=tab_title, rows=150, cols=12)

def run_backtest_pipeline():
    print("=" * 80)
    print("🏈 STARTING 5-YEAR NFL SURVIVOR BACKTESTER (CALIBRATED MODEL)")
    print("=" * 80)

    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = get_spreadsheet_with_retries(client, SHEET_TITLE)

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
        sheet_id = sheet.id

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
                mod_prob = calculate_model_prob(m_prob, True, home_spread, w)
                weekly_slates[w].append({
                    "team": h_team, "opponent": a_team,
                    "matchup": f"{a_team} @ {h_team}",
                    "is_home": True, "spread": home_spread,
                    "m_prob": m_prob, "mod_prob": mod_prob,
                    "team_score": h_score, "opp_score": a_score
                })
            else:
                away_spread = -home_spread
                m_prob = spread_to_market_prob(away_spread)
                mod_prob = calculate_model_prob(m_prob, False, away_spread, w)
                weekly_slates[w].append({
                    "team": a_team, "opponent": h_team,
                    "matchup": f"{a_team} @ {h_team}",
                    "is_home": False, "spread": away_spread,
                    "m_prob": m_prob, "mod_prob": mod_prob,
                    "team_score": a_score, "opp_score": h_score
                })

        for w in range(1, WEEKS + 1):
            weekly_slates[w].sort(
                key=lambda x: (x["mod_prob"] is not None, x["mod_prob"] if x["mod_prob"] is not None else 0),
                reverse=True
            )

        optimal_path = solve_season_survivor_path(weekly_slates)

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

                if ("LOSS" in status or "TIE" in status) and eliminated_week is None:
                    eliminated_week = w

                weekly_outcomes[w] = {
                    "pick_display": f"{pick} ({status} {t_score}-{o_score})",
                    "status": status
                }
            else:
                weekly_outcomes[w] = {"pick_display": pick, "status": ""}

        total_grid_rows = 1 + (WEEKS * 6)
        headers = [
            "Week", "Recommended Pick", "|", "Historical Outcome",
            "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %"
        ]
        matrix = [["" for _ in range(9)] for _ in range(total_grid_rows + 2)]
        matrix[0] = headers

        for w in range(1, WEEKS + 1):
            r_idx = w
            matrix[r_idx][0] = f"Week {w}"
            matrix[r_idx][1] = optimal_path.get(w, "")
            matrix[r_idx][2] = ""
            matrix[r_idx][3] = weekly_outcomes[w]["pick_display"]

        survived_text = "🏆 SURVIVED 18-0" if eliminated_week is None else f"❌ OUT WK {eliminated_week}"
        matrix[19][0] = f"🏆 Season Result ({survived_text})"
        matrix[19][1] = f"{cum_prob * 100:.2f}% Model Proj"

        yellow_row_indices = []
        merge_row_indices = []

        for w in range(1, WEEKS + 1):
            rec_team = optimal_path.get(w, "")
            cands = weekly_slates.get(w, [])
            block_start_row = 1 + (w - 1) * 6 + 1

            matrix[block_start_row - 1][4] = f"Top candidates for Week {w}"
            merge_row_indices.append(block_start_row)

            for i in range(5):
                cand_row_num = block_start_row + 1 + i
                if i < len(cands):
                    cand = cands[i]
                    is_rec = (cand["team"] == rec_team and rec_team != "")
                    if is_rec:
                        yellow_row_indices.append(cand_row_num)

                    team_display = f"**{cand['team']}**" if cand.get("is_home", False) else cand["team"]
                    spread_display = f"{cand['spread']:+.1f}" if cand["spread"] is not None else ""
                    m_prob_display = f"{cand['m_prob'] * 100:.1f}%" if cand["m_prob"] is not None else ""
                    mod_prob_display = f"{cand['mod_prob'] * 100:.1f}%" if cand["mod_prob"] is not None else ""

                    matrix[cand_row_num - 1][4] = team_display
                    matrix[cand_row_num - 1][5] = cand.get("matchup", "")
                    matrix[cand_row_num - 1][6] = spread_display
                    matrix[cand_row_num - 1][7] = m_prob_display
                    matrix[cand_row_num - 1][8] = mod_prob_display

        # 1. Update Grid Values
        sheet.update(range_name=f"A1:I{total_grid_rows + 2}", values=matrix)

        # 2. Build Single Consolidated Batch Update (Merges + Formats)
        requests_payload = []

        requests_payload.append({
            "unmergeCells": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": total_grid_rows + 20,
                    "startColumnIndex": 0, "endColumnIndex": 9
                }
            }
        })

        for r in merge_row_indices:
            requests_payload.append({
                "mergeCells": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r - 1, "endRowIndex": r,
                        "startColumnIndex": 4, "endColumnIndex": 9
                    },
                    "mergeType": "MERGE_ALL"
                }
            })

        requests_payload.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": total_grid_rows + 20,
                    "startColumnIndex": 0, "endColumnIndex": 9
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
                        "textFormat": {"bold": False, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat)"
            }
        })

        requests_payload.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": 1,
                    "startColumnIndex": 0, "endColumnIndex": 9
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
                        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                        "horizontalAlignment": "CENTER"
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
            }
        })

        requests_payload.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0, "endRowIndex": total_grid_rows + 2,
                    "startColumnIndex": 2, "endColumnIndex": 3
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}
                    }
                },
                "fields": "userEnteredFormat(backgroundColor)"
            }
        })

        requests_payload.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 19, "endRowIndex": 20,
                    "startColumnIndex": 0, "endColumnIndex": 2
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
                        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                        "horizontalAlignment": "CENTER"
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
            }
        })

        for r in merge_row_indices:
            requests_payload.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r - 1, "endRowIndex": r,
                        "startColumnIndex": 4, "endColumnIndex": 9
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 0.83, "green": 0.90, "blue": 0.95},
                            "textFormat": {"bold": True, "foregroundColor": {"red": 0.05, "green": 0.16, "blue": 0.28}},
                            "horizontalAlignment": "CENTER"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
                }
            })

        for r in yellow_row_indices:
            requests_payload.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": r - 1, "endRowIndex": r,
                        "startColumnIndex": 4, "endColumnIndex": 9
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.55},
                            "textFormat": {"bold": True}
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)"
                }
            })

        spreadsheet.batch_update({"requests": requests_payload})
        print(f"Tab '{tab_name}' generated successfully. Result: {survived_text}")
        time.sleep(2)

    print("\n" + "=" * 80)
    print("✅ 5-YEAR BACKTEST COMPLETE: ALL HISTORICAL TABS CREATED IN GOOGLE SHEETS")
    print("=" * 80)

if __name__ == "__main__":
    run_backtest_pipeline()
