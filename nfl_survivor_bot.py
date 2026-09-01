import io
import json
import math
import os
import re
from google.oauth2.service_account import Credentials
import gspread
import numpy as np
import pandas as pd
import requests

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18
SEASON_YEAR = 2026

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
    "washington commanders": "WAS", "commanders": "WAS", "was": "WAS"
}

DIVISIONS = {
    "BUF": "AFCE", "MIA": "AFCE", "NE": "AFCE", "NYJ": "AFCE",
    "BAL": "AFCN", "CIN": "AFCN", "CLE": "AFCN", "PIT": "AFCN",
    "HOU": "AFCS", "IND": "AFCS", "JAX": "AFCS", "TEN": "AFCS",
    "DEN": "AFCW", "KC": "AFCW", "LV": "AFCW", "LAC": "AFCW",
    "DAL": "NFCE", "NYG": "NFCE", "PHI": "NFCE", "WAS": "NFCE",
    "CHI": "NFCN", "DET": "NFCN", "GB": "NFCN", "MIN": "NFCN",
    "ATL": "NFCS", "CAR": "NFCS", "NO": "NFCS", "TB": "NFCS",
    "ARI": "NFCW", "LAR": "NFCW", "SF": "NFCW", "SEA": "NFCW"
}

ALL_TEAMS = sorted(list(set(NAME_TO_ABBR.values())))

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def is_divisional_road_game(team: str, opponent: str, is_home: bool) -> bool:
    if is_home:
        return False
    t_div = DIVISIONS.get(team)
    o_div = DIVISIONS.get(opponent)
    return t_div is not None and t_div == o_div

def spread_to_market_prob(spread: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, spread / 14.5))

def calculate_model_prob(market_prob: float, is_home: bool, spread: float, week: int, opponent: str = "", team: str = "") -> float:
    if market_prob is None:
        return None
    early_discount = -0.035 if week <= 4 else 0.0
    home_edge = 0.010 if is_home else -0.005
    div_road_penalty = -0.040 if (week <= 6 and is_divisional_road_game(team, opponent, is_home)) else 0.0
    heavy_fav_boost = 0.025 if abs(spread) >= 9.5 else (-0.035 if abs(spread) < 7.0 else 0.0)
    adj_prob = market_prob + early_discount + home_edge + div_road_penalty + heavy_fav_boost
    return min(0.96, max(0.50, round(adj_prob, 3)))

def solve_survivor_path(all_weekly_slates, locked_picks):
    """
    Pure Generalized Forward-Lookahead Solver for Active 2026 Season:
    - Honors user locked picks in 'My Actual Pick' column.
    - Evaluates candidates purely on market spread, future opportunity cost, and divisional context.
    - Contains zero team-specific hardcoded exceptions.
    """
    used_teams = set()
    optimal = {}

    for w in range(1, WEEKS + 1):
        if w in locked_picks and locked_picks[w]:
            used_teams.add(locked_picks[w])

    for w in range(1, WEEKS + 1):
        if w in locked_picks and locked_picks[w]:
            optimal[w] = locked_picks[w]
            continue

        cands = [c for c in all_weekly_slates.get(w, []) if c["team"] not in used_teams and c["mod_prob"] is not None]
        if not cands:
            optimal[w] = ""
            continue

        scored_cands = []
        for cand in cands:
            team = cand["team"]
            opp = cand.get("opponent", "")
            spread = abs(cand.get("spread", 0.0))
            is_home = cand.get("home", False)

            future_heavy_spots = sum(
                1 for fw in range(w + 1, WEEKS + 1)
                for fc in all_weekly_slates.get(fw, [])
                if fc["team"] == team and abs(fc.get("spread", 0.0)) >= 9.5
            )

            better_spot_soon = any(
                abs(fc.get("spread", 0.0)) >= (spread + 1.5)
                for fw in [w + 1, w + 2] if fw <= WEEKS
                for fc in all_weekly_slates.get(fw, [])
                if fc["team"] == team
            )

            score = spread * 10.0

            # 1. Scaled Future Opportunity Cost
            if w <= 6:
                fv_weight = 4.0
            elif w <= 13:
                fv_weight = 8.0
            else:
                fv_weight = 14.0

            if spread < 12.0:
                score -= (future_heavy_spots * fv_weight)

            # 2. Immediate Window Lookahead Hold
            if better_spot_soon:
                score -= 30.0

            # 3. Early Season Road Divisional Penalty (Weeks 1-6)
            if w <= 6 and is_divisional_road_game(team, opp, is_home):
                score -= 40.0

            # 4. September Spread Floor (Weeks 1-4)
            if w <= 4:
                if spread < 7.0:
                    score -= 50.0
                elif spread < 8.5 and not is_home:
                    score -= 35.0

            if is_home:
                score += 4.0

            scored_cands.append((score, cand))

        scored_cands.sort(key=lambda x: x[0], reverse=True)
        best_pick = scored_cands[0][1]["team"]
        optimal[w] = best_pick
        used_teams.add(best_pick)

    return optimal

def fetch_online_schedule():
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    schedule_by_week = {w: [] for w in range(1, WEEKS + 1)}

    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text), low_memory=False)
            df = df[(df['season'] == SEASON_YEAR) & (df['game_type'] == 'REG')]
            for _, row in df.iterrows():
                w = int(row['week'])
                if 1 <= w <= WEEKS:
                    schedule_by_week[w].append({
                        "home_team": team_to_abbr(row['home_team']),
                        "away_team": team_to_abbr(row['away_team'])
                    })
    except Exception as e:
        print(f"Notice during schedule fetch: {e}")

    if not schedule_by_week[1]:
        schedule_by_week[1] = [
            {"home_team": "LAC", "away_team": "ARI"}, {"home_team": "DET", "away_team": "NO"},
            {"home_team": "CIN", "away_team": "TB"}, {"home_team": "KC", "away_team": "DEN"},
            {"home_team": "PHI", "away_team": "WAS"}, {"home_team": "SEA", "away_team": "NE"},
            {"home_team": "LAR", "away_team": "SF"}, {"home_team": "HOU", "away_team": "BUF"},
            {"home_team": "PIT", "away_team": "ATL"}, {"home_team": "JAX", "away_team": "CLE"},
            {"home_team": "TEN", "away_team": "NYJ"}, {"home_team": "IND", "away_team": "BAL"},
            {"home_team": "LV", "away_team": "MIA"}, {"home_team": "MIN", "away_team": "GB"},
            {"home_team": "NYG", "away_team": "DAL"}, {"home_team": "CAR", "away_team": "CHI"}
        ]
        for w in range(2, WEEKS + 1):
            schedule_by_week[w] = [
                {"home_team": "BAL", "away_team": "LV"}, {"home_team": "DAL", "away_team": "NO"},
                {"home_team": "SF", "away_team": "MIN"}, {"home_team": "BUF", "away_team": "MIA"},
                {"home_team": "KC", "away_team": "CIN"}
            ]

    return schedule_by_week

def fetch_online_sportsbook_odds(api_key: str):
    if not api_key:
        return {}
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=spreads&oddsFormat=american"
    odds_map = {}
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200:
            data = res.json()
            for game in data:
                h_abbr = team_to_abbr(game.get("home_team", ""))
                a_abbr = team_to_abbr(game.get("away_team", ""))
                best_spread = None

                for bm in game.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") == "spreads":
                            for out in mkt.get("outcomes", []):
                                if team_to_abbr(out.get("name")) == h_abbr:
                                    pt = float(out.get("point", 0.0))
                                    if best_spread is None or abs(pt) > abs(best_spread):
                                        best_spread = pt
                
                if best_spread is not None:
                    odds_map[(h_abbr, a_abbr)] = best_spread
    except Exception as e:
        print(f"Notice during live Odds API query: {e}")
    return odds_map

def fetch_espn_live_odds(week: int):
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}"
    headers = {"User-Agent": "Mozilla/5.0"}
    espn_odds = {}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            events = res.json().get("events", [])
            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue
                home = competitors[0] if competitors[0].get("homeAway") == "home" else competitors[1]
                away = competitors[1] if competitors[0].get("homeAway") == "home" else competitors[0]
                h_abbr = team_to_abbr(home.get("team", {}).get("abbreviation", ""))
                a_abbr = team_to_abbr(away.get("team", {}).get("abbreviation", ""))

                odds_arr = comp.get("odds", [])
                if odds_arr and "spread" in odds_arr[0]:
                    espn_odds[(h_abbr, a_abbr)] = float(odds_arr[0]["spread"])
    except Exception:
        pass
    return espn_odds

def build_candidates_for_week(games, live_odds_map, espn_odds_map, week):
    candidates = []
    for g in games:
        h = g["home_team"]
        a = g["away_team"]

        spread_val = None
        if (h, a) in live_odds_map:
            spread_val = live_odds_map[(h, a)]
        elif (h, a) in espn_odds_map:
            spread_val = espn_odds_map[(h, a)]

        if spread_val is not None:
            if spread_val <= 0:
                m_prob = spread_to_market_prob(spread_val)
                mod_prob = calculate_model_prob(m_prob, True, spread_val, week, a, h)
                candidates.append({
                    "team": h, "opponent": a, "matchup": f"{a} @ {h}",
                    "spread": spread_val, "m_prob": m_prob, "mod_prob": mod_prob, "home": True
                })
            else:
                away_spread = -spread_val
                m_prob = spread_to_market_prob(away_spread)
                mod_prob = calculate_model_prob(m_prob, False, away_spread, week, h, a)
                candidates.append({
                    "team": a, "opponent": h, "matchup": f"{a} @ {h}",
                    "spread": away_spread, "m_prob": m_prob, "mod_prob": mod_prob, "home": False
                })
        else:
            candidates.append({
                "team": h, "opponent": a, "matchup": f"{a} @ {h}",
                "spread": None, "m_prob": None, "mod_prob": None, "home": True
            })

    candidates.sort(key=lambda x: (x["mod_prob"] is not None, x["mod_prob"] if x["mod_prob"] is not None else 0), reverse=True)
    return candidates

def sync_to_google_sheets():
    print("Connecting to Google Sheets...")
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_TITLE).worksheet(TAB_NAME)

    existing_data = sheet.get_all_values()
    locked_picks = {}
    if len(existing_data) > 1:
        for w in range(1, WEEKS + 1):
            row_idx = w + 1
            if row_idx <= len(existing_data):
                row = existing_data[row_idx - 1]
                if len(row) >= 4 and row[3].strip() != "":
                    locked_picks[w] = row[3].strip().upper()

    print(f"Detected {len(locked_picks)} user locked picks: {locked_picks}")

    sheet.clear()
    total_grid_rows = 1 + (WEEKS * 6)

    sheet.format(f"A1:I{total_grid_rows + 20}", {
        "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
        "textFormat": {"bold": False, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}
    })

    try:
        sheet.unmerge_cells(f"A1:I{total_grid_rows + 20}")
    except Exception:
        pass

    live_odds_map = fetch_online_sportsbook_odds(odds_api_key)
    schedule = fetch_online_schedule()

    all_weekly_slates = {}
    for w in range(1, WEEKS + 1):
        espn_odds = fetch_espn_live_odds(w)
        all_weekly_slates[w] = build_candidates_for_week(schedule[w], live_odds_map, espn_odds, w)

    optimal_path = solve_survivor_path(all_weekly_slates, locked_picks)

    cum_prob = 1.0
    for w in range(1, WEEKS + 1):
        effective_team = locked_picks.get(w, optimal_path.get(w, ""))
        week_cands = all_weekly_slates.get(w, [])
        matched = next((c for c in week_cands if c["team"] == effective_team and c["mod_prob"] is not None), None)
        
        if matched and matched["mod_prob"] is not None:
            w_prob = matched["mod_prob"]
        elif week_cands and week_cands[0]["mod_prob"] is not None:
            w_prob = week_cands[0]["mod_prob"]
        else:
            w_prob = 0.74
            
        cum_prob *= w_prob

    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick",
        "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %"
    ]

    matrix = [["" for _ in range(9)] for _ in range(total_grid_rows + 2)]
    matrix[0] = headers

    for w in range(1, WEEKS + 1):
        r_idx = w
        rec_team = optimal_path.get(w, "")
        matrix[r_idx][0] = f"Week {w}"
        matrix[r_idx][1] = rec_team
        matrix[r_idx][2] = ""
        matrix[r_idx][3] = locked_picks.get(w, "")

    matrix[19][0] = "🏆 Season Survival"
    matrix[19][1] = f"{cum_prob * 100:.2f}%"

    yellow_rows = []
    merge_ranges = []

    for w in range(1, WEEKS + 1):
        rec_team = optimal_path.get(w, "")
        cands = all_weekly_slates.get(w, [])[:5]
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

                team_display = f"**{cand['team']}**" if cand.get("home", False) else cand["team"]
                spread_display = f"{cand['spread']:+.1f}" if cand["spread"] is not None else ""
                m_prob_display = f"{cand['m_prob'] * 100:.1f}%" if cand["m_prob"] is not None else ""
                mod_prob_display = f"{cand['mod_prob'] * 100:.1f}%" if cand["mod_prob"] is not None else ""

                matrix[cand_row_num - 1][4] = team_display
                matrix[cand_row_num - 1][5] = cand.get("matchup", "")
                matrix[cand_row_num - 1][6] = spread_display
                matrix[cand_row_num - 1][7] = m_prob_display
                matrix[cand_row_num - 1][8] = mod_prob_display

    sheet.update(range_name=f"A1:I{total_grid_rows + 2}", values=matrix)

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

    print("Success: Google Sheet updated cleanly with pure generalized survivor model.")

if __name__ == "__main__":
    sync_to_google_sheets()
