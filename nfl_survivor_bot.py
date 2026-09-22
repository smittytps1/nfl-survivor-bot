import os
import io
import re
import json
import math
import time
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
LOG_TAB_NAME = "Adjustment Log"
LINES_TAB_NAME = "Full Season Lines"
WEEKS = 18
SEASON_YEAR = 2026
DOUBLE_PICK_WEEKS = {15, 16, 17, 18}

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

# --- 2026 POWER RATINGS FOR AUTO-POPULATION ---
POWER_RATINGS = {
    "LAR": 6.5, "SEA": 6.0, "BUF": 5.5, "HOU": 5.0, "DEN": 4.5, "NE": 4.0, 
    "PHI": 3.5, "LAC": 3.0, "KC": 2.5, "SF": 2.5, "BAL": 2.0, "DET": 2.0, 
    "CIN": 1.5, "GB": 1.0, "DAL": 0.5, "CHI": 0.5, "PIT": 0.0, "TB": 0.0, 
    "JAX": -0.5, "MIN": -1.0, "IND": -1.5, "WAS": -2.0, "NYG": -2.5, "NO": -2.5, 
    "ATL": -3.0, "LV": -3.5, "TEN": -4.0, "CLE": -4.5, "NYJ": -5.0, "ARI": -5.5, 
    "CAR": -6.0, "MIA": -6.5
}

ALL_TEAMS = sorted(list(set(NAME_TO_ABBR.values())))

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def parse_actual_picks(cell_text: str):
    if not cell_text:
        return []
    parts = re.split(r'[/,;\s]+', cell_text.strip())
    teams = [team_to_abbr(p) for p in parts if p.strip()]
    return [t for t in teams if t in ALL_TEAMS]

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

def get_safe_abs_spread(cand_dict) -> float:
    sp = cand_dict.get("spread")
    return abs(sp) if sp is not None else 0.0

def fetch_dynamic_schedule():
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    schedule_by_week = {w: [] for w in range(1, WEEKS + 1)}
    print("Fetching authentic 2026 NFL schedule from nflverse...")
    try:
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            df = pd.read_csv(io.StringIO(res.text), low_memory=False)
            df = df[(df['season'] == SEASON_YEAR) & (df['game_type'] == 'REG')]
            for _, row in df.iterrows():
                w = int(row['week'])
                if 1 <= w <= WEEKS:
                    h_abbr = team_to_abbr(row['home_team'])
                    a_abbr = team_to_abbr(row['away_team'])
                    if h_abbr in ALL_TEAMS and a_abbr in ALL_TEAMS:
                        schedule_by_week[w].append({
                            "home_team": h_abbr,
                            "away_team": a_abbr
                        })
    except Exception as e:
        print(f"Notice on dynamic schedule fetch: {e}")
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

def reconcile_and_update_lines_tab(spreadsheet, schedule_2026, live_odds_map, all_espn_odds):
    try:
        lines_sheet = spreadsheet.worksheet(LINES_TAB_NAME)
        existing_data = lines_sheet.get_all_values()
    except gspread.WorksheetNotFound:
        lines_sheet = spreadsheet.add_worksheet(title=LINES_TAB_NAME, rows=300, cols=6)
        existing_data = []

    existing_lines = {}
    if len(existing_data) > 1:
        for row in existing_data[1:]:
            if len(row) >= 6:
                try:
                    w = int(row[0].replace("Week ", "").strip())
                    a = team_to_abbr(row[1])
                    h = team_to_abbr(row[3])
                    line = float(row[4])
                    src = row[5].strip()
                    existing_lines[(w, h, a)] = {"line": line, "source": src}
                except ValueError:
                    pass

    for w in range(1, WEEKS + 1):
        if not schedule_2026[w]:
            for (ew, eh, ea), edata in existing_lines.items():
                if ew == w:
                    schedule_2026[w].append({
                        "home_team": eh,
                        "away_team": ea
                    })

    reconciled_schedule = {w: [] for w in range(1, WEEKS + 1)}
    matrix = [["Week", "Away Team", "vs", "Home Team", "Home Spread", "Data Source"]]

    for w in range(1, WEEKS + 1):
        games = schedule_2026.get(w, [])
        for g in games:
            h = g["home_team"]
            a = g["away_team"]
            
            source = ""
            if (h, a) in live_odds_map:
                chosen_spread = live_odds_map[(h, a)]
                source = "Live (Odds API)"
            elif (h, a) in all_espn_odds:
                chosen_spread = all_espn_odds[(h, a)]
                source = "Live (ESPN)"
            elif (w, h, a) in existing_lines and existing_lines[(w, h, a)]["source"] not in ["Default Baseline", "Power Rating Alg"]:
                chosen_spread = existing_lines[(w, h, a)]["line"]
                source = existing_lines[(w, h, a)]["source"]
            else:
                h_pr = POWER_RATINGS.get(h, 0.0)
                a_pr = POWER_RATINGS.get(a, 0.0)
                chosen_spread = round(a_pr - h_pr - 2.0, 1)
                source = "Power Rating Alg"

            reconciled_schedule[w].append({
                "home_team": h,
                "away_team": a,
                "line": chosen_spread
            })
            matrix.append([f"Week {w}", a, "@", h, f"{chosen_spread:+.1f}", source])

    lines_sheet.clear()
    lines_sheet.update(range_name=f"A1:F{len(matrix)}", values=matrix)
    lines_sheet.format("A1:F1", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
        "horizontalAlignment": "CENTER"
    })
    lines_sheet.format(f"A2:F{len(matrix)}", {"horizontalAlignment": "CENTER"})
    
    return reconciled_schedule

def build_slates_from_reconciled(reconciled_schedule):
    all_slates = {}
    for w in range(1, WEEKS + 1):
        all_slates[w] = []
        games = reconciled_schedule.get(w, [])
        for g in games:
            h = g["home_team"]
            a = g["away_team"]
            home_spread = g["line"]

            if home_spread <= 0:
                fav_team = h
                dog_team = a
                is_home = True
                fav_spread = home_spread
            else:
                fav_team = a
                dog_team = h
                is_home = False
                fav_spread = -home_spread

            m_prob = spread_to_market_prob(fav_spread)
            mod_prob = calculate_model_prob(m_prob, is_home, fav_spread, w, dog_team, fav_team)

            all_slates[w].append({
                "team": fav_team,
                "opponent": dog_team,
                "matchup": f"{a} @ {h}",
                "spread": fav_spread,
                "is_home": is_home,
                "m_prob": m_prob,
                "mod_prob": mod_prob
            })
        all_slates[w].sort(key=lambda x: (x["mod_prob"] is not None, x["mod_prob"]), reverse=True)
    return all_slates

def solve_survivor_path(all_weekly_slates, locked_picks):
    used_teams = set()
    optimal = {w: [] for w in range(1, WEEKS + 1)}

    for w in range(1, WEEKS + 1):
        for t in locked_picks.get(w, []):
            used_teams.add(t)

    for w in range(1, WEEKS + 1):
        picks_needed = 2 if w in DOUBLE_PICK_WEEKS else 1
        user_picks = locked_picks.get(w, [])

        for t in user_picks:
            if t not in optimal[w]:
                optimal[w].append(t)

        while len(optimal[w]) < picks_needed:
            cands = [c for c in all_weekly_slates.get(w, []) if c["team"] not in used_teams and c["mod_prob"] is not None]
            if not cands:
                break

            scored_cands = []
            for cand in cands:
                team = cand["team"]
                opp = cand.get("opponent", "")
                spread = get_safe_abs_spread(cand)
                is_home = cand.get("is_home", False)

                future_heavy_spots = sum(
                    1 for fw in range(w + 1, WEEKS + 1)
                    for fc in all_weekly_slates.get(fw, [])
                    if fc["team"] == team and get_safe_abs_spread(fc) >= 8.5
                )

                double_pick_anchors = sum(
                    1 for fw in DOUBLE_PICK_WEEKS
                    if fw > w
                    for fc in all_weekly_slates.get(fw, [])
                    if fc["team"] == team and get_safe_abs_spread(fc) >= 7.0
                )

                score = spread * 10.0

                if w < 15:
                    score -= (future_heavy_spots * 7.5)
                    score -= (double_pick_anchors * 22.0)
                else:
                    score -= (future_heavy_spots * 5.0)

                better_spot_soon = any(
                    get_safe_abs_spread(fc) >= (spread + 1.5)
                    for fw in [w + 1, w + 2] if fw <= WEEKS
                    for fc in all_weekly_slates.get(fw, [])
                    if fc["team"] == team
                )

                if better_spot_soon:
                    score -= 25.0

                if w <= 6 and is_divisional_road_game(team, opp, is_home):
                    score -= 30.0

                if is_home:
                    score += 3.0

                scored_cands.append((score, cand))

            scored_cands.sort(key=lambda x: x[0], reverse=True)
            best_pick = scored_cands[0][1]["team"]
            optimal[w].append(best_pick)
            used_teams.add(best_pick)

    optimal_display = {w: " / ".join(optimal[w]) for w in range(1, WEEKS + 1)}
    return optimal, optimal_display

def log_adjustments_to_sheet(spreadsheet, previous_picks, current_picks, previous_actuals, current_actuals, prev_prob, new_prob):
    try:
        log_sheet = spreadsheet.worksheet(LOG_TAB_NAME)
    except gspread.WorksheetNotFound:
        log_sheet = spreadsheet.add_worksheet(title=LOG_TAB_NAME, rows=300, cols=7)
        headers = [
            "Timestamp (UTC)", "Trigger Event", "Week", "Old Recommendation", 
            "New Recommendation", "Model Survival Shift", "Notes"
        ]
        log_sheet.update(range_name="A1:G1", values=[headers])
        log_sheet.format("A1:G1", {
            "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
            "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
            "horizontalAlignment": "CENTER"
        })

    newly_locked = []
    for w in range(1, WEEKS + 1):
        prev_act = previous_actuals.get(w, "")
        curr_act = current_actuals.get(w, "")
        if curr_act and curr_act != prev_act:
            newly_locked.append(f"Wk {w}: Locked {curr_act}")

    trigger_description = "; ".join(newly_locked) if newly_locked else "Power Ratings & Schedule Alignment"
    log_rows = []
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    survival_shift_str = f"{prev_prob:.2f}% -> {new_prob:.2f}%" if prev_prob is not None else f"{new_prob:.2f}%"

    for w in range(1, WEEKS + 1):
        old_rec = previous_picks.get(w, "")
        new_rec = current_picks.get(w, "")

        if old_rec and new_rec and old_rec != new_rec:
            reason = "Rerouted due to User Pick" if newly_locked else "Double-pick portfolio optimization"
            log_rows.append([
                timestamp_str, trigger_description, f"Week {w}", old_rec, new_rec, survival_shift_str, reason
            ])

    if log_rows:
        log_sheet.append_rows(log_rows, value_input_option="USER_ENTERED")

def sync_to_google_sheets():
    print("Connecting to Google Sheets...")
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    client = gspread.authorize(creds)
    spreadsheet = client.open(SHEET_TITLE)
    
    try:
        sheet = spreadsheet.worksheet(TAB_NAME)
    except gspread.WorksheetNotFound:
        sheet = spreadsheet.add_worksheet(title=TAB_NAME, rows=300, cols=10)

    existing_data = sheet.get_all_values()
    locked_picks = {}
    previous_picks = {}
    previous_actuals = {}
    prev_prob = None

    # REWRITTEN: Safely read user picks regardless of row spacing
    if len(existing_data) > 1:
        for row in existing_data[1:]:
            if not row or len(row) < 1:
                continue
                
            match = re.match(r"^Week\s+(\d+)", row[0].strip(), re.IGNORECASE)
            if match:
                w = int(match.group(1))
                if len(row) >= 2 and row[1].strip():
                    previous_picks[w] = row[1].strip()
                if len(row) >= 4 and row[3].strip():
                    cell_val = row[3].strip()
                    locked_picks[w] = parse_actual_picks(cell_val)
                    previous_actuals[w] = cell_val
                    
            if "Season Survival" in row[0]:
                if len(row) >= 2:
                    prob_raw = row[1].replace("%", "").strip()
                    try:
                        prev_prob = float(prob_raw)
                    except ValueError:
                        pass

    print(f"Detected locked user picks: {locked_picks}")

    schedule_2026 = fetch_dynamic_schedule()
    live_odds_map = fetch_online_sportsbook_odds(odds_api_key)
    
    all_espn_odds = {}
    for w in range(1, WEEKS + 1):
        all_espn_odds.update(fetch_espn_live_odds(w))

    reconciled_schedule = reconcile_and_update_lines_tab(spreadsheet, schedule_2026, live_odds_map, all_espn_odds)
    all_weekly_slates = build_slates_from_reconciled(reconciled_schedule)
    optimal_picks_by_week, optimal_display = solve_survivor_path(all_weekly_slates, locked_picks)

    cum_prob = 1.0
    for w in range(1, WEEKS + 1):
        chosen_teams = locked_picks.get(w, []) if locked_picks.get(w) else optimal_picks_by_week.get(w, [])
        week_cands = all_weekly_slates.get(w, [])
        week_joint_prob = 1.0
        for t in chosen_teams:
            matched = next((c for c in week_cands if c["team"] == t and c["mod_prob"] is not None), None)
            if matched and matched["mod_prob"] is not None:
                week_joint_prob *= matched["mod_prob"]
            else:
                week_joint_prob *= 0.74
        if not chosen_teams:
            week_joint_prob = 0.74
        cum_prob *= week_joint_prob

    new_prob = cum_prob * 100.0

    sheet_weekly_display = {}
    total_display_rows = 1
    for w in range(1, WEEKS + 1):
        rec_teams = optimal_picks_by_week.get(w, [])
        user_teams = locked_picks.get(w, [])
        chosen_teams = list(set(rec_teams + user_teams))

        # NEW LOGIC: Filter out teams you officially burned in previous weeks
        officially_taken_before = set()
        for pw in range(1, w):
            for t in locked_picks.get(pw, []):
                officially_taken_before.add(t)

        valid_cands = [c for c in all_weekly_slates.get(w, []) if c["team"] not in officially_taken_before]

        top5 = valid_cands[:5]
        top5_teams = {c["team"] for c in top5}

        outside_picks = [c for c in valid_cands if c["team"] in chosen_teams and c["team"] not in top5_teams]
        curated_slate = top5 + outside_picks
        sheet_weekly_display[w] = curated_slate
        total_display_rows += (1 + len(curated_slate))

    sheet.clear()
    total_grid_rows = total_display_rows + 5 

    sheet.format(f"A1:I{total_grid_rows + 20}", {
        "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
        "textFormat": {"bold": False, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}
    })

    try:
        sheet.unmerge_cells(f"A1:I{total_grid_rows + 20}")
    except Exception:
        pass

    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick",
        "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %"
    ]

    matrix = [["" for _ in range(9)] for _ in range(total_grid_rows + 2)]
    matrix[0] = headers

    yellow_rows = []
    merge_ranges = []

    current_start_row = 2
    for w in range(1, WEEKS + 1):
        rec_display = optimal_display.get(w, "")
        user_actual_str = previous_actuals.get(w, "")
        
        # STRUCTURAL ALIGNMENT: Lock the left column precisely to the start of its candidate block on the right
        matrix[current_start_row - 1][0] = f"Week {w} (2 Picks)" if w in DOUBLE_PICK_WEEKS else f"Week {w}"
        matrix[current_start_row - 1][1] = rec_display
        matrix[current_start_row - 1][2] = ""
        matrix[current_start_row - 1][3] = user_actual_str

        rec_teams = optimal_picks_by_week.get(w, [])
        user_teams = locked_picks.get(w, [])
        chosen_set = set(rec_teams + user_teams)

        cands = sheet_weekly_display.get(w, [])

        label_suffix = " (DOUBLE PICK ROUND)" if w in DOUBLE_PICK_WEEKS else ""
        matrix[current_start_row - 1][4] = f"Top candidates for Week {w}{label_suffix}"
        merge_ranges.append(f"E{current_start_row}:I{current_start_row}")

        for i, cand in enumerate(cands):
            cand_row_num = current_start_row + 1 + i
            if cand["team"] in chosen_set:
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

        current_start_row += (1 + len(cands))

    season_row = current_start_row + 1
    matrix[season_row - 1][0] = "🏆 Season Survival"
    matrix[season_row - 1][1] = f"{new_prob:.2f}%"

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

    sheet.format(f"A{season_row}:B{season_row}", {
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

    log_adjustments_to_sheet(
        spreadsheet, previous_picks, optimal_display, previous_actuals, locked_picks, prev_prob, new_prob
    )
    print("Success: Structural alignment fixed and passed-week candidates filtered.")

if __name__ == "__main__":
    sync_to_google_sheets()
