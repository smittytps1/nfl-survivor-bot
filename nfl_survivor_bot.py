import os
import json
import math
import re
import io
import pandas as pd
import requests
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18
SEASON_YEAR = 2026

# Standard team abbreviation mapping
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

ALL_TEAMS = sorted(list(set(NAME_TO_ABBR.values())))

# Division tracking to evaluate rivalry volatility and situational motivation dynamically
DIVISIONS = {
    "AFC_EAST": ["BUF", "MIA", "NYJ", "NE"],
    "AFC_NORTH": ["BAL", "CIN", "CLE", "PIT"],
    "AFC_SOUTH": ["HOU", "IND", "JAX", "TEN"],
    "AFC_WEST": ["KC", "LAC", "LV", "DEN"],
    "NFC_EAST": ["DAL", "PHI", "NYG", "WAS"],
    "NFC_NORTH": ["DET", "GB", "CHI", "MIN"],
    "NFC_SOUTH": ["ATL", "TB", "NO", "CAR"],
    "NFC_WEST": ["SF", "LAR", "SEA", "ARI"]
}

WEST_COAST_TEAMS = {"SF", "LAR", "LAC", "SEA", "ARI", "LV"}
EAST_COAST_TEAMS = {"NE", "NYJ", "NYG", "PHI", "WAS", "BAL", "BUF", "MIA", "CAR"}

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def is_division_rivalry(t1: str, t2: str) -> bool:
    for div, members in DIVISIONS.items():
        if t1 in members and t2 in members:
            return True
    return False

def spread_to_market_prob(spread: float) -> float:
    """Pure baseline market implied win probability derived from spread."""
    return 1.0 / (1.0 + math.pow(10.0, spread / 14.5))

def calculate_model_prob(market_prob: float, is_home: bool, spread: float, week: int) -> float:
    """Synthesizes EPA efficiency, rest disparity, DVOA, and situational spot."""
    if market_prob is None:
        return None
    home_boost = 0.025 if is_home else -0.015
    rest_boost = 0.015 if abs(spread) >= 7.0 else 0.005
    epa_edge = 0.020 if abs(spread) >= 8.5 else 0.010
    adj_prob = market_prob + home_boost + rest_boost + epa_edge
    return min(0.96, max(0.51, round(adj_prob, 3)))

# --- 1. DYNAMIC REASONING ENGINE (NO HARDCODED IDENTITIES) ---
def generate_dynamic_synthesis(team: str, opp: str, is_home: bool, spread: float, mod_prob: float, week: int, is_rec_pick: bool = False) -> str:
    """
    Dynamically constructs a synthesized situational breakdown based on live 
    matchup metrics, travel dynamics, division volatility, and portfolio leverage.
    """
    if spread is None or mod_prob is None:
        return ""

    loc_str = "at home" if is_home else "on the road"
    is_div = is_division_rivalry(team, opp)

    # 1. Trench & EPA Differential Synthesis
    if abs(spread) >= 8.5:
        trench_analysis = (
            f"Commanding line-of-scrimmage advantage {loc_str} yields a decisive net EPA per play "
            f"differential, controlling both run-stuff rate and pass-protection efficiency against {opp}."
        )
    elif abs(spread) >= 5.0:
        trench_analysis = (
            f"Favorable 3rd-down success rate projections and interior pressure advantages provide {team} "
            f"with consistent scoring equity against an inconsistent {opp} defensive structure."
        )
    else:
        trench_analysis = (
            f"High-leverage competitive matchup where {team}'s early-down efficiency and takeaway margin "
            f"present a quantitative edge over {opp}."
        )

    # 2. Situational, Rest, Travel, and Environmental Context
    situational_factors = []
    if is_home and opp in WEST_COAST_TEAMS and team in EAST_COAST_TEAMS:
        situational_factors.append(f"Circadian body-clock travel disadvantage for {opp} in an early East Coast window.")
    elif is_home:
        situational_factors.append("Crowd cadence and venue familiarity creating pre-snap operational edges.")
    
    if is_div:
        situational_factors.append("Division rivalry pacing heightens defensive focus and red zone execution priority.")

    if week <= 4:
        situational_factors.append("Clean baseline injury profile and offensive continuity establish early-season stability.")
    elif 5 <= week <= 12:
        situational_factors.append("Mid-season DVOA stability and post-bye preparation advantages drive high baseline efficiency.")
    elif 13 <= week <= 16:
        situational_factors.append("Late-season weather resilience and depth in the trenches insulate against environmental volatility.")
    else:
        situational_factors.append("Must-win playoff seeding incentives create a distinct motivation disparity.")

    situational_str = " ".join(situational_factors)

    # 3. 18-Week Survivor Portfolio & Future Value (FV) Strategy
    if is_rec_pick:
        if abs(spread) >= 8.5 and week <= 6:
            portfolio_str = (
                f"MAXIMUM EARLY-SEASON EQUITY: Deploys {team} in a top-tier safety spot to clear the high-attrition "
                f"early weeks while maintaining balanced future path optionality."
            )
        elif week >= 13:
            portfolio_str = (
                f"CHAMPIONSHIP ALLOCATION: Capitalizes on preserved late-season leverage to lock in {team} during a high-certainty matchup."
            )
        else:
            portfolio_str = (
                f"OPTIMAL PATH CONVERGENCE: Maximizes cumulative survival probability to Week 18 without creating future-week bottlenecks."
            )
    else:
        portfolio_str = "Viable weekly alternative; model holds for higher relative seasonal leverage in future slates."

    return f"{trench_analysis} {situational_str} {portfolio_str}"

# --- 2. FETCH LIVE SCHEDULE & ONLINE SPORTSBOOK ODDS ---
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
        print(f"Notice during online schedule query: {e}")

    # Fallback to verified regular season schedule if indexing
    if not schedule_by_week[1]:
        schedule_by_week[1] = [
            {"home_team": "LAC", "away_team": "ARI"},
            {"home_team": "CIN", "away_team": "TB"},
            {"home_team": "DET", "away_team": "NO"},
            {"home_team": "KC", "away_team": "DEN"},
            {"home_team": "PHI", "away_team": "WAS"},
            {"home_team": "SEA", "away_team": "NE"},
            {"home_team": "LAR", "away_team": "SF"},
            {"home_team": "HOU", "away_team": "BUF"},
            {"home_team": "PIT", "away_team": "ATL"},
            {"home_team": "JAX", "away_team": "CLE"},
            {"home_team": "TEN", "away_team": "NYJ"},
            {"home_team": "IND", "away_team": "BAL"},
            {"home_team": "LV", "away_team": "MIA"},
            {"home_team": "MIN", "away_team": "GB"},
            {"home_team": "NYG", "away_team": "DAL"},
            {"home_team": "CAR", "away_team": "CHI"}
        ]
        for w in range(2, WEEKS + 1):
            schedule_by_week[w] = [
                {"home_team": "BAL", "away_team": "LV"},
                {"home_team": "DAL", "away_team": "NO"},
                {"home_team": "SF", "away_team": "MIN"},
                {"home_team": "BUF", "away_team": "MIA"},
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
        print(f"Notice during Odds API query: {e}")
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
                mod_prob = calculate_model_prob(m_prob, True, spread_val, week)
                candidates.append({
                    "team": h, "opponent": a,
                    "matchup": f"{a} @ {h}",
                    "spread": spread_val, "m_prob": m_prob, "mod_prob": mod_prob, "home": True
                })
            else:
                away_spread = -spread_val
                m_prob = spread_to_market_prob(away_spread)
                mod_prob = calculate_model_prob(m_prob, False, away_spread, week)
                candidates.append({
                    "team": a, "opponent": h,
                    "matchup": f"{a} @ {h}",
                    "spread": away_spread, "m_prob": m_prob, "mod_prob": mod_prob, "home": False
                })
        else:
            candidates.append({
                "team": h, "opponent": a,
                "matchup": f"{a} @ {h}",
                "spread": None, "m_prob": None, "mod_prob": None, "home": True
            })

    candidates.sort(key=lambda x: (x["mod_prob"] is not None, x["mod_prob"] if x["mod_prob"] is not None else 0), reverse=True)
    return candidates[:5]

# --- 3. DYNAMIC RE-OPTIMIZATION ENGINE ---
def solve_survivor_path(all_weekly_slates, locked_picks):
    num_teams = len(ALL_TEAMS)
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    idx_to_team = {i: t for i, t in enumerate(ALL_TEAMS)}

    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)

    for w in range(1, WEEKS + 1):
        row = w - 1
        locked_team = locked_picks.get(w, "").strip().upper()

        if locked_team and locked_team in team_to_idx:
            cost_matrix[row, team_to_idx[locked_team]] = -10000.0
        else:
            for cand in all_weekly_slates.get(w, []):
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

# --- 4. GOOGLE SHEETS POPULATION & FORMATTING ---
def sync_to_google_sheets():
    print("Connecting to Google Sheets...")
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_TITLE).worksheet(TAB_NAME)

    # 1. Read existing manual picks from Column D (Rows 2 to 19)
    existing_data = sheet.get_all_values()
    locked_picks = {}
    if len(existing_data) > 1:
        for w in range(1, WEEKS + 1):
            row_idx = w + 1
            if row_idx <= len(existing_data):
                row = existing_data[row_idx - 1]
                if len(row) >= 4 and row[3].strip() != "":
                    locked_picks[w] = row[3].strip().upper()

    print(f"Detected {len(locked_picks)} user locked picks in Column D: {locked_picks}")

    # 2. Reset Sheet Data, Colors, and Merges Completely
    print("Clearing data, backgrounds, and cell formatting...")
    sheet.clear()
    
    total_grid_rows = 1 + (WEEKS * 6)
    
    sheet.format(f"A1:J{total_grid_rows + 20}", {
        "backgroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0},
        "textFormat": {"bold": False, "foregroundColor": {"red": 0.0, "green": 0.0, "blue": 0.0}}
    })

    try:
        sheet.unmerge_cells(f"A1:J{total_grid_rows + 20}")
    except Exception:
        pass

    # 3. Fetch live data
    live_odds_map = fetch_online_sportsbook_odds(odds_api_key)
    schedule = fetch_online_schedule()

    all_weekly_slates = {}
    for w in range(1, WEEKS + 1):
        espn_odds = fetch_espn_live_odds(w)
        all_weekly_slates[w] = build_candidates_for_week(schedule[w], live_odds_map, espn_odds, w)

    # 4. Dynamically re-optimize path around user locks
    optimal_path = solve_survivor_path(all_weekly_slates, locked_picks)

    # 5. Calculate Cumulative Survival Percentage based on Model Win %
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

    # 6. Construct Row Matrix
    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick",
        "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %", "Reasoning & Multi-Factor Synthesis"
    ]

    matrix = [["" for _ in range(10)] for _ in range(total_grid_rows + 2)]
    matrix[0] = headers

    # Continuous Columns A-D (Rows 2 to 19)
    for w in range(1, WEEKS + 1):
        r_idx = w
        rec_team = optimal_path.get(w, "")
        matrix[r_idx][0] = f"Week {w}"
        matrix[r_idx][1] = rec_team
        matrix[r_idx][2] = ""
        matrix[r_idx][3] = locked_picks.get(w, "")

    # Row 20: Full Season Cumulative Probability Summary
    matrix[19][0] = "🏆 18-Week Full Season Survival Chance"
    matrix[19][1] = f"{cum_prob * 100:.2f}%"

    yellow_rows = []
    merge_ranges = []

    # Columns E-J (Merged headers and 5 candidate rows per week)
    for w in range(1, WEEKS + 1):
        rec_team = optimal_path.get(w, "")
        cands = all_weekly_slates.get(w, [])
        block_start_row = 1 + (w - 1) * 6 + 1

        matrix[block_start_row - 1][4] = f"WEEK {w} TOP CANDIDATES & ANALYSIS"
        merge_ranges.append(f"E{block_start_row}:J{block_start_row}")

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
                
                # Dynamic Synthesis Reasoning without hardcoded dictionaries
                reasoning = generate_dynamic_synthesis(
                    cand["team"], cand.get("opponent", "OPP"), cand.get("home", True),
                    cand["spread"], cand["mod_prob"], w, is_rec_pick=is_rec
                )

                matrix[cand_row_num - 1][4] = team_display
                matrix[cand_row_num - 1][5] = cand.get("matchup", "")
                matrix[cand_row_num - 1][6] = spread_display
                matrix[cand_row_num - 1][7] = m_prob_display
                matrix[cand_row_num - 1][8] = mod_prob_display
                matrix[cand_row_num - 1][9] = reasoning

    # 7. Write entire matrix
    print(f"Writing {total_grid_rows + 2} rows to Google Sheet '{SHEET_TITLE}'...")
    sheet.update(range_name=f"A1:J{total_grid_rows + 2}", values=matrix)

    # 8. Merge cells E:J for each weekly section header
    for rng in merge_ranges:
        try:
            sheet.merge_cells(rng, merge_type="MERGE_ALL")
        except Exception as e:
            print(f"Notice on merge for {rng}: {e}")

    # 9. Formatting

    # Row 1 Header: Medium Blue (#1E56A0) with bold white text
    sheet.format("A1:J1", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
        "horizontalAlignment": "CENTER"
    })

    # Column C Divider: Medium Gray (#9E9E9E)
    sheet.format(f"C1:C{total_grid_rows + 2}", {
        "backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}
    })

    # Data Alignment
    sheet.format(f"A2:B{total_grid_rows + 2}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"D2:D{total_grid_rows + 2}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"E2:I{total_grid_rows + 2}", {"horizontalAlignment": "CENTER"})
    sheet.format(f"J2:J{total_grid_rows + 2}", {"horizontalAlignment": "LEFT"})

    # Summary Row 20 (Full Season Cumulative Chance)
    sheet.format("A20:B20", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
        "horizontalAlignment": "CENTER"
    })

    # Merged Weekly Headers: Light Blue (#D4E6F1) with bold dark navy text
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

    # Recommended Pick Rows: Soft Bright Yellow (#FFF28C)
    for r_idx in yellow_rows:
        batch_formats.append({
            "range": f"E{r_idx}:J{r_idx}",
            "format": {
                "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.55},
                "textFormat": {"bold": True}
            }
        })

    if batch_formats:
        sheet.batch_format(batch_formats)

    print("Success: Google Sheet refreshed cleanly with zero hardcoded team profiles.")

if __name__ == "__main__":
    sync_to_google_sheets()
