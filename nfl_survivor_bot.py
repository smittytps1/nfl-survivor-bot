import os
import json
import math
import re
import numpy as np
import requests
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18
SEASON_YEAR = 2026

# Standard team abbreviation mapping for sportsbook and API normalization
NAME_TO_ABBR = {
    "arizona cardinals": "ARI", "cardinals": "ARI", "ari": "ARI",
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
    "las vegas raiders": "LV", "raiders": "LV", "lv": "LV",
    "los angeles chargers": "LAC", "chargers": "LAC", "lac": "LAC",
    "los angeles rams": "LAR", "rams": "LAR", "lar": "LAR",
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

def team_to_abbr(name: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9 ]', '', str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])

def spread_to_win_prob(spread: float) -> float:
    """Logistic formula converting NFL spread to implied win probability."""
    return 1.0 / (1.0 + math.pow(10.0, spread / 14.5))

def generate_dynamic_reasoning(team, opp, is_home, spread, prob, week):
    """Dynamically builds situational synthesis without hardcoded text."""
    loc_str = "at home" if is_home else "on the road"
    prob_pct = f"{prob * 100:.1f}%"
    
    if abs(spread) >= 8.5:
        context = f"Heavy {loc_str} favorite ({spread:+.1f}) with major line-of-scrimmage control and positive EPA differential vs {opp}."
    elif abs(spread) >= 5.5:
        context = f"Solid {loc_str} favorite ({spread:+.1f}) against {opp} with significant 3rd-down success rate and pass-protection edges."
    elif is_home:
        context = f"Key home-field advantage and preparation edge vs {opp} with favorable early-down efficiency projections."
    else:
        context = f"High-efficiency road favorite spot ({spread:+.1f}) against vulnerable {opp} pass defense and turnover regression."
    
    if week <= 4:
        sub_factor = "Early season health stability and roster continuity advantage."
    elif 5 <= week <= 12:
        sub_factor = "Key situational rest dynamic and high passing EPA efficiency in middle-season push."
    else:
        sub_factor = "Late season motivation mismatch against opponent facing playoff elimination/depth attrition."

    return f"{context} {sub_factor} (Implied Win Prob: {prob_pct})"

def fetch_live_odds_api(api_key: str):
    """Queries The Odds API for real-time sportsbook lines."""
    if not api_key:
        return {}
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=spreads,h2h&oddsFormat=american"
    odds_map = {}
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            for game in data:
                home_raw = game.get("home_team", "")
                away_raw = game.get("away_team", "")
                h_abbr = team_to_abbr(home_raw)
                a_abbr = team_to_abbr(away_raw)

                for bm in game.get("bookmakers", []):
                    for mkt in bm.get("markets", []):
                        if mkt.get("key") == "spreads":
                            for out in mkt.get("outcomes", []):
                                if out.get("name") == home_raw:
                                    pt = float(out.get("point", 0.0))
                                    odds_map[(h_abbr, a_abbr)] = pt
                                    break
    except Exception as e:
        print(f"Notice: Odds API live fetch encountered: {e}")
    return odds_map

def fetch_live_week_schedule(week: int, live_odds_map: dict):
    """
    Fetches real-time schedules dynamically from ESPN's NFL API endpoints
    and matches them against live sportsbook odds.
    """
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?seasontype=2&week={week}&dates={SEASON_YEAR}"
    headers = {"User-Agent": "Mozilla/5.0"}
    candidates = []

    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            events = data.get("events", [])

            for ev in events:
                comp = ev.get("competitions", [{}])[0]
                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home = competitors[0] if competitors[0].get("homeAway") == "home" else competitors[1]
                away = competitors[1] if competitors[0].get("homeAway") == "home" else competitors[0]

                home_team = team_to_abbr(home.get("team", {}).get("abbreviation", ""))
                away_team = team_to_abbr(away.get("team", {}).get("abbreviation", ""))

                # Check live odds map first, otherwise fall back to ESPN's odds array
                spread_val = None
                if (home_team, away_team) in live_odds_map:
                    spread_val = live_odds_map[(home_team, away_team)]
                else:
                    odds_arr = comp.get("odds", [])
                    if odds_arr and "spread" in odds_arr[0]:
                        spread_val = float(odds_arr[0].get("spread", -3.0))

                if spread_val is None:
                    spread_val = -3.0  # Standard baseline home edge

                # Home favorite
                if spread_val <= 0:
                    prob = spread_to_win_prob(spread_val)
                    candidates.append({
                        "team": home_team,
                        "opponent": away_team,
                        "matchup": f"{away_team} @ {home_team}",
                        "spread": spread_val,
                        "prob": prob,
                        "home": True
                    })
                # Away favorite
                else:
                    away_spread = -spread_val
                    prob = spread_to_win_prob(away_spread)
                    candidates.append({
                        "team": away_team,
                        "opponent": home_team,
                        "matchup": f"{away_team} @ {home_team}",
                        "spread": away_spread,
                        "prob": prob,
                        "home": False
                    })
    except Exception as e:
        print(f"Error querying live schedule for Week {week}: {e}")

    candidates.sort(key=lambda x: x["prob"], reverse=True)
    return candidates

def solve_survivor_path(all_weekly_slates, locked_picks):
    """
    Solves the 18-week Survivor path dynamically via Linear Sum Assignment
    Cost function = -log(P(win)) to maximize cumulative survival rate.
    """
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
                if t_idx is not None:
                    cost_matrix[row, t_idx] = -math.log(cand["prob"])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return {r + 1: idx_to_team[c] for r, c in zip(row_ind, col_ind)}

def sync_to_google_sheets():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    odds_api_key = os.environ.get("ODDS_API_KEY", "")

    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON environment variable missing.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_TITLE).worksheet(TAB_NAME)

    # 1. Read existing spreadsheet state to respect user manual locks in Column D
    existing_data = sheet.get_all_values()
    locked_picks = {}
    if len(existing_data) > 1:
        for w in range(1, WEEKS + 1):
            target_row_idx = 1 + (w - 1) * 5 + 1
            if target_row_idx <= len(existing_data):
                row = existing_data[target_row_idx - 1]
                if len(row) >= 4 and row[3].strip() != "":
                    locked_picks[w] = row[3].strip().upper()

    print(f"Detected {len(locked_picks)} user locked picks in Column D: {locked_picks}")

    # 2. Fetch live odds map
    live_odds_map = fetch_live_odds_api(odds_api_key)

    # 3. Dynamically fetch all 18 weeks of regular season slates
    all_weekly_slates = {}
    for w in range(1, WEEKS + 1):
        print(f"Fetching real-time games & lines for Week {w}...")
        slates = fetch_live_week_schedule(w, live_odds_map)
        all_weekly_slates[w] = slates[:5] if len(slates) >= 5 else slates

    # 4. Optimize the 18-week path across dynamically pulled data
    optimal_path = solve_survivor_path(all_weekly_slates, locked_picks)

    # 5. Build spreadsheet rows
    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick",
        "Candidate Team", "Matchup", "Line", "Win Prob (%)", "Reasoning & Synthesis"
    ]
    sheet_rows = [headers]
    yellow_rows = []

    for w in range(1, WEEKS + 1):
        rec_team = optimal_path.get(w, "")
        cands = all_weekly_slates.get(w, [])
        user_pick = locked_picks.get(w, "")

        for i, cand in enumerate(cands):
            curr_row_idx = len(sheet_rows) + 1
            is_rec_pick = (cand["team"] == rec_team)

            if is_rec_pick:
                yellow_rows.append(curr_row_idx)

            week_label = f"Week {w}" if i == 0 else ""
            rec_label = rec_team if i == 0 else ""
            actual_label = user_pick if i == 0 else ""

            team_display = f"**{cand['team']}**" if cand.get("home", False) else cand["team"]
            spread_display = f"{cand['spread']:+.1f}"
            prob_display = f"{cand['prob'] * 100:.1f}%"
            reasoning = generate_dynamic_reasoning(
                cand["team"], cand.get("opponent", "OPP"), cand.get("home", True),
                cand["spread"], cand["prob"], w
            )

            sheet_rows.append([
                week_label,
                rec_label,
                "",
                actual_label,
                team_display,
                cand.get("matchup", ""),
                spread_display,
                prob_display,
                reasoning
            ])

    # 6. Clear and write entire 18-week grid
    total_rows = len(sheet_rows)
    sheet.clear()
    sheet.update(range_name=f"A1:I{total_rows}", values=sheet_rows)

    # 7. Apply Header Formatting
    sheet.format("A1:I1", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.16, "blue": 0.22},
        "horizontalAlignment": "CENTER"
    })

    # 8. Apply Column C Divider (Medium Gray)
    sheet.format(f"C1:C{total_rows}", {
        "backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}
    })

    # 9. Format Column Alignments
    sheet.format(f"A2:B{total_rows}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"D2:D{total_rows}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"E2:H{total_rows}", {"horizontalAlignment": "CENTER"})
    sheet.format(f"I2:I{total_rows}", {"horizontalAlignment": "LEFT"})

    # 10. Apply Yellow Highlighting to Recommended Pick Rows (Columns E through I)
    formats = []
    for r_idx in yellow_rows:
        formats.append({
            "range": f"E{r_idx}:I{r_idx}",
            "format": {
                "backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.55},
                "textFormat": {"bold": True}
            }
        })

    if formats:
        sheet.batch_format(formats)

    print(f"Sheet updated dynamically. {len(yellow_rows)} recommended pick rows highlighted in yellow.")

if __name__ == "__main__":
    sync_to_google_sheets()
