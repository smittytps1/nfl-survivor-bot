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

ALL_TEAMS = sorted(list(set(NAME_TO_ABBR.values())))

BASELINE_SEASON_SLATES = {
    1: [
        {"team": "LAC", "opponent": "ARI", "matchup": "ARI @ LAC", "spread": -10.5, "is_home": True},
        {"team": "JAX", "opponent": "CLE", "matchup": "CLE @ JAX", "spread": -9.0, "is_home": True},
        {"team": "DET", "opponent": "NO", "matchup": "NO @ DET", "spread": -7.0, "is_home": True},
        {"team": "PHI", "opponent": "WAS", "matchup": "WAS @ PHI", "spread": -5.5, "is_home": True},
        {"team": "LAR", "opponent": "SF", "matchup": "SF @ LAR", "spread": -4.0, "is_home": True},
    ],
    2: [
        {"team": "SF", "opponent": "MIA", "matchup": "MIA @ SF", "spread": -10.5, "is_home": True},
        {"team": "LAR", "opponent": "NYG", "matchup": "NYG @ LAR", "spread": -9.5, "is_home": True},
        {"team": "SEA", "opponent": "ARI", "matchup": "SEA @ ARI", "spread": -10.0, "is_home": False},
        {"team": "LAC", "opponent": "LV", "matchup": "LV @ LAC", "spread": -9.0, "is_home": True},
        {"team": "BAL", "opponent": "NO", "matchup": "NO @ BAL", "spread": -7.5, "is_home": True},
    ],
    3: [
        {"team": "SF", "opponent": "ARI", "matchup": "ARI @ SF", "spread": -11.5, "is_home": True},
        {"team": "DET", "opponent": "NYJ", "matchup": "NYJ @ DET", "spread": -10.0, "is_home": True},
        {"team": "GB", "opponent": "ATL", "matchup": "ATL @ GB", "spread": -7.5, "is_home": True},
        {"team": "KC", "opponent": "MIA", "matchup": "KC @ MIA", "spread": -7.5, "is_home": False},
        {"team": "NYG", "opponent": "TEN", "matchup": "TEN @ NYG", "spread": -4.0, "is_home": True},
    ],
    4: [
        {"team": "BAL", "opponent": "TEN", "matchup": "TEN @ BAL", "spread": -9.0, "is_home": True},
        {"team": "CHI", "opponent": "NYJ", "matchup": "NYJ @ CHI", "spread": -9.0, "is_home": True},
        {"team": "MIN", "opponent": "MIA", "matchup": "MIA @ MIN", "spread": -7.5, "is_home": True},
        {"team": "NYG", "opponent": "ARI", "matchup": "ARI @ NYG", "spread": -7.0, "is_home": True},
        {"team": "BUF", "opponent": "NE", "matchup": "NE @ BUF", "spread": -4.0, "is_home": True},
    ],
    5: [
        {"team": "NE", "opponent": "LV", "matchup": "LV @ NE", "spread": -8.5, "is_home": True},
        {"team": "DET", "opponent": "ARI", "matchup": "DET @ ARI", "spread": -8.5, "is_home": False},
        {"team": "CIN", "opponent": "MIA", "matchup": "CIN @ MIA", "spread": -6.5, "is_home": False},
        {"team": "LAR", "opponent": "BUF", "matchup": "BUF @ LAR", "spread": -4.5, "is_home": True},
        {"team": "BAL", "opponent": "ATL", "matchup": "BAL @ ATL", "spread": -4.5, "is_home": False},
    ],
    6: [
        {"team": "LAR", "opponent": "ARI", "matchup": "ARI @ LAR", "spread": -14.5, "is_home": True},
        {"team": "NE", "opponent": "NYJ", "matchup": "NYJ @ NE", "spread": -10.0, "is_home": True},
        {"team": "PHI", "opponent": "CAR", "matchup": "CAR @ PHI", "spread": -7.0, "is_home": True},
        {"team": "BUF", "opponent": "LV", "matchup": "BUF @ LV", "spread": -7.0, "is_home": True},
        {"team": "BAL", "opponent": "CLE", "matchup": "BAL @ CLE", "spread": -7.0, "is_home": False},
    ],
    7: [
        {"team": "LAR", "opponent": "LV", "matchup": "LAR @ LV", "spread": -8.5, "is_home": False},
        {"team": "DEN", "opponent": "ARI", "matchup": "DEN @ ARI", "spread": -7.5, "is_home": False},
        {"team": "HOU", "opponent": "NYG", "matchup": "NYG @ HOU", "spread": -6.0, "is_home": True},
        {"team": "SF", "opponent": "ATL", "matchup": "SF @ ATL", "spread": -4.5, "is_home": False},
        {"team": "BAL", "opponent": "CIN", "matchup": "CIN @ BAL", "spread": -4.0, "is_home": True},
    ],
    8: [
        {"team": "DAL", "opponent": "ARI", "matchup": "ARI @ DAL", "spread": -10.5, "is_home": True},
        {"team": "GB", "opponent": "CAR", "matchup": "CAR @ GB", "spread": -7.5, "is_home": True},
        {"team": "CIN", "opponent": "TEN", "matchup": "TEN @ CIN", "spread": -7.0, "is_home": True},
        {"team": "NE", "opponent": "MIA", "matchup": "NE @ MIA", "spread": -7.0, "is_home": False},
        {"team": "PIT", "opponent": "CLE", "matchup": "CLE @ PIT", "spread": -6.0, "is_home": True},
    ],
    9: [
        {"team": "SEA", "opponent": "ARI", "matchup": "ARI @ SEA", "spread": -13.5, "is_home": True},
        {"team": "KC", "opponent": "NYJ", "matchup": "NYJ @ KC", "spread": -10.0, "is_home": True},
        {"team": "SF", "opponent": "LV", "matchup": "LV @ SF", "spread": -9.5, "is_home": True},
        {"team": "DET", "opponent": "MIA", "matchup": "DET @ MIA", "spread": -6.5, "is_home": False},
        {"team": "PHI", "opponent": "NYG", "matchup": "NYG @ PHI", "spread": -6.0, "is_home": True},
    ],
    10: [
        {"team": "LAR", "opponent": "ARI", "matchup": "LAR @ ARI", "spread": -10.5, "is_home": False},
        {"team": "IND", "opponent": "MIA", "matchup": "MIA @ IND", "spread": -7.0, "is_home": True},
        {"team": "BUF", "opponent": "NYJ", "matchup": "BUF @ NYJ", "spread": -7.0, "is_home": False},
        {"team": "SEA", "opponent": "LV", "matchup": "SEA @ LV", "spread": -7.0, "is_home": False},
        {"team": "GB", "opponent": "MIN", "matchup": "MIN @ GB", "spread": -4.5, "is_home": True},
    ],
    11: [
        {"team": "BUF", "opponent": "MIA", "matchup": "MIA @ BUF", "spread": -12.5, "is_home": True},
        {"team": "KC", "opponent": "ARI", "matchup": "ARI @ KC", "spread": -11.5, "is_home": True},
        {"team": "LAC", "opponent": "NYJ", "matchup": "NYJ @ LAC", "spread": -10.5, "is_home": True},
        {"team": "DEN", "opponent": "LV", "matchup": "LV @ DEN", "spread": -8.5, "is_home": True},
        {"team": "DAL", "opponent": "TEN", "matchup": "TEN @ DAL", "spread": -7.0, "is_home": True},
    ],
    12: [
        {"team": "CIN", "opponent": "NO", "matchup": "NO @ CIN", "spread": -6.5, "is_home": True},
        {"team": "JAX", "opponent": "TEN", "matchup": "TEN @ JAX", "spread": -6.5, "is_home": True},
        {"team": "LAR", "opponent": "GB", "matchup": "GB @ LAR", "spread": -5.5, "is_home": True},
        {"team": "MIN", "opponent": "ATL", "matchup": "ATL @ MIN", "spread": -4.5, "is_home": True},
        {"team": "TB", "opponent": "CAR", "matchup": "CAR @ TB", "spread": -4.5, "is_home": True},
    ],
    13: [
        {"team": "DEN", "opponent": "MIA", "matchup": "MIA @ DEN", "spread": -9.5, "is_home": True},
        {"team": "PHI", "opponent": "ARI", "matchup": "PHI @ ARI", "spread": -8.5, "is_home": False},
        {"team": "LAR", "opponent": "KC", "matchup": "KC @ LAR", "spread": -5.0, "is_home": True},
        {"team": "SEA", "opponent": "DAL", "matchup": "DAL @ SEA", "spread": -4.5, "is_home": True},
        {"team": "CIN", "opponent": "CLE", "matchup": "CIN @ CLE", "spread": -4.5, "is_home": False},
    ],
    14: [
        {"team": "DET", "opponent": "TEN", "matchup": "TEN @ DET", "spread": -7.5, "is_home": True},
        {"team": "SEA", "opponent": "NYG", "matchup": "NYG @ SEA", "spread": -7.5, "is_home": True},
        {"team": "BAL", "opponent": "TB", "matchup": "TB @ BAL", "spread": -6.0, "is_home": True},
        {"team": "NE", "opponent": "MIN", "matchup": "MIN @ NE", "spread": -5.5, "is_home": True},
        {"team": "PHI", "opponent": "IND", "matchup": "IND @ PHI", "spread": -5.5, "is_home": True},
    ],
    15: [
        {"team": "GB", "opponent": "MIA", "matchup": "MIA @ GB", "spread": -10.5, "is_home": True},
        {"team": "LAR", "opponent": "DAL", "matchup": "DAL @ LAR", "spread": -7.5, "is_home": True},
        {"team": "NYG", "opponent": "CLE", "matchup": "CLE @ NYG", "spread": -4.5, "is_home": True},
        {"team": "DEN", "opponent": "LV", "matchup": "DEN @ LV", "spread": -4.5, "is_home": False},
        {"team": "BUF", "opponent": "CHI", "matchup": "CHI @ BUF", "spread": -4.0, "is_home": True},
    ],
    16: [
        {"team": "BAL", "opponent": "CLE", "matchup": "CLE @ BAL", "spread": -10.5, "is_home": True},
        {"team": "LAC", "opponent": "MIA", "matchup": "LAC @ MIA", "spread": -7.0, "is_home": False},
        {"team": "DET", "opponent": "NYG", "matchup": "NYG @ DET", "spread": -6.5, "is_home": True},
        {"team": "NE", "opponent": "NYJ", "matchup": "NE @ NYJ", "spread": -6.5, "is_home": False},
        {"team": "NO", "opponent": "ARI", "matchup": "ARI @ NO", "spread": -5.5, "is_home": True},
    ],
    17: [
        {"team": "BUF", "opponent": "MIA", "matchup": "BUF @ MIA", "spread": -7.5, "is_home": False},
        {"team": "DAL", "opponent": "NYG", "matchup": "NYG @ DAL", "spread": -5.5, "is_home": True},
        {"team": "SEA", "opponent": "CAR", "matchup": "SEA @ CAR", "spread": -5.5, "is_home": False},
        {"team": "LAR", "opponent": "TB", "matchup": "LAR @ TB", "spread": -4.5, "is_home": False},
        {"team": "JAX", "opponent": "WAS", "matchup": "WAS @ JAX", "spread": -3.5, "is_home": True},
    ],
    18: [
        {"team": "NE", "opponent": "MIA", "matchup": "MIA @ NE", "spread": -10.5, "is_home": True},
        {"team": "BUF", "opponent": "NYJ", "matchup": "NYJ @ BUF", "spread": -10.0, "is_home": True},
        {"team": "KC", "opponent": "LV", "matchup": "LV @ KC", "spread": -8.5, "is_home": True},
        {"team": "SF", "opponent": "ARI", "matchup": "SF @ ARI", "spread": -8.5, "is_home": False},
        {"team": "CIN", "opponent": "CLE", "matchup": "CLE @ CIN", "spread": -7.5, "is_home": True},
    ],
}

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

def read_existing_lines_from_sheet(sheet_data):
    """
    Parses Column E (Team), Column F (Matchup), and Column G (Line) from the existing
    sheet data so that prior verified spreads are never lost when live APIs lack data.
    """
    existing_lines = {}
    if not sheet_data or len(sheet_data) < 2:
        return existing_lines

    for row in sheet_data[1:]:
        if len(row) >= 7:
            team_raw = row[4].replace("*", "").strip()
            team_abbr = team_to_abbr(team_raw)
            line_str = row[6].strip()

            try:
                line_val = float(line_str)
                if team_abbr and line_val != 0.0:
                    existing_lines[team_abbr] = line_val
            except ValueError:
                pass
    return existing_lines

def build_slates_with_persistent_fallback(live_odds_map, sheet_existing_lines):
    """
    1. Check live online bookmaker odds.
    2. Fallback to the existing line recorded on the spreadsheet.
    3. Fallback to baseline opening season line.
    Only overwrites if a new valid line is found.
    """
    all_slates = {}
    for w in range(1, WEEKS + 1):
        espn_odds = fetch_espn_live_odds(w)
        all_slates[w] = []

        baseline_games = BASELINE_SEASON_SLATES.get(w, [])
        for game in baseline_games:
            team = game["team"]
            opp = game["opponent"]
            is_home = game["is_home"]
            h = team if is_home else opp
            a = opp if is_home else team

            chosen_spread = None

            # Priority 1: Check Live Online Odds
            if (h, a) in live_odds_map:
                h_spread = live_odds_map[(h, a)]
                chosen_spread = h_spread if is_home else -h_spread
            elif (h, a) in espn_odds:
                h_spread = espn_odds[(h, a)]
                chosen_spread = h_spread if is_home else -h_spread

            # Priority 2: Retain the Existing Line Recorded on the Sheet
            if chosen_spread is None and team in sheet_existing_lines:
                chosen_spread = sheet_existing_lines[team]

            # Priority 3: Fall back to Baseline Opening Season Line
            if chosen_spread is None:
                chosen_spread = game["spread"]

            m_prob = spread_to_market_prob(chosen_spread)
            mod_prob = calculate_model_prob(m_prob, is_home, chosen_spread, w, opp, team)

            all_slates[w].append({
                "team": team,
                "opponent": opp,
                "matchup": game["matchup"],
                "spread": chosen_spread,
                "is_home": is_home,
                "m_prob": m_prob,
                "mod_prob": mod_prob
            })

        all_slates[w].sort(key=lambda x: (x["mod_prob"] is not None, x["mod_prob"]), reverse=True)
    return all_slates

def solve_survivor_path(all_weekly_slates, locked_picks):
    used_teams = set()
    optimal = {w: [] for w in range(1, WEEKS + 1)}

    # Pre-seed locked user picks
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
                    if fc["team"] == team and get_safe_abs_spread(fc) >= 9.5
                )

                better_spot_soon = any(
                    get_safe_abs_spread(fc) >= (spread + 1.5)
                    for fw in [w + 1, w + 2] if fw <= WEEKS
                    for fc in all_weekly_slates.get(fw, [])
                    if fc["team"] == team
                )

                score = spread * 10.0

                if w <= 6:
                    fv_weight = 3.0
                elif w <= 14:
                    fv_weight = 8.0
                else:
                    fv_weight = 3.0

                if spread < 12.0:
                    score -= (future_heavy_spots * fv_weight)

                if better_spot_soon:
                    score -= 30.0

                if w <= 6 and is_divisional_road_game(team, opp, is_home):
                    score -= 40.0

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

    trigger_description = "; ".join(newly_locked) if newly_locked else "Live Odds Search / Persistent Sync"
    log_rows = []
    timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    survival_shift_str = f"{prev_prob:.2f}% -> {new_prob:.2f}%" if prev_prob is not None else f"{new_prob:.2f}%"

    for w in range(1, WEEKS + 1):
        old_rec = previous_picks.get(w, "")
        new_rec = current_picks.get(w, "")

        if old_rec and new_rec and old_rec != new_rec:
            reason = "Rerouted due to User Pick" if newly_locked else "Line movement / EV shift"
            log_rows.append([
                timestamp_str, trigger_description, f"Week {w}", old_rec, new_rec, survival_shift_str, reason
            ])

    if log_rows:
        log_sheet.append_rows(log_rows, value_input_option="USER_ENTERED")
        print(f"Logged {len(log_rows)} schedule adjustments to '{LOG_TAB_NAME}'.")
    else:
        print("No recommendations changed; audit log remains up to date.")

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
    sheet = spreadsheet.worksheet(TAB_NAME)

    # 1. READ EXISTING DATA PRIOR TO ANY MODIFICATIONS
    existing_data = sheet.get_all_values()
    locked_picks = {}
    previous_picks = {}
    previous_actuals = {}
    prev_prob = None

    if len(existing_data) > 1:
        for w in range(1, WEEKS + 1):
            row_idx = w + 1
            if row_idx <= len(existing_data):
                row = existing_data[row_idx - 1]
                if len(row) >= 2 and row[1].strip():
                    previous_picks[w] = row[1].strip()
                if len(row) >= 4 and row[3].strip():
                    cell_val = row[3].strip()
                    locked_picks[w] = parse_actual_picks(cell_val)
                    previous_actuals[w] = cell_val

        if len(existing_data) >= 20 and len(existing_data[19]) >= 2:
            prob_raw = existing_data[19][1].replace("%", "").strip()
            try:
                prev_prob = float(prob_raw)
            except ValueError:
                pass

    # Extract all currently recorded lines to serve as persistent fallback
    sheet_existing_lines = read_existing_lines_from_sheet(existing_data)
    print(f"Preserved {len(sheet_existing_lines)} existing game lines from spreadsheet.")
    print(f"Detected user locked picks: {locked_picks}")

    # 2. SEARCH ONLINE SOURCES FOR UPDATED ODDS
    live_odds_map = fetch_online_sportsbook_odds(odds_api_key)
    print(f"Retrieved {len(live_odds_map)} updated live sportsbook lines.")

    # 3. BUILD ALL 18 WEEKS (Search -> Sheet Fallback -> Baseline)
    all_weekly_slates = build_slates_with_persistent_fallback(live_odds_map, sheet_existing_lines)

    # 4. SOLVE SURVIVOR SCHEDULE (22 Teams, Weeks 15-18 Double Picks)
    optimal_picks_by_week, optimal_display = solve_survivor_path(all_weekly_slates, locked_picks)

    # 5. COMPUTE CUMULATIVE SURVIVAL PROBABILITY
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

    # 6. WRITE CLEAN UPDATED MATRIX TO GOOGLE SHEETS
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

    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick",
        "Candidate Team", "Matchup", "Line", "Market Win %", "Model Win %"
    ]

    matrix = [["" for _ in range(9)] for _ in range(total_grid_rows + 2)]
    matrix[0] = headers

    for w in range(1, WEEKS + 1):
        r_idx = w
        rec_display = optimal_display.get(w, "")
        user_actual_str = previous_actuals.get(w, "")
        matrix[r_idx][0] = f"Week {w} (2 Picks)" if w in DOUBLE_PICK_WEEKS else f"Week {w}"
        matrix[r_idx][1] = rec_display
        matrix[r_idx][2] = ""
        matrix[r_idx][3] = user_actual_str

    matrix[19][0] = "🏆 Season Survival"
    matrix[19][1] = f"{new_prob:.2f}%"

    yellow_rows = []
    merge_ranges = []

    for w in range(1, WEEKS + 1):
        rec_teams = optimal_picks_by_week.get(w, [])
        cands = all_weekly_slates.get(w, [])[:5]
        block_start_row = 1 + (w - 1) * 6 + 1

        label_suffix = " (DOUBLE PICK ROUND)" if w in DOUBLE_PICK_WEEKS else ""
        matrix[block_start_row - 1][4] = f"Top candidates for Week {w}{label_suffix}"
        merge_ranges.append(f"E{block_start_row}:I{block_start_row}")

        for i in range(5):
            cand_row_num = block_start_row + 1 + i
            if i < len(cands):
                cand = cands[i]
                is_rec = cand["team"] in rec_teams
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

    log_adjustments_to_sheet(
        spreadsheet, previous_picks, optimal_display, previous_actuals, locked_picks, prev_prob, new_prob
    )

    print("Success: Google Sheet updated cleanly with search-first persistent line hierarchy.")

if __name__ == "__main__":
    sync_to_google_sheets()
