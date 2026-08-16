import os
import json
import math
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18

# --- 18-WEEK SYNTHESIZED NFL DATA SLATE ---
# Format: [Team, Spread (negative = fav), Implied Prob, Is_Home, Reasoning/Situational Note]
SEASON_SLATES = {
    1: [
        {"team": "CIN", "spread": -8.5, "prob": 0.784, "home": True, "reason": "Favorable home opener vs. rebuilding pass defense; high baseline pass EPA."},
        {"team": "BUF", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Home efficiency edge against opponent with multiple OL question marks."},
        {"team": "KC", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Elite coaching and continuity advantage; strong early-season DVOA profile."},
        {"team": "MIA", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Early-season South Florida heat and humidity conditioning advantage."},
        {"team": "PHI", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Rushing EPA and line-of-scrimmage advantage against soft interior run defense."}
    ],
    2: [
        {"team": "BAL", "spread": -9.5, "prob": 0.808, "home": True, "reason": "Heavy rushing EPA advantage; opponent on short week traveling across 2 time zones."},
        {"team": "DAL", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Home pass-rush pressure rate mismatch against vulnerable road pass protection."},
        {"team": "SF", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Top-tier offensive success rate and early defensive health metrics."},
        {"team": "DET", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Indoor dome offensive efficiency against young, rebuilding secondary."},
        {"team": "LAC", "spread": -4.5, "prob": 0.655, "home": False, "reason": "Defensive front seven edge in low-scoring defensive battle."}
    ],
    3: [
        {"team": "SF", "spread": -10.5, "prob": 0.830, "home": True, "reason": "Dominant defensive front vs. opponent starting backup left tackle."},
        {"team": "NYJ", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Pressure rate mismatch against backup QB; sharp money backing home favorite."},
        {"team": "CLE", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Defensive EPA at home in physical division matchup."},
        {"team": "TB", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Turnover margin positive regression spot at home."},
        {"team": "SEA", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Loud home field environment against rookie QB on 1st road start."}
    ],
    4: [
        {"team": "MIA", "spread": -8.0, "prob": 0.771, "home": True, "reason": "Passing EPA edge against opponent missing primary boundary cornerbacks."},
        {"team": "KC", "spread": -7.5, "prob": 0.756, "home": False, "reason": "Road favorite efficiency advantage in primetime showcase."},
        {"team": "PHI", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Short-yardage success rate and defensive line depth superiority."},
        {"team": "HOU", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Pass protection metrics heavily favor Houston against weak edge rush."},
        {"team": "DAL", "spread": -4.5, "prob": 0.655, "home": True, "reason": "High-scoring dome environment favoring established offensive core."}
    ],
    5: [
        {"team": "CHI", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Opportunistic spot: Opponent coming off emotional division rivalry game."},
        {"team": "DET", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Offensive line grade advantage opening massive running lanes."},
        {"team": "BAL", "spread": -6.5, "prob": 0.725, "home": False, "reason": "Physical mismatch in trenches against bottom-tier run stop rate."},
        {"team": "GB", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Passing efficiency metrics trending up; strong 3rd-down conversion rate."},
        {"team": "ATL", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Rushing EPA edge in controlled indoor climate."}
    ],
    6: [
        {"team": "PHI", "spread": -9.0, "prob": 0.796, "home": True, "reason": "West Coast opponent traveling East for 1:00 PM ET game; Philly on +3 days extra rest."},
        {"team": "BUF", "spread": -7.5, "prob": 0.756, "home": True, "reason": "High-powered offense against defense suffering turnover luck regression."},
        {"team": "ATL", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Red zone execution advantage against penalty-prone opponent."},
        {"team": "CIN", "spread": -5.5, "prob": 0.691, "home": False, "reason": "Passing efficiency mismatch against depleted secondary."},
        {"team": "HOU", "spread": -5.0, "prob": 0.673, "home": True, "reason": "Strong pass rush pressure rate suppressing opposing pass game."}
    ],
    7: [
        {"team": "BUF", "spread": -10.0, "prob": 0.819, "home": True, "reason": "Top-tier passing EPA at home vs. bottom-5 pressure rate defense. Sharp line movement."},
        {"team": "KC", "spread": -7.0, "prob": 0.742, "home": False, "reason": "Coaching edge on road; opponent turnover margin regressing negatively."},
        {"team": "DET", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Indoor track speed mismatch against slow linebacker corps."},
        {"team": "LAR", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Scheme and coaching advantage against struggling young roster."},
        {"team": "WAS", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Positive ground EPA match against light box defensive scheme."}
    ],
    8: [
        {"team": "DET", "spread": -8.5, "prob": 0.784, "home": True, "reason": "High indoor success rate. Opposing defense suffering regression in turnover luck."},
        {"team": "BAL", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Dominant line-of-scrimmage control; opponent on 2nd straight road trip."},
        {"team": "DEN", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Mile High altitude conditioning advantage in mid-season spot."},
        {"team": "PIT", "spread": -5.0, "prob": 0.673, "home": True, "reason": "Elite pass-rush win rate disrupting rhythm of backup QB."},
        {"team": "MIN", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Passing volume edge in fast-paced indoor conditions."}
    ],
    9: [
        {"team": "KC", "spread": -11.0, "prob": 0.841, "home": True, "reason": "Andy Reid coming off Bye Week preparation edge (historically >80% win rate)."},
        {"team": "CIN", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Explosive play rate significantly higher than opponent's baseline."},
        {"team": "NO", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Dome home-field edge against cold-weather franchise traveling south."},
        {"team": "TEN", "spread": -5.0, "prob": 0.673, "home": True, "reason": "Physical rushing defense neutralizing opposing ground attack."},
        {"team": "LAC", "spread": -4.5, "prob": 0.655, "home": False, "reason": "Special teams efficiency and field position advantage."}
    ],
    10: [
        {"team": "ATL", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Target weak road opponent in dome setting. High rushing EPA efficiency."},
        {"team": "PHI", "spread": -7.0, "prob": 0.742, "home": False, "reason": "Trenches advantage on road against division rival."},
        {"team": "SF", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Elite EPA per play across passing and rushing facets."},
        {"team": "CHI", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Defensive takeaway rate creating short field positions."},
        {"team": "NYG", "spread": -3.5, "prob": 0.618, "home": True, "reason": "Opportunistic home underdog-turned-favorite line move."}
    ],
    11: [
        {"team": "HOU", "spread": -8.0, "prob": 0.771, "home": True, "reason": "Pass-rush win rate advantage against rookie QB; opponent on 3rd straight road game."},
        {"team": "MIA", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Perimeter speed mismatch against zone-heavy defensive scheme."},
        {"team": "GB", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Late-autumn Lambeau Field home-field edge."},
        {"team": "DET", "spread": -6.0, "prob": 0.708, "home": False, "reason": "Top offensive line health grade opening holes on the road."},
        {"team": "NE", "spread": -3.5, "prob": 0.618, "home": True, "reason": "Situational defensive prep advantage at home."}
    ],
    12: [
        {"team": "DAL", "spread": -9.0, "prob": 0.796, "home": True, "reason": "Thanksgiving home week rest dynamic; opponent on very short turnaround (Sun-to-Thu)."},
        {"team": "KC", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Championship pedigree and red-zone defensive efficiency edge."},
        {"team": "WAS", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Rushing success rate keeping opposing high-powered offense on sideline."},
        {"team": "TB", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Favorable match against defense allowing high YAC."},
        {"team": "SEA", "spread": -4.0, "prob": 0.636, "home": True, "reason": "12th Man crowd noise inducing pre-snap penalties on road offense."}
    ],
    13: [
        {"team": "GB", "spread": -8.0, "prob": 0.771, "home": True, "reason": "Sub-freezing Lambeau Field weather dynamic vs. warm-weather franchise."},
        {"team": "CIN", "spread": -7.0, "prob": 0.742, "home": True, "reason": "High red zone touchdown conversion percentage against soft goal-line defense."},
        {"team": "BAL", "spread": -6.5, "prob": 0.725, "home": False, "reason": "Cold weather ground-and-pound identity travels well."},
        {"team": "DEN", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Defensive EPA per play in high altitude environment."},
        {"team": "JAX", "spread": -4.0, "prob": 0.636, "home": True, "reason": "Opponent in letdown spot following high-stakes division battle."}
    ],
    14: [
        {"team": "NYJ", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Severe wind factor (>18 mph) suppressing opponent's pass-heavy offensive scheme."},
        {"team": "BUF", "spread": -7.0, "prob": 0.742, "home": False, "reason": "Physical running game and cold-weather experience edge."},
        {"team": "PHI", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Line-of-scrimmage dominance against worn-down opponent defense."},
        {"team": "MIA", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Opponent traveling from cold climate to warm Miami humidity."},
        {"team": "TEN", "spread": -4.0, "prob": 0.636, "home": True, "reason": "Strong defensive front limiting explosive rushing plays."}
    ],
    15: [
        {"team": "LAC", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Opponent in classic letdown spot following emotional division game."},
        {"team": "SF", "spread": -6.5, "prob": 0.725, "home": False, "reason": "Coaching and scheme versatility mismatch in late December."},
        {"team": "HOU", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Pass-rush win rate disrupting opposing backup quarterback."},
        {"team": "IND", "spread": -5.0, "prob": 0.673, "home": True, "reason": "Indoor offensive efficiency against injury-depleted secondary."},
        {"team": "ARI", "spread": -3.5, "prob": 0.618, "home": True, "reason": "Playoff positioning motivation vs. eliminated road squad."}
    ],
    16: [
        {"team": "DEN", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Late December Mile High cold/altitude. Opponent eliminated from playoffs on short rest."},
        {"team": "KC", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Playoff seeding motivation against division opponent."},
        {"team": "DET", "spread": -6.5, "prob": 0.725, "home": False, "reason": "Offensive line health and rushing attack travel effectively."},
        {"team": "BAL", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Severe weather conditions favor Baltimore's elite ground attack."},
        {"team": "LAR", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Tactical coaching advantage in late-season playoff push."}
    ],
    17: [
        {"team": "TB", "spread": -7.0, "prob": 0.742, "home": True, "reason": "Division title clinching motivation vs. checked-out road opponent."},
        {"team": "BUF", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Cold weather dominance in Orchard Park against dome-based team."},
        {"team": "DAL", "spread": -6.5, "prob": 0.725, "home": True, "reason": "Indoor home passing game firing on all cylinders in Week 17."},
        {"team": "GB", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Late-season Lambeau cold weather dynamic against eliminated opponent."},
        {"team": "CAR", "spread": -3.5, "prob": 0.618, "home": True, "reason": "Motivation mismatch against team resting key veterans."}
    ],
    18: [
        {"team": "LAR", "spread": -7.5, "prob": 0.756, "home": True, "reason": "Opponent resting starters for playoffs/draft position. Sean McVay tactical edge."},
        {"team": "SF", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Must-win scenario for division seeding against resting opponent."},
        {"team": "PHI", "spread": -6.0, "prob": 0.708, "home": True, "reason": "Superior depth and roster strength in regular season finale."},
        {"team": "CIN", "spread": -5.5, "prob": 0.691, "home": True, "reason": "Home finale motivation against division rival."},
        {"team": "SEA", "spread": -4.5, "prob": 0.655, "home": True, "reason": "Home crowd energy against eliminated road squad."}
    ]
}

ALL_TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
]

def format_candidate(cand):
    team = cand["team"]
    spread_str = f"{cand['spread']:+.1f}"
    prob_str = f"{cand['prob']*100:.1f}%"
    return f"**{team}** ({spread_str}, {prob_str})" if cand["home"] else f"{team} ({spread_str}, {prob_str})"

def solve_survivor_path(locked_picks):
    num_teams = len(ALL_TEAMS)
    team_to_idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    idx_to_team = {i: t for i, t in enumerate(ALL_TEAMS)}

    # Construct Cost Matrix where cost = -log(P(win))
    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)

    for w in range(1, WEEKS + 1):
        row = w - 1
        locked_team = locked_picks.get(w, "").strip().upper()

        if locked_team and locked_team in team_to_idx:
            # Force user pick by assigning massive negative cost
            cost_matrix[row, team_to_idx[locked_team]] = -1000.0
        else:
            for cand in SEASON_SLATES[w]:
                t_idx = team_to_idx.get(cand["team"])
                if t_idx is not None:
                    cost_matrix[row, t_idx] = -math.log(cand["prob"])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    optimal_path = {}
    for r, c in zip(row_ind, col_ind):
        optimal_path[r + 1] = idx_to_team[c]
        
    return optimal_path

def sync_to_google_sheets():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("Missing GCP_SERVICE_ACCOUNT_JSON environment variable.")

    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)

    sheet = client.open(SHEET_TITLE).worksheet(TAB_NAME)

    # Read existing sheet data to detect Column D user manual locks
    existing_rows = sheet.get_all_values()
    locked_picks = {}
    if len(existing_rows) > 1:
        for idx, row in enumerate(existing_rows[1:], start=1):
            if len(row) >= 4 and row[3].strip() != "":
                locked_picks[idx] = row[3].strip()

    print(f"Loaded {len(locked_picks)} user locked picks from Column D: {locked_picks}")

    # Solve the optimal 18-week path
    optimal_path = solve_survivor_path(locked_picks)

    # Build spreadsheet rows
    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick", "Pick Reasoning & Synthesis",
        "Top Pick #1", "Top Pick #2", "Top Pick #3", "Top Pick #4", "Top Pick #5"
    ]

    sheet_rows = [headers]

    for w in range(1, WEEKS + 1):
        rec_team = optimal_path.get(w, "")
        
        # Find matching reasoning
        reasoning = ""
        for cand in SEASON_SLATES[w]:
            if cand["team"] == rec_team:
                reasoning = cand["reason"]
                break
        if not reasoning:
            reasoning = f"Optimal calculated pick for Week {w} considering future week equity."

        cands = SEASON_SLATES[w]
        c1 = format_candidate(cands[0]) if len(cands) > 0 else ""
        c2 = format_candidate(cands[1]) if len(cands) > 1 else ""
        c3 = format_candidate(cands[2]) if len(cands) > 2 else ""
        c4 = format_candidate(cands[3]) if len(cands) > 3 else ""
        c5 = format_candidate(cands[4]) if len(cands) > 4 else ""

        my_actual = locked_picks.get(w, "")

        row_data = [
            f"Week {w}",
            rec_team,
            "",
            my_actual,
            reasoning,
            c1, c2, c3, c4, c5
        ]
        sheet_rows.append(row_data)

    # Write all rows to the sheet in a single batch
    sheet.update(range_name=f"A1:J{WEEKS + 1}", values=sheet_rows)

    # Apply professional styling and barrier in Column C
    sheet.format("A1:J1", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.16, "blue": 0.22},
        "horizontalAlignment": "CENTER"
    })

    # Medium grey barrier for Column C
    sheet.format(f"C1:C{WEEKS + 1}", {
        "backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}
    })

    # Center align Week, Recommended Pick, and Candidate columns
    sheet.format(f"A2:B{WEEKS + 1}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"D2:D{WEEKS + 1}", {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"F2:J{WEEKS + 1}", {"horizontalAlignment": "CENTER"})

    print("All 18 weeks populated and formatted successfully.")

if __name__ == "__main__":
    sync_to_google_sheets()
