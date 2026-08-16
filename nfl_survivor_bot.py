import os
import json
import math
import numpy as np
import pandas as pd
import requests
import gspread
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- SPREADSHEET CONFIGURATION ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18

# --- CONVERT VEGAS SPREAD TO MODEL WIN PROBABILITY ---
def spread_to_win_prob(spread: float) -> float:
    """
    Standard NFL Logistic Win Probability Function:
    P(win) = 1 / (1 + 10 ^ (spread / 14.5))
    """
    return 1.0 / (1.0 + math.pow(10.0, spread / 14.5))

def fetch_nfl_weekly_data():
    """
    Fetches real-time / lookahead odds, situational factors, EPA, and injuries.
    Can be linked to SportsDataIO, The-Odds-API, or ESPN hidden APIs.
    """
    # Sample structured matrix builder for full 18 weeks
    # In production, replace with live payload from Odds API / ESPN / nflverse
    teams = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
        "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
        "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
        "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
    ]
    
    # Mock data structure representing 18 weeks of candidate games
    weekly_slates = {}
    for w in range(1, WEEKS + 1):
        weekly_slates[w] = []
        
    return weekly_slates

def calculate_optimal_path(win_matrix, locked_picks):
    """
    Solves the 18-week Survivor path by maximizing product of win probabilities:
    max Prod(P_w) <=> min Sum(-log(P_w))
    subject to: Each chosen team used at most once.
    """
    num_teams = win_matrix.shape[0]
    cost_matrix = np.full((WEEKS, num_teams), fill_value=1e5)
    
    for w in range(WEEKS):
        week_num = w + 1
        if week_num in locked_picks:
            # Force user's actual pick
            team_idx = locked_picks[week_num]
            cost_matrix[w, team_idx] = -1000.0  # Priority lock
        else:
            for t in range(num_teams):
                prob = win_matrix[t, w]
                if prob > 0.01:
                    cost_matrix[w, t] = -math.log(prob)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    optimal_path = {}
    for r, c in zip(row_ind, col_ind):
        optimal_path[r + 1] = c
    return optimal_path

def sync_to_google_sheets():
    # Authenticate via Service Account JSON stored in GitHub Secrets
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("Missing GCP_SERVICE_ACCOUNT_JSON environment variable.")
        
    creds_dict = json.loads(creds_json)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    
    sheet = client.open(SHEET_TITLE).worksheet(TAB_NAME)
    
    # Read existing sheet data to respect user's manual picks in Column D
    records = sheet.get_all_values()
    locked_picks = {}
    
    if len(records) > 1:
        for idx, row in enumerate(records[1:], start=1):
            if len(row) >= 4 and row[3].strip() != "":
                locked_picks[idx] = row[3].strip()
    
    print(f"Detected {len(locked_picks)} user locked picks: {locked_picks}")
    
    # Headers
    headers = [
        "Week", "Recommended Pick", "|", "My Actual Pick", "Pick Reasoning & Synthesis",
        "Top Pick #1", "Top Pick #2", "Top Pick #3", "Top Pick #4", "Top Pick #5"
    ]
    
    # Batch update values and formatting
    sheet.update(range_name="A1:J1", values=[headers])
    
    # Format Column C as medium grey divider
    sheet.format("C1:C19", {
        "backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62},
        "horizontalAlignment": "CENTER"
    })
    
    # Set header row format
    sheet.format("A1:J1", {
        "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
        "backgroundColor": {"red": 0.12, "green": 0.14, "blue": 0.18},
        "horizontalAlignment": "CENTER"
    })
    
    print("Google Sheet updated successfully.")

if __name__ == "__main__":
    sync_to_google_sheets()
