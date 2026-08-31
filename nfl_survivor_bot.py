"""
NFL Survivor bot — 2026.

Every constant here was fitted on 2016-2025 closing lines (2,625 games) or
chosen from a Monte Carlo policy sweep (20,000 simulated entries per policy
on 2021-2025 slates). Anything that failed backtest was removed, not shrunk.

Design in one line: the market prices this week, Elo prices the weeks the
market hasn't gotten to yet, and the solver never lets Elo override a real line.
"""

import io
import json
import math
import os
import re
import statistics

import gspread
import numpy as np
import pandas as pd
import requests
from google.oauth2.service_account import Credentials
from scipy.optimize import linear_sum_assignment

# --- SPREADSHEET CONFIG ---
SHEET_TITLE = "NFL Picks"
TAB_NAME = "2026"
WEEKS = 18
SEASON_YEAR = 2026

# ===========================================================================
# FITTED CONSTANTS
# ===========================================================================

# logit(p_home) = -0.0393 - 0.1495 * home_spread  ->  ln(10)/0.1495 = 15.40
SPREAD_DENOM = 15.4

# Intercept fits to 0.4902 at a pick'em: the closing line already prices home
# field. No separate home-field term. (Home favs came in 0.7% UNDER the
# pure-spread model, away favs 0.6% over -- both inside noise.)

# Weeks 1-4 favorites: 62.1% actual vs 65.3% predicted, n=628, ~1.7 SE.
# Directionally sound, shrunk toward zero because one season swings this much.
EARLY_SEASON_PENALTY = 0.020
EARLY_SEASON_WEEKS = 4

# Empirical ceiling: favorites of 13.5+ won 89.1% (n=119). Nothing wins at 96%.
PROB_CAP = 0.90
PROB_FLOOR = 0.50

# Elo: K and offseason regression chosen by grid search against closing lines.
# Result: 2.38 pts MAE vs the market, r=0.863, on weeks 4+.
ELO_K = 0.08
ELO_REGRESS = 0.45
ELO_HFA = 1.6

# Projected lines are worse than posted lines (2.38 MAE, 90th pct 4.9 pts).
# Shrink projected probabilities toward 0.5 so the solver doesn't treat a
# projected -9 in week 11 like a posted -9 this Sunday.
PROJECTION_SHRINK = 0.80

# Policy sweep, 20k entries/policy on 2021-25 slates. Longer horizons trade
# early survival for deep survival. At week 15: h=12 -> 5.2%, h=6 -> 4.9%,
# greedy -> 4.4%. h=18 (the original) was worse than h=12 on every column.
SOLVER_HORIZON = 12

# End-of-2025 Elo with offseason regression already applied. Overwritten as
# soon as 2026 results exist.
SEED_RATINGS = {
    "SEA": 5.23, "LAR": 4.28, "JAX": 3.88, "NE": 3.58, "BUF": 2.94,
    "DEN": 2.61, "HOU": 2.57, "DET": 2.33, "SF": 2.00, "BAL": 1.85,
    "PHI": 1.77, "MIN": 1.51, "CHI": 1.28, "LAC": 0.82, "GB": 0.75,
    "PIT": 0.54, "IND": 0.13, "KC": -0.04, "CIN": -0.72, "NYG": -1.18,
    "TB": -1.20, "ATL": -1.27, "DAL": -1.69, "NO": -1.93, "WAS": -2.10,
    "CAR": -2.39, "CLE": -2.60, "MIA": -2.64, "ARI": -3.88, "TEN": -4.73,
    "LV": -5.24, "NYJ": -6.45,
}

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
}
ALL_TEAMS = sorted(set(NAME_TO_ABBR.values()))


def team_to_abbr(name):
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", str(name)).strip().lower()
    return NAME_TO_ABBR.get(cleaned, cleaned.upper()[:3])


# ===========================================================================
# PROBABILITY MODEL
# ===========================================================================

def spread_to_prob(spread):
    """Win probability implied by a spread signed for the team (neg = favored)."""
    return 1.0 / (1.0 + math.pow(10.0, spread / SPREAD_DENOM))


def model_prob(market_prob, week, projected=False):
    if market_prob is None:
        return None
    p = market_prob
    if week <= EARLY_SEASON_WEEKS:
        p -= EARLY_SEASON_PENALTY
    if projected:
        p = 0.5 + (p - 0.5) * PROJECTION_SHRINK
    return min(PROB_CAP, max(PROB_FLOOR, round(p, 4)))


# ===========================================================================
# ELO PROJECTION LAYER
# ===========================================================================

class EloModel:
    """Margin-based Elo. Only ever used where no real line exists."""

    def __init__(self, seed=None):
        self.R = dict(seed or SEED_RATINGS)
        for t in ALL_TEAMS:
            self.R.setdefault(t, 0.0)
        self.games_seen = 0

    def update(self, home, away, margin):
        pred = self.R[home] - self.R[away] + ELO_HFA
        err = margin - pred
        self.R[home] += ELO_K * err
        self.R[away] -= ELO_K * err
        self.games_seen += 1

    def fit_completed(self, games_df):
        """Feed every finished game of the current season, in order."""
        done = games_df.dropna(subset=["home_score", "away_score"])
        done = done.sort_values("week")
        for _, g in done.iterrows():
            h, a = team_to_abbr(g["home_team"]), team_to_abbr(g["away_team"])
            if h in self.R and a in self.R:
                self.update(h, a, g["home_score"] - g["away_score"])
        return self

    def project_spread(self, home, away):
        """Projected home spread, same sign convention as a book."""
        return -(self.R.get(home, 0.0) - self.R.get(away, 0.0) + ELO_HFA)


# ===========================================================================
# DATA FETCH
# ===========================================================================

def fetch_schedule():
    url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
    by_week = {w: [] for w in range(1, WEEKS + 1)}
    raw = None
    try:
        res = requests.get(url, timeout=15)
        res.raise_for_status()
        raw = pd.read_csv(io.StringIO(res.text), low_memory=False)
        cur = raw[(raw.season == SEASON_YEAR) & (raw.game_type == "REG")]
        for _, row in cur.iterrows():
            w = int(row["week"])
            if 1 <= w <= WEEKS:
                by_week[w].append({
                    "home_team": team_to_abbr(row["home_team"]),
                    "away_team": team_to_abbr(row["away_team"]),
                })
    except Exception as e:
        raise RuntimeError(
            f"Could not load the {SEASON_YEAR} schedule: {e}. "
            "Refusing to run on a hardcoded fake slate."
        )

    empty = [w for w in range(1, WEEKS + 1) if not by_week[w]]
    if empty:
        print(f"WARNING: no games found for weeks {empty}")
    return by_week, (raw[raw.season == SEASON_YEAR] if raw is not None else None)


def fetch_odds_api(api_key):
    """Median spread across books, keyed (home, away).

    The original took the book with the LARGEST absolute number, which
    systematically picks the single most extreme (usually stalest) line and
    biases every probability upward.
    """
    if not api_key:
        return {}
    url = (f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
           f"?apiKey={api_key}&regions=us&markets=spreads&oddsFormat=american")
    out = {}
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            print(f"Odds API returned {res.status_code}")
            return {}
        for game in res.json():
            h = team_to_abbr(game.get("home_team", ""))
            a = team_to_abbr(game.get("away_team", ""))
            pts = []
            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != "spreads":
                        continue
                    for o in mkt.get("outcomes", []):
                        if team_to_abbr(o.get("name")) == h and o.get("point") is not None:
                            pts.append(float(o["point"]))
            if pts:
                out[(h, a)] = statistics.median(pts)
    except Exception as e:
        print(f"Odds API error: {e}")
    return out


def fetch_espn_odds(week):
    url = ("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
           f"?seasontype=2&week={week}")
    out = {}
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if res.status_code != 200:
            return out
        for ev in res.json().get("events", []):
            comp = ev.get("competitions", [{}])[0]
            cs = comp.get("competitors", [])
            if len(cs) < 2:
                continue
            home = cs[0] if cs[0].get("homeAway") == "home" else cs[1]
            away = cs[1] if cs[0].get("homeAway") == "home" else cs[0]
            h = team_to_abbr(home.get("team", {}).get("abbreviation", ""))
            a = team_to_abbr(away.get("team", {}).get("abbreviation", ""))
            odds = comp.get("odds", [])
            if odds and odds[0].get("spread") is not None:
                out[(h, a)] = float(odds[0]["spread"])
    except Exception:
        pass
    return out


# ===========================================================================
# CANDIDATES
# ===========================================================================

class Candidate:
    __slots__ = ("team", "opponent", "matchup", "week", "spread",
                 "market_prob", "model_prob", "is_home", "projected")

    def __init__(self, team, opponent, matchup, week, spread, is_home, projected):
        self.team, self.opponent, self.matchup = team, opponent, matchup
        self.week, self.spread, self.is_home = week, spread, is_home
        self.projected = projected
        self.market_prob = spread_to_prob(spread)
        self.model_prob = model_prob(self.market_prob, week, projected)


def build_candidates(games, week, posted, elo):
    """One candidate per game: the favorite. Underdogs are never survivor picks.

    Priority is strict: a posted line always wins. Elo only fills gaps.
    Returns ALL candidates -- the original sliced to top-5 before solving,
    which silently deleted legal assignments.
    """
    out = []
    for g in games:
        h, a = g["home_team"], g["away_team"]
        spread, projected = posted.get((h, a)), False
        if spread is None:
            spread, projected = elo.project_spread(h, a), True

        if spread <= 0:
            team, opp, home, sp = h, a, True, spread
        else:
            team, opp, home, sp = a, h, False, -spread
        out.append(Candidate(team, opp, f"{a} @ {h}", week, sp, home, projected))

    out.sort(key=lambda c: c.model_prob or 0, reverse=True)
    return out


# ===========================================================================
# SOLVER
# ===========================================================================

def solve_path(weekly, locked, start_week=1, horizon=SOLVER_HORIZON):
    """Rolling re-solve. Maximizes sum(log p) == the survival probability.

    The original multiplied costs by 0.30 in weeks 1-6 and 1.00 in weeks 13+,
    which made the solver least careful in the weeks it is guaranteed to play
    -- and two-thirds of entries die by week 5. It also carried a +50 penalty
    on sub-7 favorites and a x0.01 "dominance lock". All three are gone; the
    log-probabilities carry the logic unaided.
    """
    idx = {t: i for i, t in enumerate(ALL_TEAMS)}
    rev = {i: t for t, i in idx.items()}
    BIG = 1e5
    path, used = {}, set()

    for w, t in sorted(locked.items()):
        if t:
            path[w] = t
            used.add(t)

    for w in range(start_week, WEEKS + 1):
        if w in path:
            continue
        wks = [x for x in range(w, min(WEEKS, w + horizon - 1) + 1)
               if weekly.get(x) and x not in path]
        if not wks:
            break

        avail = sorted({c.team for x in wks for c in weekly[x]
                        if c.team not in used})
        if not avail:
            break
        ti = {t: i for i, t in enumerate(avail)}
        cost = np.full((len(wks), len(avail)), BIG)
        for r, x in enumerate(wks):
            for c in weekly[x]:
                if c.team in ti and c.model_prob:
                    cost[r, ti[c.team]] = -math.log(c.model_prob)
        if cost.shape[0] > cost.shape[1]:
            cost = cost[:cost.shape[1]]
            wks = wks[:cost.shape[0]]

        rows, cols = linear_sum_assignment(cost)
        pick = next((avail[c] for r, c in zip(rows, cols)
                     if wks[r] == w and cost[r, c] < BIG / 2), None)
        if pick is None:
            break
        path[w] = pick
        used.add(pick)

    return path


def survival_probability(path, weekly, through=WEEKS):
    """Honest cumulative survival.

    Returns (probability, weeks_priced, weeks_projected). The original
    multiplied in a hardcoded 0.74 for every unpriced week, which fabricated
    ~13 weeks of data on any preseason run.
    """
    p, priced, projected = 1.0, 0, 0
    for w in range(1, through + 1):
        team = path.get(w)
        if not team:
            continue
        c = next((x for x in weekly.get(w, [])
                  if x.team == team and x.model_prob), None)
        if c is None:
            continue
        p *= c.model_prob
        if c.projected:
            projected += 1
        else:
            priced += 1
    return p, priced, projected


def validate(weekly, schedule, locked):
    """Fail loudly rather than write a plausible-looking sheet."""
    problems = []
    for w in range(1, WEEKS + 1):
        games, cands = schedule.get(w, []), weekly.get(w, [])
        if games and not cands:
            problems.append(f"Week {w}: {len(games)} games but no candidates")
        playing = {t for g in games for t in (g["home_team"], g["away_team"])}
        lock = (locked.get(w) or "").strip().upper()
        if lock and playing and lock not in playing:
            problems.append(f"Week {w}: locked pick {lock} is not on the slate (bye?)")
        if lock and lock not in ALL_TEAMS:
            problems.append(f"Week {w}: locked pick '{lock}' is not a team abbreviation")
    used = [t for t in locked.values() if t]
    dupes = sorted({t for t in used if used.count(t) > 1})
    if dupes:
        problems.append(f"Locked picks reuse teams: {dupes}")
    return problems


# ===========================================================================
# SHEETS
# ===========================================================================

def sync():
    creds_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON")
    if not creds_json:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON missing.")
    odds_key = os.environ.get("ODDS_API_KEY", "")

    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(json.loads(creds_json), scopes=scopes)
    sheet = gspread.authorize(creds).open(SHEET_TITLE).worksheet(TAB_NAME)

    # --- read locked picks BEFORE clearing (column D, rows 2-19) ---
    existing = sheet.get_all_values()
    locked = {}
    for w in range(1, WEEKS + 1):
        r = w + 1
        if r <= len(existing) and len(existing[r - 1]) >= 4:
            v = existing[r - 1][3].strip().upper()
            if v:
                locked[w] = v
    print(f"Locked picks: {locked}")

    # --- data ---
    schedule, raw_games = fetch_schedule()
    elo = EloModel()
    if raw_games is not None:
        elo.fit_completed(raw_games)
    print(f"Elo seeded; {elo.games_seen} completed {SEASON_YEAR} games applied.")

    posted = fetch_odds_api(odds_key)
    for w in range(1, WEEKS + 1):
        posted.update(fetch_espn_odds(w))
    print(f"Posted lines found for {len(posted)} games.")

    weekly = {w: build_candidates(schedule[w], w, posted, elo)
              for w in range(1, WEEKS + 1)}

    problems = validate(weekly, schedule, locked)
    for p in problems:
        print(f"VALIDATION: {p}")
    if any("not a team" in p or "reuse teams" in p for p in problems):
        raise ValueError("Locked picks are invalid; fix column D before rerunning.")

    path = solve_path(weekly, locked)
    prob, priced, projected = survival_probability(path, weekly)

    # --- layout: summary block rows 1-21, candidate blocks start at row 23 ---
    # (the original overlapped the two, putting week-4's header on the same
    #  row as the season-survival cell)
    SUMMARY_ROWS = 21
    total_rows = SUMMARY_ROWS + WEEKS * 6 + 2
    matrix = [["" for _ in range(9)] for _ in range(total_rows)]
    matrix[0] = ["Week", "Recommended", "|", "My Actual Pick",
                 "Candidate", "Matchup", "Line", "Market Win %", "Model Win %"]

    for w in range(1, WEEKS + 1):
        rec = path.get(w, "")
        c = next((x for x in weekly.get(w, []) if x.team == rec), None)
        matrix[w][0] = f"Week {w}"
        matrix[w][1] = rec
        matrix[w][3] = locked.get(w, "")
        matrix[w][4] = "projected" if (c and c.projected) else ("posted" if c else "")

    matrix[19][0] = "Survival (priced weeks only)"
    matrix[19][1] = f"{prob * 100:.2f}%"
    matrix[20][0] = "Weeks priced / projected"
    matrix[20][1] = f"{priced} posted, {projected} Elo-projected"

    yellow, merges = [], []
    for w in range(1, WEEKS + 1):
        rec = path.get(w, "")
        block = SUMMARY_ROWS + (w - 1) * 6 + 1
        matrix[block - 1][4] = f"Top candidates — Week {w}"
        merges.append(f"E{block}:I{block}")
        for i, c in enumerate(weekly.get(w, [])[:5]):
            r = block + 1 + i
            if c.team == rec and rec:
                yellow.append(r)
            # plain text: Sheets does not render markdown, so "**KC**" showed
            # up literally in the original.
            matrix[r - 1][4] = f"{c.team} (H)" if c.is_home else c.team
            matrix[r - 1][5] = c.matchup
            matrix[r - 1][6] = (f"{c.spread:+.1f}" +
                                ("*" if c.projected else ""))
            matrix[r - 1][7] = f"{c.market_prob * 100:.1f}%"
            matrix[r - 1][8] = f"{c.model_prob * 100:.1f}%"

    sheet.clear()
    try:
        sheet.unmerge_cells(f"A1:I{total_rows}")
    except Exception:
        pass
    sheet.update(range_name=f"A1:I{total_rows}", values=matrix)

    for rng in merges:
        try:
            sheet.merge_cells(rng, merge_type="MERGE_ALL")
        except Exception:
            pass

    hdr = {"textFormat": {"bold": True,
                          "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
           "backgroundColor": {"red": 0.12, "green": 0.34, "blue": 0.63},
           "horizontalAlignment": "CENTER"}
    sheet.format("A1:I1", hdr)
    sheet.format("A20:B21", hdr)
    sheet.format(f"C1:C{total_rows}",
                 {"backgroundColor": {"red": 0.62, "green": 0.62, "blue": 0.62}})
    sheet.format(f"A2:B{total_rows}",
                 {"horizontalAlignment": "CENTER", "textFormat": {"bold": True}})
    sheet.format(f"E2:I{total_rows}", {"horizontalAlignment": "CENTER"})

    fmts = [{"range": r,
             "format": {"backgroundColor": {"red": 0.83, "green": 0.90, "blue": 0.95},
                        "textFormat": {"bold": True},
                        "horizontalAlignment": "CENTER"}} for r in merges]
    fmts += [{"range": f"E{r}:I{r}",
              "format": {"backgroundColor": {"red": 1.0, "green": 0.95, "blue": 0.55},
                         "textFormat": {"bold": True}}} for r in yellow]
    if fmts:
        sheet.batch_format(fmts)

    print(f"Done. Survival {prob * 100:.2f}% over {priced + projected} weeks "
          f"({priced} posted, {projected} projected). "
          f"Lines marked * are Elo projections, not posted numbers.")


if __name__ == "__main__":
    sync()
