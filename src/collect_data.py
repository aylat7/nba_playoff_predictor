# src/collect_data.py
from nba_api.stats.endpoints import leaguestandings, teamyearbyyearstats
from nba_api.stats.static import teams
import pandas as pd
import time

def get_raptors_history():
    """Get full Raptors historical stats."""
    # Find the Raptors team ID
    nba_teams = teams.get_teams()
    raptors = [t for t in nba_teams if t['full_name'] == 'Toronto Raptors'][0]
    raptors_id = raptors['id']
    
    # Pull year-by-year stats
    stats = teamyearbyyearstats.TeamYearByYearStats(team_id=raptors_id)
    df = stats.get_data_frames()[0]
    return df

def get_all_teams_season(season='2024-25'):
    """Get all team stats for one season."""
    standings = leaguestandings.LeagueStandings(season=season)
    df = standings.get_data_frames()[0]
    time.sleep(1)  # Don't spam the API
    return df

def collect_all_seasons(start=2004, end=2025):
    """Loop through seasons and collect all team data."""
    all_data = []
    for year in range(start, end + 1):
        season_str = f"{year}-{str(year+1)[2:]}"  # e.g. "2024-25"
        print(f"Fetching {season_str}...")
        try:
            df = get_all_teams_season(season=season_str)
            df['season'] = year + 1  # label by end year
            all_data.append(df)
            time.sleep(2)  # Be nice to the API
        except Exception as e:
            print(f"  Skipped: {e}")
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df.to_csv("data/raw/team_stats.csv", index=False)
    print(f"Saved {len(full_df)} rows")
    return full_df

if __name__ == "__main__":
    df = collect_all_seasons(start=2004, end=2025)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")