# src/collect_current.py
from nba_api.stats.endpoints import leaguestandings
from nba_api.stats.static import teams
import pandas as pd
import os


def get_current_raptors(season='2025-26'):
    print(f"Fetching current {season} standings...")

    standings = leaguestandings.LeagueStandings(season=season)
    df = standings.get_data_frames()[0]

    raptors = df[df['TeamName'] == 'Raptors']

    if len(raptors) == 0:
        print("⚠ Raptors not found. Available teams:")
        print(df['TeamName'].unique())
        return None

    raptors = raptors.copy()
    raptors['season'] = int(season.split('-')[0]) + 1

    os.makedirs("data/current_season", exist_ok=True)
    raptors.to_csv("data/current_season/raptors_current.csv", index=False)

    print(f"\n  Record: {raptors['WINS'].values[0]}-{raptors['LOSSES'].values[0]}")
    print(f"  Win %:  {raptors['WinPCT'].values[0]}")
    print(f"  PPG:    {raptors['PointsPG'].values[0]}")
    print(f"  Diff:   {raptors['DiffPointsPG'].values[0]}")
    print(f"\n  Saved to data/current_season/raptors_current.csv")

    return raptors


if __name__ == "__main__":
    season = '2024-25'
    try:
        raptors = get_current_raptors(season)
    except Exception as e:
        print(f"  {season} not available: {e}")