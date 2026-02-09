# src/features.py
import pandas as pd
import numpy as np


FEATURE_COLUMNS = [
    'win_pct', 'point_diff', 'pts_pg', 'opp_pts_pg',
    'home_win_pct', 'road_win_pct', 'home_road_gap',
    'conf_win_pct', 'is_east',
    'close_game_win_pct', 'blowout_wins',
    'vs_good_teams_pct', 'hold_lead_pct',
    'fg_leader_win_pct', 'reb_leader_win_pct', 'tov_leader_win_pct',
    'win_pct_3yr_avg', 'point_diff_trend', 'win_pct_change', 'prev_playoffs',
]


def parse_record(record_str):
    try:
        parts = str(record_str).split('-')
        return int(parts[0]), int(parts[1])
    except:
        return 0, 0


def engineer_features(df):
    df = df.copy()

    df['win_pct'] = df['WinPCT'].astype(float)
    df['point_diff'] = df['DiffPointsPG'].astype(float)
    df['pts_pg'] = df['PointsPG'].astype(float)
    df['opp_pts_pg'] = df['OppPointsPG'].astype(float)
    df['wins'] = df['WINS'].astype(int)
    df['losses'] = df['LOSSES'].astype(int)

    df['playoff_rank'] = df['PlayoffRank'].astype(int)
    df['made_playoffs'] = (df['playoff_rank'] <= 8).astype(int)

    home = df['HOME'].apply(parse_record)
    df['home_wins'] = home.apply(lambda x: x[0])
    df['home_losses'] = home.apply(lambda x: x[1])
    df['home_win_pct'] = df['home_wins'] / (df['home_wins'] + df['home_losses']).replace(0, 1)

    road = df['ROAD'].apply(parse_record)
    df['road_wins'] = road.apply(lambda x: x[0])
    df['road_losses'] = road.apply(lambda x: x[1])
    df['road_win_pct'] = df['road_wins'] / (df['road_wins'] + df['road_losses']).replace(0, 1)

    df['home_road_gap'] = df['home_win_pct'] - df['road_win_pct']

    conf = df['ConferenceRecord'].apply(parse_record)
    df['conf_wins'] = conf.apply(lambda x: x[0])
    df['conf_losses'] = conf.apply(lambda x: x[1])
    df['conf_win_pct'] = df['conf_wins'] / (df['conf_wins'] + df['conf_losses']).replace(0, 1)

    close = df['ThreePTSOrLess'].apply(parse_record)
    df['close_wins'] = close.apply(lambda x: x[0])
    df['close_losses'] = close.apply(lambda x: x[1])
    df['close_total'] = df['close_wins'] + df['close_losses']
    df['close_game_win_pct'] = np.where(df['close_total'] > 0, df['close_wins'] / df['close_total'], 0.5)

    blowout = df['TenPTSOrMore'].apply(parse_record)
    df['blowout_wins'] = blowout.apply(lambda x: x[0])

    opp500 = df['OppOver500'].apply(parse_record)
    df['good_team_wins'] = opp500.apply(lambda x: x[0])
    df['good_team_losses'] = opp500.apply(lambda x: x[1])
    df['good_team_total'] = df['good_team_wins'] + df['good_team_losses']
    df['vs_good_teams_pct'] = np.where(df['good_team_total'] > 0, df['good_team_wins'] / df['good_team_total'], 0.5)

    ahead = df['AheadAtHalf'].apply(parse_record)
    df['ahead_wins'] = ahead.apply(lambda x: x[0])
    df['ahead_losses'] = ahead.apply(lambda x: x[1])
    df['ahead_total'] = df['ahead_wins'] + df['ahead_losses']
    df['hold_lead_pct'] = np.where(df['ahead_total'] > 0, df['ahead_wins'] / df['ahead_total'], 0.5)

    fg = df['LeadInFGPCT'].apply(parse_record)
    df['fg_lead_wins'] = fg.apply(lambda x: x[0])
    df['fg_lead_total'] = fg.apply(lambda x: x[0] + x[1])
    df['fg_leader_win_pct'] = np.where(df['fg_lead_total'] > 0, df['fg_lead_wins'] / df['fg_lead_total'], 0.5)

    reb = df['LeadInReb'].apply(parse_record)
    df['reb_lead_wins'] = reb.apply(lambda x: x[0])
    df['reb_lead_total'] = reb.apply(lambda x: x[0] + x[1])
    df['reb_leader_win_pct'] = np.where(df['reb_lead_total'] > 0, df['reb_lead_wins'] / df['reb_lead_total'], 0.5)

    tov = df['FewerTurnovers'].apply(parse_record)
    df['tov_lead_wins'] = tov.apply(lambda x: x[0])
    df['tov_lead_total'] = tov.apply(lambda x: x[0] + x[1])
    df['tov_leader_win_pct'] = np.where(df['tov_lead_total'] > 0, df['tov_lead_wins'] / df['tov_lead_total'], 0.5)

    df['is_east'] = (df['Conference'] == 'East').astype(int)

    df = df.sort_values(['TeamName', 'season'])

    df['win_pct_3yr_avg'] = df.groupby('TeamName')['win_pct'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['point_diff_trend'] = df.groupby('TeamName')['point_diff'].transform(
        lambda x: x.diff()
    )
    df['point_diff_trend'] = df['point_diff_trend'].fillna(0)

    df['win_pct_change'] = df.groupby('TeamName')['win_pct'].transform(
        lambda x: x.diff()
    )
    df['win_pct_change'] = df['win_pct_change'].fillna(0)

    df['prev_playoffs'] = df.groupby('TeamName')['made_playoffs'].transform(
        lambda x: x.shift(1)
    )
    df['prev_playoffs'] = df['prev_playoffs'].fillna(0)

    df = df.fillna(0)
    return df


def prepare_current_season(historical_df, current_stats):
    team_name = current_stats.get('TeamName', 'Raptors')
    team_history = historical_df[historical_df['TeamName'] == team_name]

    if len(team_history) > 0:
        current_stats['win_pct_3yr_avg'] = team_history['win_pct'].tail(3).mean()
        current_stats['point_diff_trend'] = current_stats.get('point_diff', 0) - team_history['point_diff'].iloc[-1]
        current_stats['win_pct_change'] = current_stats.get('win_pct', 0) - team_history['win_pct'].iloc[-1]
        current_stats['prev_playoffs'] = team_history['made_playoffs'].iloc[-1]
    else:
        current_stats['win_pct_3yr_avg'] = current_stats.get('win_pct', 0.5)
        current_stats['point_diff_trend'] = 0
        current_stats['win_pct_change'] = 0
        current_stats['prev_playoffs'] = 0

    return current_stats