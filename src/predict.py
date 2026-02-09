# src/predict.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.features import FEATURE_COLUMNS, engineer_features, prepare_current_season


def predict_all_raptors_seasons(df):
    model = joblib.load("models/best_model.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    scaler = joblib.load("models/scaler.pkl")

    is_lr = model_name == 'Logistic Regression'

    raptors = df[df['TeamName'] == 'Raptors'].sort_values('season')

    if len(raptors) == 0:
        print("⚠ No Raptors data found. Check TeamName column.")
        print(f"  Available teams: {sorted(df['TeamName'].unique()[:10])}...")
        return None

    print("\n" + "=" * 60)
    print("🦖 RAPTORS PLAYOFF PREDICTIONS — ALL SEASONS")
    print("=" * 60)
    print(f"  Model: {model_name}")

    predictions = []
    for _, row in raptors.iterrows():
        X_row = row[FEATURE_COLUMNS].values.reshape(1, -1)
        X_row = np.nan_to_num(X_row, 0)

        if is_lr:
            X_row = scaler.transform(X_row)

        prob = model.predict_proba(X_row)[0][1]
        predictions.append({
            'season': int(row['season']),
            'wins': int(row['wins']),
            'losses': int(row['losses']),
            'win_pct': row['win_pct'],
            'point_diff': row['point_diff'],
            'actual': int(row['made_playoffs']),
            'predicted_prob': prob,
            'predicted': 1 if prob >= 0.5 else 0
        })

    pred_df = pd.DataFrame(predictions)

    print(f"\n  {'Season':<8} {'Record':<8} {'Diff':>6} {'Prob':>8} {'Pred':>6} {'Actual':>8} {'':>3}")
    print("  " + "-" * 55)
    for _, r in pred_df.iterrows():
        correct = "✅" if r['predicted'] == r['actual'] else "❌"
        print(f"  {int(r['season']):<8} {int(r['wins'])}-{int(r['losses']):<5} "
              f"{r['point_diff']:>+6.1f} {r['predicted_prob']:>7.1%} "
              f"{'Yes' if r['predicted'] else 'No':>6} "
              f"{'Yes' if r['actual'] else 'No':>8} {correct:>3}")

    accuracy = (pred_df['predicted'] == pred_df['actual']).mean()
    print(f"\n  Raptors prediction accuracy: {accuracy:.1%}")

    plot_raptors_predictions(pred_df)
    return pred_df


def predict_current_raptors(df):
    model = joblib.load("models/best_model.pkl")
    model_name = joblib.load("models/best_model_name.pkl")
    scaler = joblib.load("models/scaler.pkl")

    is_lr = model_name == 'Logistic Regression'

    raptors = df[df['TeamName'] == 'Raptors'].sort_values('season')
    if len(raptors) == 0:
        print("⚠ No Raptors data found.")
        return None

    current = raptors.iloc[-1]

    print("\n" + "=" * 60)
    print("🔮 CURRENT SEASON PREDICTION")
    print("=" * 60)
    print(f"  Season:     {int(current['season'])}")
    print(f"  Record:     {int(current['wins'])}-{int(current['losses'])}")
    print(f"  Win %:      {current['win_pct']:.3f}")
    print(f"  Point Diff: {current['point_diff']:+.1f}")
    print(f"  3yr Avg:    {current['win_pct_3yr_avg']:.3f}")
    print(f"  Trend:      {current['point_diff_trend']:+.1f}")

    X_row = current[FEATURE_COLUMNS].values.reshape(1, -1)
    X_row = np.nan_to_num(X_row, 0)

    if is_lr:
        X_row = scaler.transform(X_row)

    prob = model.predict_proba(X_row)[0][1]
    prediction = "MAKING THE PLAYOFFS" if prob >= 0.5 else "MISSING THE PLAYOFFS"

    print(f"\n  🦖 Playoff Probability: {prob:.1%}")
    print(f"  📢 Prediction: {prediction}")

    return prob


def plot_raptors_predictions(pred_df):
    os.makedirs("visuals", exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Toronto Raptors — Playoff Probability Over Time",
                 fontsize=16, fontweight='bold')

    seasons = pred_df['season']
    probs = pred_df['predicted_prob']
    actuals = pred_df['actual']

    # Top: Probability bars
    ax = axes[0]
    bar_colors = ['#2ECC71' if a == 1 else '#E74C3C' for a in actuals]
    bars = ax.bar(seasons, probs, color=bar_colors, edgecolor='white', alpha=0.8)
    ax.axhline(y=0.5, color='#333', linestyle='--', alpha=0.5)

    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.0%}', ha='center', fontsize=7, fontweight='bold')

    ax.set_ylabel('Predicted Playoff Probability')
    ax.set_ylim(0, 1.15)
    made_patch = mpatches.Patch(color='#2ECC71', label='Actually Made Playoffs')
    missed_patch = mpatches.Patch(color='#E74C3C', label='Actually Missed Playoffs')
    ax.legend(handles=[made_patch, missed_patch], fontsize=9)
    ax.set_title('Model Predicted Probability vs Actual Result')

    # Bottom: Win % timeline
    ax = axes[1]
    ax.fill_between(seasons, pred_df['win_pct'], 0.5,
                     where=(pred_df['win_pct'] >= 0.5),
                     alpha=0.3, color='#2ECC71', label='Above .500')
    ax.fill_between(seasons, pred_df['win_pct'], 0.5,
                     where=(pred_df['win_pct'] < 0.5),
                     alpha=0.3, color='#E74C3C', label='Below .500')
    ax.plot(seasons, pred_df['win_pct'], 'o-', color='#7B2D8E',
            linewidth=2.5, markersize=6, label='Raptors Win %')
    ax.axhline(y=0.5, color='#333', linestyle='--', alpha=0.5)

    if 2019 in seasons.values:
        champ_pct = pred_df[pred_df['season'] == 2019]['win_pct'].values[0]
        ax.annotate('🏆', xy=(2019, champ_pct + 0.03), fontsize=14, ha='center')

    ax.set_xlabel('Season')
    ax.set_ylabel('Win Percentage')
    ax.set_title('Raptors Win % Timeline')
    ax.legend(fontsize=9)
    ax.set_ylim(0.15, 0.85)

    plt.tight_layout()
    plt.savefig('visuals/raptors_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: visuals/raptors_predictions.png")