# main.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.collect_data import collect_all_seasons
from src.features import engineer_features
from src.train import train
from src.predict import predict_all_raptors_seasons, predict_current_raptors
import pandas as pd


def main():
    print("🏀" * 30)
    print("  NBA PLAYOFF PREDICTOR — RAPTORS EDITION")
    print("🏀" * 30)

    data_path = "data/raw/team_stats.csv"
    if not os.path.exists(data_path):
        print("\n⚠ No data found. Running data collection...")
        collect_all_seasons(start=2004, end=2025)

    df = pd.read_csv(data_path)
    print(f"\n📊 Loaded {len(df)} team-seasons")

    print("\n⚙ Engineering features...")
    df = engineer_features(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/training_data.csv", index=False)

    model, scaler, results = train(df)

    pred_df = predict_all_raptors_seasons(df)
    prob = predict_current_raptors(df)

    print("\n" + "=" * 60)
    print("📁 OUTPUT FILES")
    print("=" * 60)
    print("  data/raw/team_stats.csv          — Raw NBA API data")
    print("  data/processed/training_data.csv  — Engineered features")
    print("  models/best_model.pkl             — Trained model")
    print("  visuals/model_evaluation.png      — Model comparison charts")
    print("  visuals/raptors_predictions.png   — Raptors prediction timeline")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()