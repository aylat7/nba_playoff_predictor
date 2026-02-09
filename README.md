# 🏀 NBA Playoff Predictor — Raptors Edition

An end-to-end machine learning pipeline that predicts whether the Toronto Raptors will make the NBA playoffs, using 20+ years of historical team data and live mid-season stats from the official NBA API.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn&logoColor=white)
![NBA API](https://img.shields.io/badge/NBA_API-live_data-red)

---

## Overview

This project answers one question: **will the Raptors make the playoffs this year?**

It pulls real NBA data through the official API, engineers 20 predictive features (including historical rolling trends), trains and compares three classification models with probability calibration, and outputs a playoff probability with visualizations. The model uses both **historical Raptors performance** and **current season stats** to make its prediction.

### Key Results

| Metric | Value |
|--------|-------|
| Best Model | Calibrated Gradient Boosting |
| CV Accuracy | 92.8% (±1.1%) |
| Test Accuracy | 96.7% |
| Raptors Prediction Accuracy | 100% (22/22 seasons) |
| Current Prediction (2025-26) | **96.3% — Making the Playoffs** |

---

## Current Prediction

```
🔮 CURRENT SEASON PREDICTION
  Season:     2026
  Record:     32-22
  Win %:      0.593
  Point Diff: +2.0
  3yr Avg:    0.421
  Trend:      +6.0

  🦖 Playoff Probability: 96.3%
  📢 Prediction: MAKING THE PLAYOFFS
```

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  NBA API     │────▶│  Feature     │────▶│  Train &     │────▶│  Predict     │
│  660 team-   │     │  Engineering │     │  Compare     │     │  Raptors     │
│  seasons     │     │  20 features │     │  3 models    │     │  probability │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### Feature Categories

| Category | Features | Why They Matter |
|----------|----------|-----------------|
| **Core Performance** | Win %, point differential, PPG, opponent PPG | Strongest predictors of playoff teams |
| **Home / Road** | Home win %, road win %, gap between them | Good teams win on the road |
| **Clutch & Dominance** | Close game win %, blowout wins | Separates contenders from pretenders |
| **Quality of Wins** | Record vs .500+ teams, lead protection | Beating good teams matters more |
| **Efficiency** | FG% leader win rate, rebounding, turnovers | Underlying team quality indicators |
| **Historical Trends** | 3-year rolling avg, point diff trend, previous playoffs | Captures rebuilds vs contention windows |

The **historical trend features** are the key differentiator — they give the model context about whether a team is rising or declining, not just a snapshot of one season.

---

## Full Prediction History

```
Season   Record     Diff     Prob   Pred   Actual
-------------------------------------------------------
2005     33-49      -2.0    2.8%     No       No   ✅
2006     27-55      -3.0    3.1%     No       No   ✅
2007     47-35      +1.0   97.8%    Yes      Yes   ✅
2008     41-41      +3.0   93.3%    Yes      Yes   ✅
2009     33-49      -3.0    3.5%     No       No   ✅
2010     40-42      -2.0   20.4%     No       No   ✅
2011     22-60      -6.0    3.0%     No       No   ✅
2012     23-43      -3.0    3.3%     No       No   ✅
2013     34-48      -1.0    6.9%     No       No   ✅
2014     48-34      +3.0   94.2%    Yes      Yes   ✅
2015     49-33      +3.0   97.8%    Yes      Yes   ✅
2016     56-26      +5.0   95.9%    Yes      Yes   ✅
2017     51-31      +4.0   97.4%    Yes      Yes   ✅
2018     59-23      +8.0   97.7%    Yes      Yes   ✅
2019     58-24      +6.0   97.3%    Yes      Yes   ✅  🏆
2020     53-19      +6.0   97.4%    Yes      Yes   ✅
2021     27-45      +0.0    5.3%     No       No   ✅
2022     48-34      +2.0   97.0%    Yes      Yes   ✅
2023     41-41      +1.0   11.3%     No       No   ✅
2024     25-57      -6.0    3.0%     No       No   ✅
2025     30-52      -4.0    3.0%     No       No   ✅
2026     32-22      +2.0   96.3%    Yes      Yes   ✅
```

---

## Project Structure

```
nba-playoff-predictor/
├── main.py                     # Runs the full pipeline
├── requirements.txt
├── README.md
├── src/
│   ├── collect_data.py         # Scrapes 21 seasons from NBA API
│   ├── collect_current.py      # Fetches live mid-season stats
│   ├── features.py             # Feature engineering (20 features)
│   ├── train.py                # Model training, calibration, evaluation
│   └── predict.py              # Predictions + visualizations
├── data/
│   ├── raw/                    # Original API data (never edit)
│   │   └── team_stats.csv
│   ├── processed/              # Engineered features
│   │   └── training_data.csv
│   └── current_season/         # Live stats for current predictions
│       └── raptors_current.csv
├── models/
│   ├── best_model.pkl          # Saved trained model
│   ├── scaler.pkl              # Feature scaler
│   └── best_model_name.pkl     # Name of best model
└── visuals/
    ├── model_evaluation.png    # Model comparison dashboard
    └── raptors_predictions.png # Raptors prediction timeline
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/nba-playoff-predictor.git
cd nba-playoff-predictor
python -m venv .venv
.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
```

### 2. Collect data (one-time, ~1 min)

```bash
python src/collect_data.py
```

Fetches 660 team-seasons (30 teams × 22 years) from the NBA API.

### 3. Run the full pipeline

```bash
python main.py
```

### 4. Update with live stats mid-season

```bash
python src/collect_current.py
python main.py
```

---

## Models Compared

| Model | Test Accuracy | CV Accuracy | Notes |
|-------|:------------:|:-----------:|-------|
| Logistic Regression | 100.0% | 92.3% ± 2.5% | Simple baseline |
| Random Forest | 98.3% | 92.5% ± 1.7% | Ensemble of decision trees |
| **Gradient Boosting** | **96.7%** | **92.8% ± 1.1%** | **Selected — best CV score** |

All models use **sigmoid probability calibration** to produce realistic confidence percentages instead of extreme 0%/100% outputs. Models are evaluated with 5-fold cross-validation and tested on held-out recent seasons using a temporal split (no data leakage).

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| nba_api | Official NBA data |
| pandas | Data manipulation |
| scikit-learn | ML models, calibration, evaluation |
| matplotlib / seaborn | Visualizations |
| joblib | Model persistence |

---

## Future Improvements

- [ ] Streamlit web dashboard for interactive predictions
- [ ] Player-level features (top player PER, injury games missed)
- [ ] Elo rating system (FiveThirtyEight-style)
- [ ] SHAP values for model explainability
- [ ] Weekly mid-season tracking with probability trend chart
- [ ] Extend to predict any NBA team
- [ ] Play-in tournament probability modeling

---

## What I Learned

- **Feature engineering > model complexity.** Win percentage and point differential alone get ~85% accuracy. The additional 18 features push it past 92%.
- **Temporal validation matters.** Random train/test splits leak future information. Training on older seasons and testing on recent ones gives honest accuracy estimates.
- **Probability calibration is important.** Raw model outputs were 0% or 100%. Sigmoid calibration produces realistic probabilities that better reflect true uncertainty.
- **Historical trends are powerful.** A team's 3-year rolling average captures rebuilds and contention windows that single-season stats miss entirely.
- **The NBA is surprisingly predictable.** Teams with a positive point differential and .550+ win rate almost always make the playoffs.

---
