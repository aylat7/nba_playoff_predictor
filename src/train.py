# src/train.py
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_curve, auc)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.features import FEATURE_COLUMNS, engineer_features


def train(df):
    print("\n" + "=" * 60)
    print("🤖 MODEL TRAINING")
    print("=" * 60)

    X = df[FEATURE_COLUMNS].fillna(0)
    y = df['made_playoffs']

    print(f"  Samples:  {len(X)}")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
    print(f"  Playoff rate: {y.mean():.1%}")

    cutoff_season = df['season'].max() - 2
    train_mask = df['season'] <= cutoff_season

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print(f"  Train: seasons <= {cutoff_season} ({len(X_train)} samples)")
    print(f"  Test:  seasons > {cutoff_season} ({len(X_test)} samples)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42
            ),
            cv=5, method='sigmoid'
        ),
        'Gradient Boosting': CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                min_samples_leaf=5, random_state=42
            ),
            cv=5, method='sigmoid'
        )
    }

    results = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_proba': y_proba
        }

        print(f"\n  📌 {name}")
        print(f"     Test Accuracy: {acc:.1%}")
        print(f"     CV Accuracy:   {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")

    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    best_model = results[best_name]['model']
    print(f"\n  🏆 Best: {best_name} ({results[best_name]['cv_mean']:.1%} CV)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(best_name, "models/best_model_name.pkl")
    print(f"  Saved to models/best_model.pkl")

    plot_evaluation(results, y_test, best_name)

    return best_model, scaler, results


def plot_evaluation(results, y_test, best_name):
    os.makedirs("visuals", exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Model Evaluation Dashboard", fontsize=16, fontweight='bold', y=1.02)

    model_colors = {
        'Logistic Regression': '#3498DB',
        'Random Forest': '#2ECC71',
        'Gradient Boosting': '#E67E22'
    }

    # 1. Accuracy comparison
    ax = axes[0, 0]
    names = list(results.keys())
    test_accs = [results[n]['accuracy'] for n in names]
    cv_accs = [results[n]['cv_mean'] for n in names]
    cv_stds = [results[n]['cv_std'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy',
                   color=[model_colors[n] for n in names], alpha=0.8)
    ax.bar(x + width/2, cv_accs, width, label='CV Accuracy',
           color=[model_colors[n] for n in names], alpha=0.5,
           yerr=cv_stds, capsize=5)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    for bar, val in zip(bars1, test_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', fontsize=9)

    # 2. ROC Curves
    ax = axes[0, 1]
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=model_colors[name], linewidth=2,
                label=f'{name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=8)

    # 3. Confusion Matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_test, results[best_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Missed', 'Made'], yticklabels=['Missed', 'Made'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix ({best_name})')

    # 4. Feature Importance
    ax = axes[1, 1]
    model = results[best_name]['model']
    # CalibratedClassifierCV wraps the model, so unwrap it for feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'estimators_'):
        # Calibrated model — average importance across the internal estimators
        importances = np.mean([
            e.feature_importances_ for e in model.estimators_
            if hasattr(e, 'feature_importances_')
        ], axis=0)
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(FEATURE_COLUMNS))

    feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=True)
    colors_list = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_imp)))
    feat_imp.plot(kind='barh', ax=ax, color=colors_list)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance ({best_name})')
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig('visuals/model_evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: visuals/model_evaluation.png")