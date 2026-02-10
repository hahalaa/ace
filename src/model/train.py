"""
Model training and evaluation logic.
Trains multiple classifiers and selects the best performing one.
"""
import config
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any

# Machine Learning Imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(df: pd.DataFrame) -> Any:
    """
    Train multiple models and evaluate on the test year.
    Returns the best-performing model based on test accuracy.
    """
    print("🧠 Training models...")
    
    # Split by Year (Train: <END_YEAR, Test: END_YEAR)
    train_mask = df['tourney_date'].dt.year < config.END_YEAR
    test_mask = df['tourney_date'].dt.year == config.END_YEAR
    
    X_train = df.loc[train_mask, config.MODEL_FEATURES]
    y_train = df.loc[train_mask, 'target']
    X_test  = df.loc[test_mask, config.MODEL_FEATURES]
    y_test  = df.loc[test_mask, 'target']
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    # Fit model to data
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        results[name] = acc
        print(f"   - {name}: {acc:.4f}")
        
        if acc > best_acc:
            best_model = model
            best_acc = acc

    # Save Accuracy Plot
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=list(results.values()), hue=list(results.keys()), legend=False, palette="viridis")
    plt.title(f"Model Accuracy (Test Year: {config.END_YEAR})")
    plt.ylim(config.DEFAULT_WIN_PCT, 0.75)
    plt.savefig(config.ACCURACY_PLOT)
    plt.close()
    print("   [Saved accuracy_comparison.png]")
    
    return best_model