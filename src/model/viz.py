"""
Visualization utilities for model analysis.
Generates feature importance plots.
"""
import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Any

def plot_feature_importance(model: Any) -> None:
    """
    Plot and display feature importance and save it to a PNG file.
    """
    if not hasattr(model, "feature_importances_"):
        print("⚠️ Selected model does not support feature importance.")
        return

    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': config.MODEL_FEATURES, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=df_imp, hue='Feature', legend=False, palette='viridis')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.tight_layout()
    import os
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(config.FEATURE_IMPORTANCE_PLOT)
    plt.close()
    
    print("\n📊 Feature Importance Ranking:")
    print(df_imp.to_string(index=False))