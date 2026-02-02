import joblib
from pathlib import Path

import config
import data.loader as loader
import data.preprocess as preprocess
import features.engineering as features
import model.train as train
import model.viz as viz
import cli.interactive as cli

# ==========================================
# MAIN EXECUTION
# ==========================================
def main() -> None:
    DATA_PATH = Path("atp_tennis_data.csv")
    MODEL_PATH = Path("tennis_model.pkl")

    data = loader.load_cached_data(DATA_PATH, config.START_YEAR, config.END_YEAR)

    if data is None:
        data = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
        data.to_csv(DATA_PATH, index=False)
        print(f"💾 New data saved to {DATA_PATH}")

    processed_data = preprocess.preprocess_data(data)
    final_df, surf_history, h2h_history = features.add_features(processed_data)
    
    if MODEL_PATH.exists():
        print(f"📂 Loading trained model from {MODEL_PATH}...")
        rf_model = joblib.load(MODEL_PATH)
    else:
        rf_model = train.train_and_evaluate(final_df)
        joblib.dump(rf_model, MODEL_PATH)
        print(f"💾 Model saved to {MODEL_PATH}")

    viz.plot_feature_importance(rf_model)
    cli.interactive_prediction_loop(rf_model, final_df, surf_history, h2h_history)

if __name__ == "__main__":
    main()