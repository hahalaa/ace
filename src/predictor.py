import joblib
import config
import data.loader as loader
import data.preprocess as preprocess
import features.engineering as features
import model.train as train
import model.viz as viz
import cli.interactive as cli

def main() -> None:
    """
    Main entry point for the Tennis Match Predictor pipeline.
    
    Steps:
    1. Load data (cached or download new).
    2. Preprocess and feature engineer.
    3. Train model (or load existing).
    4. Generate visualizations.
    5. Start interactive prediction loop.
    """
    # 1. Load Data
    data = loader.load_cached_data(config.DATA_PATH, config.START_YEAR, config.END_YEAR)

    if data is None:
        data = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
        data.to_csv(config.DATA_PATH, index=False)
        print(f"💾 New data saved to {config.DATA_PATH}")

    # 2. Preprocess & Feature Engineering
    processed_data = preprocess.preprocess_data(data)
    final_df, surf_history, h2h_history = features.add_features(processed_data)
    
    # 3. Model Training / Loading
    if config.MODEL_PATH.exists():
        print(f"📂 Loading trained model from {config.MODEL_PATH}...")
        rf_model = joblib.load(config.MODEL_PATH)
    else:
        rf_model = train.train_and_evaluate(final_df)
        import os
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        joblib.dump(rf_model, config.MODEL_PATH)
        print(f"💾 Model saved to {config.MODEL_PATH}")

    # 4. Visualization & Interaction
    viz.plot_feature_importance(rf_model)
    cli.interactive_prediction_loop(rf_model, final_df, surf_history, h2h_history)

if __name__ == "__main__":
    main()