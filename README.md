# Tennis Match Outcome Predictor 🎾

A machine learning pipeline for predicting ATP singles match outcomes using historical data and time-aware feature engineering.

This project explores whether engineered matchup features (surface performance, head-to-head history) can outperform simple ranking-based heuristics when predicting professional tennis matches.

It includes an interactive CLI which uses a binary classification system to predict the probability that Player A defeats Player B in a given matchup. The CLI outputs a predicted win probability (model confidence) for the selected matchup.

* **Data Source:** Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp)
* **Tech Stack:** Python, Pandas, Scikit-Learn, XGBoost, Matplotlib
* **Best Model:** Random Forest (~64% Accuracy on a chronologically held-out 2014-2024 test set)

## Features Engineered
All features are computed chronologically, using only information available prior to each match:

1.  **Surface-Specific Win Rate:** How well does the player perform specifically on Clay/Hard/Grass?
2.  **Head-to-Head (H2H):** Relative historical dominance of one player over the other.
3.  **Dynamic Rankings:** Uses the player's rank at the time of the match, not current rankings.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn
   ```
2. Set up your virtual environment: source venv/bin/activate
3. Run the script: python predictor.py
4. The script will:
    - Download ATP match data (cached locally)
    - Engineer features
    - Train four models (Logistic Regression, Decision Tree, Random Forest, XGBoost).
    - Save performance visualisations locally.
    - Launch an interactive terminal-based predictor for custom ATP singles matchups.
   
