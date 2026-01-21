# UFC Binary Classification Fight Predictor

---

This project is a machine learning algorithm designed to predict binary fight outcomes (Win/Loss) in the UFC. The model uses time-series feature engineering, including a custom-built chronological Elo rating system and temporally-decayed performance statistics.

The final model (Random Forest/XGBoost) achieves a **0.64 Weighted F1-Score**, demonstrating a significant predictive uplift over the 0.50 random-chance baseline in a highly chaotic and stochastic sports environment.

---

## Project Pipeline & Methodology

This project's core challenge was engineering non-leaky, time-dependent features. This was accomplished by processing the entire fight history chronologically.

### 1. Data Cleaning and Preparation
* Loaded and merged three raw datasets: fighter attributes, detailed fight-by-fight results, and event data.
* Cleaned and standardized all data, including:
    * Converting "n of m" string statistics (e.g., "50 of 100") into numerical `Landed` and `Attempted` columns.
    * Parsing time formats (e.g., "4:15") into total time in minutes (float datatype).
    * Standardizing categorical text features (e.g., `Method`, `Stance`).
* Addressed data sparsity by implementing **Missingness Indicator Flags** (e.g., `Reach_missing`) and using vectorized **median imputation** for physical attributes.

### 2. Chronological Feature Engineering
A single, chronological loop (`.itertuples()`) was used to iterate through the dataset (sorted from oldest to newest) to generate historical features without data leakage.

* **Elo Rating System:** A custom Elo model (K-factor=40, R0=1500) was implemented to assign a dynamic skill rating to each fighter. The pre-fight `Elo_Diff` was stored as a primary feature.
* **W-L-D Record:** A dictionary tracked each fighter's historical Win-Loss-Draw record, allowing for the calculation of features like `Win_Rate_Diff`.
* **Temporally-Decayed Stats:** To prioritize current form, cumulative stats (like `Total SS` and `Total Fight Time`) were calculated using **exponential decay** (`New Total = (Old Total * 0.9) + Current Stat`). This ensured that recent fights were weighted more heavily when calculating rate features like `SSpm_Diff` and `Finish_Rate_Diff`.

### 3. Data Augmentation & Imbalance
The `Result_1_Binary` target variable was heavily imbalanced (favoring wins). This was solved at the data level:

* **Symmetrical Data Augmentation:** The training set was "flipped" and concatenated with the original, doubling its size.
* **Feature Inversion:** Symmetrical features (e.Example, `Elo_Diff`) were multiplied by -1.
* **Feature Swapping:** Asymmetrical features (e.g., `Total_Fights_1` <-> `Total_Fights_2`) were swapped to create a perfectly balanced training set.

### 4. Model Training and Evaluation
* **Data Split:** A strict **time-based split** (80% train, 20% test) was used to ensure the model was trained only on past data and tested on unseen future fights.
* **Scaling:** `StandardScaler` (Z-score) was fit on the augmented training data and applied to both training and test sets.
* **Modelling:** `RandomForestClassifier` and `XGBClassifier` were trained to predict the binary outcome.

---

## Results

The model's performance was evaluated on the unseen 20% test set. The Random Forest Classifier achieved the most stable result.

* **Weighted F1-Score:** **0.64**
* **Overall Accuracy:** **63.5%**

Feature importance analysis confirmed that the custom-engineered features (`SSpm_Diff`, 'Elo_Diff', 'TDpm_Diff') were the most significant predictors in the final model.

---

## Tech Stack

* **Python**
* **Pandas:** Data cleaning, merging, and manipulation.
* **NumPy:** Vectorized numerical operations and data augmentation.
* **scikit-learn:** `StandardScaler`, `RandomForestClassifier`, `classification_report`.
* **XGBoost:** `XGBClassifier` for comparative modelling.

---

## ⚙️ Future Improvements

* Increase the K-Factor for more experienced fighters and decrease for newer fighters so that earlier fights have less weighting.
* Add age to the fighters table which is a considerably strong predictor and would potentially make fights involving older fighters more accurate.
* Add opponent adjusted stats, e.g. `SSpm_1/2` is heavily dependent on what calibre of opponents the fighter is up against.
* Split the data so that training and test data are not split between time (fighting game in the UFC evolves over time could be completely different across two datasets).
