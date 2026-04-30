# US Accident Severity Prediction

* This repository holds an attempt to apply a Random Forest classifier to predict accident severity using the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data) dataset from Kaggle.

## Overview

* The task is to use tabular features describing road, weather, and temporal conditions at the time of a US traffic accident to predict its severity on a scale of 1–4 (1 = least impact on traffic, 4 = most impact).
* This repository formulates the problem as a multiclass classification task. A Random Forest classifier was trained on a 50,000-row stratified random sample of the full dataset. Preprocessing included temporal feature engineering, frequency encoding of high-cardinality categoricals, median/mode imputation, and class-weight balancing to address severe class imbalance.
* The model achieved a weighted F1-score of approximately 0.8223 on the held-out test set. Class imbalance — Severity 2 accounts for ~70% of all records — makes this a challenging prediction problem, particularly for Severity 1 (< 1% of data).

## Summary of Workdone

### Data

* **Data:**
  * Type: Tabular CSV — 47 input features (numerical, boolean, categorical, datetime) and 1 target column (`Severity`)
  * Size: Full dataset ~7.7M rows, 47 columns (~1.2 GB). A random sample of 50,000 rows was used for training due to memory constraints.
  * Instances: 35,000 training / 7,500 validation / 7,500 test (70/15/15 stratified split)

#### Preprocessing / Clean Up

* **Dropped irrelevant / leaky columns:** `ID`, `Source`, `Description` (free text), `End_Lat`, `End_Lng`, `Number`, `Wind_Chill(F)`, `Weather_Timestamp`, `Country`, `Turning_Loop`, `Airport_Code`, `Zipcode`
* **Dropped high-missingness columns:** Any column with >40% missing values was removed to avoid biased imputation
* **Temporal feature engineering:** Extracted `Hour`, `DayOfWeek`, `Month`, `Year`, and `Duration_min` (accident duration in minutes, clamped to [0, 1440]) from `Start_Time` and `End_Time`, then dropped the raw datetime columns
* **Imputation:** Median for numerical columns, mode (or "Unknown") for categorical columns, and `False` for boolean columns
* **Boolean encoding:** All boolean columns cast to integer (0/1)
* **Categorical encoding:** One-hot encoding for low-cardinality features (≤ 20 unique values); frequency encoding for high-cardinality features (`State`, `City`, `Street`, `Weather_Condition`, `Timezone`)
* **Feature scaling:** `StandardScaler` applied to all non-binary numerical columns (zero mean, unit variance)
* **Class imbalance:** Handled via `class_weight='balanced'` in the Random Forest — minority classes are automatically up-weighted without synthetic oversampling

#### Data Visualization

* Histograms of each numerical feature split by Severity class revealed that `Duration_min` and `Visibility(mi)` show the strongest distributional differences across severity levels — longer-duration and lower-visibility accidents skew toward higher severity.
* The class distribution bar chart confirms severe imbalance: Severity 2 dominates (~70%), while Severity 1 is extremely rare (~1%), motivating the use of class weighting.

### Problem Formulation

* **Input:** ~40+ engineered and encoded features describing temporal context (hour, day, month), weather (temperature, visibility, wind speed, precipitation), road infrastructure (junction, crossing, traffic signal, etc.), and geographic frequency-encoded location
* **Output:** Accident severity class — 0, 1, 2, or 3 (re-encoded from original 1–4)
* **Model:** Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
  * Chosen for its robustness to mixed feature types, built-in feature importance, no requirement for feature scaling, and native support for class weighting
  * Hyperparameters: `n_estimators=100`, `max_depth=20`, `min_samples_leaf=10`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`
* **No explicit loss function** (Random Forest uses Gini impurity for splitting); class weights adjust the effective contribution of each sample

### Training

* **Software:** Python 3, scikit-learn, pandas, NumPy, matplotlib, seaborn
* **Hardware:** Local machine running Ubuntu via WSL2; training was performed on CPU with `n_jobs=-1` to parallelize across all cores
* **Training time:** Approximately 120 seconds on the 35,000-row training set
* **No epoch-based training** (Random Forest trains in a single pass); no early stopping was required
* **Difficulties:** The full 7.7M-row dataset exceeded available WSL2 memory (~8 GB). This was resolved by sampling 50,000 rows randomly with a fixed seed (`random_state=42`) for reproducibility.

### Performance Comparison

* **Primary metric:** Weighted F1-score (accounts for class imbalance by weighting each class by its support)
* **Secondary metrics:** Macro F1-score (unweighted, reveals performance on rare classes), accuracy, per-class precision and recall

| Metric | Validation Set | Test Set |
|---|---|---|
| Accuracy | [0.8125] | [0.8091] |
| Weighted F1 | [0.8260] | [0.8223] |
| Macro F1 | [0.6039] | [0.6081] |

* Confusion matrices for both the validation and test sets are saved as `confusion_matrix_val.png` and `confusion_matrix_test.png`.
* The gap between weighted F1 and macro F1 quantifies how much the model struggles with rare classes (Severity 1 and 4).

### Conclusions

* Random Forest with class weighting is a strong baseline for accident severity prediction. The model performs well on the dominant Severity 2 class but struggles with Severity 1 due to its extreme rarity (~1% of data).
* Temporal features (`Hour`, `Duration_min`) and environmental features (`Visibility(mi)`, `Temperature(F)`) are among the most predictive, consistent with domain knowledge about accident risk factors.
* Frequency-encoding high-cardinality geographic features (City, Street) proved more practical than one-hot encoding at this dataset scale.

### Future Work

* **Hyperparameter tuning** via `RandomizedSearchCV` (deeper trees, more estimators)
* **XGBoost or LightGBM** — typically superior to Random Forest on tabular imbalanced data
* **SMOTE** applied to training set only for Severity 1 and 4 oversampling
* **NLP on `Description` column** — the free-text field was dropped here but likely contains strong severity signal
* **Full dataset training** on a machine with sufficient RAM or via cloud (Google Colab + Google Drive)
* **Geospatial features** — clustering lat/lng into regions rather than dropping them

## How to Reproduce Results

### Overview of Files in Repository

* `Kaggle_Tabular_Data_completed.ipynb` — Main notebook containing all steps: data loading, EDA, cleaning, preprocessing, model training, and evaluation. Run cells top to bottom.

### Software Setup

Required packages:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn


### Data

* Download the dataset from [Kaggle: US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data)
* Place `US_Accidents_March23.csv` in a local directory
* Update the `file_path` variable in the first code cell of the notebook to point to your CSV:
* A random sample of 50,000 or 100,000 rows is drawn immediately after loading to manage memory

### Training

* Open `Kaggle_Tabular_Data_completed.ipynb` in Jupyter Notebook or JupyterLab
* Run all cells sequentially from top to bottom

#### Performance Evaluation

* Validation metrics are printed and plotted automatically after the training cell
* Final test set metrics are printed in the last evaluation cell
* Plots saved to disk: `scaling_before_after.png`, `class_distribution.png`, `feature_distributions_by_severity.png`, `feature_importance.png`, `confusion_matrix_val.png`, `confusion_matrix_test.png`

## Citations

* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", 2019.
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
* scikit-learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 2011.