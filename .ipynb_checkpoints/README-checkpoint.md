# US Accident Severity Prediction

This repository holds an attempt to apply a Random Forest classifier to predict accident severity using the [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents/data) dataset from Kaggle.

## Overview

The task is to use tabular features describing road, weather, and temporal conditions at the time of a US traffic accident to predict its severity on a scale of 1–4 (1 = least impact on traffic, 4 = most impact). A Random Forest classifier was trained on a 100,000-row stratified random sample of the full dataset. Preprocessing included temporal feature engineering, frequency encoding of high-cardinality categoricals, median/mode imputation, and class-weight balancing to address severe class imbalance. The model achieved a weighted F1-score of approximately 0.8223 on the held-out test set. Class imbalance — Severity 2 accounts for ~70% of all records — makes this a challenging prediction problem, particularly for Severity 1 (< 1% of data).

## Summary of Workdone

### Data

* **Data:**
  * Type: Tabular CSV — 47 input features (numerical, boolean, categorical, datetime) and 1 target column (`Severity`)
  * Size: Full dataset ~7.7M rows, 47 columns (~1.2 GB). A random sample of 100,000 rows was used for training due to memory constraints.
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

* Histograms of each numerical feature split by Severity class revealed that most of the numerical features do not have clear distributional differences. 
* The class distribution bar chart confirms severe imbalance: Severity 2 dominates (~70%), while Severity 1 is extremely rare (~1%), motivating the use of class weighting.

![Class Distribution](images/class_distribution.png)
*The dataset is heavily imbalanced — Severity 2 accounts for ~70% of all records.*

![Feature Distributions](images/feature_distributions_kde.png)
*KDE plots of numerical features by severity class. Severity 2 and 3 show heavy 
overlap across most features, consistent with the model's confusion matrix results.*

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
* **Difficulties:** The full 7.7M-row dataset exceeded available WSL2 memory (~8 GB). This was resolved by sampling 100,000 rows randomly with a fixed seed (`random_state=42`) for reproducibility.

### Performance Comparison

* **Primary metric:** Weighted F1-score (accounts for class imbalance by weighting each class by its support)
* **Secondary metrics:** Macro F1-score (unweighted, reveals performance on rare classes), accuracy, per-class precision and recall

| Metric | Validation Set | Test Set |
|---|---|---|
| Accuracy | 0.8125 | 0.8091 |
| Weighted F1 | 0.8260 | 0.8223 |
| Macro F1 | 0.6039 | 0.6081 |

* The gap between weighted F1 and macro F1 quantifies how much the model struggles with rare classes (Severity 1 and 4).

![Confusion Matrix - Validation](images/confusion_matrix_val.png)
*Confusion matrix on the validation set. The dominant error is Severity 3 events 
predicted as Severity 2 (1,565 misclassifications).*

![Confusion Matrix - Test](images/confusion_matrix_test.png)
*Confusion matrix on the held-out test set.*

![Feature Importance](images/feature_importance.png)
*Top 20 features by mean decrease in Gini impurity. Distance(mi), Year, and 
Duration_min account for ~41% of total importance.*

### Conclusions

The Random Forest classifier achieved **80.9% accuracy** and a **weighted F1-score 
of 0.822** on the held-out test set, with consistent results on the validation set 
(81.3% accuracy, 0.826 weighted F1), indicating the model generalizes well and is 
not overfitting.

However, the **macro F1 of 0.608** tells a more honest story — performance varies 
dramatically across severity classes, driven by class imbalance and the inherent 
difficulty of distinguishing adjacent severity levels.

**Per-class breakdown:**
- **Severity 2 (dominant class):** The model performs strongly here — F1 of 0.88, 
  precision of 0.94. This is expected given that ~70% of training data belongs to 
  this class.
- **Severity 3:** Moderate performance (F1 = 0.63 test, 0.65 val). The model 
  achieves decent recall (0.77) but low precision (0.54), meaning it over-predicts 
  Severity 3 — many Severity 2 events are pulled into this bucket.
- **Severity 1:** Despite representing only ~1% of the data (126 test samples), 
  the model achieves 0.80 recall — meaning it catches 4 out of 5 rare Severity 1 
  events. Precision is low (0.44), but this is an acceptable tradeoff: it is 
  better to over-flag minor accidents than to miss serious ones.
- **Severity 4:** The weakest class (F1 = 0.35). With only 407 test samples and 
  the highest real-world severity, this is the most concerning gap. Low precision 
  (0.30) and recall (0.43) suggest the model does not have enough signal to 
  reliably identify the most dangerous accidents.

**The key failure mode** is the Severity 2 / Severity 3 boundary. In validation, 
1,565 Severity 3 events were misclassified as Severity 2. These two classes likely 
share highly similar feature profiles — the difference between a "moderate" and 
"serious" accident in terms of weather, road features, and time of day may be 
subtle enough that tabular features alone cannot reliably separate them.

**Feature importance** reveals that the three most predictive features are 
`Distance(mi)` (0.164), `Year` (0.123), and `Duration_min` (0.123) — together 
accounting for ~41% of the model's decisions. Distance and duration are direct 
proxies for accident impact on traffic, which aligns with how Severity is defined 
in this dataset. The prominence of `Year` suggests reporting patterns or road 
conditions changed meaningfully across 2016–2023. Geographic frequency features 
(`Street_freq`, `County_freq`, `City_freq`) collectively contribute ~16%, 
confirming that location — specifically how accident-prone a given road or area is 
— is a strong predictor of severity.

**In summary:** the model is a strong baseline for Severity 2 detection and shows 
promising recall on rare classes, but the Severity 3/4 boundary remains a 
significant challenge. The results are consistent between validation and test sets, 
which validates the integrity of the preprocessing and splitting pipeline.

### Future Work

Several concrete directions could meaningfully improve on these results:

**Addressing the Severity 3/4 weakness**
The most pressing issue is the model's inability to reliably identify Severity 3 
and 4 events. One avenue is reframing the problem as **ordinal classification** — 
Severity has a natural order (1 < 2 < 3 < 4), and standard multiclass classifiers 
ignore this. Libraries like `mord` implement ordinal regression, which penalizes 
predictions that are further from the true class more heavily than adjacent 
misclassifications. Another option is **cost-sensitive learning** — manually 
specifying a misclassification cost matrix where predicting Severity 2 for a true 
Severity 4 event is penalized far more severely than predicting Severity 3 for it.

**Training on the full dataset**
This project trained on a 100,000-row sample due to WSL2 memory constraints. The 
full dataset contains ~7.7M records, and rare classes (Severity 1: ~1%, Severity 4: 
~3%) would have substantially more training examples at full scale — likely 
improving recall on those classes the most. Google Colab with a high-RAM runtime 
(up to 52GB) would allow full-dataset training without any code changes beyond 
removing the sampling line.

**Temporal drift analysis**
`Year` was the second most important feature (importance = 0.123), which likely 
reflects changes in reporting methodology, road infrastructure, or traffic patterns 
across 2016–2023 rather than a causal relationship with severity. This raises the 
question of whether a model trained on 2016–2019 data generalizes to 2022–2023 
data. A temporal train/test split — training on earlier years and testing on later 
ones — would give a more realistic estimate of real-world performance and expose 
any temporal drift in the data.

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

### Performance Evaluation

The primary evaluation metric is **weighted F1-score**, chosen over accuracy because 
the dataset is heavily imbalanced — ~70% of records are Severity 2, meaning a model 
that always predicts Severity 2 would achieve 70% accuracy while being completely 
useless. Weighted F1 accounts for this by computing the harmonic mean of precision 
and recall per class, weighted by each class's frequency. Macro F1 is reported 
alongside it as a secondary metric — it weights all classes equally, exposing 
performance on rare classes that weighted F1 can mask.

#### Results Summary

| Metric        | Validation | Test   |
|---------------|------------|--------|
| Accuracy      | 0.8125     | 0.8091 |
| F1 (weighted) | 0.8260     | 0.8223 |
| F1 (macro)    | 0.6039     | 0.6081 |---

#### Per-Class Performance (Test Set)

| Class      | Precision | Recall | F1   | Support |
|------------|-----------|--------|------|---------|
| Severity 1 | 0.44      | 0.80   | 0.56 | 126     |
| Severity 2 | 0.94      | 0.83   | 0.88 | 11,939  |
| Severity 3 | 0.54      | 0.77   | 0.63 | 2,528   |
| Severity 4 | 0.30      | 0.43   | 0.35 | 407     |

The 13-point gap between weighted F1 (0.822) and macro F1 (0.608) reflects the 
model's uneven performance across classes. Severity 2 is predicted with high 
confidence (F1 = 0.88, precision = 0.94), while Severity 4 is the weakest class 
(F1 = 0.35) despite being the most safety-critical.

The most common misclassification is **Severity 3 predicted as Severity 2** — 
1,565 such cases in validation alone. These two classes share overlapping feature 
profiles, and the boundary between them appears too subtle for tabular features 
alone to resolve reliably.

Validation and test results are nearly identical (0.826 vs. 0.822 weighted F1), 
confirming the model generalizes well and the preprocessing pipeline is sound.

See `confusion_matrix_val.png` and `confusion_matrix_test.png` for the full 
breakdown of misclassifications across all class pairs.

## Citations

* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", 2019.
* Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.
* scikit-learn: Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR 12, 2011.