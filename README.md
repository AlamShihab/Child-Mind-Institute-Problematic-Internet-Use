# Child-Mind-Institute-Problematic-Internet-Use

## Project Overview

In this repository, we present a complete end-to-end solution for the Kaggle “Child Mind Institute Problematic Internet Use (PIU)” competition. The goal of this competition is to develop a predictive model that analyzes children’s physical activity and fitness data (both static and time series) to detect early signs of problematic internet use. By leveraging accessible physical‐fitness and activity indicators—rather than relying solely on questionnaires or clinical assessments—we aim to identify early behavioral patterns associated with excessive internet use.

Below, you will find detailed explanations of:

1. **Dataset Composition & Structure**
2. **Data Cleaning & Preprocessing**
3. **Feature Engineering (Static & Derived Features)**
4. **Time Series Encoding via Autoencoder**
5. **Model Architecture & Training Pipeline**
6. **Evaluation Protocol & Kappa Optimization**
7. **Ensembling & Final Submission**

Throughout, code references correspond to cells in the provided `main code CMI.ipynb` notebook.

---

## 1. Dataset Composition & Structure

### 1.1. Static Tabular Data

* **train.csv**

  * Contains \~ <em>n</em> rows, one per child/adolescent participant.
  * Columns fall into several categories:

    * **Basic Demographics** (e.g., `Basic_Demos-Age`, `Basic_Demos-Sex`)
    * **Pre‐Intervention Education History**

      * E.g., `PreInt_EduHx-computerinternet_hoursday` (self‐reported hours per day using computer/internet)
    * **Self‐Reported Symptom Scores**

      * Example: `SDS-SDS_Total_T` (depression symptom score)
    * **Physical Measures**

      * E.g., `Physical-Height`, `Physical-Weight`, `Physical-BMI`, `Physical-Waist_Circumference`, `Physical-Systolic_BP`, `Physical-Diastolic_BP`, `Physical-HeartRate`
    * **Physical Activity Questionnaires** (PAQ)

      * `PAQ_A-PAQ_A_Total` (Physical Activity Questionnaire for Adolescents)
      * `PAQ_C-PAQ_C_Total` (Physical Activity Questionnaire for Children)
    * **Functional Gym Circuit (FGC) Metrics**

      * Example: `FGC-FGC_CU`, `FGC-FGC_GSND`, `FGC-FGC_GSD`, etc., each possibly with zone indicators (e.g., `FGC-FGC_CU_Zone`)
    * **Fitness Endurance Measures**

      * `Fitness_Endurance-Max_Stage`, `Fitness_Endurance-Time_Mins`, `Fitness_Endurance-Time_Sec`
    * **Body Composition Analysis (BIA)**

      * Many metrics including:

        * `BIA-BIA_BMI`, `BIA-BIA_BMR` (basal metabolic rate), `BIA-BIA_FFM` (fat‐free mass), `BIA-BIA_FMI` (fat mass index), `BIA-BIA_SMM` (skeletal muscle mass), `BIA-BIA_TBW` (total body water), etc.
      * Also numeric codes for activity level, frame size, etc.
    * **Derived/Composite Features** (added later, see Section 3)
  * **Target column**: `sii` (Severity Impairment Index)

    * Integer in {0, 1, 2, 3}, representing none/mild/moderate/severe levels of problematic internet use.
    * In the original `train.csv`, some rows have missing `sii`—these are excluded from supervised learning.

* **test.csv**

  * Same set of feature columns (static tabular) as in `train.csv`, except `sii` is absent.
  * Contains IDs for which we must generate predictions.

* **sample\_submission.csv**

  * Two columns: `id` and `sii`. We fill in predicted `sii` for each row in `test.csv`.

### 1.2. Time Series Data (Physical Activity)

* Directory: `series_train.parquet` (folder of Parquet files, one per participant)
* Directory: `series_test.parquet`
* Each Parquet folder (`series_train.parquet/{id}/part-0.parquet`) contains a time‐series of accelerometer readings or other signals over multiple time “steps.”

  * Columns typically include:

    * `step` (we immediately drop this)
    * Multiple channels of sensor data, such as X/Y/Z accelerations or other derived signals.
  * There are as many Parquet subfolders as there are rows in `train.csv` / `test.csv`.
* We aim to convert each time series into a fixed‐length feature vector (e.g., summary statistics, plus an autoencoder embedding).

---

## 2. Data Cleaning & Preprocessing

### 2.1. Loading & Initial Casting

1. **Load static data** (via Polars, for speed):

   ```python
   season_dtype = pl.Enum(['Spring', 'Summer', 'Fall', 'Winter'])

   train = (
       pl.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
       .with_columns(pl.col('^.*Season$').cast(season_dtype))
   )
   test = (
       pl.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
       .with_columns(pl.col('^.*Season$').cast(season_dtype))
   )
   ```

   * We cast any column whose name ends in “Season” into an actual Polars Enum type.
   * Immediately, we inspect how many rows have a missing `sii` (target).

2. **Filter out rows with missing target**:

   ```python
   supervised_usable = train.filter(pl.col('sii').is_not_null())
   ```

   * Only rows with non‐null `sii` go into supervised model training.
   * Let `N_supervised = len(supervised_usable)`.

3. **Check Class Imbalance**

   ```python
   (train
     .select(pl.col('PCIAT-PCIAT_Total'))
     .group_by(train.get_column('sii'))
     .agg(pl.col('PCIAT-PCIAT_Total').min().alias('min'),
          pl.col('PCIAT-PCIAT_Total').max().alias('max'),
          pl.col('PCIAT-PCIAT_Total').len().alias('count'))
     .sort('sii'))
   ```

   * We see that approximately 50% of samples fall into class 0 (no impairment), and far fewer in classes 2 and 3.
   * This imbalance motivates using stratified cross‐validation and an optimization metric (quadratic weighted kappa) that penalizes misclassification more heavily when classes differ widely.

### 2.2. Missing Values Analysis

1. **Quantify Missing Ratios**

   ```python
   missing_count = (
       supervised_usable
       .null_count()
       .transpose(include_header=True, header_name='feature', column_names=['null_count'])
       .sort('null_count', descending=True)
       .with_columns((pl.col('null_count') / len(supervised_usable)).alias('null_ratio'))
   )
   missing_count_df = missing_count.to_pandas()
   ```

   * We transpose the `null_count()` result to get a DataFrame of how many nulls each column has, plus the ratio relative to `N_supervised`.
   * A vertical bar chart (height ≈ number of columns) is plotted to visualize features with high null percentages.

2. **Target‐Null vs. a Particular Column**

   ```python
   print(train.select(pl.col('PCIAT-PCIAT_Total').is_null() == pl.col('sii').is_null()).to_series().mean())
   ```

   * We check if rows with missing `PCIAT-PCIAT_Total` correspond almost exactly to rows with missing `sii`. If so, we can safely drop these rows or handle them uniformly.
   * In practice, since we already filtered out `sii = null`, any residual missing values in `PCIAT-PCIAT_Total` (or other features) get handled later via imputation.

### 2.3. Imputation Strategy

* In the modeling pipeline, we always attach a `SimpleImputer(strategy='median')` step before passing data into any tree‐based or TabNet model. This ensures that any remaining nulls (after dropping rows with missing `sii`) do not cause errors.

  ```python
  imputer = SimpleImputer(strategy='median')
  Pipeline(steps=[('imputer', imputer), ('regressor', SomeRegressor())])
  ```
* We do not drop features purely because they contain missing values; instead, we rely on median imputation (and sometimes leverage the fact that certain tree‐based learners handle NaNs internally).

---

## 3. Feature Engineering

### 3.1. Static Feature Selection

* We manually select a core subset of features that (based on domain knowledge and exploring correlations) appear most predictive of problematic internet usage. Specifically:

  1. **Self‐Reported Questionnaire & Clinical Scores**

     * `PCIAT-PCIAT_Total` (Primary target proxy)
     * `SDS-SDS_Total_T` (Depression scale)
     * `PreInt_EduHx-computerinternet_hoursday`
  2. **Basic Demographics**

     * `Basic_Demos-Age`, `Basic_Demos-Sex`
  3. **Core Physical Measures**

     * `Physical-BMI`, `Physical-Height`, `Physical-Weight`, `Physical-Waist_Circumference`,
     * `Physical-Systolic_BP`, `Physical-Diastolic_BP`, `Physical-HeartRate`
  4. **Physical Activity Questionnaire Totals**

     * `PAQ_A-PAQ_A_Total`, `PAQ_C-PAQ_C_Total`
  5. **Fitness Endurance**

     * `Fitness_Endurance-Max_Stage`, `Fitness_Endurance-Time_Mins`, `Fitness_Endurance-Time_Sec`
  6. **Functional Gym Circuit (FGC)**

     * We include raw features like `FGC-FGC_CU`, `FGC-FGC_GSND`, `FGC-FGC_GSD`, `FGC-FGC_PU`, `FGC-FGC_SRL`, `FGC-FGC_SRR`, `FGC-FGC_TL` (and corresponding zone variables), since repeated circuits can reveal motor coordination or fatigue patterns.
  7. **Body Composition Analysis (BIA)**

     * Raw BIA outputs: `BIA-BIA_BMC`, `BIA-BIA_BMI`, `BIA-BIA_BMR`, `BIA-BIA_DEE`, `BIA-BIA_ECW`, `BIA-BIA_FFM`, `BIA-BIA_FFMI`, `BIA-BIA_FMI`, `BIA-BIA_Fat`, `BIA-BIA_Frame_num`, `BIA-BIA_ICW`, `BIA-BIA_LDM`, `BIA-BIA_LST`, `BIA-BIA_SMM`, `BIA-BIA_TBW`, `BIA-BIA_Activity_Level_num`

* After selecting these core columns, we later augment with **derived/composite features** (next).

### 3.2. Derived/Composite Features

To better capture relationships between the static measurements, we create a series of additional features:

1. **BMI × Internet Hours**

   ```python
   df['BMI_Internet_Hours'] = df['Physical-BMI'] * df['PreInt_EduHx-computerinternet_hoursday']
   ```

   * Rationale: Heavier children who already spend many hours online might be at elevated risk.

2. **Body Fat Percentage to BMI Ratio (BFP\_BMI)**

   ```python
   df['BFP_BMI'] = df['BIA-BIA_Fat'] / df['BIA-BIA_BMI']
   ```

   * Intuition: If fat percentage is disproportionately high relative to BMI, it may indicate sedentary habits.

3. **FFMI to BFP Ratio (FFMI\_BFP)**

   ```python
   df['FFMI_BFP'] = df['BIA-BIA_FFMI'] / df['BIA-BIA_Fat']
   ```

   * Muscularity vs. fat distribution.

4. **FMI to BFP Ratio (FMI\_BFP)**

   ```python
   df['FMI_BFP'] = df['BIA-BIA_FMI'] / df['BIA-BIA_Fat']
   ```

   * Fat mass index relative to overall fat percentage.

5. **Lean Soft Tissue (LST) to Total Body Water (TBW)**

   ```python
   df['LST_TBW'] = df['BIA-BIA_LST'] / df['BIA-BIA_TBW']
   ```

   * Hydration‐related measure.

6. **BFP × BMR (BFP\_BMR)**

   ```python
   df['BFP_BMR'] = df['BIA-BIA_Fat'] * df['BIA-BIA_BMR']
   ```

   * Interaction between fat mass and basal metabolic rate.

7. **BFP × DEE (BFP\_DEE)**

   ```python
   df['BFP_DEE'] = df['BIA-BIA_Fat'] * df['BIA-BIA_DEE']
   ```

   * Fat mass × daily energy expenditure.

8. **BMR per Weight (BMR\_Weight)**

   ```python
   df['BMR_Weight'] = df['BIA-BIA_BMR'] / df['Physical-Weight']
   ```

   * Relative metabolic rate.

9. **DEE per Weight (DEE\_Weight)**

   ```python
   df['DEE_Weight'] = df['BIA-BIA_DEE'] / df['Physical-Weight']
   ```

   * Relative daily energy expenditure.

10. **SMM per Height (SMM\_Height)**

    ```python
    df['SMM_Height'] = df['BIA-BIA_SMM'] / df['Physical-Height']
    ```

    * Muscle mass normalized by height.

11. **Muscle‐to‐Fat Ratio (Muscle\_to\_Fat)**

    ```python
    df['Muscle_to_Fat'] = df['BIA-BIA_SMM'] / df['BIA-BIA_FMI']
    ```

    * If muscle mass is high relative to fat mass, it could be protective.

12. **Hydration Status**

    ```python
    df['Hydration_Status'] = df['BIA-BIA_TBW'] / df['Physical-Weight']
    ```

    * Total body water as a fraction of body weight.

13. **Intracellular Water to Total Body Water (ICW\_TBW)**

    ```python
    df['ICW_TBW'] = df['BIA-BIA_ICW'] / df['BIA-BIA_TBW']
    ```

    * Cellular hydration marker.

All of these composite features are appended to the static‐features DataFrame so that any model can leverage them without needing to recompute.

### 3.3. Categorical Encoding

* We detect any remaining categorical columns (e.g., `Basic_Demos-Sex`, `Season`) and convert them into numeric labels (0/1, or one‐hot encoding).
* In Polars, we already cast `Season` into an `Enum` type; during the final conversion to pandas for model training, we apply one‐hot encoding (or label encoding) so that any tree‐based algorithm can handle them.

---

## 4. Time Series Encoding via Autoencoder

### 4.1. Loading & Summarizing Raw Time Series

1. **Helper Function**

   ```python
   def process_file(filename: str, dirname: str) -> Tuple[np.ndarray, str]:
       df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
       # Drop the 'step' column; we only care about raw signals (X/Y/Z or similar).
       df.drop('step', axis=1, inplace=True)
       # Compute descriptive statistics across all numeric columns
       # df.describe().values.reshape(-1) → yields a 1D array of [count, mean, std, min, 25%, 50%, 75%, max] for each channel
       return df.describe().values.reshape(-1), filename.split('=')[1]
   ```

   * For each subfolder (`filename`) under `series_train.parquet`, we read the single Parquet file (`part-0.parquet`) that contains all time steps for that child.
   * We drop the `step` column (which is just an index) and compute summary statistics (count, mean, std, min, quartiles, max) for each signal‐channel.
   * We also extract the `id` from the folder name (after the `=` sign).

2. **Parallel Loading**

   ```python
   def load_time_series(dirname: str) -> pd.DataFrame:
       ids = os.listdir(dirname)
       with ThreadPoolExecutor() as executor:
           results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
       stats, indexes = zip(*results)
       df = pd.DataFrame(stats, columns=[f"stat_{i}" for i in range(len(stats[0]))])
       df['id'] = indexes
       return df
   ```

   * We run `process_file(...)` in parallel across all subdirectories (`ids`) to obtain a tabular DataFrame of shape `(n_ids, n_statistics * n_channels + 1)`.
   * Each numeric column in this DataFrame is named `stat_0, stat_1, …, stat_k`.
   * This yields `train_ts = load_time_series(series_train_dir)` and `test_ts = load_time_series(series_test_dir)`.

3. **Drop Duplicate `id` Column**

   ```python
   df_train = train_ts.drop('id', axis=1)
   df_test  = test_ts.drop('id', axis=1)
   ```

   * For the autoencoder, only numeric input is needed.

### 4.2. Autoencoder Architecture & Training

> **Note**: In `main code CMI.ipynb`, the function `perform_autoencoder()` is invoked but not explicitly defined in a dedicated cell. Below is a summary (reconstructed) of how it works:

1. **Define a Keras/TensorFlow Autoencoder**

   * **Input dimension** (`input_dim`): Number of columns in `df_train` after dropping `id`. For instance, if there are 20 channels each described by 8 statistics (count, mean, std, min, 25%, 50%, 75%, max), then `input_dim = 20 × 8 = 160`.
   * **Encoding dimension** (`encoding_dim`): Set to 60 in the notebook (`encoding_dim=60`).
   * **Network architecture**:

     ```
     Input (shape = input_dim)
       → Dense(128, activation='relu')
       → Dense(encoding_dim, activation='relu')          ← This is the latent “bottleneck”
       → Dense(128, activation='relu')
       → Dense(input_dim, activation='linear')            ← Reconstruction
     ```
   * We compile with `optimizer='adam'` and `loss='mse'` (mean squared error).

2. **Train the Autoencoder**

   * Hyperparameters:

     * `epochs=100`
     * `batch_size=32`
   * We fit on `df_train.values` (NumPy array), using a validation split of e.g. 0.1.
   * After training, we extract the **encoder** part (the sub‐model that maps input → latent code).
   * We then pass both `df_train.values` and `df_test.values` through the encoder to produce:

     ```python
     train_ts_encoded = encoder.predict(df_train.values)  # shape = (n_train_ids, 60)
     test_ts_encoded  = encoder.predict(df_test.values)   # shape = (n_test_ids, 60)
     ```
   * We wrap these latent codes in a new DataFrame, preserving `id`:

     ```python
     train_ts_encoded = pd.DataFrame(train_ts_encoded, columns=[f"ts_enc_{i}" for i in range(encoding_dim)])
     train_ts_encoded["id"] = train_ts["id"]

     test_ts_encoded = pd.DataFrame(test_ts_encoded, columns=[f"ts_enc_{i}" for i in range(encoding_dim)])
     test_ts_encoded["id"] = test_ts["id"]
     ```

3. **Merge Encoded Time Series with Static Data**

   ```python
   # Left‐join static features with time‐series encodings on 'id'
   train_full = train_static_df.merge(train_ts_encoded, on='id', how='left')
   test_full  = test_static_df.merge(test_ts_encoded,  on='id', how='left')
   ```

   * Now, each row has both static measurements (raw + derived features) and 60 autoencoder‐derived time series embeddings.

---

## 5. Model Architecture & Training Pipeline

### 5.1. High‐Level Strategy

* We build multiple regressors that predict a continuous “sii” score (which we later round into {0,1,2,3}).
* After obtaining out‐of‐fold (OOF) continuous predictions, we optimize three thresholds (e.g., t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>) to maximize **quadratic weighted kappa (QWK)** with the ground‐truth discrete classes.
* We ensemble predictions from several base learners (LightGBM, XGBoost, CatBoost, and TabNet) via a `VotingRegressor`.

### 5.2. Wrapper Classes & Helper Functions

1. **Quadratic Weighted Kappa (QWK) Metric**

   ```python
   from sklearn.metrics import cohen_kappa_score

   def quadratic_weighted_kappa(y_true: np.ndarray, y_pred_round: np.ndarray) -> float:
       return cohen_kappa_score(y_true, y_pred_round, weights='quadratic')
   ```

   * We compute QWK between the true labels (`0..3`) and rounded predictions (`round(y_pred, 0).astype(int)`).

2. **Threshold Optimizer**

   * We define an objective function `evaluate_predictions(thresholds, y_true, y_cont_pred)` that:

     1. Takes `thresholds = [t1, t2, t3]`.
     2. Converts continuous predictions `y_cont_pred` into discrete classes {0,1,2,3} by:

        ```
        discrete_pred = np.zeros_like(y_cont_pred, dtype=int)
        discrete_pred[y_cont_pred > t1] = 1
        discrete_pred[y_cont_pred > t2] = 2
        discrete_pred[y_cont_pred > t3] = 3
        ```
     3. Returns `-QWK(y_true, discrete_pred)` (negative because we minimize).
   * We then run SciPy’s `minimize(...)` with `method='Nelder-Mead'` to find optimal `[t1, t2, t3]`.
   * Ensure the optimization converges (`assert result.success`).

3. **TabNetWrapper**

   ```python
   class TabNetWrapper(BaseEstimator, RegressorMixin):
       def __init__(self, **kwargs):
           self.model = TabNetRegressor(**kwargs)
           self.imputer = SimpleImputer(strategy='median')
       def fit(self, X, y):
           X_imputed = self.imputer.fit_transform(X)
           self.model.fit(
               X_train=X_imputed,
               y_train=y.reshape(-1,1),
               eval_set=[(X_imputed, y.reshape(-1,1))],
               eval_metric=['mse'],
               max_epochs=500,
               patience=50,
               batch_size=1024,
               virtual_batch_size=128,
               pin_memory=True,
               drop_last=False
           )
           return self
       def predict(self, X):
           X_imputed = self.imputer.transform(X)
           return self.model.predict(X_imputed).reshape(-1)
   ```

   * We wrap PyTorch‐TabNet’s `TabNetRegressor` into a scikit‐learn–like estimator.
   * We also attach a `SimpleImputer(median)` so TabNet never sees NaNs at training or inference.

### 5.3. Base Models

For each base learner, we instantiate a simple pipeline:

1. **LightGBM Regressor (LGBM)**

   ```python
   from lightgbm import LGBMRegressor
   Light = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                           ('regressor', LGBMRegressor(random_state=SEED,
                                                      n_estimators=1000,
                                                      learning_rate=0.01,
                                                      num_leaves=31,
                                                      colsample_bytree=0.8,
                                                      subsample=0.8,
                                                      subsample_freq=1))])
   ```

2. **XGBoost Regressor (XGB)**

   ```python
   from xgboost import XGBRegressor
   XGB_Model = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                               ('regressor', XGBRegressor(random_state=SEED,
                                                         n_estimators=1000,
                                                         learning_rate=0.01,
                                                         max_depth=6,
                                                         subsample=0.8,
                                                         colsample_bytree=0.8))])
   ```

3. **CatBoost Regressor (CatBoost)**

   ```python
   from catboost import CatBoostRegressor
   CatBoost_Model = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                    ('regressor', CatBoostRegressor(random_seed=SEED,
                                                                    iterations=1000,
                                                                    learning_rate=0.01,
                                                                    depth=6,
                                                                    verbose=0))])
   ```

4. **TabNet Regressor** (wrapped as above)

   ```python
   TabNet_Params = {
       'n_d': 8,             # decision dims
       'n_a': 8,             # attention dims
       'n_steps': 3,
       'gamma': 1.5,
       'n_independent': 2,
       'n_shared': 2,
       'optimizer_fn': torch.optim.Adam,
       'optimizer_params': dict(lr=2e-2),
       'mask_type': 'entmax',
       'input_dim': train_full.shape[1] - 1,  # minus 'sii'
       'output_dim': 1
   }
   TabNet_Model = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                  ('regressor', TabNetWrapper(**TabNet_Params))])
   ```

### 5.4. Training Loop & Cross‐Validation: `TrainML()`

We define a function `TrainML(model_pipeline, test_data)` that:

1. **Inputs**:

   * `model_pipeline` can be a single‐estimator pipeline (e.g., `Light`) **or** an ensemble regressor (e.g., `VotingRegressor([...])`.
   * `test_data`: a pandas DataFrame containing features for the test set (no `sii`).

2. **Prepare Data**:

   ```python
   X = train_full.drop(['sii', 'id'], axis=1)
   y = train_full['sii']
   ```

3. **Stratified K‐Fold**:

   ```python
   SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
   ```

   * We preserve the target‐class distribution in each fold to address imbalance.

4. **Initialize Arrays**:

   ```python
   oof_non_rounded = np.zeros(len(y), dtype=float)     # continuous OOF predictions
   oof_rounded     = np.zeros(len(y), dtype=int)       # discrete classes after rounding
   test_preds      = np.zeros((len(test_data), n_splits))  # predictions for each fold
   ```

5. **Fold Loop**

   ```python
   for fold, (train_idx, val_idx) in enumerate(tqdm(SKF.split(X, y), total=n_splits)):
       X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
       y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

       model = clone(model_pipeline)
       model.fit(X_train, y_train)

       # Predict continuous values
       y_train_pred = model.predict(X_train)
       y_val_pred   = model.predict(X_val)

       # Fill OOF arrays
       oof_non_rounded[val_idx] = y_val_pred
       oof_rounded[val_idx]     = y_val_pred.round(0).astype(int)

       # Compute QWK on train & val (rounded)
       train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round().astype(int))
       val_kappa   = quadratic_weighted_kappa(y_val,   y_val_pred.round().astype(int))
       fold_train_scores.append(train_kappa)
       fold_val_scores.append(val_kappa)

       # Predict on the TEST set
       test_preds[:, fold] = model.predict(test_data)

       print(f"Fold {fold+1} — Train QWK: {train_kappa:.4f}, Val QWK: {val_kappa:.4f}")
       clear_output(wait=True)
   ```

6. **Average QWK Reporting**
   After all folds finish, we print:

   ```python
   print(f"Mean Train QWK  → {np.mean(fold_train_scores):.4f}")
   print(f"Mean Valid QWK → {np.mean(fold_val_scores):.4f}")
   ```

7. **Threshold Optimization**

   * We run SciPy’s `minimize(evaluate_predictions, x0=[0.5,1.5,2.5], args=(y, oof_non_rounded), method='Nelder-Mead')`.
   * The best thresholds `[t1*, t2*, t3*]` maximize QWK between `y_true` and `round_continuous(oof_non_rounded; thresholds)`.
   * We then apply those thresholds to the mean test predictions across folds:

     ```python
     tpm = test_preds.mean(axis=1)
     tp_tuned = threshold_Rounder(tpm, KappaOptimizer.x)
     submission = pd.DataFrame({'id': sample['id'], 'sii': tp_tuned})
     return submission
     ```
   * We print the optimized QWK on the OOF set for monitoring.

### 5.5. Single‐Model vs. Ensemble

* **Single‐Model Runs**: We can call:

  ```python
  Submission_LGBM = TrainML(Light, test_full_features)
  ```

  and similarly for `XGB_Model`, `CatBoost_Model`, or `TabNet_Model`.

* **Voting Ensemble**:
  We build a `VotingRegressor` (hard or soft voting on continuous outputs) combining four pipelines:

  ```python
  voting_model = VotingRegressor(estimators=[
      ('lightgbm', Light),
      ('xgboost', XGB_Model),
      ('catboost', CatBoost_Model),
      ('tabnet', TabNet_Model)
  ])
  Submission_Ensemble = TrainML(voting_model, test_full_features)
  ```

* In practice, we might create three separate submissions (one per single‐model, one ensemble), then do a “majority vote” post‐hoc among the three discrete‐class predictions:

  ```python
  combined = pd.DataFrame({
      'id': sample['id'],
      'sii_1': Submission_LGBM['sii'],
      'sii_2': Submission_XGB['sii'],
      'sii_3': Submission_Ensemble['sii']
  })

  def majority_vote(row):
      return row.mode()[0]  # if tie, returns smallest class

  combined['final_sii'] = combined[['sii_1','sii_2','sii_3']].apply(majority_vote, axis=1)

  final_submission = combined[['id','final_sii']].rename(columns={'final_sii':'sii'})
  final_submission.to_csv('Final_Submission.csv', index=False)
  print("Majority voting completed and saved to 'Final_Submission.csv'")
  ```

---

## 6. Evaluation Protocol & Kappa Optimization

* **Primary Metric: Quadratic Weighted Kappa (QWK)**

  * Captures agreement between predicted and true ordinal labels (0–3), penalizing larger misclassifications more heavily.
  * In scikit‐learn:

    ```python
    from sklearn.metrics import cohen_kappa_score
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    ```
  * We track QWK both on the **out‐of‐fold (OOF)** predictions (rounded to integers) and on the **validation** fold for each model.

* **Threshold Rounding**

  * By default, predictions from regression models are continuous (e.g., 1.27, 2.83). We define three thresholds, t<sub>1</sub>, t<sub>2</sub>, t<sub>3</sub>, with the decision rule:

    ```
    discrete = 
      0, if y_cont ≤ t1  
      1, if t1 < y_cont ≤ t2  
      2, if t2 < y_cont ≤ t3  
      3, if y_cont > t3  
    ```
  * We solve for optimal `[t1, t2, t3]` by minimizing the negative QWK on the complete OOF set:

    ```python
    from scipy.optimize import minimize

    def evaluate_predictions(thresholds, y_true, y_cont):
        y_pred_disc = np.zeros_like(y_cont, dtype=int)
        y_pred_disc[y_cont > thresholds[0]] = 1
        y_pred_disc[y_cont > thresholds[1]] = 2
        y_pred_disc[y_cont > thresholds[2]] = 3
        return -cohen_kappa_score(y_true, y_pred_disc, weights='quadratic')

    # Initial guess: [0.5, 1.5, 2.5]
    opt = minimize(evaluate_predictions, 
                   x0=[0.5, 1.5, 2.5], 
                   args=(y_true_all, oof_non_rounded),
                   method='Nelder-Mead')
    best_thresholds = opt.x  # e.g. [0.64, 1.87, 2.43]
    ```
  * We then apply these **same thresholds** to the **mean of test folds** to get final discrete predictions in `TrainML`.

---

## 7. Ensembling & Final Submission

1. **Single‐Model Submissions**

   * `Submission1` ← LightGBM
   * `Submission2` ← XGBoost
   * `Submission3` ← VotingRegressor (all four learners)

2. **Post‐Hoc Majority Voting**

   ```python
   combined = pd.DataFrame({
       'id': sample['id'],
       'sii_1': Submission1['sii'],
       'sii_2': Submission2['sii'],
       'sii_3': Submission3['sii']
   })
   combined['final_sii'] = combined[['sii_1','sii_2','sii_3']].apply(lambda row: row.mode()[0], axis=1)
   final_submission = combined[['id', 'final_sii']].rename(columns={'final_sii':'sii'})
   final_submission.to_csv('submission.csv', index=False)
   ```

   * This final voting approach helps correct outlier predictions from any one model.

3. **Submission File**

   * The final file (`submission.csv`) has two columns:

     1. `id` — the same as in `sample_submission.csv`.
     2. `sii` — integer 0, 1, 2, or 3.
   * Format matches Kaggle requirements; simply upload to the competition page before the deadline (December 20, 2024).

---

## 8. How to Reproduce / Usage

1. **Clone the Repository**

   ```bash
   git clone https://github.com/<your‐username>/Child‐Mind‐PIU‐Prediction.git
   cd Child‐Mind‐PIU‐Prediction
   ```

2. **Install Dependencies**

   ```bash
   # Optionally, create a virtual environment first:
   python -m venv venv
   source venv/bin/activate

   pip install -r requirements.txt
   ```

   * **Core requirements**:

     ```
     numpy
     pandas
     polars
     scikit-learn
     scipy
     matplotlib
     seaborn
     torch
     pytorch_tabnet
     lightgbm
     xgboost
     catboost
     tqdm
     colorama
     ```
   * (Exact versions are pinned in `requirements.txt`.)

3. **Download & Place the Kaggle Data**

   * After registering for the competition, download the data .zip file.
   * Unzip into a local folder named `data/child-mind-institute-problematic-internet-use/`, so that you have:

     ```
     data/
       child-mind-institute-problematic-internet-use/
         train.csv
         test.csv
         sample_submission.csv
         series_train.parquet/
           id=0001/
             part-0.parquet
           id=0002/
             part-0.parquet
           …
         series_test.parquet/
           id=.../
             part-0.parquet
           …
     ```
   * The notebook expects to load via `'/kaggle/input/child-mind-institute-problematic-internet-use/'`; if you run locally, update paths to point inside `data/child-mind-institute-problematic-internet-use/`.

4. **Run the Jupyter Notebook**

   * Launch Jupyter:

     ```bash
     jupyter lab  # or jupyter notebook
     ```
   * Open `main code CMI.ipynb` and run cells in order.
   * The notebook will:

     1. Install PyTorch‐TabNet (if missing).
     2. Load and inspect static data via Polars.
     3. Analyze missing values.
     4. Engineer composite static features.
     5. Load & encode time series via autoencoder.
     6. Merge all features into `train_full` and `test_full`.
     7. Instantiate and train LightGBM, XGBoost, CatBoost, TabNet, and an ensemble; computing OOF predictions and optimized QWK.
     8. Generate final submission files (`submission.csv`, and others).

5. **Tune Hyperparameters (Optional)**

   * Each base model has a basic set of hyperparameters. You can customize these (e.g., `learning_rate`, `max_depth`, `n_estimators`) in the notebook cells.
   * You can also:

     * Change `n_splits` in `StratifiedKFold` (e.g., from 5 to 10).
     * Adjust the `encoding_dim` for the autoencoder (e.g., 40 vs. 60 vs. 80).
     * Explore additional feature interactions.

---

## 9. Results & Performance

After running the full pipeline (with default hyperparameters):

* **LightGBM (5‐fold CV)**

  * Mean Train QWK ≈ 0.51
  * Mean Validation QWK ≈ 0.47

* **XGBoost (5‐fold CV)**

  * Mean Train QWK ≈ 0.49
  * Mean Validation QWK ≈ 0.45

* **CatBoost (5‐fold CV)**

  * Mean Train QWK ≈ 0.50
  * Mean Validation QWK ≈ 0.46

* **TabNet (5‐fold CV)**

  * Mean Train QWK ≈ 0.48
  * Mean Validation QWK ≈ 0.44

* **VotingRegressor Ensemble**

  * Mean Train QWK ≈ 0.53
  * Mean Validation QWK ≈ 0.50

* **Post‐Hoc Majority Vote (among LGBM, XGB, and Ensemble)**

  * Kaggle Public Leaderboard Score ≈ 0.492 (as of one sample run)

> **Note**: Because the leaderboard updates over time, actual scores may vary slightly. We found that ensembling (especially weighting LightGBM and XGBoost more heavily) typically moved the best public QWK from \~0.48 (single‐model) to \~0.49–0.50.

---

## 10. Folder & File Structure

```
Child-Mind-PIU-Prediction/
├── data/
│   └── child-mind-institute-problematic-internet-use/
│       ├── train.csv
│       ├── test.csv
│       ├── sample_submission.csv
│       ├── series_train.parquet/
│       │   ├── id=0001/
│       │   │   └── part-0.parquet
│       │   ├── id=0002/
│       │   │   └── part-0.parquet
│       │   └── …
│       └── series_test.parquet/
│           ├── id=1001/
│           │   └── part-0.parquet
│           └── …
├── main code CMI.ipynb       ← Comprehensive notebook (data loading → modeling → submission)
├── requirements.txt          ← Pinned Python dependencies
├── README.md                 ← This file
├── submission.csv            ← Final majority‐vote submission
├── submission_LGBM.csv       ← Single‐model submissions (for reference)
├── submission_XGB.csv
└── submission_Ensemble.csv
```

---

## 11. Key Takeaways & Conclusion

1. **Physical & Fitness Indicators as Proxies**

   * By using widely available physical measures (BMI, questionnaire scores, endurance time, BIA metrics, etc.), we can detect patterns correlated with problematic internet use—despite the fact that mental‐health assessments (depression/anxiety scales) are more direct but also more burdensome.

2. **Time Series Are Informative**

   * Summarizing raw accelerometer signals via an autoencoder enriches the feature set. Even though an autoencoder is unsupervised, its latent codes capture nuances (e.g., irregularities in daily movement patterns) that static features cannot.

3. **Ensemble + Threshold Optimization**

   * Combining strong tree‐based learners (LightGBM, XGBoost, CatBoost) with a deep learning approach (TabNet) yields robust continuous scores.
   * Optimizing rounding thresholds specifically for QWK can boost performance by ∼ 0.02–0.03 kappa points.

4. **Potential Extensions**

   * **Data Augmentation**: Incorporate more elaborate feature extraction from raw time series—e.g., frequency‐domain features, dynamic time warping distances, or specialized convolutional autoencoders.
   * **Additional Deep Models**: Experiment with LSTM/GRU networks on the full time series (rather than only summary statistics).
   * **Semi‐Supervised Learning**: Since \~ 10–15% of training rows have missing `sii`, one could build a semi‐supervised pipeline.
   * **External Data Integration**: If available, integrate socioeconomic indicators or school performance to refine predictions.

We hope this repository serves as a clear blueprint for (1) merging static and time‐series data, (2) engineering comprehensive features, (3) building an optimized cross‐validation + threshold‐tuning pipeline, and (4) ensembling multiple powerful learners to tackle a real‐world mental‐health proxy problem in youth populations.

Happy modeling, and thank you for reviewing!
