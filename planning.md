# Titanic — Machine Learning from Disaster (CRISP-DM)

## 0. Project Setup

### Repository Structure

```text
titanic-ml/
├── data/                # raw/processed/submission
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment_submission.ipynb
├── submission/
└── README.md
```

## 1. Business Understanding

### Goal

Predict **Survived** for passengers in `test.csv` and submit a `submission.csv` with columns `PassengerId`, `Survived`.

### Success Criteria

**Primary:** Kaggle public LB accuracy ≥ simple baseline (e.g., > 0.76555 "gender baseline").

**Secondary:** Cross-validation accuracy stable (std ≤ ~0.02), minimal gap vs LB (overfit risk ≤ ~0.03).

### Constraints & Risks

- Small tabular dataset (891 train rows) → high variance; prefer simple, regularized models and careful CV.
- Potential leakage via engineered features if built from test set stats—avoid global stats that peek at test.

### Deliverables

- Short executive summary (what moved the needle).
- Reproducible pipeline + notebook links.
- Submission file in `submission/`.

## 2. Data Understanding

### Inputs

- `train.csv` (891 rows): includes Survived
- `test.csv` (418 rows): no Survived
- `gender_submission.csv` (example format)

### First Pass Checks

- Schema & types; missingness (notably Age, Cabin, some Embarked)
- Unique IDs: PassengerId
- Class balance: proportion survived
- Data leakage scan (e.g., Ticket groups that span train/test)

### Exploration Checklist

- Univariate distributions (Age, Fare, Pclass, Sex, Embarked)
- Bivariate with target: survival rate by Sex, Pclass, Embarked, Age buckets, Fare quantiles
- Interaction plots: Sex × Pclass, FamilySize × Pclass
- Outliers: extreme Fare, implausible Age
- Text fields (Name, Ticket, Cabin) quick look for signal

### Artifacts

- `reports/figures/*.png` (or inline figs)
- Notes on early signals (e.g., strong survival lift for Sex=female, Pclass=1)

## 3. Data Preparation

### Plan

**Target:** Survived (binary)

**Train/Validation split:** Stratified K-fold (k=5 or 10), grouped if you decide to group by Ticket to reduce leakage.

### Imputation

- **Age:** model-based imputation (e.g., RandomForestRegressor) or stratified median by Sex × Pclass
- **Embarked:** mode
- **Fare:** median (or leave; some models handle)

### Encoding/Scaling

- **Categorical:** one-hot (Sex, Embarked, extracted Title, Deck)
- **Numeric:** standardize for linear models/SVM; tree models don't require scaling

### Feature Ideas (proven winners)

- Title from Name (Mr, Miss, Mrs, Master, Rare)
- FamilySize = SibSp + Parch + 1
- IsAlone = 1 if FamilySize==1 else 0
- TicketGroupSize (count passengers sharing Ticket)
- CabinKnown = 1/0; Deck from Cabin first letter
- FarePerPerson = Fare / FamilySize
- AgeBin, FareBin (quantile bins for linear models)
- Interactions: Sex × Pclass, Title × Pclass

### Pipeline Skeleton (sklearn)

Use ColumnTransformer + Pipeline, so the exact same steps fit on train and transform test without drift.

### Quality Gates

- No data from test.csv used to compute training statistics
- Consistent column order between train and test after transforms
- Save fitted pipeline (pickle) and the feature columns list

## 4. Modeling

### Baselines (establish floor)

- Majority class
- Simple rules: predict female=1, male=0 (or gender_submission.csv)

### Models to try (in order)

- Logistic Regression (with/without interactions)
- Tree-based: RandomForest, GradientBoosting, XGBoost/LightGBM
- Linear SVM (after scaling)
- Stacking (only if you've stabilized CV)

### Cross-Validation

- Stratified K-fold
- If using TicketGroupSize, consider GroupKFold by Ticket to reduce optimistic bias

### Hyperparameters (compact grids)

- **Logistic:** C ∈ {0.1, 1, 3, 10}, penalty {l2}
- **RF:** n_estimators ∈ {300, 800}, max_depth ∈ {None, 5, 8}, min_samples_leaf ∈ {1, 3, 5}
- **GBDT/LGBM:** n_estimators ∈ {300, 800}, learning_rate ∈ {0.03, 0.1}, max_depth ∈ {3, -1}, num_leaves (LGBM) ∈ {15, 31, 63}
- Calibrate thresholds only if needed (usually not for accuracy)

### Tracking

- Keep a results table: model, features set, CV mean, CV std, LB score, notes
- Save the best pipeline + params JSON

## 5. Evaluation

### What to check

- CV performance vs hold-out (if used) vs Public LB
- Error analysis: who's misclassified?
  - False negatives among women/children in 1st class?
  - Specific Ticket groups consistently wrong?
- Feature importance (for trees) or coefficients (for logistic) to sanity-check signals

### Robustness tests

- Remove a high-leakage feature (e.g., TicketGroupSize) → does score collapse?
- Vary bins/thresholds slightly → stability?

### Accept/Revise Decision

- If CV ≈ LB and above baseline, proceed to submit.
- If LB << CV, look for leakage/data shift; simplify features; tighten CV (grouped folds).

## 6. Deployment (Kaggle Submission)

### Build submission.csv

- Exactly 2 columns: PassengerId, Survived
- 418 rows (matches test.csv)
- No extra columns/index; integer {0,1}

### Submission protocol

- Name files clearly: `submission/submission_YYYYMMDD_modelname.csv`
- Log Kaggle LB score and diff vs CV
- Keep top 2–3 submissions and notes on what changed

## Working Checklist

### Business Understanding

- [ ] Write success criteria and baseline target
- [ ] Risks/leakage notes
- [ ] Executive summary stub created

### Data Understanding

- [ ] Load `train.csv`/`test.csv`
- [ ] Missingness table & class balance
- [ ] Targeted EDA figs saved

### Data Preparation

- [ ] Feature extraction: Title, FamilySize, IsAlone, Deck, TicketGroupSize, FarePerPerson
- [ ] Imputation plan implemented
- [ ] ColumnTransformer + Pipeline ready
- [ ] Train/test transforms verified

### Modeling

- [ ] Baselines recorded
- [ ] CV scheme chosen (Stratified/GroupKFold)
- [ ] Compact hyperparam search run
- [ ] Results table updated, best pipeline saved

### Evaluation

- [ ] Error analysis notes
- [ ] Importance/coef sanity check
- [ ] LB vs CV gap assessed

### Deployment

- [ ] submission.csv created (418 rows)
- [ ] Submitted; LB score logged
- [ ] Short write-up of what worked

## Code Snippets

### Feature Engineering

```python
# Extract title from name
title = name.split(',')[1].split('.')[0].strip()

# Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Ticket group size
df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')

# Deck letter from Cabin
df['Deck'] = df['Cabin'].str[0].fillna('U')
```

### Model Setup

```python
# Stratified K-fold
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# One-hot with ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Submission formatting
df[['PassengerId','Survived']].to_csv(path, index=False)
```

## Stretch Ideas (after a solid baseline)

- Target encoding (careful, use CV-safe scheme)
- Rare-title grouping (e.g., Lady/Countess/Capt → "Rare")
- GroupKFold by Ticket to combat family/group leakage
- Simple stacking: LR on top of GBDT + RF predictions (out-of-fold)
