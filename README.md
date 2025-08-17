# Titanic: Machine Learning from Disaster

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF.svg)](https://www.kaggle.com/c/titanic)

A comprehensive machine learning project predicting passenger survival on the Titanic using the **CRISP-DM methodology**. This project demonstrates end-to-end data science workflow from business understanding to deployment, with professional documentation and reproducible results.

## 🎯 Project Overview

This project tackles the famous Kaggle Titanic competition using a systematic approach based on the Cross-Industry Standard Process for Data Mining (CRISP-DM). The goal is to predict passenger survival with high accuracy while maintaining model interpretability and business alignment.

### 🏆 Key Achievements

- **Methodology**: Complete CRISP-DM implementation across 6 phases
- **Performance**: Gradient Boosting model with 84.1% CV accuracy
- **Features**: Advanced feature engineering (Title, FamilySize, FarePerPerson, etc.)
- **Reproducibility**: Professional notebooks and automated pipeline
- **Documentation**: Comprehensive planning and evaluation reports

## 📊 Business Problem

**Objective**: Predict passenger survival (binary classification) for Titanic disaster

- **Dataset**: 891 training samples, 418 test samples
- **Target**: Achieve >76.6% accuracy (gender baseline)
- **Constraints**: Small dataset, potential overfitting, feature leakage risks

## 🛠️ Technical Stack

- **Languages**: Python 3.8+
- **Libraries**: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost
- **Tools**: Jupyter Notebooks, Git
- **Methodology**: CRISP-DM (6 phases)
- **ML Techniques**: Gradient Boosting, Random Forest, Logistic Regression

## 📁 Project Structure

```
titanic-ml/
├── 📊 data/
│   ├── raw/                    # Original Kaggle datasets
│   ├── interim/                # Intermediate processing results  
│   └── processed/              # Final feature-engineered datasets
├── 📓 notebooks/
│   ├── 01_business_understanding.ipynb    # CRISP-DM Phase 1
│   ├── 02_data_understanding.ipynb       # CRISP-DM Phase 2  
│   ├── 03_data_preparation.ipynb         # CRISP-DM Phase 3
│   ├── 04_modeling.ipynb                 # CRISP-DM Phase 4
│   ├── 05_evaluation.ipynb               # CRISP-DM Phase 5
│   └── 06_deployment_submission.ipynb    # CRISP-DM Phase 6
├── 🤖 models/                  # Saved model artifacts
├── 📈 submission/              # Kaggle submission files
├── 📋 planning.md              # Detailed project roadmap
└── 📖 README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn xgboost jupyter
```

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/ejparnell/titanic-ml.git
   cd titanic-ml
   ```

2. **Download Kaggle data**
   - Download `train.csv` and `test.csv` from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
   - Place files in `data/raw/`

3. **Run the notebooks**

   ```bash
   jupyter notebook
   ```

   - Execute notebooks in order (01 → 06) for complete pipeline
   - Each notebook represents one CRISP-DM phase

## 🔬 Methodology: CRISP-DM Implementation

### Phase 1: Business Understanding

- **Goal**: Binary classification for passenger survival
- **Success Criteria**: >76.6% accuracy, stable CV, minimal overfitting
- **Notebook**: `01_business_understanding.ipynb`

### Phase 2: Data Understanding

- **Exploratory Analysis**: Missing values, class distribution, feature relationships
- **Key Insights**: Strong survival patterns by Sex, Pclass, Age, Fare
- **Notebook**: `02_data_understanding.ipynb`

### Phase 3: Data Preparation

- **Feature Engineering**: Title extraction, FamilySize, FarePerPerson, Deck
- **Imputation**: Age (stratified), Embarked (mode), Fare (median)
- **Encoding**: One-hot for categoricals, standardization for numerics
- **Notebook**: `03_data_preparation.ipynb`

### Phase 4: Modeling

- **Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Validation**: Stratified 5-fold cross-validation
- **Hyperparameters**: Grid search with compact parameter spaces
- **Best Model**: Gradient Boosting (84.1% CV accuracy)
- **Notebook**: `04_modeling.ipynb`

### Phase 5: Evaluation

- **Performance**: 84.1% CV accuracy ± 2.2%
- **Error Analysis**: Overfitting detection, feature importance
- **Robustness**: Stability tests, leakage checks
- **Notebook**: `05_evaluation.ipynb`

### Phase 6: Deployment

- **Submission**: Kaggle-ready CSV generation
- **Validation**: Format checks, reproducibility
- **Archiving**: Model artifacts, documentation
- **Notebook**: `06_deployment_submission.ipynb`

## 📈 Model Performance

| Model | CV Accuracy | CV Std | Training Accuracy | Notes |
|-------|-------------|---------|-------------------|--------|
| **Gradient Boosting** | **84.1%** | **±2.2%** | **93.9%** | **Final Model** |
| Random Forest | 82.7% | ±2.5% | 89.2% | Good baseline |
| Logistic Regression | 81.3% | ±1.8% | 82.1% | Most stable |
| Gender Baseline | 78.7% | - | - | Simple rule |

### 🎯 Feature Importance (Top 5)

1. **Sex_male** (15.4%) - Gender remains strongest predictor
2. **Title_Mr** (15.0%) - Social status indicator
3. **Age** (12.2%) - Age groups (children priority)
4. **FarePerPerson** (12.0%) - Economic status per family member
5. **FamilySize_Cat** (8.7%) - Optimal family size for survival

## 📋 Key Features Engineered

- **Title**: Extracted from Name (Mr, Miss, Mrs, Master, Rare)
- **FamilySize**: SibSp + Parch + 1
- **IsAlone**: Binary indicator for solo travelers
- **FarePerPerson**: Fare / FamilySize (economic status)
- **TicketGroupSize**: Passengers sharing tickets
- **Deck**: Cabin deck letter (A-G, Unknown)
- **CabinKnown**: Binary indicator for cabin information
- **AgeGroup**: Binned age categories
- **Sex_Pclass**: Interaction between gender and class

## 🔍 Key Insights

### Business Insights

- **"Women and children first"** policy clearly evident in survival patterns
- **Class matters**: First-class passengers had 63% survival rate vs 24% in third class
- **Family dynamics**: Optimal family size (2-4 members) had highest survival rates
- **Economic factors**: Higher fare per person correlates with survival

### Technical Insights

- **Feature engineering** drove most performance gains over baseline
- **Overfitting risk**: 9.9% train-CV gap suggests model complexity optimization needed
- **CV stability**: 2.2% standard deviation indicates reasonable but improvable robustness
- **Interpretability**: Tree-based models provide clear feature importance rankings

## 📊 Visualizations

The project includes comprehensive visualizations:

- Survival rates by demographic groups
- Feature correlation heatmaps  
- Model performance comparisons
- Feature importance plots
- Error analysis and confusion matrices

## 🔮 Future Improvements

### Model Enhancements

- **Ensemble methods**: Stacking multiple algorithms
- **Feature selection**: Recursive feature elimination
- **Hyperparameter optimization**: Bayesian optimization
- **Cross-validation**: GroupKFold by Ticket to reduce leakage

### Engineering Improvements

- **Pipeline automation**: End-to-end ML pipeline
- **Model monitoring**: Performance tracking over time
- **A/B testing**: Compare model versions
- **Production deployment**: API for real-time predictions

## 📝 Documentation

- **`planning.md`**: Comprehensive project roadmap and methodology
- **Jupyter Notebooks**: Step-by-step analysis with markdown explanations
- **Inline Comments**: Detailed code documentation
- **README.md**: Project overview and usage instructions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for hosting the Titanic competition and providing the dataset
- **CRISP-DM** methodology for providing a structured approach to data mining
- **Titanic Historical Society** for preserving the historical context
- **Open Source Community** for the excellent Python data science ecosystem

## 📞 Contact

**Elizabeth Parnell** - [@ejparnell](https://github.com/ejparnell)

Project Link: [https://github.com/ejparnell/titanic-ml](https://github.com/ejparnell/titanic-ml)

---

⭐ **Star this repository if you found it helpful!**

*This project demonstrates professional data science practices and serves as a portfolio piece showcasing end-to-end machine learning workflow implementation.*
