# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project focused on depression prediction using various classification algorithms. The project implements a comprehensive analysis pipeline with hyperparameter optimization, data balancing, and model interpretability features.

## Project Structure

```
machine-learning/
├── src/
│   ├── models/
│   │   └── depression_classification.py    # Main classification pipeline
│   └── utils/
│       └── preprocessing.py                # Data preprocessing utilities
├── notebooks/
│   └── exploratory_analysis.ipynb         # Jupyter notebook for EDA
├── requirements.txt                        # Python dependencies
└── README.md                              # Project documentation (Turkish)
```

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For Jupyter notebooks
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Running Analysis
```bash
# Run the main depression classification analysis
python src/models/depression_classification.py

# The script will prompt for data file path, typically:
# src/data/depresyon.xlsx
```

## Core Architecture

### Main Analysis Pipeline (`depression_classification.py`)
- **DepressionClassifier class**: Main orchestrator for the entire ML pipeline
- **Data Loading**: Handles Excel file input with error handling
- **Preprocessing**: Missing data handling, SMOTEENN balancing, categorical encoding
- **Model Training**: Supports Logistic Regression, Gradient Boosting, and SVM
- **Hyperparameter Optimization**: Uses RandomizedSearchCV with cross-validation
- **Evaluation**: Comprehensive metrics including accuracy, ROC-AUC, F1, PPV, NPV, Brier score
- **Interpretability**: SHAP analysis for feature importance
- **Output Generation**: Saves models, visualizations, and results to `outputs/models/randomized_search/`

### Preprocessing Utilities (`preprocessing.py`)
- Modular functions for data preprocessing tasks
- Missing data analysis and filling strategies
- Class imbalance detection and SMOTEENN balancing
- Categorical feature encoding (label and one-hot)
- Complete preprocessing pipeline function

### Key Features
- **Multi-model comparison**: Automatically trains and compares multiple algorithms
- **Data balancing**: Addresses class imbalance using SMOTEENN technique
- **Cross-validation**: 5-fold stratified cross-validation for robust evaluation
- **Hyperparameter tuning**: RandomizedSearchCV for optimization
- **Model interpretability**: SHAP analysis for feature importance
- **Comprehensive reporting**: Automated generation of results and visualizations

## Data Requirements

- Input: Excel (.xlsx) format
- Target column must be named "Depression"
- Supports both numerical and categorical features
- Automatic handling of missing values

## Output Structure

Results are saved to `outputs/models/randomized_search/`:
- `model_comparison.png`: Performance comparison visualization
- `shap_summary.png`: SHAP feature importance plots
- `shap_importance_bar.png`: SHAP bar chart
- `model_results.txt`: Detailed results and feature importance rankings

## Development Notes

- Project uses Turkish language in comments and documentation
- Implements defensive machine learning practices with comprehensive error handling
- Follows scikit-learn conventions and patterns
- Uses randomized search for efficient hyperparameter optimization
- SHAP analysis currently supports tree-based models (Gradient Boosting)