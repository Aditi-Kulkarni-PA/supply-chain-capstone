"""
Hyperparameter configurations for regression and classification models.

This file contains all hyperparameter grids and settings for model tuning.
Import this file and use the parameter dictionaries in your model training code.
"""
# ============================================================================
# MODEL SELECTION FLAGS
# ============================================================================
# Set to True to run the model, False to skip it
# This allows you to selectively run only the models you want to test

# Regression Models Selection
run_regression_models = {
    "linear": True,          # Baseline Linear Regression
    "ridge": True,           # Ridge Regression
    "lasso": True,           # Lasso Regression
    "decision_tree": True,   # Decision Tree Regressor
    "random_forest": True,   # Random Forest Regressor
    "adaboost": True,        # AdaBoost Regressor
    "xgboost": True,        # XGBoost Regressor
    "lightgbm": True,       # LightGBM Regressor
    "svm": False,            # SVM Regressor
    "naive_bayes": True,    # Naive Bayes Regressor (using binning)
}

# Classification Models Selection
run_classification_models = {
    "logistic_baseline": True,    # Baseline Logistic Regression (no regularization)
    "logistic_regression": True,  # Logistic Regression with regularization
    "decision_tree": True,        # Decision Tree Classifier
    "random_forest": True,       # Random Forest Classifier
    "adaboost": True,             # AdaBoost Classifier
    "xgboost": True,              # XGBoost Classifier
    "lightgbm": True,            # LightGBM Classifier
    "svm": False,                  # SVM Classifier
    "naive_bayes": True,          # Naive Bayes Classifier
}

# ============================================================================
# RIDGE REGRESSION
# ============================================================================
param_ridge = {
    "alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "cv": 5
}

# ============================================================================
# LASSO REGRESSION
# ============================================================================
param_lasso = {
    "alphas": [0.001, 0.01, 0.1, 1.0, 10.0],
    "cv": 5
}

# ============================================================================
# DECISION TREE REGRESSOR
# ============================================================================
param_decision_tree = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
}

# ============================================================================
# RANDOM FOREST REGRESSOR
# ============================================================================
param_random_forest = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

# ============================================================================
# ADABOOST REGRESSOR
# ============================================================================
# Full grid: 48 combinations (~12 seconds) - Already fast!
param_adaboost = {
    "n_estimators": [50, 100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
    "loss": ["linear", "square", "exponential"],
}

# Fast grid: 12 combinations (~3-5 seconds) - Optional, for even faster testing
param_adaboost_fast = {
    "n_estimators": [100, 200],
    "learning_rate": [0.1, 0.5],
    "loss": ["linear", "square"],
}

# ============================================================================
# XGBOOST REGRESSOR
# ============================================================================
# Full grid: 1,728 combinations (~5-10 minutes)
param_xgboost = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

# Fast grid: 54 combinations (~30-60 seconds) - Use this for quicker results
param_xgboost_fast = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1, 3],
}

# ============================================================================
# LIGHTGBM REGRESSOR
# ============================================================================
# Full grid: 8,640 combinations (~20-25 minutes with early stopping) - Still slow, use fast version
param_lightgbm = {
    "n_estimators": [50, 100, 200, 300],  # Reduced: early stopping will optimize
    "max_depth": [3, 5, 7, 10, -1],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "num_leaves": [31, 50, 100, 200],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_samples": [20, 30, 50],
}

# Fast grid: 432 combinations (~1-2 minutes with early stopping) - RECOMMENDED
param_lightgbm_fast = {
    "n_estimators": [50, 100, 150],  # Reduced: early stopping will optimize
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.2],
    "num_leaves": [31, 50, 100],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_samples": [20, 30],
}

# Ultra-fast grid: 64 combinations (~20-30 seconds with early stopping) - Use this if fast is still too slow
param_lightgbm_ultra_fast = {
    "n_estimators": [50, 100],  # Reduced: early stopping will optimize
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.2],
    "num_leaves": [31, 50],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_samples": [20],
}

# ============================================================================
# SVM REGRESSOR
# ============================================================================
# Full grid: 360 combinations (~1-2 minutes)
param_svm = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1.0],
    "kernel": ["rbf", "linear", "poly"],
}

# Fast grid: 48 combinations (~12-15 seconds) - Use this for quicker results
param_svm_fast = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.01, 0.1],
    "epsilon": [0.1, 0.5],
    "kernel": ["rbf", "linear"],
}

# ============================================================================
# NAIVE BAYES REGRESSOR (using binning)
# ============================================================================
param_naive_bayes = {
    "n_bins": 10,  # Number of bins for target discretization
}

# ============================================================================
# CROSS-VALIDATION STRATEGIES
# ============================================================================
from sklearn.model_selection import KFold, StratifiedKFold

# For REGRESSION: Use KFold (standard for continuous targets)
cv_regression = KFold(n_splits=3, shuffle=True, random_state=42)

# For CLASSIFICATION: Use StratifiedKFold (maintains class distribution)
cv_classification = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Alternative: Simple integer (defaults to KFold for regression)
cv_regression_simple = 3  # Just number of folds
cv_classification_simple = 3  # Just number of folds

# ============================================================================
# COMMON HYPERPARAMETERS
# ============================================================================
# Common settings for GridSearchCV
common_search_params = {
    "cv": cv_regression,  # Use KFold for regression (can also use integer like 3)
    "scoring": "neg_mean_squared_error",
    "n_jobs": -1,
}

# Common settings for Classification GridSearchCV
# "recall" maximises recall for the positive class (delayed=1) by default — catches the most delays
# Other options: "accuracy", "f1", "f1_macro", "roc_auc", "precision"
common_search_params_classification = {
    "cv": cv_classification,  # Use StratifiedKFold for classification
    "scoring": "recall",  # Maximize delayed detection — missing a delay is costlier than a false alarm
    "n_jobs": -1,
}



# ============================================================================
# LOGISTIC REGRESSION (with regularization)
# ============================================================================
param_logistic_regression = {
    "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l1", "l2", "elasticnet"],
    "solver": ["lbfgs", "liblinear", "saga"],
}

# Fast grid: 18 combinations (~5-10 seconds)
param_logistic_regression_fast = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "lbfgs"],
}

# ============================================================================
# DECISION TREE CLASSIFIER
# ============================================================================
param_decision_tree_classifier = {
    "max_depth": [3, 5, 7, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
    "criterion": ["gini", "entropy"],
}

# ============================================================================
# RANDOM FOREST CLASSIFIER
# ============================================================================
# Full grid: 567 combinations — balanced weighting always on for delay detection
param_random_forest_classifier = {
    "n_estimators": [150, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [3, 5, 7],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced"],  # Always balanced — delayed class is the priority, "None" will remove minority class upweighting and reduce recall 
}

# Fast grid: 24 combinations (~30-60 seconds) — tuned around best params
param_random_forest_classifier_fast = {
    "n_estimators": [150, 200, 300],
    "max_depth": [None],
    "min_samples_split": [3, 5, 7],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "class_weight": ["balanced"],
}

# ============================================================================
# ADABOOST CLASSIFIER
# ============================================================================
# Full grid: 48 combinations (~12 seconds)
param_adaboost_classifier = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.5, 1.0],
}

# Fast grid: 4 combinations (~3-5 seconds)
param_adaboost_classifier_fast = {
    "n_estimators": [100, 200],
    "learning_rate": [0.1, 0.5],
}

# ============================================================================
# XGBOOST CLASSIFIER
# ============================================================================
# Full grid: 1,728 combinations (~5-10 minutes)
param_xgboost_classifier = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}

# Fast grid: 54 combinations (~30-60 seconds)
param_xgboost_classifier_fast = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.2],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_weight": [1, 3],
}

# ============================================================================
# LIGHTGBM CLASSIFIER
# ============================================================================
# Full grid: 8,640 combinations (~20-25 minutes) - Use fast version
param_lightgbm_classifier = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7, 10, -1],
    "learning_rate": [0.01, 0.1, 0.2, 0.3],
    "num_leaves": [31, 50, 100, 200],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_samples": [20, 30, 50],
}

# Fast grid: 432 combinations (~1-2 minutes) - RECOMMENDED
param_lightgbm_classifier_fast = {
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.2],
    "num_leaves": [31, 50, 100],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_samples": [20, 30],
}

# Ultra-fast grid: 64 combinations (~20-30 seconds)
param_lightgbm_classifier_ultra_fast = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.2],
    "num_leaves": [31, 50],
    "subsample": [0.8, 0.9],
    "colsample_bytree": [0.8, 0.9],
    "min_child_samples": [20],
}

# ============================================================================
# SVM CLASSIFIER
# ============================================================================
# Full grid: 360 combinations (~1-2 minutes)
param_svm_classifier = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1, 1],
    "kernel": ["rbf", "linear"],
}

# Fast grid: 48 combinations (~12-15 seconds)
param_svm_classifier_fast = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1],
    "kernel": ["rbf", "linear"],
}

# ============================================================================
# NAIVE BAYES CLASSIFIER
# ============================================================================
param_naive_bayes_classifier = {
    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
}

# For models that use GridSearchCV, you can combine param_grid with common_search_params
# Example: {**param_decision_tree, **common_search_params} won't work directly
# Instead, use param_grid=param_decision_tree and pass other params separately

