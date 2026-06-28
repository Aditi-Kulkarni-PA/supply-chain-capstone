"""
Classification Models Module

This module contains functions for various classification models with
hyperparameter optimization and evaluation.
"""


import pandas as pd
import numpy as np
import time
from typing import List, Optional, Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from .model_evaluation_5 import ModelEvaluation


class ClassificationModels:
    """Class for various classification models with hyperparameter optimization and evaluation."""

    @staticmethod
    def _get_column_types(X: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Helper method to identify numeric and categorical columns.
        
        Parameters:
        X: DataFrame to analyze
        exclude_cols: List of columns to exclude from analysis
        
        Returns:
        numeric_cols: List of numeric column names
        categorical_cols: List of categorical column names
        """
        if exclude_cols is None:
            exclude_cols = []
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove excluded columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        return numeric_cols, categorical_cols

    @staticmethod
    def _plot_feature_importances(feature_names, importances, model_name: str, top_n: int = 15):
        """Plot horizontal bar chart of top feature importances."""
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(data=feature_imp.head(top_n), x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances — {model_name}', fontsize=13, fontweight='bold')
        plt.xlabel('Importance', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def logistic_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Logistic Regression with regularization and hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
                 Options: 'accuracy', 'recall', 'f1', 'f1_macro', 'roc_auc', etc.
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("LOGISTIC REGRESSION")
            print("=" * 80)

        if param_grid is None:
            param_grid = {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l1", "l2", "elasticnet"],
                "solver": ["lbfgs", "liblinear", "saga"],
            }

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (scaling + encoding for linear models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        # Update param_grid keys if they don't have 'lr__' prefix
        if param_grid and not any(k.startswith('lr__') for k in param_grid.keys()):
            param_grid = {f'lr__{k}': v for k, v in param_grid.items()}
        
        # Handle solver compatibility with penalty
        if 'lr__penalty' in param_grid and 'lr__solver' in param_grid:
            penalty_values = param_grid['lr__penalty']
            if any(p in ['l1', 'elasticnet'] for p in penalty_values):
                compatible_solvers = ['saga', 'liblinear']
                param_grid['lr__solver'] = [s for s in param_grid['lr__solver'] if s in compatible_solvers]
                if not param_grid['lr__solver']:
                    param_grid['lr__solver'] = ['saga']
        
        # Handle f1_macro scoring (needs make_scorer)
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def decision_tree_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Decision Tree Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("DECISION TREE CLASSIFIER")
            print("=" * 80)

        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
                "criterion": ["gini", "entropy"],
            }

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('dt', DecisionTreeClassifier(random_state=42))
        ])
        
        # Update param_grid keys if they don't have 'dt__' prefix
        if param_grid and not any(k.startswith('dt__') for k in param_grid.keys()):
            param_grid = {f'dt__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        best_dt = model.best_estimator_.named_steps['dt']
        try:
            feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_dt.feature_importances_))]
        metrics_dict["feature_importances"] = dict(zip(feature_names, best_dt.feature_importances_))

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ClassificationModels._plot_feature_importances(
                feature_names, best_dt.feature_importances_, 'Decision Tree')
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def random_forest_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Random Forest Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("RANDOM FOREST CLASSIFIER")
            print("=" * 80)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "criterion": ["gini", "entropy"],
            }

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        # Update param_grid keys if they don't have 'rf__' prefix
        if param_grid and not any(k.startswith('rf__') for k in param_grid.keys()):
            param_grid = {f'rf__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        # Extract feature importances
        best_rf = model.best_estimator_.named_steps['rf']
        try:
            feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_rf.feature_importances_))]
        metrics_dict["feature_importances"] = dict(zip(feature_names, best_rf.feature_importances_))

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ClassificationModels._plot_feature_importances(
                feature_names, best_rf.feature_importances_, 'Random Forest')
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def adaboost_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        AdaBoost Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("ADABOOST CLASSIFIER")
            print("=" * 80)

        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
            }

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ab', AdaBoostClassifier(random_state=42))
        ])
        
        # Update param_grid keys if they don't have 'ab__' prefix
        if param_grid and not any(k.startswith('ab__') for k in param_grid.keys()):
            param_grid = {f'ab__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        best_ab = model.best_estimator_.named_steps['ab']
        try:
            feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_ab.feature_importances_))]
        metrics_dict["feature_importances"] = dict(zip(feature_names, best_ab.feature_importances_))

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ClassificationModels._plot_feature_importances(
                feature_names, best_ab.feature_importances_, 'AdaBoost')
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def xgboost_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        XGBoost Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("XGBOOST CLASSIFIER")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('xgb', xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss'))
        ])
        
        # Update param_grid keys if they don't have 'xgb__' prefix
        if param_grid and not any(k.startswith('xgb__') for k in param_grid.keys()):
            param_grid = {f'xgb__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        best_xgb = model.best_estimator_.named_steps['xgb']
        try:
            feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_xgb.feature_importances_))]
        metrics_dict["feature_importances"] = dict(zip(feature_names, best_xgb.feature_importances_))

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ClassificationModels._plot_feature_importances(
                feature_names, best_xgb.feature_importances_, 'XGBoost')
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def lightgbm_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        LightGBM Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("LIGHTGBM CLASSIFIER")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10, -1],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "num_leaves": [31, 50, 100, 200],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_samples": [20, 30, 50],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('lgb', lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1, force_col_wise=True))
        ])
        
        # Update param_grid keys if they don't have 'lgb__' prefix
        if param_grid and not any(k.startswith('lgb__') for k in param_grid.keys()):
            param_grid = {f'lgb__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        best_lgb = model.best_estimator_.named_steps['lgb']
        try:
            feature_names = model.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_lgb.feature_importances_))]
        metrics_dict["feature_importances"] = dict(zip(feature_names, best_lgb.feature_importances_))

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ClassificationModels._plot_feature_importances(
                feature_names, best_lgb.feature_importances_, 'LightGBM')
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def svm_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        SVM Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("SVM CLASSIFIER")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "kernel": ["rbf", "linear", "poly"],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (scaling + encoding)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('svm', SVC(random_state=42, probability=True))
        ])
        
        # Update param_grid keys if they don't have 'svm__' prefix
        if param_grid and not any(k.startswith('svm__') for k in param_grid.keys()):
            param_grid = {f'svm__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def naive_bayes_classification(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        scoring: str = 'accuracy',
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Naive Bayes Classification with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        scoring: Scoring metric for GridSearchCV (default: 'accuracy')
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("NAIVE BAYES CLASSIFIER")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = ClassificationModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (scaling + encoding)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('nb', GaussianNB())
        ])
        
        # Update param_grid keys if they don't have 'nb__' prefix
        if param_grid and not any(k.startswith('nb__') for k in param_grid.keys()):
            param_grid = {f'nb__{k}': v for k, v in param_grid.items()}
        
        # Handle f1_macro scoring
        if scoring == 'f1_macro':
            scoring_func = make_scorer(f1_score, average='macro')
        else:
            scoring_func = scoring
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring=scoring_func,
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get probabilities for ROC-AUC
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred, y_test, y_test_pred,
            y_train_proba=y_train_proba, y_test_proba=y_test_proba,
            time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": model.best_score_,
            "train_accuracy": round(accuracy_score(y_train, y_train_pred), 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "train_precision": round(precision_score(y_train, y_train_pred, zero_division=0), 4),
            "test_precision": round(precision_score(y_test, y_test_pred, zero_division=0), 4),
            "train_recall": round(recall_score(y_train, y_train_pred, zero_division=0), 4),
            "test_recall": round(recall_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1": round(f1_score(y_train, y_train_pred, zero_division=0), 4),
            "test_f1": round(f1_score(y_test, y_test_pred, zero_division=0), 4),
            "train_f1_macro": round(f1_score(y_train, y_train_pred, average='macro', zero_division=0), 4),
            "test_f1_macro": round(f1_score(y_test, y_test_pred, average='macro', zero_division=0), 4),
            "train_roc_auc": round(roc_auc_score(y_train, y_train_proba), 4),
            "test_roc_auc": round(roc_auc_score(y_test, y_test_proba), 4),
            "time_taken": round(time_taken, 4),
        }

        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ModelEvaluation.classification_plots(y_test, y_test_pred, y_test_proba, display=display)

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict
