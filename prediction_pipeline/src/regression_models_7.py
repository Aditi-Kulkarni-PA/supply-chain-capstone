"""
Regression Models Module

This module contains functions for various regression models with
hyperparameter optimization and evaluation.
"""


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Any
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
from .model_evaluation_5 import ModelEvaluation


class RegressionModels:
    """Class for various regression models with hyperparameter optimization and evaluation."""

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
    def ridge_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        alphas: List[float] = [0.1, 1.0, 10.0, 100.0, 1000.0],
        cv: int = 5,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:

        if display:
            print("=" * 80)
            print("RIDGE REGRESSION")
            print("=" * 80)

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
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
            ('ridge', Ridge())
        ])
        
        # Use GridSearchCV for hyperparameter tuning
        param_grid = {'ridge__alpha': alphas}
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_alpha": model.best_params_['ridge__alpha'],
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
        }

        if display:
            print(f"\nBest alpha: {model.best_params_['ridge__alpha']}")
            ModelEvaluation.regression_prediction_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_residuals_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def lasso_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        alphas: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0],
        cv: int = 5,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:

        if display:
            print("=" * 80)
            print("LASSO REGRESSION")
            print("=" * 80)

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
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
            ('lasso', Lasso(random_state=42, max_iter=2000))
        ])
        
        # Use GridSearchCV for hyperparameter tuning
        param_grid = {'lasso__alpha': alphas}
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)

        # Get number of features selected from the best estimator
        best_estimator = model.best_estimator_.named_steps['lasso']
        n_features_selected = np.sum(best_estimator.coef_ != 0)

        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_alpha": model.best_params_['lasso__alpha'],
            "n_features_selected": n_features_selected,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
        }

        if display:
            print(f"\nBest alpha: {model.best_params_['lasso__alpha']}")
            print(
                f"Features selected: {metrics_dict['n_features_selected']} "
                f"out of {len(best_estimator.coef_)}"
            )
            ModelEvaluation.regression_prediction_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_residuals_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def decision_tree_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:

        if display:
            print("=" * 80)
            print("DECISION TREE REGRESSOR")
            print("=" * 80)

        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8],
            }

        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Numeric columns pass through unchanged
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('dt', DecisionTreeRegressor(random_state=42))
        ])
        
        # Update param_grid keys if they don't have 'dt__' prefix
        if param_grid and not any(k.startswith('dt__') for k in param_grid.keys()):
            param_grid = {f'dt__{k}': v for k, v in param_grid.items()}

        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time

        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)

        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )

        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
        }

        if display:
            print(f"\nBest parameters: {model.best_params_}")

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def adaboost_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        AdaBoost Regression with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        from sklearn.ensemble import AdaBoostRegressor
        
        if display:
            print("=" * 80)
            print("ADABOOST REGRESSOR")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.5, 1.0],
                "loss": ["linear", "square", "exponential"],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Numeric columns pass through unchanged
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ab', AdaBoostRegressor(random_state=42))
        ])
        
        # Update param_grid keys if they don't have 'ab__' prefix
        if param_grid and not any(k.startswith('ab__') for k in param_grid.keys()):
            param_grid = {f'ab__{k}': v for k, v in param_grid.items()}
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get feature importances
        best_ab = model.best_estimator_.named_steps['ab']
        preprocessor_fitted = model.best_estimator_.named_steps['preprocessor']
        
        try:
            feature_names = preprocessor_fitted.get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_ab.feature_importances_))]
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
            "feature_importances": dict(zip(feature_names, best_ab.feature_importances_))
        }
        
        if display:
            print(f"\nBest parameters: {model.best_params_}")
            
            # Plot feature importances
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': best_ab.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(15), x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importances - AdaBoost', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def xgboost_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        XGBoost Regression with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        import xgboost as xgb
        
        if display:
            print("=" * 80)
            print("XGBOOST REGRESSOR")
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
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Numeric columns pass through unchanged
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('xgb', xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))
        ])
        
        # Update param_grid keys if they don't have 'xgb__' prefix
        if param_grid and not any(k.startswith('xgb__') for k in param_grid.keys()):
            param_grid = {f'xgb__{k}': v for k, v in param_grid.items()}
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get feature importances
        best_xgb = model.best_estimator_.named_steps['xgb']
        preprocessor_fitted = model.best_estimator_.named_steps['preprocessor']
        
        try:
            feature_names = preprocessor_fitted.get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_xgb.feature_importances_))]
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
            "feature_importances": dict(zip(feature_names, best_xgb.feature_importances_))
        }
        
        if display:
            print(f"\nBest parameters: {model.best_params_}")
            
            # Plot feature importances
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': best_xgb.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(15), x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importances - XGBoost', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def lightgbm_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        LightGBM Regression with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        import lightgbm as lgb
        
        if display:
            print("=" * 80)
            print("LIGHTGBM REGRESSOR")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [3, 5, 7, 10, -1],
                "learning_rate": [0.01, 0.1, 0.2, 0.3],
                "num_leaves": [31, 50, 100, 200],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_samples": [20, 30, 50],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Numeric columns pass through unchanged
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('lgb', lgb.LGBMRegressor(
                random_state=42, 
                n_jobs=-1, 
                verbosity=-1,
                force_col_wise=True
            ))
        ])
        
        # Update param_grid keys if they don't have 'lgb__' prefix
        if param_grid and not any(k.startswith('lgb__') for k in param_grid.keys()):
            param_grid = {f'lgb__{k}': v for k, v in param_grid.items()}
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get feature importances
        best_lgb = model.best_estimator_.named_steps['lgb']
        preprocessor_fitted = model.best_estimator_.named_steps['preprocessor']
        
        try:
            feature_names = preprocessor_fitted.get_feature_names_out()
        except AttributeError:
            feature_names = [f'feature_{i}' for i in range(len(best_lgb.feature_importances_))]
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
            "feature_importances": dict(zip(feature_names, best_lgb.feature_importances_))
        }
        
        if display:
            print(f"\nBest parameters: {model.best_params_}")
            
            # Plot feature importances
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': best_lgb.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(15), x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importances - LightGBM', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def random_forest_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Random Forest Regression with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("RANDOM FOREST REGRESSOR")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
        # Create preprocessing pipeline (encoding only, no scaling for tree-based models)
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'  # Numeric columns pass through unchanged
        )
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('rf', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        
        # Update param_grid keys if they don't have 'rf__' prefix
        if param_grid and not any(k.startswith('rf__') for k in param_grid.keys()):
            param_grid = {f'rf__{k}': v for k, v in param_grid.items()}
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        # Get feature importances - need to get feature names from preprocessor
        best_rf = model.best_estimator_.named_steps['rf']
        preprocessor_fitted = model.best_estimator_.named_steps['preprocessor']
        
        # Get feature names after preprocessing
        try:
            feature_names = preprocessor_fitted.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            feature_names = [f'feature_{i}' for i in range(len(best_rf.feature_importances_))]
        
        feature_importances_dict = dict(zip(feature_names, best_rf.feature_importances_))
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
            "feature_importances": feature_importances_dict
        }
        
        if display:
            print(f"\nBest parameters: {model.best_params_}")
            
            # Plot feature importances
            feature_imp = pd.DataFrame({
                'feature': feature_names,
                'importance': best_rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_imp.head(15), x='importance', y='feature', palette='viridis')
            plt.title('Top 15 Feature Importances - Random Forest', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def svm_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        param_grid: Optional[Dict] = None,
        cv: int = 3,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Support Vector Machine Regression with hyperparameter optimization.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        param_grid: Optional parameter grid for GridSearchCV
        cv: Number of cross-validation folds
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("SVM REGRESSOR")
            print("=" * 80)
        
        if param_grid is None:
            param_grid = {
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
                "epsilon": [0.01, 0.1, 0.5, 1.0],
                "kernel": ["rbf", "linear", "poly"],
            }
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
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
            ('svm', SVR())
        ])
        
        # Update param_grid keys if they don't have 'svm__' prefix
        if param_grid and not any(k.startswith('svm__') for k in param_grid.keys()):
            param_grid = {f'svm__{k}': v for k, v in param_grid.items()}
        
        model = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "best_params": model.best_params_,
            "best_score": -model.best_score_,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
        }
        
        if display:
            print(f"\nBest parameters: {model.best_params_}")
            ModelEvaluation.regression_prediction_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_residuals_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict

    @staticmethod
    def naive_bayes_regression(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_bins: int = 10,
        display: bool = True
    ) -> Tuple[Any, pd.Series, pd.Series, pd.DataFrame, Dict]:
        """
        Naive Bayes for Regression using binning approach.
        
        Note: Naive Bayes is inherently a classification algorithm. This function
        converts the regression problem to classification by binning the target variable.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        n_bins: Number of bins for target variable discretization
        display: Whether to display results
        
        Returns:
        model, y_train_pred, y_test_pred, comparison_df, metrics_dict
        """
        if display:
            print("=" * 80)
            print("NAIVE BAYES REGRESSOR (using binning approach)")
            print("=" * 80)
        
        # Bin the target variable for classification
        from sklearn.preprocessing import KBinsDiscretizer
        
        binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_train_binned = binner.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_test_binned = binner.transform(y_test.values.reshape(-1, 1)).ravel()
        
        # Convert to integers for Naive Bayes
        y_train_binned = y_train_binned.astype(int)
        y_test_binned = y_test_binned.astype(int)
        
        # Identify column types
        exclude_cols = ['delayed', 'delay_hours', 'slack_time']
        numeric_cols, categorical_cols = RegressionModels._get_column_types(X_train, exclude_cols)
        
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
        
        # Build Naive Bayes model
        model = pipeline
        
        start_time = time.time()
        model.fit(X_train, y_train_binned)
        time_taken = time.time() - start_time
        
        # Get predictions (class probabilities)
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
        
        # Convert probabilities back to continuous values using bin centers
        bin_centers = binner.bin_edges_[0][:-1] + (binner.bin_edges_[0][1:] - binner.bin_edges_[0][:-1]) / 2
        
        # Weighted average of bin centers using probabilities
        y_train_pred = pd.Series(
            np.dot(y_train_proba, bin_centers),
            index=X_train.index
        )
        y_test_pred = pd.Series(
            np.dot(y_test_proba, bin_centers),
            index=X_test.index
        )
        
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred, y_test, y_test_pred, time_taken=time_taken, display=display
        )
        
        metrics_dict = {
            "n_bins": n_bins,
            "train_r2": r2_score(y_train, y_train_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "time_taken": time_taken,
        }
        
        if display:
            print(f"\nNumber of bins used: {n_bins}")
            print("Note: Naive Bayes is a classification algorithm. This implementation")
            print("uses binning to convert regression to classification.")
            ModelEvaluation.regression_prediction_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_residuals_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )

            ModelEvaluation.regression_overlay_plot(
                y_train, y_train_pred, y_test, y_test_pred, display=display
            )
        
        # Store binner and model for potential future use
        model.binner = binner
        model.bin_centers = bin_centers
        
        return model, y_train_pred, y_test_pred, comparison_df, metrics_dict    

