"""
Baseline Models Module

This module contains functions for baseline machine learning models:
Linear Regression and Logistic Regression.
"""

from sklearn.linear_model import LinearRegression, LogisticRegression
import time
from typing import Tuple
import pandas as pd
from .model_evaluation_5 import ModelEvaluation

class BaselineModels:
    """Class for baseline machine learning models: Linear Regression and Logistic Regression."""
    
    @staticmethod
    def run_regression_baseline(X_train: pd.DataFrame,
                                y_train: pd.Series,
                                X_test: pd.DataFrame,
                                y_test: pd.Series,
                                display: bool = True) -> Tuple[LinearRegression, pd.Series, pd.Series, pd.DataFrame]:
        """
        Run complete regression baseline: train Linear Regression, predict, and compare train/test results.
        
        Parameters:
        X_train: pandas DataFrame, training features
        y_train: pandas Series, training target
        X_test: pandas DataFrame, test features
        y_test: pandas Series, test target
        display: bool, whether to print information (default: True)
        
        Returns:
        model: Trained LinearRegression model
        y_train_pred: pandas Series, training predictions
        y_test_pred: pandas Series, test predictions
        comparison_df: DataFrame with train/test comparison
        """
        if display:
            print("=" * 80)
            print("BASELINE REGRESSION MODEL - LINEAR REGRESSION")
            print("=" * 80)
            print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        
        # Train model
        model = LinearRegression()
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        # Make predictions
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        
        if display:
            print("Model trained and predictions made.\n")
        
        # Compare train and test results
        comparison_df = ModelEvaluation.compare_train_test_regression(
            y_train, y_train_pred,
            y_test, y_test_pred,
            time_taken=time_taken,
            display=display
        )
        
        return model, y_train_pred, y_test_pred, comparison_df
    
    @staticmethod
    def run_classification_baseline(X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_test: pd.DataFrame,
                                   y_test: pd.Series,
                                   max_iter: int = 1000,
                                   display: bool = True) -> Tuple[LogisticRegression, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
        """
        Run complete classification baseline: train Logistic Regression, predict, and compare train/test results.
        
        Parameters:
        X_train: pandas DataFrame, training features
        y_train: pandas Series, training target
        X_test: pandas DataFrame, test features
        y_test: pandas Series, test target
        max_iter: int, maximum iterations for convergence (default: 1000)
        display: bool, whether to print information (default: True)
        
        Returns:
        model: Trained LogisticRegression model
        y_train_pred: pandas Series, training predictions
        y_test_pred: pandas Series, test predictions
        y_train_proba: pandas Series, training probabilities
        y_test_proba: pandas Series, test probabilities
        comparison_df: DataFrame with train/test comparison
        """
        if display:
            print("=" * 80)
            print("BASELINE CLASSIFICATION MODEL - LOGISTIC REGRESSION")
            print("=" * 80)
            print(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}, Max iterations: {max_iter}")
        
        # Train model
        model = LogisticRegression(max_iter=max_iter, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        
        # Make predictions
        y_train_pred = pd.Series(model.predict(X_train), index=X_train.index)
        y_test_pred = pd.Series(model.predict(X_test), index=X_test.index)
        y_train_proba = pd.Series(model.predict_proba(X_train)[:, 1], index=X_train.index)
        y_test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)
        
        if display:
            print("Model trained and predictions made.\n")
        
        # Compare train and test results
        comparison_df = ModelEvaluation.compare_train_test_classification(
            y_train, y_train_pred,
            y_test, y_test_pred,
            y_train_proba=y_train_proba,
            y_test_proba=y_test_proba,
            time_taken=time_taken,
            display=display
        )
        
        return model, y_train_pred, y_test_pred, y_train_proba, y_test_proba, comparison_df


