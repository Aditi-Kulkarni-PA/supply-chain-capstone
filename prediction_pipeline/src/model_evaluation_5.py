"""
Model Evaluation Module

This module contains functions for evaluating model performance for both
regression and classification tasks, including metrics, plots, and comparisons.
"""

# Import statements needed:
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
)
import pandas as pd
import numpy as np


class ModelEvaluation:
    """Class for evaluating model performance for both regression and classification tasks."""
    
    @staticmethod
    def evaluate_classification(y_true: pd.Series,
                                y_pred: pd.Series,
                                y_pred_proba: Optional[pd.Series] = None,
                                display: bool = True) -> Dict[str, float]:
        """
        Evaluate classification model performance.
        
        Parameters:
        y_true: pandas Series, true labels
        y_pred: pandas Series, predicted labels
        y_pred_proba: Optional pandas Series, predicted probabilities (for ROC-AUC)
        display: bool, whether to print metrics (default: True)
        
        Returns:
        metrics: Dictionary with classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = None
                if display:
                    print("Warning: ROC-AUC cannot be calculated (requires binary classification or probabilities)")
        else:
            metrics['roc_auc'] = None
        
        if display:
            print("=" * 80)
            print("CLASSIFICATION MODEL EVALUATION")
            print("=" * 80)
            print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc'] is not None:
                print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            print("=" * 80)
        
        return metrics
    
    @staticmethod
    def confusion_matrix_plot(y_true: pd.Series,
                              y_pred: pd.Series,
                              labels: Optional[list] = None,
                              figsize: Tuple[int, int] = (8, 6),
                              display: bool = True,
                              ax=None) -> np.ndarray:
        """
        Plot confusion matrix for classification.
        
        Parameters:
        y_true: pandas Series, true labels
        y_pred: pandas Series, predicted labels
        labels: Optional list, class labels (default: None, auto-detect)
        figsize: Tuple, figure size (default: (8, 6))
        display: bool, whether to display the plot (default: True)
        ax: matplotlib Axes, if provided plot on this axes (default: None)
        
        Returns:
        cm: numpy array, confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        if display:
            tick_labels = labels if labels else sorted(set(y_true) | set(y_pred))
            if ax is None:
                plt.figure(figsize=figsize)
                ax_plot = plt.gca()
            else:
                ax_plot = ax
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=tick_labels, yticklabels=tick_labels, ax=ax_plot)
            ax_plot.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
            ax_plot.set_ylabel('True Label', fontsize=11)
            ax_plot.set_xlabel('Predicted Label', fontsize=11)
            if ax is None:
                plt.tight_layout()
                plt.show()
        
        return cm
    
    @staticmethod
    def classification_report_detailed(y_true: pd.Series,
                                      y_pred: pd.Series,
                                      target_names: Optional[list] = None,
                                      display: bool = True) -> str:
        """
        Generate detailed classification report.
        
        Parameters:
        y_true: pandas Series, true labels
        y_pred: pandas Series, predicted labels
        target_names: Optional list, names of classes (default: None)
        display: bool, whether to print report (default: True)
        
        Returns:
        report: str, classification report
        """
        report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        
        if display:
            print("=" * 80)
            print("DETAILED CLASSIFICATION REPORT")
            print("=" * 80)
            print(report)
            print("=" * 80)
        
        return report
    
    @staticmethod
    def roc_curve_plot(y_true: pd.Series,
                       y_pred_proba: pd.Series,
                       figsize: Tuple[int, int] = (8, 6),
                       display: bool = True,
                       ax=None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Plot ROC curve for binary classification.
        
        Parameters:
        y_true: pandas Series, true labels (binary)
        y_pred_proba: pandas Series, predicted probabilities
        figsize: Tuple, figure size (default: (8, 6))
        display: bool, whether to display the plot (default: True)
        ax: matplotlib Axes, if provided plot on this axes (default: None)
        
        Returns:
        fpr: numpy array, false positive rates
        tpr: numpy array, true positive rates
        auc: float, area under the curve
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        if display:
            if ax is None:
                plt.figure(figsize=figsize)
                ax_plot = plt.gca()
            else:
                ax_plot = ax
            ax_plot.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
            ax_plot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            ax_plot.set_xlim([0.0, 1.0])
            ax_plot.set_ylim([0.0, 1.05])
            ax_plot.set_xlabel('False Positive Rate', fontsize=11)
            ax_plot.set_ylabel('True Positive Rate', fontsize=11)
            ax_plot.set_title('ROC Curve', fontsize=13, fontweight='bold')
            ax_plot.legend(loc="lower right", fontsize=10)
            ax_plot.grid(alpha=0.3)
            if ax is None:
                plt.tight_layout()
                plt.show()
        
        return fpr, tpr, auc
    
    @staticmethod
    def classification_plots(y_true: pd.Series,
                             y_pred: pd.Series,
                             y_pred_proba: pd.Series,
                             labels: Optional[list] = None,
                             figsize: Tuple[int, int] = (12, 4),
                             display: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Plot confusion matrix and ROC curve side by side as 2 subplots.
        
        Parameters:
        y_true: true labels
        y_pred: predicted labels
        y_pred_proba: predicted probabilities for positive class
        labels: class labels
        figsize: figure size for the combined plot
        display: whether to display
        
        Returns:
        cm, fpr, tpr, auc
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        if display:
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            # Confusion matrix
            tick_labels = labels if labels else sorted(set(y_true) | set(y_pred))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=tick_labels, yticklabels=tick_labels, ax=axes[0])
            axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
            axes[0].set_ylabel('True Label', fontsize=11)
            axes[0].set_xlabel('Predicted Label', fontsize=11)

            # ROC curve
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate', fontsize=11)
            axes[1].set_ylabel('True Positive Rate', fontsize=11)
            axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
            axes[1].legend(loc="lower right", fontsize=10)
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()

        return cm, fpr, tpr, auc
    
    @staticmethod
    def evaluate_regression(y_true: pd.Series,
                          y_pred: pd.Series,
                          display: bool = True) -> Dict[str, float]:
        """
        Evaluate regression model performance.
        
        Parameters:
        y_true: pandas Series, true values
        y_pred: pandas Series, predicted values
        display: bool, whether to print metrics (default: True)
        
        Returns:
        metrics: Dictionary with regression metrics
        """
        metrics = {}
        
        # Calculate metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error) - handle division by zero
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Manual calculation to handle zeros
            mask = y_true != 0
            if mask.sum() > 0:
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                metrics['mape'] = np.nan
        
        if display:
            print("=" * 80)
            print("REGRESSION MODEL EVALUATION")
            print("=" * 80)
            print(f"\nMAE  (Mean Absolute Error):      {metrics['mae']:.4f}")
            print(f"MSE  (Mean Squared Error):         {metrics['mse']:.4f}")
            print(f"RMSE (Root Mean Squared Error):   {metrics['rmse']:.4f}")
            print(f"R²   (R-squared):                  {metrics['r2']:.4f}")
            if not np.isnan(metrics['mape']):
                print(f"MAPE (Mean Absolute % Error):    {metrics['mape']:.2f}%")
            print("=" * 80)
        
        return metrics
    
    @staticmethod
    def regression_residuals_plot(y_train: pd.Series,
                                  y_train_pred: pd.Series,
                                  y_test: pd.Series,
                                  y_test_pred: pd.Series,
                                  figsize: Tuple[int, int] = (12, 5),
                                  display: bool = True) -> None:
        """
        Plot residual analysis for regression (train and test together).
        
        Parameters:
        y_train: pandas Series, true training values
        y_train_pred: pandas Series, predicted training values
        y_test: pandas Series, true test values
        y_test_pred: pandas Series, predicted test values
        figsize: Tuple, figure size (default: (12, 5))
        time_taken: Optional float, time taken to train model (default: None)
        display: bool, whether to display the plot (default: True)
        """
        # Calculate residuals (actual - prediction)
        residuals_train = y_train - y_train_pred
        residuals_test = y_test - y_test_pred
        
        if display:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Residuals vs Predicted
            sns.scatterplot(x=y_train_pred, y=residuals_train, ax=axes[0], label='Train', alpha=0.6)
            sns.scatterplot(x=y_test_pred, y=residuals_test, ax=axes[0], label='Test', alpha=0.6)
            axes[0].axhline(0, color='red', linestyle='--')
            axes[0].set_xlabel('Predicted Values', fontsize=12)
            axes[0].set_ylabel('Residuals', fontsize=12)
            axes[0].set_title('Residuals vs Predicted', fontsize=13, fontweight='bold')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # Residuals Distribution
            sns.histplot(residuals_train, kde=True, bins=30, ax=axes[1], color='blue', label='Train', alpha=0.5)
            sns.histplot(residuals_test, kde=True, bins=30, ax=axes[1], color='orange', label='Test', alpha=0.5)
            axes[1].set_xlabel('Residuals', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
            
            plt.tight_layout()
            plt.show()
    
    @staticmethod
    def regression_prediction_plot(y_train_true: pd.Series,
                                y_train_pred: pd.Series,
                                y_test_true: pd.Series,
                                y_test_pred: pd.Series,
                                figsize: Tuple[int, int] = (12, 5),
                                display: bool = True) -> None:
        """
        Plot predicted vs actual values for regression (train and test).
        
        Parameters:
        y_train_true: pandas Series, true training values
        y_train_pred: pandas Series, predicted training values
        y_test_true: pandas Series, true test values
        y_test_pred: pandas Series, predicted test values
        figsize: Tuple, figure size (default: (12, 5))
        display: bool, whether to display the plot (default: True)
        """
        if display:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Calculate common min/max for consistent axis ranges
            all_true = pd.concat([y_train_true, y_test_true])
            all_pred = pd.concat([y_train_pred, y_test_pred])
            min_val = min(all_true.min(), all_pred.min())
            max_val = max(all_true.max(), all_pred.max())
            
            # Train data plot
            axes[0].scatter(y_train_true, y_train_pred, alpha=0.5, color='blue', label='Train')
            axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            axes[0].set_xlabel('True Values', fontsize=12)
            axes[0].set_ylabel('Predicted Values', fontsize=12)
            axes[0].set_title('Train: Predicted vs Actual', fontsize=13, fontweight='bold')
            axes[0].legend(fontsize=11)
            axes[0].grid(alpha=0.3)
            
            # Test data plot
            axes[1].scatter(y_test_true, y_test_pred, alpha=0.5, color='orange', label='Test')
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            axes[1].set_xlabel('True Values', fontsize=12)
            axes[1].set_ylabel('Predicted Values', fontsize=12)
            axes[1].set_title('Test: Predicted vs Actual', fontsize=13, fontweight='bold')
            axes[1].legend(fontsize=11)
            axes[1].grid(alpha=0.3)
            
            plt.tight_layout()
            plt.show()
                
    @staticmethod
    def regression_overlay_plot(y_train_true: pd.Series,
                                y_train_pred: pd.Series,
                                y_test_true: pd.Series,
                                y_test_pred: pd.Series,
                                max_samples: int = 300,
                                figsize: Tuple[int, int] = (14, 5),
                                display: bool = True) -> None:
        """
        Overlay actual vs predicted values sorted by true value.
        Useful for tree-based models to spot stepped predictions and deviation patterns.
        
        Parameters:
        y_train_true: true training values
        y_train_pred: predicted training values
        y_test_true: true test values
        y_test_pred: predicted test values
        max_samples: max points to plot (subsampled for readability)
        figsize: figure size
        display: whether to display the plot
        """
        if not display:
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, y_true, y_pred, label in [
            (axes[0], y_train_true, y_train_pred, "Train"),
            (axes[1], y_test_true, y_test_pred, "Test"),
        ]:
            y_true = np.array(y_true).ravel()
            y_pred = np.array(y_pred).ravel()
            sort_idx = np.argsort(y_true)
            y_true_s = y_true[sort_idx]
            y_pred_s = y_pred[sort_idx]

            if len(y_true_s) > max_samples:
                step = len(y_true_s) // max_samples
                y_true_s = y_true_s[::step]
                y_pred_s = y_pred_s[::step]

            x = np.arange(len(y_true_s))
            ax.plot(x, y_true_s, color='steelblue', linewidth=1.2, label='Actual', alpha=0.9)
            ax.plot(x, y_pred_s, color='tomato', linewidth=1.2, label='Predicted', alpha=0.8)
            ax.fill_between(x, y_true_s, y_pred_s, color='gray', alpha=0.15)
            ax.set_xlabel('Samples (sorted by actual)', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.set_title(f'{label}: Actual vs Predicted', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_models_classification(results: Dict[str, Dict[str, float]],
                                     display: bool = True) -> pd.DataFrame:
        """
        Compare multiple classification models.
        
        Parameters:
        results: Dictionary with model names as keys and metrics dictionaries as values
        display: bool, whether to print comparison (default: True)
        
        Returns:
        comparison_df: DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results).T
        
        if display:
            print("=" * 80)
            print("MODEL COMPARISON - CLASSIFICATION")
            print("=" * 80)
            print(comparison_df.to_string())
            print("=" * 80)
        
        return comparison_df
    
    @staticmethod
    def compare_models_regression(results: Dict[str, Dict[str, float]],
                                 display: bool = True) -> pd.DataFrame:
        """
        Compare multiple regression models.
        
        Parameters:
        results: Dictionary with model names as keys and metrics dictionaries as values
        display: bool, whether to print comparison (default: True)
        
        Returns:
        comparison_df: DataFrame with model comparison
        """
        comparison_df = pd.DataFrame(results).T
        
        if display:
            print("=" * 80)
            print("MODEL COMPARISON - REGRESSION")
            print("=" * 80)
            print(comparison_df.to_string())
            print("=" * 80)
        
        return comparison_df

    @staticmethod
    def display_all_model_comparisons(comparison_dict: Dict[str, pd.DataFrame],
                                        display: bool = True) -> pd.DataFrame:
        """
        Display all model comparison DataFrames side by side in a formatted way.
        
        Parameters:
        comparison_dict: Dictionary with model names as keys and comparison DataFrames as values
        display: bool, whether to print comparison (default: True)
        
        Returns:
        combined_df: DataFrame with all models combined
        """
        import pandas as pd
        
        # Combine all comparison DataFrames
        combined_data = []
        
        for model_name, df in comparison_dict.items():
            # Add model name to each row
            df_copy = df.copy()
            df_copy.insert(0, "Model", model_name)
            combined_data.append(df_copy)
        
        # Concatenate all DataFrames
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        if display:
            print("=" * 120)
            print("ALL REGRESSION MODELS COMPARISON - SIDE BY SIDE")
            print("=" * 120)
            # Format the DataFrame for better display
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", None)
            print(combined_df.to_string(index=False))
            print("=" * 120)
        
        return combined_df

    @staticmethod
    def calculate_metrics_regression(y_true: pd.Series,
                                    y_pred: pd.Series) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate comprehensive regression metrics including variance and statistics.
        
        Parameters:
        y_true: pandas Series, true values
        y_pred: pandas Series, predicted values
        
        Returns:
        Tuple of (mse, variance, rmse, std_dev, mae, mean_val, mape, r2)
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE calculation with error handling
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Manual calculation to handle zeros
            mask = y_true != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
        
        r2 = r2_score(y_true, y_pred)
        
        # Convert y_true to a 1D array before passing to np functions
        y_true_array = np.ravel(y_true)
        var = np.var(y_true_array, ddof=1).item()
        std_dev = np.std(y_true_array, ddof=1).item()
        mean_val = np.mean(y_true_array).item()
        
        return mse, var, rmse, std_dev, mae, mean_val, mape, r2
    
    @staticmethod
    def compare_train_test_regression(y_train_true: pd.Series,
                                     y_train_pred: pd.Series,
                                     y_test_true: pd.Series,
                                     y_test_pred: pd.Series,
                                     time_taken: Optional[float] = None,
                                     display: bool = True) -> pd.DataFrame:
        """
        Compare train and test results for regression models.
        
        Parameters:
        y_train_true: pandas Series, true training values
        y_train_pred: pandas Series, predicted training values
        y_test_true: pandas Series, true test values
        y_test_pred: pandas Series, predicted test values
        display: bool, whether to print comparison (default: True)
        
        Returns:
        results_df: DataFrame comparing train and test metrics
        """
        # Calculate metrics for training set
        mse_train, var_train, rmse_train, std_dev_train, mae_train, mean_val_train, mape_train, r2_train = \
            ModelEvaluation.calculate_metrics_regression(y_train_true, y_train_pred)
        
        # Calculate metrics for test set
        mse_test, var_test, rmse_test, std_dev_test, mae_test, mean_val_test, mape_test, r2_test = \
            ModelEvaluation.calculate_metrics_regression(y_test_true, y_test_pred)
        
        # Create comparison DataFrame
        # Create comparison DataFrame
        results_data = {
            'Dataset': ['Training', 'Testing'],
            'MSE': [mse_train, mse_test],
            'Variance': [var_train, var_test],
            'RMSE': [rmse_train, rmse_test],
            'Std_Dev': [std_dev_train, std_dev_test],
            'MAE': [mae_train, mae_test],
            'Mean': [mean_val_train, mean_val_test],
            'MAPE': [mape_train, mape_test],
            'R²': [r2_train, r2_test]
        }
        
        # Add time_taken if provided
        if time_taken is not None:
            results_data['Time (Sec)'] = [time_taken, time_taken]
        
        results_df = pd.DataFrame(results_data)
        
        if display:
            print("=" * 80)
            print("TRAIN vs TEST COMPARISON - REGRESSION")
            print("=" * 80)
            print(results_df.to_string(index=False))
            print("=" * 80)
        
        return results_df
    
    @staticmethod
    def compare_train_test_classification(y_train_true: pd.Series,
                                         y_train_pred: pd.Series,
                                         y_test_true: pd.Series,
                                         y_test_pred: pd.Series,
                                         y_train_proba: Optional[pd.Series] = None,
                                         y_test_proba: Optional[pd.Series] = None,
                                         time_taken: Optional[float] = None,
                                         display: bool = True) -> pd.DataFrame:
        """
        Compare train and test results for classification models.
        
        Parameters:
        y_train_true: pandas Series, true training labels
        y_train_pred: pandas Series, predicted training labels
        y_test_true: pandas Series, true test labels
        y_test_pred: pandas Series, predicted test labels
        y_train_proba: Optional pandas Series, training predicted probabilities (for ROC-AUC)
        y_test_proba: Optional pandas Series, test predicted probabilities (for ROC-AUC)
        display: bool, whether to print comparison (default: True)
        
        Returns:
        results_df: DataFrame comparing train and test metrics
        """
        # Calculate metrics for training set
        train_metrics = ModelEvaluation.evaluate_classification(
            y_train_true, y_train_pred, y_train_proba, display=False
        )
        
        # Calculate metrics for test set
        test_metrics = ModelEvaluation.evaluate_classification(
            y_test_true, y_test_pred, y_test_proba, display=False
        )
        
        # Class 1 (Delayed) specific metrics
        train_p1 = precision_score(y_train_true, y_train_pred, pos_label=1, zero_division=0)
        train_r1 = recall_score(y_train_true, y_train_pred, pos_label=1, zero_division=0)
        train_f1 = f1_score(y_train_true, y_train_pred, pos_label=1, zero_division=0)
        test_p1 = precision_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
        test_r1 = recall_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)

        # Create comparison DataFrame — overall + class 1 columns
        results_data = {
            'Dataset': ['Training', 'Testing'],
            'Accuracy': [train_metrics['accuracy'], test_metrics['accuracy']],
            'Precision': [train_metrics['precision'], test_metrics['precision']],
            'Recall': [train_metrics['recall'], test_metrics['recall']],
            'F1-Score': [train_metrics['f1_score'], test_metrics['f1_score']],
            'Delayed Precision': [train_p1, test_p1],
            'Delayed Recall': [train_r1, test_r1],
            'Delayed F1': [train_f1, test_f1],
        }
        # Add time_taken if provided
        if time_taken is not None:
            results_data['Time (Sec)'] = [time_taken, time_taken]

        # Add ROC-AUC if available
        if train_metrics['roc_auc'] is not None and test_metrics['roc_auc'] is not None:
            results_data['ROC-AUC'] = [train_metrics['roc_auc'], test_metrics['roc_auc']]

        results_df = pd.DataFrame(results_data)

        if display:
            print("=" * 80)
            print("TRAIN vs TEST COMPARISON - CLASSIFICATION")
            print("=" * 80)
            print(results_df.to_string(index=False))
            print("=" * 80)

        return results_df



