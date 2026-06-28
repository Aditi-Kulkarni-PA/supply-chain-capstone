"""
Exploratory Data Analysis (EDA) Module

This module contains functions for exploratory data analysis operations including
visualizations, descriptive statistics, and outlier detection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple
from scipy import stats
from IPython.display import display as ipython_display
import logging
logger = logging.getLogger(__name__)
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 5)



# DataEDA Class
class DataEDA:
    """Class for exploratory data analysis operations."""
    
    @staticmethod
    def plot_numeric_features(df: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> None:
        """
        Plot numeric quantitative features with histplot, histplot (log scale), and boxplot.
        Uses viridis color palette - each feature gets a unique color from the palette.
        
        Parameters:
        df: pandas DataFrame
        numeric_columns: list of numeric column names (default: None - all numeric columns)
        """
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to only numeric columns
        numeric_columns = [col for col in numeric_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_columns) == 0:
            return
        
        # Generate unique colors from viridis palette for each feature
        # Each feature will get a different color based on its position
        n_features = len(numeric_columns)
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_features))
        
        for idx, col in enumerate(numeric_columns):
            if df[col].dtype in ['int64', 'float64']:
                # Each feature gets its unique color from viridis
                feature_color = viridis_colors[idx]
                
                fig, axes = plt.subplots(1, 3, figsize=(24, 7))  # Increased from (18, 5)
                fig.suptitle(f'Distribution Analysis: {col}', fontsize=18, fontweight='bold')
                
                # Histplot - use feature's unique color
                sns.histplot(data=df, x=col, kde=True, ax=axes[0], color=feature_color, alpha=0.7, edgecolor='white', linewidth=1.5)
                axes[0].set_title('Histogram', fontsize=14, fontweight='bold')
                axes[0].set_xlabel(col, fontsize=13)  # Increased from 11
                axes[0].set_ylabel('Frequency', fontsize=13)  # Increased from 11
                axes[0].tick_params(axis='both', labelsize=12)  # Increased tick label size
                axes[0].grid(axis='y', alpha=0.3, linestyle='--')
                
                # Histplot with log scale on y-axis - use same feature color
                sns.histplot(data=df, x=col, kde=True, ax=axes[1], color=feature_color, alpha=0.7, edgecolor='white', linewidth=1.5)
                axes[1].set_yscale('log')
                axes[1].set_title('Histogram (Log Y-scale)', fontsize=14, fontweight='bold')
                axes[1].set_xlabel(col, fontsize=13)  # Increased from 11
                axes[1].set_ylabel('Frequency (log scale)', fontsize=13)  # Increased from 11
                axes[1].tick_params(axis='both', labelsize=12)  # Increased tick label size
                axes[1].grid(axis='y', alpha=0.3, linestyle='--')
                
                # Boxplot - use same feature color
                box_plot = sns.boxplot(data=df, y=col, ax=axes[2], color=feature_color, width=0.6)
                # Customize boxplot colors
                for patch in box_plot.artists:
                    patch.set_facecolor(feature_color)
                    patch.set_alpha(0.8)
                axes[2].set_title('Boxplot', fontsize=14, fontweight='bold')
                axes[2].set_ylabel(col, fontsize=13)  # Increased from 11
                axes[2].tick_params(axis='both', labelsize=12)  # Increased tick label size
                axes[2].grid(axis='y', alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                plt.show()
    
    @staticmethod
    def plot_categorical_features(df: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> None:
        """
        Plot categorical features with top-10 value counts and cumulative percentage (dual y-axis).
        Uses magma color palette - each feature gets a unique color from the palette.
        Displays 2 features per row.
        
        Parameters:
        df: pandas DataFrame
        categorical_columns: list of categorical column names (default: None - all object columns)
        """
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Filter to only categorical columns that exist
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        if len(categorical_columns) == 0:
            return
        
        # Generate unique colors from magma palette for each feature
        n_features = len(categorical_columns)
        magma_colors = plt.cm.magma(np.linspace(0.2, 0.8, n_features))
        
        # Plot 2 features per row
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 8 * n_rows))  # Increased from (18, 6 * n_rows)
        fig.suptitle('Categorical Analysis (Top-10 with Cumulative %)', fontsize=18, fontweight='bold')
        
        # Flatten axes if needed
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, col in enumerate(categorical_columns):
            ax1 = axes[idx]
            feature_color = magma_colors[idx]
            bar_color = feature_color
            line_color = plt.cm.magma(min(0.95, np.linspace(0.2, 0.8, n_features)[idx] + 0.1))
            
            value_counts = df[col].value_counts().head(10)
            
            # Create a temporary dataframe for seaborn
            plot_df = pd.DataFrame({
                col: value_counts.index,
                'count': value_counts.values
            })
            
            # Calculate cumulative percentage
            cumulative_pct = (value_counts.cumsum() / value_counts.sum() * 100)
            
            # Bar plot for counts using seaborn (left y-axis) with feature's unique magma color
            bars = sns.barplot(data=plot_df, x=col, y='count', ax=ax1, color=bar_color, 
                              order=plot_df[col], alpha=0.85, edgecolor='white', linewidth=1.5)
            ax1.set_xlabel(col, fontsize=13, fontweight='bold')  # Increased from 11
            ax1.set_ylabel('Count', color=bar_color, fontsize=13, fontweight='bold')  # Increased from 11
            ax1.tick_params(axis='x', rotation=45, labelsize=12)  # Added labelsize
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)  # Added fontsize
            ax1.tick_params(axis='y', labelcolor=bar_color, labelsize=12)  # Added labelsize
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            ax1.set_title(f'{col}', fontsize=14, fontweight='bold')  # Increased from 12
            
            # Create second y-axis for cumulative percentage
            ax2 = ax1.twinx()
            
            # Line plot for cumulative percentage using seaborn (right y-axis) with feature's unique magma color
            x_positions = range(len(cumulative_pct))
            sns.lineplot(x=x_positions, y=cumulative_pct.values, 
                        ax=ax2, color=line_color, marker='o', linewidth=2, markersize=8, 
                        label='Cumulative %', zorder=5, markerfacecolor=line_color, 
                        markeredgecolor='white', markeredgewidth=1.5)
            ax2.set_ylabel('Cumulative Percentage (%)', color=line_color, fontsize=13, fontweight='bold')  # Increased from 10
            ax2.tick_params(axis='y', labelcolor=line_color, labelsize=12)  # Added labelsize
            ax2.set_ylim(0, 100)
            ax2.set_xlim(-0.5, len(cumulative_pct) - 0.5)  # Match bar plot x-axis range
            ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)  # Increased from 9
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_slack_time_correlation(df: pd.DataFrame, 
                                     y_column: str,
                                     numeric_columns: Optional[List[str]] = None,
                                     filter_condition: Optional[str] = None,
                                     figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot a correlation heatmap of numeric columns in the DataFrame.
        
        Parameters:
        df: pandas DataFrame
        y_column: str, name of the target column (used for filter defaults and title)
        numeric_columns: list of numeric column names to include. If None, uses all numeric columns.
        filter_condition: str, pandas query string to filter data before computing (default: None)
                         If None and y_column is 'slack_time', defaults to filtering negative slack_time only
        figsize: Optional Tuple, figure size (default: auto-sized based on number of columns)
        """
        if y_column not in df.columns:
            print(f"'{y_column}' column not found in dataframe.")
            return
        
        # Apply filter condition
        df_filtered = df.copy()
        if filter_condition is not None:
            try:
                df_filtered = df.query(filter_condition)
                print(f"Applied filter condition: {filter_condition}")
                print(f"Rows before filter: {len(df)}, Rows after filter: {len(df_filtered)}")
            except Exception as e:
                print(f"Error applying filter condition '{filter_condition}': {e}")
                print("Using unfiltered data.")
                df_filtered = df
        elif y_column == 'slack_time':
            df_filtered = df[df[y_column] < 0]
            print(f"Default filter applied: {y_column} < 0 (only negative values)")
            print(f"Rows before filter: {len(df)}, Rows after filter: {len(df_filtered)}")
        
        if len(df_filtered) == 0:
            print(f"No data remaining after filter. Cannot create correlation chart.")
            return
        
        # Use specified numeric_columns if provided, otherwise auto-detect
        # Exclude binary flags (only 2 unique values like 0/1) — they clutter the heatmap
        if numeric_columns is not None:
            candidates = [col for col in numeric_columns if col in df_filtered.columns]
        else:
            candidates = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        all_numeric = [col for col in candidates if df_filtered[col].nunique() > 2]
        
        if len(all_numeric) <= 1:
            print("Not enough numeric columns for a correlation chart.")
            return
        
        # Place target column first so it sits on the top-left diagonal
        if y_column in all_numeric:
            all_numeric.remove(y_column)
            all_numeric.insert(0, y_column)

        corr_matrix = df_filtered[all_numeric].corr()

        # Build title with filter info
        if filter_condition is not None:
            title = f'Correlation Matrix — All Numeric Features (Filtered: {filter_condition})'
        elif y_column == 'slack_time' and len(df_filtered) < len(df):
            title = f'Correlation Matrix — All Numeric Features (Filtered: {y_column} < 0)'
        else:
            title = f'Correlation Matrix — All Numeric Features'

        n = len(all_numeric)
        if figsize is None:
            side = max(7, min(12, n + 2))
            figsize = (side + 1, side)  # extra width for colorbar

        font_size = 10 if n <= 8 else 9 if n <= 12 else 8

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax,
                    annot_kws={'fontsize': font_size},
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size)
        plt.subplots_adjust(bottom=0.22, left=0.18, right=0.95, top=0.92)
        plt.show()
    
    @staticmethod
    def plot_correlation_to_target(df: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_column: str = 'delay_hours',
                                   title: Optional[str] = None,
                                   figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Horizontal bar chart showing each feature's Pearson correlation with the target.
        Bars are sorted by absolute correlation and colour-coded (positive=steelblue, negative=salmon).

        Parameters:
        df: pandas DataFrame
        feature_columns: columns to correlate against the target
        target_column: the target column (default: 'delay_hours')
        title: plot title (default: auto-generated)
        figsize: figure size (default: auto-sized)
        """
        cols = [c for c in feature_columns if c in df.columns and c != target_column]
        if not cols or target_column not in df.columns:
            print("Required columns missing — skipping correlation-to-target plot.")
            return

        corrs = df[cols].corrwith(df[target_column]).dropna().sort_values(key=abs, ascending=True)

        if figsize is None:
            figsize = (8, max(3, len(corrs) * 0.45))
        if title is None:
            title = f'Feature Correlation with {target_column}'

        colours = ['steelblue' if v >= 0 else 'salmon' for v in corrs]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.barh(corrs.index, corrs.values, color=colours, edgecolor='black', height=0.6)
        ax.set_xlabel(f'Pearson r  (with {target_column})')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axvline(0, color='grey', linewidth=0.8)

        for bar, val in zip(bars, corrs.values):
            offset = 0.01 if val >= 0 else -0.01
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}', va='center', ha=ha, fontsize=10)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_delay_rate_by_feature(df: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_column: str = 'delayed',
                                   figsize_per_col: int = 6) -> None:
        """
        Bar chart showing delay rate (%) at each level of the given features.

        Parameters:
        df: pandas DataFrame
        feature_columns: list of column names to plot (one subplot each)
        target_column: binary target column (default: 'delayed')
        figsize_per_col: width per subplot in inches (default: 6)
        """
        import math
        existing = [c for c in feature_columns if c in df.columns]
        if not existing or target_column not in df.columns:
            print("Required columns missing — skipping delay-rate plot.")
            return

        max_bins = 15
        ncols = min(3, len(existing))
        nrows = math.ceil(len(existing) / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(figsize_per_col * ncols, 5 * nrows))
        axes = np.array(axes).flatten()

        for ax, col in zip(axes, existing):
            series = df[col]
            if series.nunique() > max_bins:
                series = pd.cut(series, bins=max_bins)
            grouped = df.groupby(series)[target_column].mean() * 100
            grouped.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Delay Rate by {col}', fontsize=13, fontweight='bold')
            ax.set_ylabel('Delay Rate (%)')
            ax.set_xlabel(col)
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(grouped):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

        for ax in axes[len(existing):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_feature_correlation(df: pd.DataFrame,
                                 feature_columns: List[str],
                                 extra_columns: Optional[List[str]] = None,
                                 title: str = 'Feature Correlation Matrix',
                                 figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Heatmap of correlations among the given features (and optional extras).

        Parameters:
        df: pandas DataFrame
        feature_columns: primary columns of interest
        extra_columns: additional columns to include (e.g. targets, raw features)
        title: plot title
        figsize: figure size (default: auto-sized)
        """
        cols = list(feature_columns)
        if extra_columns:
            cols += [c for c in extra_columns if c not in cols]
        cols = [c for c in cols if c in df.columns]

        if len(cols) < 2:
            print("Not enough columns for a correlation matrix.")
            return

        corr_matrix = df[cols].corr()
        n = len(cols)
        if figsize is None:
            side = max(7, min(12, n + 2))
            figsize = (side + 1, side)
        font_size = 10 if n <= 8 else 9 if n <= 12 else 8

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax,
                    annot_kws={'fontsize': font_size},
                    cbar_kws={'shrink': 0.8})
        ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=font_size)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=font_size)
        plt.subplots_adjust(bottom=0.22, left=0.18, right=0.95, top=0.92)
        plt.show()

    @staticmethod
    def show_describe(df: pd.DataFrame, display: bool = True) -> None:
        """
        Display descriptive statistics for the DataFrame (transposed for better readability).
        Formats numeric statistics to 2 decimal places.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to display information (default: True)
        """
        from IPython.display import display as ipython_display
        
        describe_df = df.describe().T
        
        # Format numeric statistics to 2 decimal places
        stats_to_format = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']
        for stat in stats_to_format:
            if stat in describe_df.columns:
                describe_df[stat] = describe_df[stat].apply(lambda x: round(x, 2) if pd.notna(x) else x)
        
        if display:
            print("Descriptive Statistics (Transposed):")
            ipython_display(describe_df)
    
    @staticmethod
    def detect_outliers_comprehensive(df: pd.DataFrame, columns: Optional[List[str]] = None, 
                                     zscore_threshold: float = 3.0,
                                     display: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Detect outliers using both IQR and z-score methods and print comprehensive summary.
        
        Parameters:
        df: pandas DataFrame
        columns: list of specific columns to check (default: None - all numerical columns)
        zscore_threshold: float, threshold for z-score method (default: 3.0)
        display: bool, whether to print the results (default: True)
        
        Returns:
        outliers_info: dict with outlier information for each column and method
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_info = {}
        
        if display:
            print("Comprehensive Outlier Detection Summary:")
        
        for col in columns:
            if df[col].dtype in ['int64', 'float64']:
                # IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound_iqr = Q1 - 1.5 * IQR
                upper_bound_iqr = Q3 + 1.5 * IQR
                outliers_iqr = df[(df[col] < lower_bound_iqr) | (df[col] > upper_bound_iqr)]
                
                # Z-score method
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_zscore = df[z_scores > zscore_threshold]
                
                outliers_info[col] = {
                    'iqr': {
                        'count': len(outliers_iqr),
                        'percentage': (len(outliers_iqr) / len(df)) * 100,
                        'lower_bound': lower_bound_iqr,
                        'upper_bound': upper_bound_iqr,
                        'outlier_indices': outliers_iqr.index.tolist()
                    },
                    'zscore': {
                        'count': len(outliers_zscore),
                        'percentage': (len(outliers_zscore) / len(df)) * 100,
                        'outlier_indices': outliers_zscore.index.tolist()
                    }
                }
                
                if display:
                    print(f"\n{col}:")
                    print(f"  IQR Method:")
                    print(f"    Outliers: {outliers_info[col]['iqr']['count']} ({outliers_info[col]['iqr']['percentage']:.2f}%)")
                    print(f"    Bounds: [{lower_bound_iqr:.2f}, {upper_bound_iqr:.2f}]")
                    print(f"  Z-Score Method:")
                    print(f"    Outliers: {outliers_info[col]['zscore']['count']} ({outliers_info[col]['zscore']['percentage']:.2f}%)")
        
        return outliers_info
    
    @staticmethod
    def is_skewed(series: pd.Series, threshold: float = 0.5) -> bool:
        """
        Check if a series is skewed using skewness coefficient.
        
        Parameters:
        series: pandas Series
        threshold: absolute skewness threshold (default: 0.5)
        
        Returns:
        bool: True if skewed, False otherwise
        """
        skewness = stats.skew(series.dropna())
        return abs(skewness) > threshold
    
    @staticmethod
    def handle_outliers_smart(df: pd.DataFrame, columns: List[str], 
                              zscore_threshold: float = 3.0,
                              display: bool = True) -> pd.DataFrame:
        """
        Handle outliers intelligently: use IQR for skewed data, z-score for normal data.
        
        Parameters:
        df: pandas DataFrame
        columns: list of specific columns to handle (required - only these columns will be processed)
        zscore_threshold: float, threshold for z-score method (default: 3.0)
        display: bool, whether to print the handling strategy (default: True)
        
        Returns:
        df: DataFrame with outliers removed
        """
        df_cleaned = df.copy()
        
        if columns is None or len(columns) == 0:
            if display:
                print("No columns specified for outlier removal.")
            return df_cleaned
        
        # Filter to only columns that exist in dataframe
        columns = [col for col in columns if col in df_cleaned.columns]
        
        if display:
            print("Smart Outlier Handling:")
        
        for col in columns:
            if df_cleaned[col].dtype in ['int64', 'float64']:
                initial_count = len(df_cleaned)
                
                # Check if column has zero or near-zero variance (constant or near-constant values)
                col_std = df_cleaned[col].std()
                if col_std == 0 or pd.isna(col_std) or col_std < 1e-10:
                    # Skip outlier handling for constant columns (no outliers possible)
                    if display:
                        print(f"\n{col}:")
                        print(f"  Skipped: Constant or near-constant column (std={col_std:.2e})")
                        print(f"  Rows removed: 0 (0.00%)")
                    continue
                
                # Check if data is skewed
                skewed = DataEDA.is_skewed(df_cleaned[col])
                
                if skewed:
                    # Use IQR method for skewed data
                    Q1 = df_cleaned[col].quantile(0.25)
                    Q3 = df_cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Skip if IQR is 0 (constant values in quartiles)
                    if IQR == 0 or pd.isna(IQR):
                        if display:
                            print(f"\n{col}:")
                            print(f"  Skipped: Constant values in quartiles (IQR={IQR:.2e})")
                            print(f"  Rows removed: 0 (0.00%)")
                        continue
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
                    removed_count = initial_count - len(df_cleaned)
                    method_used = 'IQR'
                    
                else:
                    # Use z-score method for normal data
                    z_scores = np.abs((df_cleaned[col] - df_cleaned[col].mean()) / col_std)
                    # Keep rows where z_score is valid (not NaN) and <= threshold
                    # If z_score is NaN, keep the row (shouldn't happen if std check passed, but safety check)
                    valid_mask = (z_scores <= zscore_threshold) | pd.isna(z_scores)
                    df_cleaned = df_cleaned[valid_mask]
                    removed_count = initial_count - len(df_cleaned)
                    method_used = 'Z-Score'
                
                if display:
                    print(f"\n{col}:")
                    print(f"  Skewed: {skewed}")
                    print(f"  Method used: {method_used}")
                    print(f"  Rows removed: {removed_count} ({removed_count/initial_count*100:.2f}%)")
        
        if display:
            print(f"\nFinal dataset shape: {df_cleaned.shape}")
        
        return df_cleaned
    
    @staticmethod
    def pre_outlier_removal(df: pd.DataFrame,
                            numeric_columns: Optional[List[str]] = None,
                            categorical_columns: Optional[List[str]] = None,
                            zscore_threshold: float = 3.0,
                            display: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        EDA pipeline part 1: Pre-outlier removal analysis.
        Includes: descriptive statistics, plots, and outlier detection.
        
        Parameters:
        df: pandas DataFrame
        numeric_columns: list of numeric column names (default: None - all numeric columns)
        categorical_columns: list of categorical column names (default: None - all object columns)
        zscore_threshold: float, threshold for z-score method (default: 3.0)
        display: bool, whether to display progress (default: True)
        
        Returns:
        df: DataFrame (unchanged)
        outlier_summary: dict with outlier detection summary
        """
        if display:
            print("=" * 80)
            print("EDA PIPELINE - PART 1: PRE-OUTLIER REMOVAL")
            print("=" * 80)
        
        logger.info("Starting pre-outlier removal EDA")
        
        # Step 1: Show descriptive statistics
        if display:
            print("\nSTEP 1: DESCRIPTIVE STATISTICS")
            print("=" * 80)
        logger.info("Displaying descriptive statistics")
        DataEDA.show_describe(df, display=display)
        
        # Step 2: Plot numeric features (before outlier removal)
        if display:
            print("\nSTEP 2: PLOTTING NUMERIC FEATURES (BEFORE OUTLIER REMOVAL)")
            print("=" * 80)
        logger.info("Plotting numeric features before outlier removal")
        DataEDA.plot_numeric_features(df, numeric_columns)
        
        # Step 3: Plot categorical features (before outlier removal)
        if display:
            print("\nSTEP 3: PLOTTING CATEGORICAL FEATURES (BEFORE OUTLIER REMOVAL)")
            print("=" * 80)
        logger.info("Plotting categorical features before outlier removal")
        DataEDA.plot_categorical_features(df, categorical_columns)
        
        # Step 4: Detect outliers using both methods
        if display:
            print("\nSTEP 4: DETECTING OUTLIERS")
            print("=" * 80)
        logger.info("Detecting outliers using IQR and z-score methods")
        outlier_summary = DataEDA.detect_outliers_comprehensive(df, 
                                                               columns=numeric_columns,
                                                               zscore_threshold=zscore_threshold,
                                                               display=display)
        
        if display:
            print("\n" + "=" * 80)
            print("PRE-OUTLIER REMOVAL COMPLETED!")
            print("=" * 80)
        
        return df, outlier_summary
    
    @staticmethod
    def outlier_removal(df: pd.DataFrame,
                       columns_for_outlier_removal: List[str],
                       zscore_threshold: float = 3.0,
                       display: bool = True) -> pd.DataFrame:
        """
        EDA pipeline part 2: Outlier removal.
        Removes outliers from specified columns.
        
        Parameters:
        df: pandas DataFrame
        columns_for_outlier_removal: list of column names to apply outlier removal to
        zscore_threshold: float, threshold for z-score method (default: 3.0)
        display: bool, whether to display progress (default: True)
        
        Returns:
        df_cleaned: DataFrame with outliers removed
        """
        if display:
            print("=" * 80)
            print("EDA PIPELINE - PART 2: OUTLIER REMOVAL")
            print("=" * 80)
        
        logger.info("Handling outliers intelligently")
        
        if display:
            print(f"\nColumns for outlier removal: {columns_for_outlier_removal}")
            print("=" * 80)
        
        df_cleaned = DataEDA.handle_outliers_smart(df, 
                                                   columns=columns_for_outlier_removal,
                                                   zscore_threshold=zscore_threshold,
                                                   display=display)
        
        if display:
            print("\n" + "=" * 80)
            print("OUTLIER REMOVAL COMPLETED!")
            print(f"Rows before: {len(df)}, Rows after: {len(df_cleaned)}")
            print("=" * 80)
        
        return df_cleaned
    
    @staticmethod
    def post_outlier_removal(df: pd.DataFrame,
                             numeric_columns: Optional[List[str]] = None,
                             categorical_columns: Optional[List[str]] = None,
                             y_column: Optional[str] = None,
                             filter_condition: Optional[str] = None,
                             display: bool = True) -> pd.DataFrame:
        """
        EDA pipeline part 3: Post-outlier removal analysis.
        Includes: plots, descriptive statistics, and correlation plots.
        
        Parameters:
        df: pandas DataFrame
        numeric_columns: list of numeric column names (default: None - all numeric columns)
        categorical_columns: list of categorical column names (default: None - all object columns)
        y_column: str, name of the column to plot on Y-axis for correlation plots (default: None)
        filter_condition: str, pandas query string to filter data for correlation plots (default: None)
                         If None and y_column is 'slack_time', defaults to filtering negative slack_time only
        display: bool, whether to display progress (default: True)
        
        Returns:
        df: DataFrame (unchanged)
        """
        if display:
            print("=" * 80)
            print("EDA PIPELINE - PART 3: POST-OUTLIER REMOVAL")
            print("=" * 80)
        
        logger.info("Starting post-outlier removal EDA")
        
        # Step 1: Plot numeric features again (after outlier removal)
        if display:
            print("\nSTEP 1: PLOTTING NUMERIC FEATURES (AFTER OUTLIER REMOVAL)")
            print("=" * 80)
        logger.info("Plotting numeric features after outlier removal")
        DataEDA.plot_numeric_features(df, numeric_columns)
        
        # Step 2: Plot categorical features again (after outlier removal)
        if display:
            print("\nSTEP 2: PLOTTING CATEGORICAL FEATURES (AFTER OUTLIER REMOVAL)")
            print("=" * 80)
        logger.info("Plotting categorical features after outlier removal")
        DataEDA.plot_categorical_features(df, categorical_columns)
        
        # Step 3: Show descriptive statistics again (after outlier removal)
        if display:
            print("\nSTEP 3: DESCRIPTIVE STATISTICS (AFTER OUTLIER REMOVAL)")
            print("=" * 80)
        logger.info("Displaying descriptive statistics after outlier removal")
        DataEDA.show_describe(df, display=display)
        
        # Step 4: Plot y_column correlation with numerical features
        if y_column is not None and y_column in df.columns:
            if display:
                print(f"\nSTEP 4: {y_column.upper()} CORRELATION WITH NUMERICAL FEATURES")
                print("=" * 80)
            logger.info(f"Plotting {y_column} correlation with numerical features")
            DataEDA.plot_slack_time_correlation(df, y_column, numeric_columns, filter_condition=filter_condition)
        else:
            if display:
                if y_column is None:
                    print("\nSTEP 4: SKIPPING CORRELATION PLOTS (y_column not specified)")
                else:
                    print(f"\nSTEP 4: SKIPPING CORRELATION PLOTS ({y_column} column not found)")
                print("=" * 80)
            logger.info("Skipping correlation plots")
        
        if display:
            print("\n" + "=" * 80)
            print("POST-OUTLIER REMOVAL COMPLETED!")
            print("=" * 80)
        
        return df
    
    @staticmethod
    def eda_pipeline(df: pd.DataFrame, 
                     numeric_columns: Optional[List[str]] = None,
                     categorical_columns: Optional[List[str]] = None,
                     columns_for_outlier_removal: Optional[List[str]] = None,
                     y_column: Optional[str] = None,
                     handle_outliers: bool = True,
                     zscore_threshold: float = 3.0,
                     display: bool = True) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
        """
        Complete EDA pipeline: calls pre_outlier_removal, outlier_removal, and post_outlier_removal.
        
        Parameters:
        df: pandas DataFrame
        numeric_columns: list of numeric column names (default: None - all numeric columns)
        categorical_columns: list of categorical column names (default: None - all object columns)
        columns_for_outlier_removal: list of column names to apply outlier removal to (default: None - no outlier removal)
        y_column: str, name of the column to plot on Y-axis for correlation plots (default: None)
        handle_outliers: bool, whether to handle outliers (default: True)
        zscore_threshold: float, threshold for z-score method (default: 3.0)
        display: bool, whether to display progress (default: True)
        
        Returns:
        df_cleaned: DataFrame with outliers removed (if handle_outliers=True and columns_for_outlier_removal provided)
        outlier_summary: dict with outlier detection summary
        """
        logger.info("Starting EDA Pipeline...")
        df_cleaned = df.copy()
        
        if display:
            print("=" * 80)
            print("EDA PIPELINE (COMPLETE)")
            print("=" * 80)
        
        # Part 1: Pre-outlier removal
        df_cleaned, outlier_summary = DataEDA.pre_outlier_removal(
            df_cleaned,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            zscore_threshold=zscore_threshold,
            display=display
        )
        
        # Part 2: Outlier removal
        if handle_outliers and columns_for_outlier_removal is not None and len(columns_for_outlier_removal) > 0:
            df_cleaned = DataEDA.outlier_removal(
                df_cleaned,
                columns_for_outlier_removal=columns_for_outlier_removal,
                zscore_threshold=zscore_threshold,
                display=display
            )
        else:
            if display:
                if not handle_outliers:
                    print("\n" + "=" * 80)
                    print("SKIPPING OUTLIER REMOVAL (handle_outliers=False)")
                    print("=" * 80)
                elif columns_for_outlier_removal is None or len(columns_for_outlier_removal) == 0:
                    print("\n" + "=" * 80)
                    print("SKIPPING OUTLIER REMOVAL (no columns specified)")
                    print("=" * 80)
            logger.info("Skipping outlier removal")
        
        # Part 3: Post-outlier removal
        df_cleaned = DataEDA.post_outlier_removal(
            df_cleaned,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            y_column=y_column,
            display=display
        )
        
        if display:
            print("\n" + "=" * 80)
            print("EDA PIPELINE COMPLETED!")
            print("=" * 80)
        logger.info("EDA Pipeline completed successfully")
        
        return df_cleaned, outlier_summary

    

