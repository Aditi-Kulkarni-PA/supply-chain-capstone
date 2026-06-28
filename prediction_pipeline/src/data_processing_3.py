"""
Data Processing Module

This module contains functions for data cleaning and preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional

# Perform normalization, check missing values, standardize column names
from typing import Optional, List, Tuple, Dict, Any


class DataProcessing:
    """Class for data cleaning and preprocessing operations."""
    
    @staticmethod
    def standardize_index(df: pd.DataFrame, display: bool = True) -> pd.DataFrame:
        """
        Standardize the DataFrame index.
        Checks if index is monotonically increasing, starts with 0, and is unique.
        If any condition fails, resets the index.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to print information about index standardization (default: True)
        
        Returns:
        df: DataFrame with standardized index
        """
        df_cleaned = df.copy()
        
        # Check conditions
        is_monotonic = df_cleaned.index.is_monotonic_increasing
        starts_with_zero = len(df_cleaned) > 0 and df_cleaned.index[0] == 0
        is_unique = df_cleaned.index.is_unique
        
        # Determine if reset is needed
        needs_reset = not (is_monotonic and starts_with_zero and is_unique)
        
        if needs_reset:
            initial_index_info = f"Range: [{df_cleaned.index.min()}, {df_cleaned.index.max()}], Unique: {is_unique}, Monotonic: {is_monotonic}"
            df_cleaned = df_cleaned.reset_index(drop=True)
            if display:
                print(f"  Index reset: {initial_index_info} -> Reset to range [0, {len(df_cleaned)-1}]")
        else:
            if display:
                print(f"  Index is already standardized (range: [0, {len(df_cleaned)-1}], unique: True, monotonic: True)")
        
        return df_cleaned
    
    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase with underscores.
        
        Parameters:
        df: pandas DataFrame
        
        Returns:
        df: DataFrame with standardized column names
        """
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        return df


    @staticmethod
    def drop_unwanted_columns(df: pd.DataFrame, columns_to_drop: Optional[List[str]] = None, 
                              display: bool = True) -> pd.DataFrame:
        """
        Drop unwanted columns from the dataset.
        
        Parameters:
        df: pandas DataFrame
        columns_to_drop: list of column names to drop (default: None - no columns dropped)
        display: bool, whether to print information about dropped columns (default: True)
        
        Returns:
        df: DataFrame with unwanted columns removed
        """
        if columns_to_drop is None:
            columns_to_drop = []
        
        if len(columns_to_drop) == 0:
            if display:
                print("No columns specified for dropping.")
            return df
        
        # Check which columns actually exist
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        missing_cols = [col for col in columns_to_drop if col not in df.columns]
        
        if len(missing_cols) > 0 and display:
            print(f"Warning: The following columns were not found in the dataset: {missing_cols}")
        
        if len(existing_cols) > 0:
            initial_shape = df.shape
            df_cleaned = df.drop(columns=existing_cols)
            final_shape = df_cleaned.shape
            
            if display:
                print(f"\nDropped {len(existing_cols)} column(s): {existing_cols}")
                print(f"Dataset shape changed from {initial_shape} to {final_shape}")
            
            return df_cleaned
        
        return df



    @staticmethod
    def check_missing_values(df: pd.DataFrame, display: bool = True) -> pd.DataFrame:
        """
        Check for missing values in the dataset.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to print the results (default: True)
        
        Returns:
        missing_info: DataFrame with missing value information
        """
        missing_info = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
            'Data_Type': df.dtypes
        })
        missing_info = missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if display:
            if len(missing_info) == 0:
                print("✓ No missing values found in the dataset!")
            else:
                print("Missing Values Summary:")
                print(missing_info.to_string(index=False))
                print(f"\nTotal missing values: {df.isnull().sum().sum()}")
        
        return missing_info


    @staticmethod
    def handle_missing_values(df: pd.DataFrame, missing_info: pd.DataFrame, 
                              cat_features: Optional[List[str]] = None, 
                              num_quant_median: Optional[List[str]] = None, 
                              num_quant_mean: Optional[List[str]] = None, 
                              num_quant_mode: Optional[List[str]] = None, 
                              target_variable: str = 'delayed', 
                              display: bool = True, 
                              manual_analysis_action: str = 'drop',
                              allow_row_drop: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Handle missing values in the dataset based on feature types and percentage thresholds.
        
        Rules by feature type:
        
        For cat_features:
        - <= 5%: drop rows (or impute with mode if allow_row_drop=False)
        - 5% < missing <= 20%: impute with mode
        - 20% < missing <= 40%: manual analysis required
        - > 40%: drop column
        
        For num_quant_median:
        - <= 5%: drop rows (or impute with median if allow_row_drop=False)
        - 5% < missing <= 20%: impute with median
        - 20% < missing <= 40%: manual analysis required
        - > 40%: drop column
        
        For num_quant_mean:
        - <= 5%: drop rows (or impute with mean if allow_row_drop=False)
        - 5% < missing <= 20%: impute with mean
        - 20% < missing <= 40%: manual analysis required
        - > 40%: drop column
        
        For num_quant_mode:
        - <= 5%: drop rows (or impute with mode if allow_row_drop=False)
        - 5% < missing <= 20%: impute with mode
        - 20% < missing <= 40%: manual analysis required
        - > 40%: drop column
        
        Parameters:
        df: pandas DataFrame
        missing_info: DataFrame from check_missing_values function with missing value percentages
        cat_features: list of categorical feature names
        num_quant_median: list of numerical features to impute with median
        num_quant_mean: list of numerical features to impute with mean
        num_quant_mode: list of numerical features to impute with mode
        target_variable: str, name of the target variable to exclude from imputation
        display: bool, whether to print the handling strategy for each column
        manual_analysis_action: str, deprecated - not used (columns with 20-40% missing require manual intervention)
        allow_row_drop: bool, if False impute instead of dropping rows for <=5% missing (default: True)
        
        Returns:
        df_cleaned: DataFrame with handled missing values
        summary_df: DataFrame with feature information and applied strategies
        """
        df_cleaned = df.copy()
        
        # Initialize feature type lists if None
        if cat_features is None:
            cat_features = []
        if num_quant_median is None:
            num_quant_median = []
        if num_quant_mean is None:
            num_quant_mean = []
        if num_quant_mode is None:
            num_quant_mode = []
        
        if len(missing_info) == 0:
            if display:
                print("No missing values to handle.")
            # Return empty summary
            summary_df = pd.DataFrame(columns=['Feature', 'Dtype', 'Feature_Type', 'Missing_Count', 
                                              'Missing_Percentage', 'Applied_Strategy'])
            return df_cleaned, summary_df
        
        if display:
            print("\nHandling missing values based on feature types and percentage thresholds:")
        
        # Track summary information
        summary_data = []
        manual_analysis_cols = []
        dropped_columns_list = []
        
        for idx, row in missing_info.iterrows():
            col = row['Column']
            missing_pct = row['Missing_Percentage']
            missing_count = row['Missing_Count']
            dtype = str(row['Data_Type'])
            
            # Skip target variable
            if col == target_variable:
                if display:
                    print(f"  {col}: Skipped (target variable)")
                summary_data.append({
                    'Feature': col,
                    'Dtype': dtype,
                    'Feature_Type': 'target',
                    'Missing_Count': missing_count,
                    'Missing_Percentage': round(missing_pct, 2),
                    'Applied_Strategy': 'skipped'
                })
                continue
            
            # Determine feature type
            if col in cat_features:
                feature_type = 'cat_features'
            elif col in num_quant_median:
                feature_type = 'num_quant_median'
            elif col in num_quant_mean:
                feature_type = 'num_quant_mean'
            elif col in num_quant_mode:
                feature_type = 'num_quant_mode'
            else:
                feature_type = 'unknown'
            
            # Apply strategy based on missing percentage
            if missing_pct <= 5:
                if allow_row_drop:
                    # Drop rows
                    initial_shape = df_cleaned.shape[0]
                    df_cleaned = df_cleaned.dropna(subset=[col])
                    dropped_rows = initial_shape - df_cleaned.shape[0]
                    strategy = 'drop_rows'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Dropped {dropped_rows} rows")
                else:
                    # Impute instead of dropping (same strategy as 5-20% bracket)
                    if feature_type == 'cat_features':
                        mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                        df_cleaned[col].fillna(mode_value, inplace=True)
                        strategy = 'mode_imputation'
                        if display:
                            print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mode imputation (mode={mode_value}) [no-drop mode]")
                    elif feature_type == 'num_quant_median':
                        median_value = df_cleaned[col].median()
                        df_cleaned[col].fillna(median_value, inplace=True)
                        strategy = 'median_imputation'
                        if display:
                            print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Median imputation (median={median_value:.2f}) [no-drop mode]")
                    elif feature_type == 'num_quant_mean':
                        mean_value = df_cleaned[col].mean()
                        df_cleaned[col].fillna(mean_value, inplace=True)
                        strategy = 'mean_imputation'
                        if display:
                            print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mean imputation (mean={mean_value:.2f}) [no-drop mode]")
                    elif feature_type == 'num_quant_mode':
                        mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else df_cleaned[col].median()
                        df_cleaned[col].fillna(mode_value, inplace=True)
                        strategy = 'mode_imputation'
                        if display:
                            print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mode imputation (mode={mode_value}) [no-drop mode]")
                    else:
                        # Unknown feature type — fill with mode for cat-like, median otherwise
                        if dtype == 'object':
                            mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                            df_cleaned[col].fillna(mode_value, inplace=True)
                            strategy = 'mode_imputation'
                        else:
                            median_value = df_cleaned[col].median()
                            df_cleaned[col].fillna(median_value, inplace=True)
                            strategy = 'median_imputation'
                        if display:
                            print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Imputed [no-drop mode]")
            
            elif 5 < missing_pct <= 20:
                # Impute based on feature type
                if feature_type == 'cat_features':
                    mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else 'Unknown'
                    df_cleaned[col].fillna(mode_value, inplace=True)
                    strategy = 'mode_imputation'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mode imputation (mode={mode_value})")
                
                elif feature_type == 'num_quant_median':
                    median_value = df_cleaned[col].median()
                    df_cleaned[col].fillna(median_value, inplace=True)
                    strategy = 'median_imputation'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Median imputation (median={median_value:.2f})")
                
                elif feature_type == 'num_quant_mean':
                    mean_value = df_cleaned[col].mean()
                    df_cleaned[col].fillna(mean_value, inplace=True)
                    strategy = 'mean_imputation'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mean imputation (mean={mean_value:.2f})")
                
                elif feature_type == 'num_quant_mode':
                    mode_value = df_cleaned[col].mode()[0] if len(df_cleaned[col].mode()) > 0 else df_cleaned[col].median()
                    df_cleaned[col].fillna(mode_value, inplace=True)
                    strategy = 'mode_imputation'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Mode imputation (mode={mode_value})")
                else:
                    # Unknown feature type - default to drop rows
                    initial_shape = df_cleaned.shape[0]
                    df_cleaned = df_cleaned.dropna(subset=[col])
                    dropped_rows = initial_shape - df_cleaned.shape[0]
                    strategy = 'drop_rows'
                    if display:
                        print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Dropped {dropped_rows} rows (unknown type)")
            
            elif 20 < missing_pct <= 40:
                # Manual analysis required - NO AUTOMATIC ACTION
                manual_analysis_cols.append((col, feature_type, missing_pct, missing_count))
                strategy = 'manual_analysis_required'
                if display:
                    print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> ⚠️  MANUAL ANALYSIS REQUIRED (No automatic action taken)")
                # Column is left unchanged - user must decide on handling strategy
            
            else:  # missing_pct > 40
                # Drop column
                df_cleaned = df_cleaned.drop(columns=[col])
                dropped_columns_list.append(col)
                strategy = 'drop_column'
                if display:
                    print(f"  {col} ({feature_type}): {missing_pct:.2f}% missing -> Dropped column (>40% threshold)")
            
            # Add to summary (only if column still exists)
            if col in df_cleaned.columns or strategy == 'drop_column':
                summary_data.append({
                    'Feature': col,
                    'Dtype': dtype,
                    'Feature_Type': feature_type,
                    'Missing_Count': missing_count,
                    'Missing_Percentage': round(missing_pct, 2),
                    'Applied_Strategy': strategy
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Display manual analysis columns if any
        if len(manual_analysis_cols) > 0 and display:
            print("\n⚠️  MANUAL ANALYSIS REQUIRED for the following columns:")
            manual_df = pd.DataFrame(manual_analysis_cols, 
                                    columns=['Feature', 'Feature_Type', 'Missing_Percentage', 'Missing_Count'])
            print(manual_df.to_string(index=False))
            print("\nPlease review these columns and decide on appropriate handling strategy.")
        
        if display:
            print("\nSummary:")
            print(f"  Columns with dropped rows: {len(summary_df[summary_df['Applied_Strategy'].str.contains('drop_rows', na=False)])}")
            print(f"  Columns with imputation: {len(summary_df[summary_df['Applied_Strategy'].str.contains('imputation', na=False)])}")
            print(f"  Columns dropped: {len(dropped_columns_list)}")
            print(f"  Columns requiring manual analysis: {len(manual_analysis_cols)}")
            print(f"  Final dataset shape: {df_cleaned.shape}")
        
        return df_cleaned, summary_df

    @staticmethod
    def get_data_summary(df: pd.DataFrame, display: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive summary of the dataset.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to print the results (default: True)
        
        Returns:
        summary: dict with summary information
        """
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        if display:
            print("Dataset Summary:")
            print(f"Shape: {summary['shape']}")
            print(f"Memory Usage: {summary['memory_usage']:.2f} MB")
            print(f"\nNumeric Columns ({len(summary['numeric_columns'])}): {summary['numeric_columns']}")
            print(f"Categorical Columns ({len(summary['categorical_columns'])}): {summary['categorical_columns']}")
            if summary['datetime_columns']:
                print(f"Datetime Columns ({len(summary['datetime_columns'])}): {summary['datetime_columns']}")
        
        return summary

    @staticmethod
    def clean_date_fields(df: pd.DataFrame, date_features: Optional[List[str]] = None, 
                          display: bool = True) -> Tuple[pd.DataFrame, Optional[Dict[str, List[str]]]]:
        """
        Clean date fields by converting datetime strings encoding durations to hours.
        The datetime strings are in format "1970-01-01 00:00:00.000000008" where the 
        number after the decimal point (e.g., 000000008) directly represents the duration in hours.
        Creates new numeric columns and calculates slack_time (expected - actual).
        
        Parameters:
        df: pandas DataFrame
        date_features: list of 2 date field names [actual_date_field, expected_date_field] 
                      (default: None - no date fields processed)
        display: bool, whether to print the processing information (default: True)
        
        Returns:
        df: DataFrame with new numeric date columns and slack_time column
        new_columns_info: dict with new column names for updating feature lists, or None if failed
        """
        import re  # Import at method level for regex operations
        
        if date_features is None or len(date_features) != 2:
            if display:
                print("Warning: date_features must be a list of exactly 2 date field names.")
                print("Expected format: [actual_date_field, expected_date_field]")
            return df, None
        
        df_cleaned = df.copy()
        actual_field = date_features[0]
        expected_field = date_features[1]
        
        # Check if fields exist
        missing_fields = [f for f in date_features if f not in df_cleaned.columns]
        if missing_fields:
            if display:
                print(f"Warning: The following date fields were not found: {missing_fields}")
            return df_cleaned, None
        
        if display:
            print(f"Processing date fields: {actual_field} (actual), {expected_field} (expected)")
        
        try:
            # Create numeric column names: replace '_hours' with '_hrs_num', or append '_num' if not found
            if '_hours' in actual_field:
                actual_num_col = actual_field.replace('_hours', '_hrs_num')
            else:
                actual_num_col = f"{actual_field}_num"
            
            if '_hours' in expected_field:
                expected_num_col = expected_field.replace('_hours', '_hrs_num')
            else:
                expected_num_col = f"{expected_field}_num"
            
            # Step 1: Extract duration from datetime strings
            # The format is "1970-01-01 00:00:00.000000008" where the number after the decimal
            # point represents the duration in hours (not nanoseconds)
            # Extract the numeric part after the decimal point and use it directly as hours
            
            def extract_hours_from_datetime_string(dt_str):
                """Extract hours from datetime string where duration is encoded in fractional seconds."""
                try:
                    if pd.isna(dt_str):
                        return np.nan
                    
                    # Convert to string if not already
                    dt_str = str(dt_str)
                    
                    # Extract the number after the decimal point
                    # Format: "1970-01-01 00:00:00.000000008" -> extract "000000008"
                    match = re.search(r'\.(\d+)$', dt_str)
                    if match:
                        # Extract the numeric part and convert to float
                        # The number directly represents hours
                        hours = float(match.group(1))
                        return hours
                    else:
                        # Fallback: try parsing as datetime and calculate from epoch
                        dt = pd.to_datetime(dt_str, errors='coerce')
                        if pd.isna(dt):
                            return np.nan
                        epoch = pd.Timestamp('1970-01-01 00:00:00')
                        delta = dt - epoch
                        hours = delta.total_seconds() / 3600.0
                        return hours
                except Exception as e:
                    return np.nan
            
            # Apply the conversion function to both columns
            df_cleaned[expected_num_col] = df_cleaned[expected_field].apply(extract_hours_from_datetime_string)
            df_cleaned[actual_num_col] = df_cleaned[actual_field].apply(extract_hours_from_datetime_string)
            
            # Check if conversion was successful (not all NaN)
            expected_nan_count = df_cleaned[expected_num_col].isna().sum()
            actual_nan_count = df_cleaned[actual_num_col].isna().sum()
            total_rows = len(df_cleaned)
            
            if display:
                print(f"  Conversion results:")
                print(f"    Expected field: {total_rows - expected_nan_count}/{total_rows} values converted, {expected_nan_count} NaN")
                print(f"    Actual field: {total_rows - actual_nan_count}/{total_rows} values converted, {actual_nan_count} NaN")
            
            # Only proceed if at least some values were converted successfully
            if expected_nan_count == total_rows or actual_nan_count == total_rows:
                if display:
                    print(f"  ❌ Error: Conversion failed - all values are NaN. Sample values:")
                    print(f"    Expected field sample: {df_cleaned[expected_field].head(3).tolist()}")
                    print(f"    Actual field sample: {df_cleaned[actual_field].head(3).tolist()}")
                raise ValueError("All values converted to NaN - datetime conversion failed")
            
            if display:
                print(f"  ✓ Converted to hours: {expected_num_col}, {actual_num_col}")
            
            # Step 2: Add slack time column (expected - actual)
            df_cleaned['slack_time'] = df_cleaned[expected_num_col] - df_cleaned[actual_num_col]
            if display:
                print(f"  ✓ Created slack_time = {expected_num_col} - {actual_num_col}")
            
            # Step 3: Drop original columns (only if conversion was successful)
            df_cleaned = df_cleaned.drop(columns=[actual_field, expected_field])
            if display:
                print(f"  ✓ Dropped original columns: {actual_field}, {expected_field}")
                print(f"\nNew columns created: {actual_num_col}, {expected_num_col}, slack_time")
                print(f"Columns dropped: {actual_field}, {expected_field}")
                print(f"Dataset shape: {df_cleaned.shape}")
            
            # Return information about new columns for updating feature lists
            new_columns_info = {
                'new_numeric_columns': [actual_num_col, expected_num_col, 'slack_time'],
                'dropped_columns': [actual_field, expected_field]
            }
            return df_cleaned, new_columns_info
        
        except Exception as e:
            if display:
                print(f"Error processing date fields: {str(e)}")
            return df_cleaned, None

    @staticmethod
    def clean_data_pipeline(df: pd.DataFrame, standardize_index: bool = True,
                            standardize_names: bool = True, 
                            handle_missing: bool = True, 
                            columns_to_drop: Optional[List[str]] = None,
                            cat_features: Optional[List[str]] = None,
                            num_quant_median: Optional[List[str]] = None,
                            num_quant_mean: Optional[List[str]] = None,
                            num_quant_mode: Optional[List[str]] = None,
                            target_variable: str = 'delayed',
                            date_features: Optional[List[str]] = None,
                            num_quant_features: Optional[List[str]] = None,
                            display: bool = True,
                            allow_row_drop: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[Dict[str, List[str]]]]:
        """
        Complete data cleaning pipeline.
        
        Parameters:
        df: pandas DataFrame
        standardize_index: bool, whether to standardize index (default: True)
        standardize_names: bool, whether to standardize column names (default: True)
        handle_missing: bool, whether to handle missing values (default: True)
        columns_to_drop: list of column names to drop (default: None - no columns dropped)
        cat_features: list of categorical feature names (default: None)
        num_quant_median: list of numerical features to impute with median (default: None)
        num_quant_mean: list of numerical features to impute with mean (default: None)
        num_quant_mode: list of numerical features to impute with mode (default: None)
        target_variable: str, name of the target variable (default: 'delayed')
        date_features: list of 2 date field names [actual_date_field, expected_date_field] (default: None)
        num_quant_features: list of all numeric quantitative features (default: None) - will be updated if date cleaning succeeds
        display: bool, whether to display progress (default: True)
        
        Returns:
        df_cleaned: cleaned DataFrame
        summary: dict with data summary information
        new_columns_info: dict with new column names for updating feature lists, or None
        """
        df_cleaned = df.copy()
        new_columns_info = None  # Store info about new columns created during date cleaning
        
        if display:
            print("Starting Data Cleaning Pipeline...")
            print("=" * 80)
        
        # Step 1: Standardize index
        if standardize_index:
            if display:
                print("\n1. Standardizing index...")
            df_cleaned = DataProcessing.standardize_index(df_cleaned, display=display)
            if display:
                print("   ✓ Index standardized")
        else:
            if display:
                print("\n1. Skipping index standardization")
        
        # Step 2: Standardize column names
        if standardize_names:
            if display:
                print("\n2. Standardizing column names...")
            df_cleaned = DataProcessing.standardize_column_names(df_cleaned)
            if display:
                print("   ✓ Column names standardized")
        else:
            if display:
                print("\n2. Skipping column name standardization")
        
        # Step 3: Drop unwanted columns
        if columns_to_drop is not None and len(columns_to_drop) > 0:
            if display:
                print("\n3. Dropping unwanted columns...")
            df_cleaned = DataProcessing.drop_unwanted_columns(df_cleaned, columns_to_drop=columns_to_drop, display=display)
            if display:
                print("   ✓ Unwanted columns dropped")
        else:
            if display:
                print("\n3. Skipping column dropping (no columns specified)")
        
        # Step 4: Clean date fields (before missing values handling)
        if date_features is not None and len(date_features) == 2:
            if display:
                print("\n4. Cleaning date fields...")
            df_cleaned, new_columns_info = DataProcessing.clean_date_fields(df_cleaned, date_features=date_features, display=display)
            # Store new_columns_info for return value
            
            # Update feature lists if date cleaning was successful
            if new_columns_info is not None:
                new_numeric_cols = new_columns_info['new_numeric_columns']
                dropped_cols = new_columns_info['dropped_columns']
                
                # new_numeric_cols = [actual_num_col, expected_num_col, 'slack_time']
                actual_num_col = new_numeric_cols[0]
                expected_num_col = new_numeric_cols[1]
                slack_time_col = new_numeric_cols[2]
                
                # Update num_quant_features (add all new numeric columns)
                if num_quant_features is not None:
                    num_quant_features.extend(new_numeric_cols)
                
                # Update num_quant_median (add actual and expected time columns - use median for time)
                if num_quant_median is not None:
                    num_quant_median.extend([actual_num_col, expected_num_col])
                
                # Update num_quant_mean (add slack_time - it's a difference, so mean makes sense)
                if num_quant_mean is not None:
                    num_quant_mean.append(slack_time_col)
                
                # Update date_features (remove dropped columns)
                if date_features is not None:
                    for col in dropped_cols:
                        if col in date_features:
                            date_features.remove(col)
                
                if display:
                    print("   ✓ Date fields cleaned")
                    print(f"   ✓ Updated feature lists:")
                    print(f"      - Added {new_numeric_cols} to num_quant_features")
                    print(f"      - Added {actual_num_col}, {expected_num_col} to num_quant_median")
                    print(f"      - Added {slack_time_col} to num_quant_mean")
                    print(f"      - Removed {dropped_cols} from date_features")
            else:
                if display:
                    print("   ⚠️ Date field cleaning completed with warnings")
        else:
            if display:
                print("\n4. Skipping date field cleaning (no date features specified)")
        
        # Step 5: Check and handle missing values
        if handle_missing:
            if display:
                print("\n5. Checking missing values...")
            missing_info = DataProcessing.check_missing_values(df_cleaned, display=display)
            
            if len(missing_info) > 0:
                if display:
                    print("\n   Handling missing values...")
                df_cleaned, _ = DataProcessing.handle_missing_values(
                    df_cleaned, missing_info, 
                    cat_features=cat_features,
                    num_quant_median=num_quant_median,
                    num_quant_mean=num_quant_mean,
                    num_quant_mode=num_quant_mode,
                    target_variable=target_variable,
                    display=display,
                    allow_row_drop=allow_row_drop,
                )
                if display:
                    print("   ✓ Missing values handled")
            else:
                if display:
                    print("   ✓ No missing values found")
        
        # Step 6: Get data summary
        if display:
            print("\n6. Generating data summary...")
        summary = DataProcessing.get_data_summary(df_cleaned, display=display)
        
        if display:
            print("\n" + "=" * 80)
            print("Data cleaning pipeline completed!")
            print(f"Final shape: {df_cleaned.shape}")
            print("=" * 80)
        
        return df_cleaned, summary, new_columns_info


