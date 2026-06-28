"""
Data Extraction Module

This module contains functions for data loading and basic inspection operations.
"""

import pandas as pd
from typing import Optional, Tuple
from IPython.display import display as ipython_display

class DataExtract:
    """Class for data extraction and basic inspection operations."""
    
    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.
        
        Parameters:
        file_path: str, path to the CSV file
        **kwargs: additional arguments to pass to pd.read_csv()
        
        Returns:
        df: pandas DataFrame
        """
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def show_head(df: pd.DataFrame, n: int = 5, display: bool = True) -> pd.DataFrame:
        """
        Display the first n rows of the DataFrame with formatted output.
        
        Parameters:
        df: pandas DataFrame
        n: int, number of rows to display (default: 5)
        display: bool, whether to display information (default: True)
        
        Returns:
        df: DataFrame with first n rows
        """
        result_df = df.head(n)
        if display:
            print(f"First {n} rows:")
            ipython_display(result_df)
        return result_df
    
    @staticmethod
    def show_tail(df: pd.DataFrame, n: int = 5, display: bool = True) -> pd.DataFrame:
        """
        Display the last n rows of the DataFrame with formatted output.
        
        Parameters:
        df: pandas DataFrame
        n: int, number of rows to display (default: 5)
        display: bool, whether to display information (default: True)
        
        Returns:
        df: DataFrame with last n rows
        """
        result_df = df.tail(n)
        if display:
            print(f"Last {n} rows:")
            ipython_display(result_df)
        return result_df
    
    @staticmethod
    def show_info(df: pd.DataFrame, display: bool = True) -> None:
        """
        Display information about the DataFrame including data types and non-null counts.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to display information (default: True)
        """
        if display:
            df.info()
    
    @staticmethod
    def show_shape(df: pd.DataFrame, display: bool = True) -> tuple:
        """
        Display the shape (rows, columns) of the DataFrame.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to display information (default: True)
        
        Returns:
        tuple: (rows, columns)
        """
        shape = df.shape
        if display:
            print(f"Dataset shape: {shape} (rows: {shape[0]}, columns: {shape[1]})")
        return shape
    
    @staticmethod
    def show_dtypes(df: pd.DataFrame, display: bool = True) -> pd.Series:
        """
        Display the data types of each column with formatted output.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to display information (default: True)
        
        Returns:
        dtypes: pandas Series with column data types
        """
        dtypes = df.dtypes
        if display:
            print("Column Data Types:")
            ipython_display(dtypes)
        return dtypes
    
    @staticmethod
    def show_null_info(df: pd.DataFrame, display: bool = True) -> pd.DataFrame:
        """
        Display null value information for the DataFrame.
        
        Parameters:
        df: pandas DataFrame
        display: bool, whether to display information (default: True)
        
        Returns:
        null_counts_df: DataFrame with null counts (only columns with nulls)
        """
        null_counts_series = df.isnull().sum()
        null_counts_df = pd.DataFrame({
            'Column': null_counts_series.index,
            'Null Count': null_counts_series.values
        })
        null_counts_df = null_counts_df[null_counts_df['Null Count'] > 0]
        
        if display:
            print("Null Counts:")
            if len(null_counts_df) > 0:
                ipython_display(null_counts_df)
            else:
                print("  No null values found")
        
        return null_counts_df
    
    @staticmethod
    def data_extract_pipeline(file_path: str,
                               show_shape: bool = True,
                               show_head: bool = True,
                               show_info: bool = True,
                               show_null_info: bool = True,
                               head_n: int = 5,
                               **read_csv_kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Complete data extraction and inspection pipeline.
        
        Parameters:
        file_path: str, path to the CSV file
        show_shape: bool, whether to display dataset shape (default: True)
        show_head: bool, whether to display first n rows (default: True)
        show_info: bool, whether to display DataFrame info (default: True)
        show_null_info: bool, whether to display null value information (default: True)
        head_n: int, number of rows to display in head (default: 5)
        **read_csv_kwargs: additional arguments to pass to pd.read_csv()
        
        Returns:
        df: pandas DataFrame
        null_info_df: DataFrame with null counts (if show_null_info=True), None otherwise
        """
        if True in [show_shape, show_head, show_info, show_null_info]:
            print("=" * 80)
            print("DATA EXTRACTION PIPELINE")
            print("=" * 80)
        
        # Step 1: Read CSV file
        print("\n1. Reading CSV file...")
        df = DataExtract.read_csv(file_path, **read_csv_kwargs)
        print(f"   ✓ File loaded: {file_path}")
        
        null_info_df = None
        
        # Step 2: Display shape
        if show_shape:
            print("\n2. Displaying dataset shape...")
            DataExtract.show_shape(df, display=True)
            print("   ✓ Shape displayed")
        else:
            print("\n2. Skipping shape display")
        
        # Step 3: Display head
        if show_head:
            print("\n3. Displaying first few rows...")
            DataExtract.show_head(df, n=head_n, display=True)
            print("   ✓ Head displayed")
        else:
            print("\n3. Skipping head display")
        
        # Step 4: Display info
        if show_info:
            print("\n4. Displaying DataFrame info...")
            DataExtract.show_info(df, display=True)
            print("   ✓ Info displayed")
        else:
            print("\n4. Skipping info display")
        
        # Step 5: Display null information
        if show_null_info:
            print("\n5. Displaying null value information...")
            null_info_df = DataExtract.show_null_info(df, display=True)
            print("   ✓ Null information displayed")
        else:
            print("\n5. Skipping null information display")
        
        if True in [show_shape, show_head, show_info, show_null_info]:
            print("\n" + "=" * 80)
            print("Data extraction pipeline completed!")
            print(f"Final shape: {df.shape}")
            print("=" * 80)
        
        return df, null_info_df