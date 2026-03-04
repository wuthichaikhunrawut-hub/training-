"""
NeuroCore - Data Engine Module
Handles data loading, cleaning, and preprocessing for Perceptron Dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO
import arff


def load_arff_file(file):
    """Load and parse ARFF file using liac-arff"""
    try:
        # Read the file content
        file_content = file.getvalue().decode('utf-8')
        # Parse ARFF
        decoder = arff.ArffDecoder()
        data = decoder.decode(file_content, encode_nominal=True)
        # Convert to DataFrame
        df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])
        return df, None
    except Exception as e:
        return None, str(e)


def load_csv_file(file):
    """Load CSV file into DataFrame"""
    try:
        df = pd.read_csv(file)
        return df, None
    except Exception as e:
        return None, str(e)


def load_data(file):
    """
    Main data loading function
    Supports CSV and ARFF formats
    """
    file_name = file.name.lower()

    if file_name.endswith('.arff'):
        return load_arff_file(file)
    elif file_name.endswith('.csv'):
        return load_csv_file(file)
    else:
        return None, "Unsupported file format. Please upload CSV or ARFF files."


def clean_missing_values(df):
    """
    Auto-detect and remove missing values
    Returns cleaned DataFrame and notification message
    """
    initial_rows = len(df)

    # Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()

    if total_missing == 0:
        return df, None, 0

    # Drop rows with missing values
    df_cleaned = df.dropna()
    rows_removed = initial_rows - len(df_cleaned)

    # Create notification message
    missing_info = []
    for col, count in missing_counts.items():
        if count > 0:
            missing_info.append(f"{col}: {count} missing")

    notification = f"🗑️ **Auto-Cleaning Complete** | Removed {rows_removed} rows with missing values\n\n"
    notification += " | ".join(missing_info)

    return df_cleaned, notification, rows_removed


def min_max_scaler(df):
    """
    Custom Min-Max Scaler
    Transforms all numeric columns to [0, 1] range
    """
    df_scaled = df.copy()
    numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()

        if col_max - col_min != 0:
            df_scaled[col] = (df_scaled[col] - col_min) / (col_max - col_min)
        else:
            df_scaled[col] = 0

    return df_scaled


def get_data_statistics(df):
    """Generate basic statistics for the dataset"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return None

    stats = df[numeric_cols].describe().T
    stats['range'] = stats['max'] - stats['min']
    return stats


def get_feature_columns(df):
    """Extract feature columns (numeric, excluding target if possible)"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Try to identify target column (often named 'class', 'target', 'label', or 'output')
    target_candidates = ['class', 'target', 'label', 'output', 'y', 'Class', 'Target', 'Label']
    target_col = None

    for col in df.columns:
        if col.lower() in [t.lower() for t in target_candidates]:
            target_col = col
            break

    # If target found and it's in numeric columns, remove it from features
    if target_col and target_col in numeric_cols:
        feature_cols = [col for col in numeric_cols if col != target_col]
    else:
        feature_cols = numeric_cols

    return feature_cols, target_col


def auto_label_encode(df):
    """
    Automatically encode categorical (object/string) columns into numeric labels.
    Returns the encoded DataFrame and a dictionary of mappings.
    """
    df_encoded = df.copy()
    mappings = {}
    
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        # Factorize returns (labels, uniques)
        labels, uniques = pd.factorize(df_encoded[col])
        df_encoded[col] = labels
        # Create mapping dictionary: {index: string_value}
        mappings[col] = {i: val for i, val in enumerate(uniques)}
        
    return df_encoded, mappings
