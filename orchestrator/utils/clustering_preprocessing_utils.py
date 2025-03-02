import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

def perform_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Perform simple imputation based on data types"""
    imputed_df = df.copy()
    
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                # Use median for numeric
                imputed_df[column] = df[column].fillna(df[column].median())
            else:
                # Use mode for categorical
                imputed_df[column] = df[column].fillna(df[column].mode()[0])
                
    return imputed_df

def perform_encoding(
    df: pd.DataFrame,
    categorical_columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Perform one-hot encoding for categorical features"""
    if not categorical_columns:
        return df, []
        
    encoded_df = df.copy()
    encoded_features = []
    
    for col in categorical_columns:
        # Create dummy variables
        dummies = pd.get_dummies(
            df[col],
            prefix=col,
            drop_first=True  # Drop one category to avoid multicollinearity
        )
        
        # Add dummy columns to dataframe
        encoded_df = pd.concat([encoded_df, dummies], axis=1)
        encoded_features.extend(dummies.columns)
        
        # Drop original column
        encoded_df.drop(col, axis=1, inplace=True)
        
    return encoded_df, encoded_features

def perform_scaling(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """Scale numeric features using StandardScaler"""
    if not numeric_columns:
        return df
        
    scaled_df = df.copy()
    scaler = StandardScaler()
    
    scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return scaled_df 