import pandas as pd
import numpy as np
import os
import pytest
from scripts.data_cleaning import DataCleaner

def test_missing_value_imputation():
    # Setup dummy data with missing values
    test_df = pd.DataFrame({
        'sale_price': [100, 200, np.nan],
        'sqft': [1000, np.nan, 3000],
        'neighborhood': ['A', 'A', 'B']
    })
    
    # Save temp raw data
    os.makedirs('data/raw', exist_ok=True)
    test_df.to_csv('data/raw/raw_market_data.csv', index=False)
    
    cleaner = DataCleaner()
    df_cleaned = cleaner.handle_missing_values(test_df.copy())
    
    assert df_cleaned['sale_price'].isnull().sum() == 0
    assert df_cleaned['sqft'].isnull().sum() == 0
    
def test_outlier_detection():
    # Provide enough variance so that 10 million is definitely an outlier
    prices = [100000, 110000, 120000, 130000, 90000] * 20
    sqfts = [2000, 2100, 1900, 2200, 1800] * 20
    
    prices.append(10000000) # The price outlier
    sqfts.append(2000)
    
    test_df = pd.DataFrame({
        'sale_price': prices,
        'sqft': sqfts,
        'neighborhood': ['Suburbs'] * 101
    })
    
    cleaner = DataCleaner()
    df_no_outliers = cleaner.detect_outliers(test_df)
    
    # Check that at least the outlier was removed and some data remains
    assert len(df_no_outliers) > 0
    assert len(df_no_outliers) < len(test_df)
    assert df_no_outliers['sale_price'].max() < 1000000
