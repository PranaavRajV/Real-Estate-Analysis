import pandas as pd
import numpy as np
import yaml
import logging
import os
from scipy import stats
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Production-grade data cleaning pipeline for Real Estate datasets.
    Handles missing values, outliers, duplicates, and type enforcement.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.raw_path = os.path.join(self.config['paths']['raw_data_path'], "raw_market_data.csv")
        self.processed_path = os.path.join(self.config['paths']['processed_data_path'], "clean_data.csv")
        
    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            logger.error(f"File not found at {self.raw_path}")
            raise FileNotFoundError(f"Raw data missing at {self.raw_path}")
        df = pd.read_csv(self.raw_path, parse_dates=['sale_date'])
        logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies cleaning strategies:
        - Numerical: Median imputation (robust to outliers)
        - Categorical: Mode imputation
        """
        logger.info("Handling missing values...")
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {missing_count} values in {col} with median: {median_val}")
                
        for col in cat_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed {missing_count} values in {col} with mode: {mode_val}")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(df)
        df = df.drop_duplicates()
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    def detect_outliers(self, df: pd.DataFrame, method: str = 'both') -> pd.DataFrame:
        """
        Uses IQR and Z-Score to filter extreme anomalies in sale_price and sqft.
        """
        logger.info(f"Detecting outliers using {method} method...")
        cols_to_check = ['sale_price', 'sqft']
        mask = pd.Series([True] * len(df))

        for col in cols_to_check:
            # IQR Method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            iqr_mask = (df[col] >= (Q1 - 1.5 * IQR)) & (df[col] <= (Q3 + 1.5 * IQR))
            
            # Z-Score Method
            z_scores = np.abs(stats.zscore(df[col]))
            z_mask = z_scores < 3
            
            if method == 'iqr':
                mask &= iqr_mask
            elif method == 'zscore':
                mask &= z_mask
            else: # both
                mask &= (iqr_mask & z_mask)
                
        cleaned_df = df[mask].copy()
        logger.info(f"Outlier removal complete. Rows dropped: {len(df) - len(cleaned_df)}")
        return cleaned_df

    def enforce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Enforcing data types...")
        df['bedrooms'] = df['bedrooms'].astype(int)
        df['garage'] = df['garage'].astype(int)
        df['pool'] = df['pool'].astype(int)
        # Handle bedrooms -1 logic
        df.loc[df['bedrooms'] < 0, 'bedrooms'] = df['bedrooms'].median()
        return df

    def validate_health(self, df: pd.DataFrame):
        """Prints a comprehensive data health report."""
        print("\n" + "="*30)
        print("DATA HEALTH REPORT")
        print("="*30)
        print(f"Total Records: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        print("-" * 30)
        print("MISSING VALUES:")
        print(df.isnull().sum())
        print("-" * 30)
        print("DESCRIPTIVE STATS (Price & Sqft):")
        print(df[['sale_price', 'sqft']].describe())
        print("="*30 + "\n")

    def run_pipeline(self):
        try:
            df = self.load_data()
            df = self.remove_duplicates(df)
            df = self.handle_missing_values(df)
            df = self.enforce_types(df)
            df = self.detect_outliers(df)
            
            self.validate_health(df)
            
            df.to_csv(self.processed_path, index=False)
            logger.info(f"Cleaned data successfully saved to {self.processed_path}")
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run_pipeline()
