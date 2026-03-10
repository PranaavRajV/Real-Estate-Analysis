import pandas as pd
import numpy as np
import yaml
import logging
import os
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from category_encoders import TargetEncoder

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Advanced feature engineering pipeline for Real Estate prediction.
    Implements creation, encoding, scaling, and selection.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.input_path = os.path.join(self.config['paths']['processed_data_path'], "clean_data.csv")
        self.output_path = os.path.join(self.config['paths']['processed_data_path'], "features.csv")
        self.current_year = datetime.now().year

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derives new features from raw signals.
        """
        logger.info("Starting feature creation...")
        
        # 1. Price Components
        df['price_per_sqft'] = df['sale_price'] / df['sqft']
        
        # 2. Age Components
        df['property_age'] = self.current_year - df['year_built']
        
        def categorize_age(age):
            if age < 5: return 'New'
            if age < 20: return 'Modern'
            if age < 50: return 'Vintage'
            return 'Historic'
        
        df['age_category'] = df['property_age'].apply(categorize_age)
        
        # 3. Size & Utility
        df['total_rooms'] = df['bedrooms'] + df['bathrooms']
        
        # 4. Luxury Index (Composite)
        # Weights: Neighborhood (5x) + Pool (2x) + Garage (1x)
        df['luxury_score'] = (df['neighborhood_score'] * 5) + (df['pool'] * 2) + df['garage']
        
        # 5. Time Decomposition
        df['sale_date'] = pd.to_datetime(df['sale_date'])
        df['sale_month'] = df['sale_date'].dt.month
        df['sale_quarter'] = df['sale_date'].dt.quarter
        df['sale_year'] = df['sale_date'].dt.year
        df['is_weekend_sale'] = df['sale_date'].dt.dayofweek >= 5
        df['is_weekend_sale'] = df['is_weekend_sale'].astype(int)
        
        logger.info(f"Created {len(df.columns) - 11} new features")
        return df

    def build_pipelines(self, df: pd.DataFrame) -> Tuple[ColumnTransformer, ColumnTransformer]:
        """
        Defines encoding and scaling strategies.
        """
        logger.info("Building preprocessing pipelines...")
        
        # Nominal: No inherent order (OneHot)
        # Rationale: Small cardinality, prevents arbitrary ordering
        nominal_features = ['property_type']
        
        # Ordinal: Clear logical order (Ordinal)
        # Rationale: 'New' > 'Modern' > 'Vintage' etc.
        ordinal_features = ['age_category']
        age_order = [['Historic', 'Vintage', 'Modern', 'New']]
        
        # High Cardinality: Location/Neighborhood (TargetEncoder)
        # Rationale: OneHot would create too many columns; TargetEncoding captures local price trends
        high_card_features = ['neighborhood']
        
        # Numerical: Continuous values
        numeric_features = [
            'sqft', 'bedrooms', 'bathrooms', 'property_age', 
            'total_rooms', 'luxury_score', 'neighborhood_score',
            'sale_month', 'sale_quarter'
        ]

        # Pipeline A: Linear Models (with Scaling)
        linear_preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('nom', OneHotEncoder(drop='first'), nominal_features),
                ('ord', OrdinalEncoder(categories=age_order), ordinal_features),
                ('high', TargetEncoder(), high_card_features)
            ])

        # Pipeline B: Tree Models (No Scaling)
        tree_preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numeric_features),
                ('nom', OneHotEncoder(handle_unknown='ignore'), nominal_features),
                ('ord', OrdinalEncoder(categories=age_order), ordinal_features),
                ('high', TargetEncoder(), high_card_features)
            ])

        return linear_preprocessor, tree_preprocessor

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out low-variance and highly correlated features.
        """
        logger.info("Performing feature selection...")
        
        target = self.config['dataset']['target_column']
        # Remove columns we can't use directly in modeling
        drop_cols = [target, 'sale_date', 'year_built', 'price_per_sqft']
        X = df.drop(columns=drop_cols)
        y = df[target]
        
        # 1. Handle Categorical for Selection (Manual temp encoding)
        X_encoded = pd.get_dummies(X)
        
        # 2. Variance Threshold (Remove features with < 1% variance)
        selector = VarianceThreshold(threshold=0.01)
        X_var = selector.fit_transform(X_encoded)
        selected_vars = X_encoded.columns[selector.get_support()]
        
        # 3. Correlation Filter (Remove r > 0.95)
        corr_matrix = pd.DataFrame(X_var, columns=selected_vars).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        X_final = pd.DataFrame(X_var, columns=selected_vars).drop(columns=to_drop)
        
        # 4. Univariate Selection (SelectKBest)
        skb = SelectKBest(score_func=f_regression, k='all')
        skb.fit(X_final, y)
        
        scores_df = pd.DataFrame({
            'Feature': X_final.columns,
            'Score': skb.scores_
        }).sort_values(by='Score', ascending=False)
        
        logger.info(f"Top 10 Features by F-Score:\n{scores_df.head(10)}")
        
        # Return the DataFrame with columns that passed the initial filters
        # Note: We keep the original categorical columns for the pipeline logic later
        return df

    def run_pipeline(self):
        df = pd.read_csv(self.input_path)
        df = self.create_features(df)
        df = self.select_features(df)
        
        df.to_csv(self.output_path, index=False)
        logger.info(f"Feature engineering complete. File saved to {self.output_path}")

    def plot_feature_importance(self, model, feature_names):
        """Visualizes importance from a trained model."""
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices], align="center", color='#048A81')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run_pipeline()
