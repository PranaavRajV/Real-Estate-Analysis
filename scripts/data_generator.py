import pandas as pd
import numpy as np
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealEstateDataGenerator:
    """
    Generates synthetic real estate data for development and testing.
    Includes price trends, seasonality, and realistic feature distributions.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def generate_data(self, n_rows: int = 10000) -> pd.DataFrame:
        """
        Generates a synthetic dataset with realistic correlations.
        """
        logger.info(f"Generating {n_rows} synthetic real estate records...")
        
        np.random.seed(self.config['dataset']['random_state'])
        
        # Base Features
        sqft = np.random.normal(2000, 800, n_rows).clip(500, 10000)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        # Bathrooms usually follow bedrooms
        bathrooms = (bedrooms * 0.7 + np.random.normal(0.5, 0.3, n_rows)).round(1).clip(1, 5)
        
        neighborhoods = ['Downtown', 'Suburbs', 'Riverside', 'Hillside', 'Industrial']
        location = np.random.choice(neighborhoods, n_rows)
        neighborhood_scores = {
            'Downtown': 0.9, 'Suburbs': 0.7, 'Riverside': 0.85, 'Hillside': 0.95, 'Industrial': 0.4
        }
        scores = np.array([neighborhood_scores[loc] for loc in location])
        
        year_built = np.random.randint(1950, 2024, n_rows)
        garage = np.random.choice([0, 1, 2, 3], n_rows, p=[0.2, 0.4, 0.3, 0.1])
        pool = np.random.choice([0, 1], n_rows, p=[0.8, 0.2])
        property_type = np.random.choice(['Single Family', 'Condo', 'Townhouse', 'Multi-Family'], n_rows)
        
        # Time Component
        start_date = datetime(2020, 1, 1)
        sale_dates = [start_date + timedelta(days=np.random.randint(0, 1500)) for _ in range(n_rows)]
        
        # Price Logic (The Target)
        # Base price + sqft premium + bedroom/bath bonus + age penalty + location premium + trend
        base_price = 100000
        sqft_premium = sqft * 150
        bed_bath_bonus = (bedrooms * 20000) + (bathrooms * 15000)
        age_factor = (year_built - 1950) * 1000
        loc_premium = scores * 200000
        
        # Add a time-based trend (3% annual appreciation) and seasonality
        days_since_start = np.array([(d - start_date).days for d in sale_dates])
        trend = days_since_start * 50 
        seasonality = np.sin(2 * np.pi * days_since_start / 365) * 15000
        
        noise = np.random.normal(0, 25000, n_rows)
        
        price = base_price + sqft_premium + bed_bath_bonus + age_factor + loc_premium + trend + seasonality + noise
        
        df = pd.DataFrame({
            'sale_price': price.round(2),
            'sqft': sqft.astype(int),
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'neighborhood': location,
            'neighborhood_score': scores,
            'year_built': year_built,
            'garage': garage,
            'pool': pool,
            'sale_date': sale_dates,
            'property_type': property_type
        })
        
        # Introduce some realistic dirtiness
        logger.info("Injecting noise/missing values for cleaning pipeline testing...")
        df.loc[df.sample(frac=0.02).index, 'sqft'] = np.nan
        df.loc[df.sample(frac=0.01).index, 'bedrooms'] = -1 # Outlier/Error
        df.loc[df.sample(frac=0.03).index, 'sale_price'] = df['sale_price'] * 10 # Massive outliers
        
        save_path = f"{self.config['paths']['raw_data_path']}raw_market_data.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"Data saved to {save_path}")
        return df

if __name__ == "__main__":
    generator = RealEstateDataGenerator()
    generator.generate_data(10000)
