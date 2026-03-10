import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
import joblib
from typing import Dict, Any, List

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Import our custom preprocessor builder
import sys
sys.path.append('scripts')
from feature_engineering import FeatureEngineer

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Production-grade model training and evaluation suite.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_path = os.path.join(self.config['paths']['processed_data_path'], "features.csv")
        self.model_save_path = self.config['paths']['model_path']
        self.metrics_path = os.path.join(self.model_save_path, "model_metrics.json")
        self.fe = FeatureEngineer(config_path)
        
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def load_and_split(self) -> tuple:
        df = pd.read_csv(self.data_path)
        target = self.config['dataset']['target_column']
        
        # Features to drop before preprocessing
        X = df.drop(columns=[target, 'sale_date', 'price_per_sqft', 'year_built'])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['dataset']['test_size'], 
            random_state=self.config['dataset']['random_state']
        )
        return X_train, X_test, y_train, y_test

    def get_models(self) -> Dict[str, Any]:
        return {
            "LinearRegression": {
                "model": LinearRegression(),
                "params": {},
                "type": "linear"
            },
            "Ridge": {
                "model": Ridge(),
                "params": {"model__alpha": [0.1, 1.0, 10.0]},
                "type": "linear"
            },
            "Lasso": {
                "model": Lasso(),
                "params": {"model__alpha": [0.1, 1.0, 10.0]},
                "type": "linear"
            },
            "RandomForest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [10, 20, None]
                },
                "type": "tree"
            },
            "XGBoost": {
                "model": XGBRegressor(random_state=42),
                "params": {
                    "model__n_estimators": [500],
                    "model__learning_rate": [0.05, 0.1],
                    "model__max_depth": [6]
                },
                "type": "tree"
            },
            "LightGBM": {
                "model": LGBMRegressor(random_state=42, verbose=-1),
                "params": {
                    "model__n_estimators": [500],
                    "model__learning_rate": [0.05, 0.1]
                },
                "type": "tree"
            }
        }

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.load_and_split()
        linear_prep, tree_prep = self.fe.build_pipelines(X_train)
        
        models_dict = self.get_models()
        results = []
        best_r2 = -np.inf
        best_model_name = ""
        best_pipeline = None

        for name, config in models_dict.items():
            logger.info(f"Training {name}...")
            
            # Select appropriate preprocessor
            preprocessor = linear_prep if config['type'] == 'linear' else tree_prep
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', config['model'])
            ])
            
            # Hyperparameter Tuning
            grid = GridSearchCV(
                pipeline, 
                config['params'], 
                cv=5, 
                scoring='r2', 
                n_jobs=-1
            )
            grid.fit(X_train, y_train)
            
            # Best Estimator evaluation
            best_pipe = grid.best_estimator_
            y_pred = best_pipe.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            mape = self.mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "model": name,
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "mape": round(mape, 2),
                "r2": round(r2, 4)
            }
            results.append(metrics)
            logger.info(f"{name} Metrics: R2={metrics['r2']}, RMSE={metrics['rmse']}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_pipeline = best_pipe

        # Save Best Model Artifacts
        logger.info(f"Saving best model: {best_model_name} with R2={best_r2}")
        joblib.dump(best_pipeline, os.path.join(self.model_save_path, "best_model.pkl"))
        joblib.dump(best_pipeline.named_steps['preprocessor'], os.path.join(self.model_save_path, "preprocessor.pkl"))
        
        with open(self.metrics_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate()
