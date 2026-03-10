import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketForecaster:
    """
    Time-series forecasting suite for Real Estate Market Trends.
    Compares ARIMA, Prophet, and Moving Average models.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.input_path = os.path.join(self.config['paths']['processed_data_path'], "clean_data.csv")
        self.output_path = os.path.join(self.config['paths']['processed_data_path'], "forecast.csv")
        self.figures_path = self.config['paths']['figures_path']
        
    def prepare_data(self) -> pd.Series:
        df = pd.read_csv(self.input_path, parse_dates=['sale_date'])
        ts = df.set_index('sale_date')['sale_price'].resample('MS').median()
        if ts.isnull().any():
            ts = ts.interpolate(method='linear')
        return ts

    def train_arima(self, data: pd.Series, periods: int = 12):
        model = auto_arima(data, seasonal=True, m=12, suppress_warnings=True)
        forecast = model.predict(n_periods=periods)
        return forecast

    def train_prophet(self, data: pd.Series, periods: int = 12):
        df_p = data.reset_index()
        df_p.columns = ['ds', 'y']
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.add_country_holidays(country_name='US')
        model.fit(df_p)
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        return forecast.iloc[-periods:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    def moving_average(self, data: pd.Series, periods: int = 12, window: int = 6):
        last_val = data.rolling(window=window).mean().iloc[-1]
        return pd.Series([last_val] * periods)

    def run_pipeline(self):
        ts = self.prepare_data()
        
        # 1. VALUATION (Holdout 6 months)
        train = ts[:-6]
        test = ts[-6:]
        
        ar_val = self.train_arima(train, 6)
        pr_val = self.train_prophet(train, 6)['yhat']
        ma_val = self.moving_average(train, 6)
        
        metrics = []
        for name, pred in zip(['ARIMA', 'Prophet', 'MA'], [ar_val, pr_val, ma_val]):
            rmse = np.sqrt(mean_squared_error(test, pred))
            metrics.append({'Model': name, 'RMSE': rmse})
        logger.info(f"Cross-Validation:\n{pd.DataFrame(metrics)}")

        # 2. FINAL FORECAST (Full Data)
        logger.info("Generating 12-month future forecast...")
        horizon = 12
        final_arima = self.train_arima(ts, horizon)
        final_prophet = self.train_prophet(ts, horizon)
        final_ma = self.moving_average(ts, horizon)
        
        future_dates = pd.date_range(ts.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq='MS')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'arima': final_arima.values,
            'prophet': final_prophet['yhat'].values,
            'ma_benchmark': final_ma.values
        })
        forecast_df.to_csv(self.output_path, index=False)
        
        # Plot
        plt.figure(figsize=(14, 7))
        plt.plot(ts.index, ts.values, label='History', color='#2E4057', linewidth=2)
        plt.plot(future_dates, forecast_df['arima'], label='ARIMA Forecast', linestyle='--', color='#EF626C')
        plt.plot(future_dates, forecast_df['prophet'], label='Prophet Forecast', linestyle='--', color='#048A81')
        plt.fill_between(future_dates, final_prophet['yhat_lower'], final_prophet['yhat_upper'], color='#048A81', alpha=0.1)
        plt.title('Real Estate Market Trends: 12-Month Price Forecast')
        plt.legend()
        plt.savefig(os.path.join(self.figures_path, 'forecast_comparison.png'), dpi=300)
        logger.info(f"Forecast saved to {self.output_path}")

if __name__ == "__main__":
    forecaster = MarketForecaster()
    forecaster.run_pipeline()
