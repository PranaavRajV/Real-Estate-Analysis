import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml

# Load Config
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Settings
processed_data_path = "data/processed/clean_data.csv"
figures_path = "reports/figures/"
palette = ["#2E4057", "#048A81", "#54C6EB", "#EF626C"]
sns.set_theme(style="whitegrid")
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=palette)

# Ensure directory exists
os.makedirs(figures_path, exist_ok=True)

# Load Data
df = pd.read_csv(processed_data_path, parse_dates=['sale_date'])
df['price_per_sqft'] = df['sale_price'] / df['sqft']
df['year'] = df['sale_date'].dt.year

def save_fig(name):
    plt.savefig(os.path.join(figures_path, f"{name}.png"), bbox_inches='tight', dpi=300)
    plt.close()

# Q1: Price Distribution
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df['sale_price'], kde=True, color=palette[0])
plt.title('Sale Price Distribution')
plt.subplot(1, 2, 2)
sns.boxplot(y=df['sale_price'], color=palette[1])
plt.title('Sale Price Boxplot')
save_fig('q1_price_dist')

# Q2: Correlations
plt.figure(figsize=(10, 8))
# Select only numeric for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()['sale_price'].sort_values(ascending=False).head(11)
sns.barplot(x=corr.values, y=corr.index, palette='viridis')
plt.title('Top Feature Correlations with Sale Price')
save_fig('q2_correlations')

# Q3: Price by Neighborhood
plt.figure(figsize=(12, 6))
sns.boxplot(x='neighborhood', y='sale_price', data=df, palette=palette)
plt.title('Price Variation by Neighborhood')
plt.xticks(rotation=45)
save_fig('q3_price_neighborhood')

# Q4: Price per Sqft across Property Types
plt.figure(figsize=(12, 6))
sns.violinplot(x='property_type', y='price_per_sqft', data=df, palette=palette)
plt.title('Price per Square Foot by Property Type')
save_fig('q4_price_sqft_type')

# Q5: Time Series Trends
plt.figure(figsize=(14, 6))
df_monthly = df.set_index('sale_date').resample('M')['sale_price'].mean().reset_index()
sns.lineplot(x='sale_date', y='sale_price', data=df_monthly, color=palette[3], linewidth=2)
plt.title('Monthly Average Sale Price Trend (2020-2024)')
plt.xlabel('Date')
plt.ylabel('Average Price')
save_fig('q5_price_trend')

# Q6: Amenities (Pool vs Garage)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(x='pool', y='sale_price', data=df, ax=axes[0], palette=[palette[1], palette[2]])
axes[0].set_title('Price Comparison: Pool vs No Pool')
sns.barplot(x='garage', y='sale_price', data=df, ax=axes[1], palette='flare')
axes[1].set_title('Price Comparison by Garage Capacity')
save_fig('q6_amenities')

# Q7: Geographic Proxy (Neighborhood Score vs Price)
plt.figure(figsize=(10, 8))
sns.scatterplot(x='neighborhood_score', y='sale_price', hue='neighborhood', size='sqft', data=df, alpha=0.6, palette='deep')
plt.title('Price Clusters: Neighborhood Score vs Valuation')
save_fig('q7_geographic_clusters')

# Q8: Price by Bedrooms
plt.figure(figsize=(12, 6))
sns.boxenplot(x='bedrooms', y='sale_price', data=df, palette='coolwarm')
plt.title('Price Distribution by Number of Bedrooms')
save_fig('q8_bedrooms_dist')

print("All EDA visualizations generated and saved to reports/figures/")
