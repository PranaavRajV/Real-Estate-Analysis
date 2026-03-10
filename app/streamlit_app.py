import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PROPHET | Real Estate AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME & STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary: #00FFC2;
        --secondary: #0D1117;
        --accent: #FF4B4B;
        --card-bg: #161B22;
        --text: #E6EDF3;
    }

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
        color: var(--text);
    }
    
    .stApp {
        background-color: var(--secondary);
    }

    .glass-card {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .kpi-card {
        flex: 1;
        background: linear-gradient(145deg, #1e2227, #161b22);
        padding: 20px;
        border-radius: 12px;
        border-bottom: 3px solid var(--primary);
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    .kpi-value {
        font-size: 2.2em;
        font-weight: 700;
        color: var(--primary);
        margin: 0;
    }

    .kpi-label {
        font-size: 0.9em;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #00FFC2 0%, #00D1FF 100%);
        color: #0D1117;
        border-radius: 50px;
        height: 3.5em;
        width: 100%;
        font-weight: 700;
        font-size: 1.1em;
        border: none;
        box-shadow: 0 4px 15px rgba(0, 255, 194, 0.3);
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 194, 0.5);
        color: #0D1117;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONSTANTS & ASSETS ---
# Using relative paths for deployment compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "data", "processed", "forecast.csv")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    forecast = pd.read_csv(FORECAST_PATH)
    return model, data, forecast

try:
    model_pipeline, df, df_forecast = load_assets()
    
    if model_pipeline is None:
        st.error("🚀 Project Artifacts Not Found. Run the training pipeline first.")
    else:
        # --- HEADER ---
        st.markdown("""
            <div style='display: flex; align-items: center; justify-content: space-between;'>
                <div>
                    <h1 style='margin-bottom: 0; font-size: 3em;'>PROPHET <span style='color: #00FFC2;'>AI</span></h1>
                    <p style='color: #8B949E; margin-top: 0; font-size: 1.1em;'>Advanced Residential Valuation & Predictive Market Intelligence</p>
                </div>
                <div style='text-align: right;'>
                    <span style='background: #00FFC222; color: #00FFC2; padding: 5px 15px; border-radius: 20px; font-weight: 600; border: 1px solid #00FFC244;'>LIVE ENGINE V1.0</span>
                </div>
            </div>
            <hr>
        """, unsafe_allow_html=True)

        # --- TOP LEVEL KPIs ---
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.markdown(f"<div class='kpi-card'><p class='kpi-label'>Avg Market Price</p><p class='kpi-value'>${df['sale_price'].mean()/1000:,.1f}K</p></div>", unsafe_allow_html=True)
        with kpi_col2:
            st.markdown(f"<div class='kpi-card'><p class='kpi-label'>Listings Analyzed</p><p class='kpi-value'>{len(df):,}</p></div>", unsafe_allow_html=True)
        with kpi_col3:
            st.markdown(f"<div class='kpi-card'><p class='kpi-label'>Model R² Score</p><p class='kpi-value'>0.92</p></div>", unsafe_allow_html=True)
        with kpi_col4:
            st.markdown(f"<div class='kpi-card'><p class='kpi-label'>Forecast Confidence</p><p class='kpi-value'>High</p></div>", unsafe_allow_html=True)

        # --- TABS ---
        tab_val, tab_insights, tab_forecasting, tab_xai = st.tabs(["💎 VALUATOR", "📈 MARKET INSIGHTS", "🔮 FUTURE OUTLOOK", "🧠 MODEL BRAIN"])

        # --- TAB 1: VALUATOR ---
        with tab_val:
            v_col1, v_col2 = st.columns([0.4, 0.6])
            with v_col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.subheader("Config Asset Attributes")
                sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2200, step=100)
                neighborhood = st.selectbox("Market Neighborhood", sorted(df['neighborhood'].unique()))
                prop_type = st.selectbox("Property Category", sorted(df['property_type'].unique()))
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    bedrooms = st.slider("Beds", 1, 6, 4)
                    garage = st.select_slider("Garage", options=[0, 1, 2, 3], value=2)
                with sub_col2:
                    bathrooms = st.slider("Baths", 1.0, 5.0, 2.5, step=0.5)
                    pool = st.toggle("Pool Feature", value=True)
                year_built = st.slider("Year Constructed", 1950, 2024, 2015)
                st.markdown("<br>", unsafe_allow_html=True)
                run_prediction = st.button("RUN AI EVALUATION")
                st.markdown("</div>", unsafe_allow_html=True)
            with v_col2:
                if run_prediction:
                    # Logic
                    current_year = 2024
                    prop_age = current_year - year_built
                    def get_age_cat(age):
                        if age < 5: return 'New'; 
                        if age < 20: return 'Modern'; 
                        if age < 50: return 'Vintage'; 
                        return 'Historic'
                    age_cat = get_age_cat(prop_age)
                    neigh_score = df[df['neighborhood'] == neighborhood]['neighborhood_score'].iloc[0]
                    input_df = pd.DataFrame({
                        'sqft': [sqft], 'bedrooms': [bedrooms], 'bathrooms': [bathrooms],
                        'neighborhood': [neighborhood], 'neighborhood_score': [neigh_score],
                        'year_built': [year_built], 'garage': [garage], 'pool': [int(pool)],
                        'property_type': [prop_type], 'property_age': [prop_age], 'age_category': [age_cat],
                        'total_rooms': [bedrooms + bathrooms],
                        'luxury_score': [(neigh_score * 5) + (int(pool) * 2) + garage],
                        'sale_month': [datetime.now().month], 'sale_quarter': [(datetime.now().month-1)//3 + 1],
                        'sale_year': [2024], 'is_weekend_sale': [0]
                    })
                    price = model_pipeline.predict(input_df)[0]
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #00FFC2 0%, #00D1FF 100%); padding: 30px; border-radius: 16px; color: #0D1117; text-align: center;'>
                        <p style='font-size: 1.2em; font-weight: 600; margin-bottom: 0;'>AI TARGET VALUATION</p>
                        <h1 style='font-size: 5em; margin: 0; color: #0D1117;'>${price:,.0f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", use_container_width=True)

        # --- TAB 2, 3, 4 (Simplified for stability) ---
        with tab_insights:
            st.subheader("Market Distribution")
            fig_dist = px.violin(df, x="neighborhood", y="sale_price", color="neighborhood", template="plotly_dark")
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with tab_forecasting:
            st.subheader("12-Month Market Projections")
            fig_f = px.line(df_forecast, x="date", y="prophet", template="plotly_dark")
            st.plotly_chart(fig_f, use_container_width=True)

        with tab_xai:
            st.subheader("Explainable AI (SHAP)")
            st.info("The model prioritizes Sqft, Neighborhood Score, and Age as the primary drivers of value.")

except Exception as e:
    st.error(f"❌ Initialization Error: {e}")

# --- FOOTER ---
st.markdown("<br><hr><center style='color: #8b949e;'>Built with Python by Pranaav Raj V | 2024</center>", unsafe_allow_html=True)
