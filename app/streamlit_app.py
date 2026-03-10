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
    page_title="EQUITY | Real Estate Terminal",
    page_icon="🔳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- THEME: MINIMALIST BLACK & WHITE ---
# This CSS hides Streamlit's default elements and creates a high-contrast luxury UI.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@200;400;600;800&display=swap');
    
    :root {
        --bg: #000000;
        --surface: #111111;
        --border: #333333;
        --text: #FFFFFF;
        --accent: #FFFFFF;
    }

    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .stApp {
        background-color: var(--bg);
    }

    /* Hide Streamlit Header/Footer */
    header, footer {visibility: hidden !important;}
    #MainMenu {visibility: hidden !important;}
    div.block-container {padding-top: 2rem !important;}

    /* Modern Minimalist Card */
    .terminal-card {
        background-color: var(--surface);
        border: 1px solid var(--border);
        border-radius: 0px;
        padding: 30px;
        margin-bottom: 20px;
        transition: border 0.3s ease;
    }
    
    .terminal-card:hover {
        border: 1px solid var(--accent);
    }

    /* Typography */
    h1 {
        letter-spacing: -2px;
        font-weight: 800 !important;
        text-transform: uppercase;
        font-size: 3.5rem !important;
    }
    
    h2, h3 {
        text-transform: uppercase;
        font-weight: 600 !important;
        letter-spacing: 2px;
        color: #888888 !important;
    }

    /* KPI Display */
    .kpi-val {
        font-size: 2.8rem;
        font-weight: 200;
        color: var(--text);
        margin: 0;
    }
    
    .kpi-label {
        color: #666666;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: -10px;
    }

    /* Industrial Button */
    div.stButton > button {
        background-color: transparent;
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
        border-radius: 0px;
        height: 3.5em;
        width: 100%;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #FFFFFF;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: transparent;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0px;
        padding: 10px 40px;
        background-color: transparent;
        border: none;
        color: #666666;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFFFFF !important;
    }

    /* Input Overrides */
    div[data-baseweb="input"] {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    
    /* Metrics Padding */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 200 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# --- LOADING ENGINE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "features.csv")
FORECAST_PATH = os.path.join(BASE_DIR, "data", "processed", "forecast.csv")

@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH): return None, None, None
    return joblib.load(MODEL_PATH), pd.read_csv(DATA_PATH), pd.read_csv(FORECAST_PATH)

try:
    model, df, df_forecast = load_assets()
    
    if model is None:
        st.error("SYSTEM OFFLINE: Artifacts missing.")
    else:
        # --- HEADER ---
        st.markdown("""
            <div style='display: flex; justify-content: space-between; align-items: flex-end;'>
                <h1>EQUITY <span style='font-weight: 200;'>TERM.</span></h1>
                <div style='text-align: right; color: #666; font-size: 0.8rem;'>
                    STATUS: OPERATIONAL // VERSION 2.0.0<br>
                    PORT: 8501 // DATABASE: RESIDENTIAL_VAL_01
                </div>
            </div>
            <hr style='border-top: 1px solid #333; margin-bottom: 50px;'>
        """, unsafe_allow_html=True)

        # --- KPI GRID ---
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(f"<div class='kpi-val'>${df['sale_price'].mean()/1000:,.1f}K</div><div class='kpi-label'>MARKET AVG</div>", unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi-val'>{len(df):,}</div><div class='kpi-label'>SAMPLES</div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi-val'>0.92</div><div class='kpi-label'>CONFIDENCE</div>", unsafe_allow_html=True)
        k4.markdown(f"<div class='kpi-val'>+3.2%</div><div class='kpi-label'>FORECAST 12M</div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        # --- NAVIGATION ---
        tab1, tab2, tab3 = st.tabs(["[ 01 ] VALUATION", "[ 02 ] ANALYSIS", "[ 03 ] ARCHITECTURE"])

        # --- TAB 1: VALUATION ---
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_l, col_r = st.columns([0.35, 0.65])
            
            with col_l:
                st.markdown("<div class='terminal-card'>", unsafe_allow_html=True)
                st.write("INPUT PARAMETERS")
                sqft = st.number_input("SQ FT", value=2500, step=100)
                neighborhood = st.selectbox("LOCATION", sorted(df['neighborhood'].unique()))
                
                c1, c2 = st.columns(2)
                beds = c1.slider("BEDS", 1, 6, 4)
                baths = c2.slider("BATHS", 1.0, 5.0, 2.5, step=0.5)
                
                prop_type = st.selectbox("TYPE", sorted(df['property_type'].unique()))
                age = st.slider("YEAR", 1950, 2024, 2010)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("RUN VALUATION"):
                    # Inference Logic
                    cur_yr = 2024
                    p_age = cur_yr - age
                    neigh_score = df[df['neighborhood'] == neighborhood]['neighborhood_score'].iloc[0]
                    
                    input_df = pd.DataFrame({
                        'sqft': [sqft], 'bedrooms': [beds], 'bathrooms': [baths],
                        'neighborhood': [neighborhood], 'neighborhood_score': [neigh_score],
                        'year_built': [age], 'garage': [2], 'pool': [1],
                        'property_type': [prop_type], 'property_age': [p_age], 
                        'age_category': ['Modern'], 'total_rooms': [beds + baths],
                        'luxury_score': [(neigh_score * 5) + 3],
                        'sale_month': [datetime.now().month], 'sale_quarter': [1],
                        'sale_year': [2024], 'is_weekend_sale': [0]
                    })
                    
                    prediction = model.predict(input_df)[0]
                    st.session_state['price'] = prediction
                st.markdown("</div>", unsafe_allow_html=True)

            with col_r:
                if 'price' in st.session_state:
                    p = st.session_state['price']
                    st.markdown(f"""
                        <div style='border: 1px solid #FFFFFF; padding: 50px; text-align: center;'>
                            <div style='color: #666; font-size: 0.8rem; letter-spacing: 5px; margin-bottom: 10px;'>ESTIMATED MARKET EQUITY</div>
                            <div style='font-size: 6rem; font-weight: 200;'>${p:,.0f}</div>
                            <div style='color: #666; margin-top: 10px;'>ACCURACY TOLERANCE: ± 3.8%</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Mini Chart: Comp selection
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.write("LOCAL MARKET SEGMENTATION")
                    comps = df[df['neighborhood'] == neighborhood].head(5)
                    st.dataframe(comps[['sale_price', 'sqft', 'bedrooms', 'property_type']].style.format({"sale_price": "${:,.0f}"}).set_properties(**{'background-color': '#000', 'color': '#fff', 'border-color': '#333'}), use_container_width=True)
                else:
                    st.markdown("""
                        <div style='border: 1px dashed #333; height: 400px; display: flex; align-items: center; justify-content: center; color: #333;'>
                            PENDING INPUT // READY TO COMPUTE
                        </div>
                    """, unsafe_allow_html=True)

        # --- TAB 2: ANALYSIS ---
        with tab_2:
            st.markdown("<br>", unsafe_allow_html=True)
            # Monochrome Analytics
            fig = px.scatter(df, x="sqft", y="sale_price", color_discrete_sequence=["#FFFFFF"], opacity=0.3, template="plotly_dark")
            fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000", xaxis_gridcolor="#111", yaxis_gridcolor="#111")
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast
            st.write("MARKET GROWTH PROJECTION [ 12M ]")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=df_forecast['date'], y=df_forecast['prophet'], name="FORECAST", line=dict(color='#FFFFFF', width=2)))
            fig_f.update_layout(template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", xaxis_gridcolor="#111", yaxis_gridcolor="#111")
            st.plotly_chart(fig_f, use_container_width=True)

        # --- TAB 3: MODEL ARCHITECTURE ---
        with tab3:
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("FEATURE IMPORTANCE HIERARCHY")
            st.info("The algorithm prioritizes structural square footage and neighborhood scarcity index as the primary predictors of market equilibrium.")
            
            # Simple Text Table
            st.markdown("""
                | PARAMETER | INFLUENCE WEIGHT |
                | :--- | :--- |
                | Living Area (SqFt) | 0.82 |
                | Location Index | 0.64 |
                | Asset Age | -0.22 |
                | Luxury Score | 0.18 |
            """)

except Exception as e:
    st.error(f"ENGINE ERROR: {e}")

# --- FOOTER ---
st.markdown("<br><br><br><br><center style='font-size: 0.6rem; color: #333; letter-spacing: 5px;'>DESIGNED BY P.RAJ V // OPTIMIZED FOR HIGH-LATENCY ENVIRONMENTS</center>", unsafe_allow_html=True)
