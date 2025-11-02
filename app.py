import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(page_title="FedEx ReturnIQ", layout="wide")

# ======================
# FEDEx BRANDING & CSS
# ======================
# st.markdown("""
# <style>
# /* --- Top Bar --- */
# .stApp header {visibility: hidden;}
# .top-bar {
#     background-color: #4D148C;
#     padding: 16px 25px;
#     font-size: 26px;
#     font-weight: 900;
#     color: white;
#     border-bottom: 3px solid #FF6600;
# }
# .fedex-orange {color: #FF6600;}
# .top-subtitle {
#     font-size: 14px;
#     color: #FFE0B2;
#     margin-left: 8px;
# }

# /* --- Sidebar Styling --- */
# section[data-testid="stSidebar"] {
#     background-color: #F5F5F5 !important;
#     border-right: 2px solid #E0E0E0;
#     padding: 20px 15px !important;
# }
# section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label {
#     color: #000 !important;
#     font-weight: 600 !important;
# }
# .stSidebar .stButton>button {
#     background-color: #4D148C;
#     color: white;
#     border-radius: 8px;
#     border: none;
#     font-weight: 600;
# }
# .stSidebar .stButton>button:hover {
#     background-color: #FF6600;
#     color: white;
# }

# /* --- General UI & Predictors --- */
# body, .stApp {
#     background-color: #FFFFFF !important;
#     color: #222 !important;
# }
# h1, h2, h3 {
#     color: #4D148C !important;
#     font-weight: 800 !important;
# }
# hr {
#     border: 2px solid #4D148C;
#     margin-top: 10px;
#     margin-bottom: 10px;
# }
# .metric-label {
#     color: #4D148C;
#     font-weight: 700;
# }
# </style>

# <div class="top-bar">
#   Fed<span class="fedex-orange">Ex</span>
#   <span class="top-subtitle">| Global Returns Intelligence</span>
# </div>
# """, unsafe_allow_html=True)

# ======================
# LOAD MODEL AND DATA
# ======================
@st.cache_data
def load_assets():
    return pd.read_csv("data/fedex_rates.csv"), joblib.load("models/returniq_model.pkl")

rates, model = load_assets()
import streamlit as st

st.header(body="Global Returns Intelligence", divider="orange")


# st.title(" Global Returns Intelligence")

st.logo("images/logo.svg",size="large")

# ======================
# SIDEBAR (GLOBAL CONTEXT)
# ======================
st.sidebar.markdown('**Return Context**')

country_to_zone = {
    "India": "A", "United States": "B", "France": "C", "Australia": "D",
    "Japan": "E", "United Kingdom": "F", "Russia": "G", "Brazil": "H"
}
countries = list(country_to_zone.keys())

origin_country = st.sidebar.selectbox("Origin (Customer Country)", countries)
destination_country = st.sidebar.selectbox("Destination (Merchant Country)", countries, index=1)
tariff_rate = st.sidebar.slider("Import Tariff Rate", 0.0, 0.3, 0.05)
condition_score = st.sidebar.slider("Condition Score", 0.3, 1.0, 0.8)
customs_delay_days = st.sidebar.slider("Customs Delay (days)", 0, 15, 3)

st.sidebar.markdown("---")
st.sidebar.info("Set product parameters in the main area and click **Estimate Return Feasibility**.")

# ======================
# MAIN CONTENT
# ======================
st.subheader("Product Parameters")
# st.markdown("<hr>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    product_price = st.number_input("Product Price (₹)", 100.0, 50000000000000.0, 2500.0)
with col2:
    weight_kg = st.number_input("Weight (kg)", 0.1, 30.0, 2.5)
with col3:
    distance_km = st.number_input("Approx. Distance (km)", 50.0, 20000.0, 5000.0)

col4, col5 = st.columns(2)
with col4:
    fragile = st.checkbox("Fragile item", value=False)
with col5:
    valuable = st.checkbox("High-value item", value=False)

st.markdown("<hr>", unsafe_allow_html=True)

if "predict" not in st.session_state:
    st.session_state.predict = False

if st.button("🚀 Estimate Return Feasibility"):
    st.session_state.predict = True

# ======================
# MODEL INFERENCE
# ======================
if st.session_state.predict:
    origin_zone = country_to_zone[origin_country]
    dest_zone = country_to_zone[destination_country]

    match = rates[(rates.origin_zone == origin_zone) & (rates.destination_zone == dest_zone)]
    if match.empty:
        base_cost, perkg, fragile_fee, val_pct = 1000, 200, 150, 0.02
    else:
        r = match.iloc[0]
        base_cost, perkg, fragile_fee, val_pct = (
            r.base_cost_inr, r.per_kg_cost_inr, r.fragile_surcharge_inr, r.valuable_surcharge_percent
        )

    shipping_cost = base_cost + perkg * weight_kg
    if fragile:
        shipping_cost += fragile_fee
    if valuable:
        shipping_cost += val_pct * product_price
    if origin_zone == dest_zone:
        shipping_cost *= 0.5

    co2_kg = 0.5 * (weight_kg / 1000.0) * distance_km
    resale_value = product_price * 0.6 * condition_score
    profit = resale_value - (shipping_cost + tariff_rate * product_price + 100)
    is_domestic = 1 if origin_zone == dest_zone else 0

    features = {
        "return_cost_ratio": shipping_cost / product_price,
        "resale_ratio": resale_value / product_price,
        "condition_score": condition_score,
        "customs_delay_days": customs_delay_days,
        "co2_kg": co2_kg,
        "distance_km": distance_km,
        "fragile": int(fragile),
        "valuable": int(valuable),
        "is_domestic": is_domestic,
        "cvar_rel": 0.1,
        "e_profit_rel": profit / product_price,
    }

    X = pd.DataFrame([features])
    pred = int(model.predict(X)[0])
    prob = model.predict_proba(X)[0]
    conf = float(max(prob)) * 100

    decision_map = {1: "✅ Accept Return", 0: "💸 Refund Without Return", -1: "❌ Reject Return"}
    decision = decision_map.get(pred, "Unknown")

    # ----- Results -----
    st.subheader("Prediction Result")
    st.markdown(f"<h3 style='text-align:center; color:#4D148C; font-weight:800;'>{decision}</h3>", unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)
    colA.metric("Shipping Cost (₹)", f"{shipping_cost:,.0f}")
    colB.metric("Estimated Profit (₹)", f"{profit:,.0f}")
    colC.metric("CO₂ Emissions (kg)", f"{co2_kg:.2f}")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf,
        number={'suffix': "%", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#FF6600"},
            'steps': [
                {'range': [0, 50], 'color': "#FFF3E0"},
                {'range': [50, 80], 'color': "#FFE0B2"},
                {'range': [80, 100], 'color': "#FFD180"}
            ]
        },
        title={'text': "Model Confidence", 'font': {'size': 16, 'color': '#333'}}
    ))
    fig.update_layout(height=180, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # Route Map
    st.subheader("Shipment Route Visualization")
    zone_to_coords = {
        "A": (28.6, 77.2), "B": (40.7, -74.0), "C": (48.8, 2.35),
        "D": (-33.8, 151.2), "E": (35.6, 139.7), "F": (51.5, -0.12),
        "G": (55.7, 37.6), "H": (-23.5, -46.6)
    }
    m = folium.Map(location=[
        (zone_to_coords[origin_zone][0] + zone_to_coords[dest_zone][0]) / 2,
        (zone_to_coords[origin_zone][1] + zone_to_coords[dest_zone][1]) / 2
    ], zoom_start=3)
    folium.Marker(zone_to_coords[origin_zone], popup=origin_country,
                  icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(zone_to_coords[dest_zone], popup=destination_country,
                  icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine([zone_to_coords[origin_zone], zone_to_coords[dest_zone]],
                    color="#4D148C", weight=3).add_to(m)
    st_folium(m, width=750, height=350)

    st.markdown("---")
    if pred in [0, -1]:
        st.warning("⚠️ High return cost detected — consider resale or thrift alternatives.")
        st.info("💡 Suggestion: Sell locally or offer store credit.")
    else:
        st.success("✅ Return is profitable and sustainable.")

    st.caption("Prototype by Siddhant Dange — FedEx Global Returns Hackathon 🚀")
