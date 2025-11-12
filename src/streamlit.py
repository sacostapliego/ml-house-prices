import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Atlanta Home Price Predictor", page_icon="🏠")

st.title("Atlanta Home Value Predictor")
st.markdown("Predict future home values for Atlanta ZIP codes using machine learning")

# Sidebar inputs
st.sidebar.header("Prediction Settings")

zip_code = st.sidebar.selectbox(
    "Select ZIP Code",
    [30305, 30306, 30307, 30308, 30309, 30310, 30311, 30312, 30313, 30314]
)

months_ahead = st.sidebar.slider("Months to predict ahead", 1, 12, 1)

if st.sidebar.button("Predict", type="primary"):
    # Call API
    with st.spinner("Making prediction..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/simple",
                json={"zip_code": zip_code, "months_ahead": months_ahead}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Current Value",
                        f"${data['current_value']:,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Predicted Value",
                        f"${data['predicted_value']:,.0f}",
                        delta=f"{data['percent_change']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Change",
                        f"${data['change']:,.0f}"
                    )
                
                st.success(f"✓ Prediction for {data['prediction_date']}")
                
                # Show historical trend
                st.subheader("📈 Historical Trend")
                df = pd.read_csv("./data/processed/atl_long.csv")
                df["Date"] = pd.to_datetime(df["Date"])
                zip_data = df[df["RegionName"] == zip_code]
                
                fig = px.line(
                    zip_data,
                    x="Date",
                    y="ZHVI",
                    title=f"Home Value History - ZIP {zip_code}",
                    labels={"ZHVI": "Home Value Index", "Date": "Date"}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error(f"Error: {response.json()['detail']}")
        
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            st.info("Make sure the API server is running: `python src/serve_fastapi.py`")

# Info section
with st.expander("How it works"):
    st.markdown("""
    This model predicts Atlanta home values using:
    - **Historical price trends** (1, 3, 6 months ago)
    - **Rolling averages** (3, 6 months)
    - **Seasonal patterns** (month/year)
    
    The predictions are based on Zillow Home Value Index (ZHVI) data.
    """)