# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ======================
# Load Data
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("dubai_rent_predictions_with_status.csv")
    return df

df = load_data()

# ======================
# App Layout
# ======================
st.title("🏙️ UAE Rent Prediction Analysis Dashboard")
st.markdown(
    """
    This dashboard provides insights into **rental predictions vs actual rents** across the UAE.
    It highlights **prediction accuracy, over/underpricing trends, community-level insights, 
    and feature correlations** to help both employers and real estate professionals.
    """
)

# ======================
# Key Metrics
# ======================
st.header("📌 Key Stats")
mae = np.mean(np.abs(df["predicted_rent"] - df["actual_rent"]))
mape = np.mean(np.abs(df["%Error"])) * 100
overpriced_pct = (df["status"] == "Overpriced").mean() * 100
underpriced_pct = (df["status"] == "Underpriced").mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean Absolute Error", f"AED {mae:,.0f}")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("Overpriced %", f"{overpriced_pct:.1f}%")
col4.metric("Underpriced %", f"{underpriced_pct:.1f}%")

# ======================
# Predicted vs Actual Scatter
# ======================
st.header("🔍 Predicted vs Actual Rent")
fig_scatter = px.scatter(
    df,
    x="actual_rent",
    y="predicted_rent",
    color="status",
    opacity=0.6,
    labels={"actual_rent": "Actual Rent (AED)", "predicted_rent": "Predicted Rent (AED)"},
    title="Predicted vs Actual Rent with Ideal Fit Line"
)
fig_scatter.add_shape(
    type="line", line=dict(dash="dash", color="black"),
    x0=0, y0=df["actual_rent"].max(), y0=0, x1=df["actual_rent"].max(), y1=df["actual_rent"].max()
)
fig_scatter.update_xaxes(range=[0, 5_000_000])
fig_scatter.update_yaxes(range=[0, 5_000_000])
st.plotly_chart(fig_scatter, use_container_width=True)

# ======================
# Error Distribution
# ======================
st.header("📉 Distribution of Prediction Errors")
fig_err = px.histogram(
    df,
    x="%Error",
    nbins=100,
    title="Distribution of % Error",
    labels={"%Error": "Prediction Error (%)"},
    color_discrete_sequence=["indianred"]
)
fig_err.update_xaxes(tickformat=".0%")
st.plotly_chart(fig_err, use_container_width=True)

# ======================
# Boxplot of Error by Bedrooms
# ======================
st.header("🏠 Over/Underpricing by Bedrooms")
fig_box = px.box(
    df,
    x="bedrooms",
    y="%Error",
    labels={"bedrooms": "Number of Bedrooms", "%Error": "Error %"},
    title="Error Distribution by Bedrooms"
)
fig_box.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_box, use_container_width=True)

# ======================
# Correlation Heatmap
# ======================
st.header("📊 Feature Correlation with Prediction Error")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()["%Error"].sort_values(ascending=False)
fig_corr = px.bar(
    corr.reset_index(),
    x="index", y="%Error",
    title="Feature Correlation with Prediction Error",
    labels={"index": "Feature", "%Error": "Correlation"}
)
st.plotly_chart(fig_corr, use_container_width=True)

# ======================
# Choropleth: Avg Price per Sqft by Community
# ======================
st.header("🌍 Average Price per Sqft Across UAE Communities")
avg_price = df.groupby("community")["Rent_per_sqft"].mean().reset_index()
fig_choro_price = px.choropleth(
    avg_price,
    geojson="uae_geo.json",  # Replace with your UAE GeoJSON
    featureidkey="properties.community",
    locations="community",
    color="Rent_per_sqft",
    color_continuous_scale="Viridis",
    title="Average Price per Sqft by Community (UAE)"
)
fig_choro_price.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_choro_price, use_container_width=True)

# ======================
# Choropleth: Rent Volatility by Community
# ======================
st.header("🌍 Rent Volatility Across UAE Communities")
volatility = df.groupby("community")["actual_rent"].std().reset_index()
fig_choro_vol = px.choropleth(
    volatility,
    geojson="uae_geo.json",
    featureidkey="properties.community",
    locations="community",
    color="actual_rent",
    color_continuous_scale="RdBu",
    title="Standard Deviation of Rent (Volatility) by Community (UAE)"
)
fig_choro_vol.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_choro_vol, use_container_width=True)

# ======================
# Top 10 Overpriced & Underpriced
# ======================
st.header("🏆 Top 10 Most Overpriced & Underpriced Properties")
df["error_abs"] = df["predicted_rent"] - df["actual_rent"]

over_top = df.nlargest(10, "error_abs")
under_top = df.nsmallest(10, "error_abs")

col1, col2 = st.columns(2)

with col1:
    fig_over = px.bar(
        over_top,
        x="community", y="error_abs",
        color="error_abs",
        title="Top 10 Overpriced Properties",
        labels={"error_abs": "Overpricing (AED)"}
    )
    st.plotly_chart(fig_over, use_container_width=True)

with col2:
    fig_under = px.bar(
        under_top,
        x="community", y="error_abs",
        color="error_abs",
        title="Top 10 Underpriced Properties",
        labels={"error_abs": "Underpricing (AED)"}
    )
    st.plotly_chart(fig_under, use_container_width=True)
