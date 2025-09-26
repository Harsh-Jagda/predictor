# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ===============================
# Load Data
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("dubai_rent_predictions_with_status.csv")
    df["error"] = df["Predicted Rent"] - df["Actual Rent"]
    df["pct_error"] = df["error"] / df["Actual Rent"] * 100
    df["Price per Sqft"] = df["Actual Rent"] / df["Size"]
    return df

df = load_data()

st.set_page_config(page_title="UAE Rent Analysis", layout="wide")
st.title("UAE Rental Market Predictions Dashboard")

# ===============================
# Predicted vs Actual Scatter + Density
# ===============================
st.subheader("Predicted vs Actual Rent Distribution")
fig_scatter = px.scatter(
    df, x="Actual Rent", y="Predicted Rent", color="Status",
    color_discrete_map={"Overpriced":"red","Underpriced":"green","Fair":"blue"},
    hover_data=["Property Type","Bedrooms","Bathrooms","Location","Size"],
    opacity=0.6
)
fig_scatter.add_trace(go.Scatter(
    x=[df["Actual Rent"].min(), df["Actual Rent"].max()],
    y=[df["Actual Rent"].min(), df["Actual Rent"].max()],
    mode="lines", name="Ideal Fit", line=dict(color="black", dash="dash")
))
fig_scatter.update_layout(xaxis_title="Actual Rent (AED)", yaxis_title="Predicted Rent (AED)")
st.plotly_chart(fig_scatter, use_container_width=True)

# ===============================
# Boxplot of Prediction Error
# ===============================
st.subheader("Prediction Error by Property Type")
fig_box = px.box(
    df, x="Property Type", y="pct_error", color="Property Type",
    title="Distribution of % Error Across Property Types"
)
fig_box.update_yaxes(title="% Error (Predicted vs Actual)", zeroline=True)
st.plotly_chart(fig_box, use_container_width=True)

# ===============================
# Top 10 Over/Underpriced Properties
# ===============================
st.subheader("Top 10 Overpriced & Underpriced Properties")

top_over = df.nlargest(10, "pct_error")[["Location","Property Type","Bedrooms","Actual Rent","Predicted Rent","pct_error"]]
top_under = df.nsmallest(10, "pct_error")[["Location","Property Type","Bedrooms","Actual Rent","Predicted Rent","pct_error"]]

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔴 Overpriced")
    st.dataframe(top_over)
    fig_over = px.bar(top_over, x="pct_error", y="Location", orientation="h", color="pct_error", title="Top 10 Overpriced")
    st.plotly_chart(fig_over, use_container_width=True)

with col2:
    st.markdown("### 🟢 Underpriced")
    st.dataframe(top_under)
    fig_under = px.bar(top_under, x="pct_error", y="Location", orientation="h", color="pct_error", title="Top 10 Underpriced")
    st.plotly_chart(fig_under, use_container_width=True)

# ===============================
# Correlation Heatmap (Features vs Error)
# ===============================
st.subheader("Feature Correlation with Prediction Error")
num_cols = df.select_dtypes(include=np.number).drop(columns=["error","pct_error"]).columns
corr_matrix = df[num_cols.to_list() + ["pct_error"]].corr()
fig_corr = px.imshow(
    corr_matrix, text_auto=True, aspect="auto",
    color_continuous_scale="RdBu_r", title="Correlation Heatmap"
)
st.plotly_chart(fig_corr, use_container_width=True)

# ===============================
# Geospatial Analysis
# ===============================
st.subheader("Geospatial Insights")

# Choropleth: Avg Price per Sqft
st.markdown("###Average Price per Sqft by Community")
price_sqft = df.groupby("Location")["Price per Sqft"].mean().reset_index()
fig_map_price = px.choropleth(
    price_sqft, geojson="dubai_communities.geojson", locations="Location",
    featureidkey="properties.name", color="Price per Sqft",
    color_continuous_scale="Viridis", title="Avg Price per Sqft"
)
fig_map_price.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map_price, use_container_width=True)

# Choropleth: Rent Volatility
st.markdown("###Rent Volatility by Community")
rent_vol = df.groupby("Location")["Actual Rent"].std().reset_index().rename(columns={"Actual Rent":"Rent Volatility"})
fig_map_vol = px.choropleth(
    rent_vol, geojson="dubai_communities.geojson", locations="Location",
    featureidkey="properties.name", color="Rent Volatility",
    color_continuous_scale="Reds", title="Rent Volatility (Std Dev)"
)
fig_map_vol.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map_vol, use_container_width=True)
