# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("dubai_rent_predictions_with_status.csv")

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
rent_min, rent_max = int(df['Rent'].min()), int(df['Rent'].max())
rent_range = st.sidebar.slider("Rent Range", rent_min, rent_max, (rent_min, rent_max), step=1000)

area_min, area_max = int(df['Area_in_sqft'].min()), int(df['Area_in_sqft'].max())
area_range = st.sidebar.slider("Area (sqft) Range", area_min, area_max, (area_min, area_max), step=100)

property_types = df['Type'].unique().tolist()
selected_types = st.sidebar.multiselect("Property Type", property_types, default=property_types)

# Apply filters
df_filtered = df[
    (df['Rent'] >= rent_range[0]) & (df['Rent'] <= rent_range[1]) &
    (df['Area_in_sqft'] >= area_range[0]) & (df['Area_in_sqft'] <= area_range[1]) &
    (df['Type'].isin(selected_types))
]

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Heatmaps", "Scatter Analysis", "Distributions", "Top 10 Insights"])

# -----------------------------
# TAB 1: Heatmaps
# -----------------------------
with tab1:
    st.subheader("Predicted Rent Heatmap")
    fig_rent = px.density_mapbox(
        df_filtered, lat="Latitude", lon="Longitude", z="Predicted_Rent",
        radius=20, center=dict(lat=25.276987, lon=55.296249),
        zoom=10, mapbox_style="carto-positron",
        color_continuous_scale="Viridis",
        title="Predicted Rent Density"
    )
    st.plotly_chart(fig_rent, use_container_width=True)

    st.subheader("Over/Underpricing Heatmap")
    # Normalize Over/Under for color
    norm = (df_filtered['Over_Under'] - df_filtered['Over_Under'].min()) / (df_filtered['Over_Under'].max() - df_filtered['Over_Under'].min())
    df_filtered['OU_Norm'] = norm
    fig_ou = px.density_mapbox(
        df_filtered, lat="Latitude", lon="Longitude", z="OU_Norm",
        radius=20, center=dict(lat=25.276987, lon=55.296249),
        zoom=10, mapbox_style="carto-positron",
        color_continuous_scale="RdBu_r",
        title="Over/Underpricing Intensity (Red=Overpriced, Blue=Underpriced)"
    )
    st.plotly_chart(fig_ou, use_container_width=True)

# -----------------------------
# TAB 2: Scatter Analysis
# -----------------------------
with tab2:
    st.subheader("Rent vs Predicted Rent Scatter Plot")
    fig_scatter = px.scatter(
        df_filtered, x="Rent", y="Predicted_Rent",
        color="Price_Status", hover_data=["Address", "Type", "Area_in_sqft", "%Error"],
        title="Actual vs Predicted Rent by Property",
        trendline="ols"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Scatter without Trendline")
    fig_scatter2 = px.scatter(
        df_filtered, x="Rent", y="Predicted_Rent",
        color="Price_Status", hover_data=["Address", "Type", "Area_in_sqft", "%Error"],
        title="Scatter Plot without Trendline"
    )
    st.plotly_chart(fig_scatter2, use_container_width=True)

# -----------------------------
# TAB 3: Distributions
# -----------------------------
with tab3:
    st.subheader("Distribution of Rents")
    fig_rent_dist = px.histogram(df_filtered, x="Rent", nbins=50, color="Price_Status",
                                 title="Distribution of Actual Rents", marginal="box")
    st.plotly_chart(fig_rent_dist, use_container_width=True)

    st.subheader("Distribution of % Error")
    fig_err = px.histogram(df_filtered, x="%Error", nbins=50, color="Price_Status",
                           title="Distribution of % Error", marginal="box")
    st.plotly_chart(fig_err, use_container_width=True)

# -----------------------------
# TAB 4: Top 10 Over/Underpriced
# -----------------------------
with tab4:
    st.subheader("Top 10 Underpriced Properties")
    top_under = df_filtered.nsmallest(10, "Error")[["Address", "Type", "Area_in_sqft", "Rent", "Predicted_Rent", "Error", "%Error", "Price_Status"]]
    st.dataframe(top_under)

    fig_under_bar = px.bar(top_under, x="Address", y="Error", color="Type", title="Top 10 Underpriced Properties (by Error)")
    st.plotly_chart(fig_under_bar, use_container_width=True)

    st.subheader("Top 10 Overpriced Properties")
    top_over = df_filtered.nlargest(10, "Error")[["Address", "Type", "Area_in_sqft", "Rent", "Predicted_Rent", "Error", "%Error", "Price_Status"]]
    st.dataframe(top_over)

    fig_over_bar = px.bar(top_over, x="Address", y="Error", color="Type", title="Top 10 Overpriced Properties (by Error)")
    st.plotly_chart(fig_over_bar, use_container_width=True)

# -----------------------------
# Summary Stats
# -----------------------------
st.sidebar.header("Summary Stats")
total_props = len(df_filtered)
overpriced = len(df_filtered[df_filtered['Price_Status'] == "Overpriced"])
underpriced = len(df_filtered[df_filtered['Price_Status'] == "Underpriced"])
fair = len(df_filtered[df_filtered['Price_Status'] == "Fair"])

st.sidebar.markdown(f"**Total Listings:** {total_props}")
st.sidebar.markdown(f"**Overpriced:** {overpriced}")
st.sidebar.markdown(f"**Underpriced:** {underpriced}")
st.sidebar.markdown(f"**Fairly Priced:** {fair}")

avg_error = df_filtered["Error"].mean()
avg_pct_error = df_filtered["%Error"].mean()
st.sidebar.markdown(f"**Average Error:** {avg_error:,.0f}")
st.sidebar.markdown(f"**Average % Error:** {avg_pct_error:.2f}%")

# -----------------------------
# Notes:
# -----------------------------
# - Make sure you have 'df_map.csv' in the same folder as this script.
# - Install required packages: streamlit, plotly, pandas, numpy
# - Run: streamlit run streamlit_app.py
# - Interactive filters allow dynamic exploration of the dataset
