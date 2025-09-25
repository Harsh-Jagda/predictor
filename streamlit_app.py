# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# streamlit_app.py

# predictor/streamlit_app.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Load Data ---
df_map = pd.read_csv("dubai_rent_predictions_with_status.csv")

# --- Filters ---
st.sidebar.header("Filters")
types = st.sidebar.multiselect("Property Type", df_map["Type"].unique(), default=df_map["Type"].unique())
rent_min, rent_max = st.sidebar.slider("Rent Range", float(df_map["Rent"].min()), float(df_map["Rent"].max()), (0.0, float(df_map["Rent"].max())))
area_min, area_max = st.sidebar.slider("Area (sqft)", float(df_map["Area_in_sqft"].min()), float(df_map["Area_in_sqft"].max()), (0.0, float(df_map["Area_in_sqft"].max())))

df_filtered = df_map[(df_map["Type"].isin(types)) &
                     (df_map["Rent"] >= rent_min) & (df_map["Rent"] <= rent_max) &
                     (df_map["Area_in_sqft"] >= area_min) & (df_map["Area_in_sqft"] <= area_max)]

# --- Calculate Price Status ---
df_filtered["Error"] = df_filtered["Predicted_Rent"] - df_filtered["Rent"]
df_filtered["Error_Percent"] = (df_filtered["Error"] / df_filtered["Rent"]) * 100
def classify_price(err_pct, threshold=5):
    if abs(err_pct) <= threshold:
        return "Fair"
    elif err_pct > threshold:
        return "Overpriced"
    else:
        return "Underpriced"
df_filtered["Price_Status"] = df_filtered["Error_Percent"].apply(lambda x: classify_price(x, threshold=5))

# --- Heatmap: Rent ---
st.header("Heatmap: Predicted Rent")
# Clip extreme values for better visualization
rent_clip = df_filtered["Predicted_Rent"].clip(0, 5000000)
heat_data = df_filtered.copy()
heat_data["Predicted_Rent_Clip"] = rent_clip

fig_rent_heat = px.density_mapbox(
    heat_data,
    lat="Latitude",
    lon="Longitude",
    z="Predicted_Rent_Clip",
    radius=20,
    center=dict(lat=25.276987, lon=55.296249),
    zoom=10,
    mapbox_style="carto-positron",
    color_continuous_scale="Viridis",
    hover_name="Type",
    hover_data=["Address", "Rent", "Predicted_Rent", "Error"]
)
st.plotly_chart(fig_rent_heat, use_container_width=True)

# --- Heatmap: Over/Underpricing ---
st.header("Heatmap: Over/Underpricing")
# Clip values for visualization (-1M to +1M)
ou_clip = df_filtered["Error"].clip(-1000000, 1000000)
heat_data["Error_Clip"] = ou_clip

fig_ou_heat = px.density_mapbox(
    heat_data,
    lat="Latitude",
    lon="Longitude",
    z="Error_Clip",
    radius=20,
    center=dict(lat=25.276987, lon=55.296249),
    zoom=10,
    mapbox_style="carto-positron",
    color_continuous_scale="RdBu_r",
    hover_name="Type",
    hover_data=["Address", "Rent", "Predicted_Rent", "Error"]
)
st.plotly_chart(fig_ou_heat, use_container_width=True)

# --- Distribution: % Error ---
st.header("Distribution of % Error")
fig_err = px.histogram(df_filtered, x="Error_Percent", nbins=50, color="Price_Status",
                       color_discrete_map={"Underpriced":"blue","Fair":"green","Overpriced":"red"},
                       title="Distribution of % Error by Price Status")
fig_err.update_layout(bargap=0.05, xaxis_title="% Error", yaxis_title="Count")
st.plotly_chart(fig_err, use_container_width=True)

# --- Scatter: Rent vs Predicted Rent ---
st.header("Rent vs Predicted Rent")
fig_scatter = px.scatter(df_filtered, x="Rent", y="Predicted_Rent",
                         color="Price_Status",
                         hover_data=["Address", "Type", "Area_in_sqft"],
                         title="Rent vs Predicted Rent")
st.plotly_chart(fig_scatter, use_container_width=True)

# --- Additional Visualizations ---
st.header("Additional Insights")

# 1️⃣ Average Rent per Property Type
st.subheader("Average Rent per Property Type")
avg_rent_type = df_filtered.groupby("Type")["Rent"].mean().reset_index()
fig_avg_type = px.bar(avg_rent_type, x="Type", y="Rent", color="Type", title="Average Rent per Property Type")
st.plotly_chart(fig_avg_type, use_container_width=True)

# 2️⃣ Average %Error per Geo Cluster
st.subheader("Average % Error per Geo Cluster")
avg_err_cluster = df_filtered.groupby("Geo_Cluster")["Error_Percent"].mean().reset_index()
fig_err_cluster = px.bar(avg_err_cluster, x="Geo_Cluster", y="Error_Percent",
                         color="Error_Percent", color_continuous_scale="RdBu_r",
                         title="Average % Error per Geo Cluster")
st.plotly_chart(fig_err_cluster, use_container_width=True)

# 3️⃣ Scatter: Area vs Rent colored by Price Status
st.subheader("Area vs Rent by Price Status")
fig_area_scatter = px.scatter(df_filtered, x="Area_in_sqft", y="Rent", color="Price_Status",
                              hover_data=["Address", "Type", "Predicted_Rent"], title="Area vs Rent by Price Status")
st.plotly_chart(fig_area_scatter, use_container_width=True)

# --- Top 10 Underpriced & Overpriced Properties ---
st.header("Top 10 Underpriced Properties")
top_under = df_filtered.nsmallest(10, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]
st.dataframe(top_under)

st.header("Top 10 Overpriced Properties")
top_over = df_filtered.nlargest(10, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]
st.dataframe(top_over)
