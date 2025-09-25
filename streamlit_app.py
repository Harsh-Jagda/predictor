import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"  # Fix inotify issue

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(layout="wide", page_title="Dubai Rental Insights")

# --- Load CSV ---
@st.cache_data
def load_data():
    df = pd.read_csv("df_map.csv")  # your CSV with all columns
    # Ensure numeric columns are correct
    numeric_cols = ["Rent", "Predicted_Rent", "Area_in_sqft", "Beds", "Baths", "Error", "Error_Percent"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df_map = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
property_types = st.sidebar.multiselect("Property Type", df_map["Type"].unique(), df_map["Type"].unique())
rent_min, rent_max = st.sidebar.slider("Rent Range", int(df_map["Rent"].min()), int(df_map["Rent"].max()), (int(df_map["Rent"].min()), int(df_map["Rent"].max())))
area_min, area_max = st.sidebar.slider("Area (sqft)", int(df_map["Area_in_sqft"].min()), int(df_map["Area_in_sqft"].max()), (int(df_map["Area_in_sqft"].min()), int(df_map["Area_in_sqft"].max())))

df_filtered = df_map[
    (df_map["Type"].isin(property_types)) &
    (df_map["Rent"] >= rent_min) & (df_map["Rent"] <= rent_max) &
    (df_map["Area_in_sqft"] >= area_min) & (df_map["Area_in_sqft"] <= area_max)
]

# --- Map ---
st.header("Dubai Rental Map")
if df_filtered.empty:
    st.warning("No properties match the filters.")
else:
    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        color="Price_Status",
        size="Abs_Error",
        hover_data=["Address", "Type", "Area_in_sqft", "Rent", "Predicted_Rent", "Error", "Error_Percent"],
        color_discrete_map={"Underpriced":"green","Overpriced":"red","Fair":"blue"},
        zoom=10,
        height=700
    )
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# --- Density Map / Heatmap for Errors ---
st.subheader("Error Density Heatmap")
fig_density = px.density_mapbox(
    df_filtered,
    lat='Latitude',
    lon='Longitude',
    z='Abs_Error',
    radius=20,
    center=dict(lat=25.276987, lon=55.296249),
    zoom=10,
    mapbox_style="open-street-map",
    height=600
)
st.plotly_chart(fig_density, use_container_width=True)

# --- Top/Bottom Properties ---
st.header("Top Property Insights")

top_underpriced = df_filtered.nsmallest(15, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]
top_overpriced = df_filtered.nlargest(15, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 15 Underpriced")
    st.dataframe(top_underpriced.style.format({"Rent": "{:,.0f}", "Predicted_Rent":"{:,.0f}", "Error":"{:,.0f}", "Area_in_sqft":"{:,.0f}"}))

with col2:
    st.subheader("Top 15 Overpriced")
    st.dataframe(top_overpriced.style.format({"Rent": "{:,.0f}", "Predicted_Rent":"{:,.0f}", "Error":"{:,.0f}", "Area_in_sqft":"{:,.0f}"}))

# --- Metrics ---
st.header("Summary Metrics")
avg_error = df_filtered["Error"].mean()
avg_error_pct = df_filtered["Error_Percent"].mean()
st.metric("Average Absolute Error (AED)", f"{avg_error:,.0f}")
st.metric("Average % Error", f"{avg_error_pct:.2f}%")
