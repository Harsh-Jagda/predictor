# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# ==============================
# streamlit_app.py
# ==============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dubai_rent_predictions_with_status.csv")
    # Ensure numeric columns are proper types
    for col in ["Rent","Predicted_Rent","Error","Error_Percent","Abs_Error","Area_in_sqft","Rent_per_sqft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data()

# ------------------------------
# Filters
# ------------------------------
st.sidebar.header("Filters")
types_selected = st.sidebar.multiselect("Property Type", df["Type"].unique(), default=df["Type"].unique())
rent_min, rent_max = st.sidebar.slider("Rent Range", float(df["Rent"].min()), float(df["Rent"].max()), (0.0, float(df["Rent"].max())))
area_min, area_max = st.sidebar.slider("Area Range (sqft)", int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max()), (0, int(df["Area_in_sqft"].max())))

df_filtered = df[
    (df["Type"].isin(types_selected)) &
    (df["Rent"] >= rent_min) & (df["Rent"] <= rent_max) &
    (df["Area_in_sqft"] >= area_min) & (df["Area_in_sqft"] <= area_max)
]

# ------------------------------
# Title
# ------------------------------
st.title("Dubai Real Estate Rent Insights")
st.markdown("Interactive dashboard with heatmaps, statistics, and over/underpricing analysis.")

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Heatmaps","Distributions","Top Properties","Extra Insights"])

# ------------------------------
# Tab 1: Heatmaps
# ------------------------------
with tabs[0]:
    st.subheader("Predicted Rent Heatmap")
    # Clip predicted rent to 0-5M for better visualization
    df_filtered["Predicted_Rent_Clip"] = df_filtered["Predicted_Rent"].clip(0, 5e6)
    fig_heat_rent = px.density_mapbox(
        df_filtered, lat="Latitude", lon="Longitude", z="Predicted_Rent_Clip",
        radius=25, center=dict(lat=25.276987, lon=55.296249),
        zoom=10, mapbox_style="carto-positron",
        color_continuous_scale="Viridis", hover_data=["Address","Type","Rent","Predicted_Rent","Error"]
    )
    st.plotly_chart(fig_heat_rent, use_container_width=True)

    st.subheader("Over/Underpricing Heatmap")
    # Diverging color scale
    df_filtered["Over_Under_Clip"] = df_filtered["Over_Under"].clip(-5e6, 5e6)
    fig_heat_ou = px.density_mapbox(
        df_filtered, lat="Latitude", lon="Longitude", z="Over_Under_Clip",
        radius=25, center=dict(lat=25.276987, lon=55.296249),
        zoom=10, mapbox_style="carto-positron",
        color_continuous_scale="RdBu_r", hover_data=["Address","Type","Rent","Predicted_Rent","Error","Price_Status"]
    )
    st.plotly_chart(fig_heat_ou, use_container_width=True)

# ------------------------------
# Tab 2: Distributions
# ------------------------------
with tabs[1]:
    st.subheader("% Error Distribution by Property Type")
    fig_err = px.violin(df_filtered, y="Error_Percent", x="Type", color="Price_Status",
                        box=True, points="all", hover_data=["Address","Rent","Predicted_Rent"])
    st.plotly_chart(fig_err, use_container_width=True)

    st.subheader("Rent vs Predicted Rent")
    fig_scatter = px.scatter(df_filtered, x="Rent", y="Predicted_Rent",
                             color="Price_Status",
                             hover_data=["Address","Type","Area_in_sqft","Error","Error_Percent"],
                             labels={"Rent":"Actual Rent","Predicted_Rent":"Predicted Rent"})
    # Add diagonal line
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=df_filtered["Rent"].max(), y1=df_filtered["Rent"].max(),
                          line=dict(color="Black", dash="dash"))
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Area vs Rent colored by Price Status")
    fig_area = px.scatter(df_filtered, x="Area_in_sqft", y="Rent", color="Price_Status",
                          size="Rent_per_sqft", hover_data=["Address","Type","Predicted_Rent","Error"])
    st.plotly_chart(fig_area, use_container_width=True)

# ------------------------------
# Tab 3: Top Properties
# ------------------------------
with tabs[2]:
    st.subheader("Top 10 Underpriced Properties")
    top_under = df_filtered.nsmallest(10, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Error_Percent","Price_Status"]]
    st.dataframe(top_under.style.applymap(lambda x: 'color: blue' if x=="Underpriced" else ''))

    st.subheader("Top 10 Overpriced Properties")
    top_over = df_filtered.nlargest(10, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Error_Percent","Price_Status"]]
    st.dataframe(top_over.style.applymap(lambda x: 'color: red' if x=="Overpriced" else ''))

# ------------------------------
# Tab 4: Extra Insights
# ------------------------------
with tabs[3]:
    st.subheader("Average % Error per Geo Cluster")
    geo_err = df_filtered.groupby("Geo_Cluster")["Error_Percent"].mean().reset_index()
    fig_geo_err = px.bar(geo_err, x="Geo_Cluster", y="Error_Percent", color="Error_Percent",
                         color_continuous_scale="RdBu_r", text="Error_Percent")
    st.plotly_chart(fig_geo_err, use_container_width=True)

    st.subheader("Median Rent vs Area per Type")
    median_rent = df_filtered.groupby("Type")[["Rent","Area_in_sqft"]].median().reset_index()
    fig_median = px.scatter(median_rent, x="Area_in_sqft", y="Rent", color="Type", size="Rent",
                            hover_data=["Type"])
    st.plotly_chart(fig_median, use_container_width=True)

    st.subheader("Price Status Counts per City")
    city_status = df_filtered.groupby(["City","Price_Status"]).size().reset_index(name="Counts")
    fig_city = px.bar(city_status, x="City", y="Counts", color="Price_Status", barmode="stack")
    st.plotly_chart(fig_city, use_container_width=True)

# ------------------------------
# End of App
# ------------------------------
st.markdown("---")
st.markdown("Made for portfolio demonstration: Dubai Real Estate Rent Analysis")

