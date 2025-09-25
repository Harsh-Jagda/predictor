import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"  # Fix inotify issue

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster, FeatureGroupSubGroup
from streamlit_folium import st_folium

# --- Load data ---
@st.cache_data
def load_data():
    return pd.read_csv("dubai_rent_predictions_with_status.csv")

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filters")
property_type = st.sidebar.multiselect("Property Type", df["Type"].unique())
min_rent, max_rent = st.sidebar.slider("Rent Range", int(df["Rent"].min()), int(df["Rent"].max()), (int(df["Rent"].min()), int(df["Rent"].max())))
min_area, max_area = st.sidebar.slider("Area Size (sqft)", int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max()), (int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max())))

filtered_df = df.copy()
if property_type:
    filtered_df = filtered_df[filtered_df["Type"].isin(property_type)]
filtered_df = filtered_df[(filtered_df["Rent"] >= min_rent) & (filtered_df["Rent"] <= max_rent)]
filtered_df = filtered_df[(filtered_df["Area_in_sqft"] >= min_area) & (filtered_df["Area_in_sqft"] <= max_area)]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📍 Map", "📊 Plots", "📈 Insights"])

with tab1:
    st.subheader("Property Map")
    m = folium.Map(location=[filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()], zoom_start=11, tiles="CartoDB positron")

    # --- Marker Clusters by Price Status ---
    cluster = MarkerCluster(name="All Properties").add_to(m)

    # Separate feature groups for Fair, Overpriced, Underpriced
    fg_fair = folium.FeatureGroup(name="Fair Deals").add_to(m)
    fg_over = folium.FeatureGroup(name="Overpriced").add_to(m)
    fg_under = folium.FeatureGroup(name="Underpriced").add_to(m)

    for _, row in filtered_df.iterrows():
        popup = f"""
        <b>Address:</b> {row['Address']}<br>
        <b>Type:</b> {row['Type']}<br>
        <b>Area (sqft):</b> {row['Area_in_sqft']}<br>
        <b>Rent:</b> {row['Rent_fmt']}<br>
        <b>Predicted:</b> {row['Predicted_Rent_fmt']}<br>
        <b>Status:</b> {row['Price_Status']}
        """
        color = "green" if row["Price_Status"] == "Fair" else ("red" if row["Price_Status"] == "Overpriced" else "blue")
        marker = folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=6,
            popup=popup,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        )
        cluster.add_child(marker)
        if row["Price_Status"] == "Fair":
            fg_fair.add_child(marker)
        elif row["Price_Status"] == "Overpriced":
            fg_over.add_child(marker)
        else:
            fg_under.add_child(marker)

    # --- Heatmaps ---
    HeatMap(
        filtered_df[['Latitude', 'Longitude', 'Predicted_Rent']].values.tolist(),
        name="Rent Heatmap",
        min_opacity=0.4, radius=12, blur=18, max_zoom=15
    ).add_to(m)

    HeatMap(
        filtered_df[['Latitude', 'Longitude', 'Abs_Error']].values.tolist(),
        name="Error Heatmap",
        min_opacity=0.4, radius=12, blur=18, max_zoom=15
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1000, height=700)

with tab2:
    st.subheader("Distribution Plots")
    import plotly.express as px
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(filtered_df, x="Rent", nbins=40, title="Rent Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(filtered_df, x="Area_in_sqft", nbins=40, title="Area Size Distribution")
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Top Properties & Insights")
    st.write("### Top 15 Underpriced Properties")
    underpriced = filtered_df.nsmallest(15, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]
    st.dataframe(underpriced.style.format({"Rent":"{:,}","Predicted_Rent":"{:,}","Error":"{:,}"}))

    st.write("### Top 15 Overpriced Properties")
    overpriced = filtered_df.nlargest(15, "Error")[["Address","Type","Area_in_sqft","Rent","Predicted_Rent","Error","Price_Status"]]
    st.dataframe(overpriced.style.format({"Rent":"{:,}","Predicted_Rent":"{:,}","Error":"{:,}"}))

    st.metric("Average Error (%)", f"{filtered_df['Error_Percent'].mean():.2f}%")
    st.metric("Fair Deals", (filtered_df['Price_Status'] == "Fair").sum())
    st.metric("Overpriced", (filtered_df['Price_Status'] == "Overpriced").sum())
    st.metric("Underpriced", (filtered_df['Price_Status'] == "Underpriced").sum())
