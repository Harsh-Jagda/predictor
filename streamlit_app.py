import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"  # Fix inotify issue

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(layout="wide")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("dubai_rent_predictions_with_status.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
types = st.sidebar.multiselect("Property Type", df["Type"].unique())
rent_range = st.sidebar.slider("Rent Range", int(df["Rent"].min()), int(df["Rent"].max()), (int(df["Rent"].min()), int(df["Rent"].max())))
area_range = st.sidebar.slider("Area Size (sqft)", int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max()), (int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max())))

filtered = df.copy()
if types:
    filtered = filtered[filtered["Type"].isin(types)]
filtered = filtered[(filtered["Rent"] >= rent_range[0]) & (filtered["Rent"] <= rent_range[1])]
filtered = filtered[(filtered["Area_in_sqft"] >= area_range[0]) & (filtered["Area_in_sqft"] <= area_range[1])]

# Tabs
tab1, tab2, tab3 = st.tabs(["Map", "Plots", "Insights"])

with tab1:
    st.subheader("Rental Property Map")
    # Create map
    m = folium.Map(location=[filtered["Latitude"].mean(), filtered["Longitude"].mean()], zoom_start=11, tiles="CartoDB positron")
    
    # MarkerCluster
    cluster = MarkerCluster().add_to(m)

    # Add markers
    for _, row in filtered.iterrows():
        color = "green" if row["Price_Status"]=="Fair" else ("red" if row["Price_Status"]=="Overpriced" else "blue")
        popup = f"{row['Address']}<br>Rent: {row['Rent']:,}<br>Predicted: {row['Predicted_Rent']:,}<br>Status: {row['Price_Status']}"
        folium.CircleMarker([row["Latitude"], row["Longitude"]],
                            radius=5, color=color, fill=True, fill_opacity=0.7,
                            popup=popup).add_to(cluster)
    
    # Heatmap
    HeatMap(filtered[['Latitude','Longitude','Abs_Error']].values.tolist(),
            name="Error Heatmap", min_opacity=0.4, radius=12, blur=15).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=1000, height=700)

with tab2:
    st.subheader("Distributions")
    import plotly.express as px
    fig_rent = px.histogram(filtered, x="Rent", nbins=40, title="Rent Distribution")
    fig_area = px.histogram(filtered, x="Area_in_sqft", nbins=40, title="Area Size Distribution")
    st.plotly_chart(fig_rent, use_container_width=True)
    st.plotly_chart(fig_area, use_container_width=True)

with tab3:
    st.subheader("Top Properties")
    st.write("### Top 10 Underpriced")
    st.dataframe(filtered.nsmallest(10,"Error")[["Address","Type","Rent","Predicted_Rent","Error","Price_Status"]].style.format({"Rent":"{:,}","Predicted_Rent":"{:,}","Error":"{:,}"}))
    
    st.write("### Top 10 Overpriced")
    st.dataframe(filtered.nlargest(10,"Error")[["Address","Type","Rent","Predicted_Rent","Error","Price_Status"]].style.format({"Rent":"{:,}","Predicted_Rent":"{:,}","Error":"{:,}"}))



