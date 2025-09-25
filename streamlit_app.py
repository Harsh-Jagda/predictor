import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
from streamlit_folium import st_folium

# ------------------------------
# Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dubai_rent_predictions_with_status.csv")
    return df

df = load_data()

st.sidebar.header("Filters")

types = st.sidebar.multiselect("Property Type", options=df["Type"].unique(), default=df["Type"].unique())
rent_range = st.sidebar.slider("Rent Range (AED)", int(df["Rent"].min()), int(df["Rent"].max()),
                               (int(df["Rent"].min()), int(df["Rent"].max())))
area_range = st.sidebar.slider("Area Range (sqft)", int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max()),
                               (int(df["Area_in_sqft"].min()), int(df["Area_in_sqft"].max())))

filtered = df[
    (df["Type"].isin(types)) &
    (df["Rent"].between(rent_range[0], rent_range[1])) &
    (df["Area_in_sqft"].between(area_range[0], area_range[1]))
]

def create_sophisticated_map(df):
    m = folium.Map(location=[df["Latitude"].mean(), df["Longitude"].mean()],
                   zoom_start=11, tiles="cartodbpositron")

    minimap = MiniMap(toggle_display=True)
    m.add_child(minimap)

    marker_cluster = MarkerCluster().add_to(m)
    color_map = {"Fair": "green", "Overpriced": "red", "Underpriced": "blue"}

    for _, row in df.iterrows():
        popup_html = f"""
        <b>{row['Type']} ({row['Furnishing']})</b><br>
         <b>Location:</b> {row['Address']}<br>
         Beds/Baths: {row['Beds']} / {row['Baths']}<br>
         Area: {row['Area_in_sqft']} sqft<br>
         Rent: {row['Rent_fmt']}<br>
         Predicted: {row['Predicted_Rent_fmt']}<br>
         Error: {row['Error_fmt']} ({row['Error_Percent']:.1f}%)<br>
         Status: <b>{row['Price_Status']}</b><br>
         Posted: {row['Posted_date']} ({row['Age_of_listing_in_days']} days ago)
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=max(4, min(12, abs(row['Error']) / 2000)),
            color=color_map.get(row["Price_Status"], "gray"),
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{row['Type']} - {row['Rent_fmt']}",
            popup=folium.Popup(popup_html, max_width=350)
        ).add_to(marker_cluster)

    # Heatmap layers
    HeatMap(df[["Latitude", "Longitude", "Predicted_Rent"]].values.tolist(),
            name="Predicted Rent Heatmap", radius=15, blur=20, max_zoom=12).add_to(m)

    HeatMap(df[["Latitude", "Longitude", "Abs_Error"]].values.tolist(),
            name="Error Heatmap", radius=15, blur=20, max_zoom=12).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding:10px;">
    <b>Price Status Legend</b><br>
    <i style="background:green; width:10px; height:10px; float:left; margin-right:5px;"></i> Fair<br>
    <i style="background:red; width:10px; height:10px; float:left; margin-right:5px;"></i> Overpriced<br>
    <i style="background:blue; width:10px; height:10px; float:left; margin-right:5px;"></i> Underpriced<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

# ------------------------------
# Streamlit Layout
# ------------------------------
st.title("Dubai Rent Predictor Dashboard")
st.markdown("Explore actual vs predicted rents, errors, and market insights.")

tab1, tab2, tab3 = st.tabs(["Map", "Plots", "Insights"])

with tab1:
    st.subheader("Interactive Map")
    m = create_sophisticated_map(filtered)
    st_folium(m, width=900, height=600)

with tab2:
    st.subheader("Data Visualizations")
    fig1 = px.histogram(filtered, x="Error_Percent", nbins=50, title="Distribution of % Error")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(filtered, x="Rent", y="Predicted_Rent", color="Price_Status",
                      title="Actual vs Predicted Rent")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(filtered, x="Type", y="Error_Percent", color="Price_Status",
                  title="Error % by Property Type")
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Top Properties Insights")

    st.markdown("**Top 10 Underpriced Properties**")
    st.dataframe(filtered.nsmallest(10, "Error")[["Address", "Type", "Rent_fmt", "Predicted_Rent_fmt", "Error_fmt", "Price_Status"]])

    st.markdown("**Top 10 Overpriced Properties**")
    st.dataframe(filtered.nlargest(10, "Error")[["Address", "Type", "Rent_fmt", "Predicted_Rent_fmt", "Error_fmt", "Price_Status"]])

