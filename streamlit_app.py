# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("dubai_rent_predictions_with_status.csv")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
city_filter = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
status_filter = st.sidebar.multiselect("Select Price Status", options=df["Price_Status"].unique(), default=df["Price_Status"].unique())

df_filtered = df[(df["City"].isin(city_filter)) & (df["Price_Status"].isin(status_filter))]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📋 Property Lookup", "📈 Error Analysis", "📊 Market Insights"])

# --- HEATMAP TAB ---
with tab1:
    st.subheader("Price Status Heatmap")

    fig_heat = px.density_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        z=None,
        radius=15,
        hover_name="City",
        hover_data=["Price_Status"],
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        height=600
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# --- PROPERTY LOOKUP TAB ---
with tab2:
    st.subheader("Search & Explore Properties")

    min_rent, max_rent = st.slider("Rent Range", int(df["Rent"].min()), int(df["Rent"].max()), (int(df["Rent"].min()), int(df["Rent"].max())))
    df_lookup = df_filtered[(df_filtered["Rent"] >= min_rent) & (df_filtered["Rent"] <= max_rent)]

    st.dataframe(
        df_lookup[["Address","City","Type","Beds","Baths","Area_in_sqft","Rent","Predicted_Rent","Price_Status"]]
        .sort_values(by="Rent", ascending=False),
        use_container_width=True
    )

# --- ERROR ANALYSIS TAB ---
with tab3:
    st.subheader("Prediction Error Analysis")

    st.write("**Distribution of Percentage Error**")
    fig_err = px.histogram(
        df_filtered,
        x="Error_Percent",
        color="Price_Status",
        nbins=50,
        color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
    )
    fig_err.update_layout(bargap=0.05)
    st.plotly_chart(fig_err, use_container_width=True)

    st.write("**Absolute Error vs Actual Rent**")
    fig_scatter = px.scatter(
        df_filtered,
        x="Rent",
        y="Abs_Error",
        color="Price_Status",
        hover_data=["Address","Predicted_Rent","City"],
        color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- MARKET INSIGHTS TAB ---
with tab4:
    st.subheader("Market-Level Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Overpriced vs Underpriced Breakdown**")
        status_counts = df_filtered["Price_Status"].value_counts().reset_index()
        status_counts.columns = ["Price_Status","Count"]
        fig_pie = px.pie(status_counts, values="Count", names="Price_Status",
                         color="Price_Status",
                         color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.write("**Average Rent vs Predicted Rent by City**")
        city_stats = df_filtered.groupby("City").agg(
            Avg_Rent=("Rent","mean"),
            Avg_Predicted=("Predicted_Rent","mean")
        ).reset_index()
        fig_city = go.Figure(data=[
            go.Bar(name="Actual Avg Rent", x=city_stats["City"], y=city_stats["Avg_Rent"]),
            go.Bar(name="Predicted Avg Rent", x=city_stats["City"], y=city_stats["Avg_Predicted"])
        ])
        fig_city.update_layout(barmode="group", height=400)
        st.plotly_chart(fig_city, use_container_width=True)

    st.write("**Detailed Breakdown Table**")
    summary = df_filtered.groupby(["City","Price_Status"]).size().reset_index(name="Count")
    total = df_filtered.groupby("City").size().reset_index(name="Total")
    merged = summary.merge(total, on="City")
    merged["Percent"] = (merged["Count"]/merged["Total"]*100).round(2)
    st.dataframe(merged.pivot(index="City", columns="Price_Status", values="Percent").fillna(0))


