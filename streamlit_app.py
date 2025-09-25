import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"  # Fix inotify issue

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
tab1, tab2, tab3, tab4 = st.tabs(["📍 Map", "📊 Insights", "📈 Error Analysis", "📌 Aggregations"])

# --- MAP TAB ---
with tab1:
    st.subheader("Property Listings Map")

    fig_map = px.scatter_mapbox(
        df_filtered,
        lat="Latitude",
        lon="Longitude",
        hover_name="Address",
        hover_data={
            "Rent": True,
            "Predicted_Rent": True,
            "Beds": True,
            "Baths": True,
            "Price_Status": True,
        },
        color="Price_Status",
        color_discrete_map={
            "Fair": "green",
            "Overpriced": "red",
            "Underpriced": "blue",
        },
        zoom=10,
        height=600
    )
    fig_map.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True)

# --- INSIGHTS TAB ---
with tab2:
    st.subheader("Market Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rent Distribution by Price Status**")
        fig_hist = px.histogram(
            df_filtered,
            x="Rent",
            color="Price_Status",
            nbins=50,
            barmode="overlay",
            color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        st.write("**Boxplot of Predicted vs Actual Rent**")
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=df_filtered["Rent"], name="Actual Rent"))
        fig_box.add_trace(go.Box(y=df_filtered["Predicted_Rent"], name="Predicted Rent"))
        st.plotly_chart(fig_box, use_container_width=True)

# --- ERROR ANALYSIS TAB ---
with tab3:
    st.subheader("Prediction Error Analysis")

    st.write("**Error Percent Distribution**")
    fig_err = px.histogram(
        df_filtered,
        x="Error_Percent",
        color="Price_Status",
        nbins=50,
        color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
    )
    st.plotly_chart(fig_err, use_container_width=True)

    st.write("**Absolute Error vs Rent**")
    fig_scatter = px.scatter(
        df_filtered,
        x="Rent",
        y="Abs_Error",
        color="Price_Status",
        hover_data=["Address","Predicted_Rent"],
        color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- AGGREGATIONS TAB ---
with tab4:
    st.subheader("Aggregated Market Statistics")

    # City-level avg rent vs predicted rent
    city_stats = df_filtered.groupby("City").agg(
        Avg_Rent=("Rent", "mean"),
        Avg_Predicted=("Predicted_Rent", "mean"),
        Count=("Rent", "count")
    ).reset_index()

    fig_city = go.Figure(data=[
        go.Bar(name="Actual Avg Rent", x=city_stats["City"], y=city_stats["Avg_Rent"]),
        go.Bar(name="Predicted Avg Rent", x=city_stats["City"], y=city_stats["Avg_Predicted"])
    ])
    fig_city.update_layout(barmode="group", title="Average Rent vs Predicted Rent by City")
    st.plotly_chart(fig_city, use_container_width=True)

    # Overpriced/Underpriced breakdown
    status_counts = df_filtered.groupby(["City", "Price_Status"]).size().reset_index(name="Count")
    fig_status = px.bar(
        status_counts,
        x="City",
        y="Count",
        color="Price_Status",
        barmode="stack",
        color_discrete_map={"Fair":"green","Overpriced":"red","Underpriced":"blue"}
    )
    fig_status.update_layout(title="Overpriced vs Underpriced Listings by City")
    st.plotly_chart(fig_status, use_container_width=True)

    # Percentages
    st.write("**Summary Table (City-level % Breakdown)**")
    total_counts = df_filtered.groupby("City")["Price_Status"].count().reset_index(name="Total")
    merged = status_counts.merge(total_counts, on="City")
    merged["Percent"] = (merged["Count"] / merged["Total"]) * 100
    st.dataframe(merged.pivot(index="City", columns="Price_Status", values="Percent").round(2))

