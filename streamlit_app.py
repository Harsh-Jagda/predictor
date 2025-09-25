# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(layout="wide", page_title="Dubai Rent Insights (Plotly)")

# -----------------------
# Data loader (robust)
# -----------------------
@st.cache_data
df = pd.read_csv("dubai_rent_predictions_with_status.csv")
if df.empty:
    st.stop()

# -----------------------
# Ensure required columns & types
# -----------------------
# Normalize column names (common variants)
col_map = {}
for c in df.columns:
    lc = c.lower()
    if lc in ["lat", "latitude"]:
        col_map[c] = "Latitude"
    if lc in ["lon", "lng", "longitude"]:
        col_map[c] = "Longitude"
    if lc == "predicted_rent" or lc == "predictedrent":
        col_map[c] = "Predicted_Rent"
    if lc in ["error_percent", "%error", "error%"]:
        col_map[c] = "Error_Percent"
    if lc in ["error", "err"]:
        col_map[c] = "Error"
    if lc in ["abs_error","abserror"]:
        col_map[c] = "Abs_Error"
    if lc in ["price_status","status"]:
        col_map[c] = "Price_Status"
    if lc in ["type","property_type"]:
        col_map[c] = "Type"
    if lc in ["rent","price","monthly_rent"]:
        col_map[c] = "Rent"
    if lc in ["area_in_sqft","area","area_sqft","area_in_sqft"]:
        col_map[c] = "Area_in_sqft"
    if lc in ["city"]:
        col_map[c] = "City"
    if lc in ["geo_cluster","cluster","geocluster"]:
        col_map[c] = "Geo_Cluster"
    if lc in ["address"]:
        col_map[c] = "Address"
    if lc in ["beds","bedrooms"]:
        col_map[c] = "Beds"
    if lc in ["baths","bathrooms"]:
        col_map[c] = "Baths"
    if lc in ["rent_fmt","rent_formatted"]:
        col_map[c] = "Rent_fmt"
    if lc in ["predicted_rent_fmt","predicted_fmt"]:
        col_map[c] = "Predicted_Rent_fmt"

df = df.rename(columns=col_map)

# Ensure numeric types
for num_col in ["Rent", "Predicted_Rent", "Area_in_sqft", "Error", "Error_Percent", "Abs_Error", "Latitude", "Longitude"]:
    if num_col in df.columns:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

# Some fallback columns / formatting if missing
if "Abs_Error" not in df.columns and "Error" in df.columns:
    df["Abs_Error"] = df["Error"].abs()
if "Error_Percent" not in df.columns and ("Error" in df.columns and "Rent" in df.columns):
    df["Error_Percent"] = (df["Error"] / df["Rent"]) * 100

# Provide nicely formatted rent strings if not present
if "Rent_fmt" not in df.columns and "Rent" in df.columns:
    df["Rent_fmt"] = df["Rent"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")
if "Predicted_Rent_fmt" not in df.columns and "Predicted_Rent" in df.columns:
    df["Predicted_Rent_fmt"] = df["Predicted_Rent"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "")

# Fill missing Price_Status if not present (use 5% threshold)
if "Price_Status" not in df.columns and "Error_Percent" in df.columns:
    def classify(err_pct, t=5):
        if pd.isna(err_pct):
            return "Unknown"
        if abs(err_pct) <= t:
            return "Fair"
        return "Overpriced" if err_pct > t else "Underpriced"
    df["Price_Status"] = df["Error_Percent"].apply(classify)

# If Geo_Cluster missing, create coarse geohash-like clusters using rounding
if "Geo_Cluster" not in df.columns:
    df["Geo_Cluster"] = (df["Latitude"].round(3).astype(str) + "_" + df["Longitude"].round(3).astype(str)).astype("category").cat.codes

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.title("Controls")
# sampling toggle (for performance)
sample_opt = st.sidebar.checkbox("Use sampling for map (faster)", value=False)
if sample_opt:
    sample_size = st.sidebar.slider("Sample size (map only)", min_value=100, max_value=5000, value=2000, step=100)
else:
    sample_size = None

type_choices = df["Type"].dropna().unique().tolist() if "Type" in df.columns else []
city_choices = df["City"].dropna().unique().tolist() if "City" in df.columns else []
status_choices = df["Price_Status"].dropna().unique().tolist()

sel_types = st.sidebar.multiselect("Property Type", options=type_choices, default=type_choices)
sel_cities = st.sidebar.multiselect("City", options=city_choices, default=city_choices if city_choices else None)
sel_status = st.sidebar.multiselect("Price Status", options=status_choices, default=status_choices)

rent_min = int(df["Rent"].min(skipna=True)) if "Rent" in df.columns else 0
rent_max = int(df["Rent"].max(skipna=True)) if "Rent" in df.columns else 1
sel_rent = st.sidebar.slider("Rent Range (AED)", rent_min, rent_max, (rent_min, rent_max))

area_min = int(df["Area_in_sqft"].min(skipna=True)) if "Area_in_sqft" in df.columns else 0
area_max = int(df["Area_in_sqft"].max(skipna=True)) if "Area_in_sqft" in df.columns else 1
sel_area = st.sidebar.slider("Area Range (sqft)", area_min, area_max, (area_min, area_max))

# map view mode: listings or cluster aggregates
view_mode = st.sidebar.radio("Map View", options=["Individual Listings", "Cluster Aggregates"])

# error% threshold filter
error_slider = st.sidebar.slider("Min |Error %| to display (map markers)", 0.0, 100.0, 0.0, step=0.5)

# -----------------------
# Apply filters
# -----------------------
masked = df.copy()
if sel_types:
    masked = masked[masked["Type"].isin(sel_types)]
if sel_cities and len(sel_cities) > 0:
    masked = masked[masked["City"].isin(sel_cities)]
if sel_status:
    masked = masked[masked["Price_Status"].isin(sel_status)]

if "Rent" in masked.columns:
    masked = masked[(masked["Rent"] >= sel_rent[0]) & (masked["Rent"] <= sel_rent[1])]
if "Area_in_sqft" in masked.columns:
    masked = masked[(masked["Area_in_sqft"] >= sel_area[0]) & (masked["Area_in_sqft"] <= sel_area[1])]
if "Error_Percent" in masked.columns:
    masked = masked[masked[masked["Error_Percent"].abs() >= error_slider].index]  # filter by abs(Error%)

if masked.empty:
    st.warning("No data after applying filters.")
    st.stop()

# -----------------------
# Utility functions / metrics
# -----------------------
def compute_metrics(df_):
    metrics = {}
    if "Error" in df_.columns:
        metrics["RMSE"] = np.sqrt(mean_squared_error(df_["Rent"], df_["Predicted_Rent"])) if ("Rent" in df_.columns and "Predicted_Rent" in df_.columns) else np.nan
        metrics["MAE"] = mean_absolute_error(df_["Rent"], df_["Predicted_Rent"]) if ("Rent" in df_.columns and "Predicted_Rent" in df_.columns) else np.nan
        metrics["Mean_Error_AED"] = df_["Error"].mean()
        metrics["Mean_Abs_Error_AED"] = df_["Abs_Error"].mean() if "Abs_Error" in df_.columns else np.nan
        metrics["Mean_Error_%"] = df_["Error_Percent"].mean() if "Error_Percent" in df_.columns else np.nan
    return metrics

global_metrics = compute_metrics(masked)

# -----------------------
# Layout / Tabs
# -----------------------
st.title("🏙️ Dubai Rent Predictor — Interactive Dashboard (Plotly)")
st.markdown("Interactive map + aggregations + error analysis. Toggle view to 'Cluster Aggregates' to see geocluster summaries.")

tab_map, tab_plots, tab_aggregations, tab_insights = st.tabs(["📍 Map", "📊 Plots", "📌 Aggregations", "📈 Over/Underpricing Stats"])

# -----------------------
# MAP TAB
# -----------------------
with tab_map:
    st.subheader("Map — choose view and interact with filters")
    if view_mode == "Individual Listings":
        map_df = masked.copy()
        if sample_opt and sample_size is not None and len(map_df) > sample_size:
            map_df = map_df.sample(sample_size, random_state=42)

        # size scaling for scatter markers (Plotly size) - scale by Abs_Error but cap and ensure min size
        if "Abs_Error" in map_df.columns:
            size_col = map_df["Abs_Error"].fillna(0)
            # normalize to 4-20
            sizes = np.interp(size_col, (size_col.min(), size_col.max()), (4, 20))
        else:
            sizes = 6

        fig_map = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Price_Status",
            size=size_col if "Abs_Error" in map_df.columns else None,
            size_max=20,
            hover_name="Address" if "Address" in map_df.columns else None,
            hover_data={
                "Type": True,
                "Beds": True if "Beds" in map_df.columns else False,
                "Baths": True if "Baths" in map_df.columns else False,
                "Area_in_sqft": True if "Area_in_sqft" in map_df.columns else False,
                "Rent": True if "Rent" in map_df.columns else False,
                "Predicted_Rent": True if "Predicted_Rent" in map_df.columns else False,
                "Error": True if "Error" in map_df.columns else False,
                "Error_Percent": True if "Error_Percent" in map_df.columns else False,
            },
            color_discrete_map={"Underpriced":"green","Overpriced":"red","Fair":"blue"},
            zoom=10,
            height=700
        )
        fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_map, use_container_width=True)

        # density / heat-like visualization (error intensity)
        if "Abs_Error" in masked.columns:
            st.subheader("Error Density (heat)")
            fig_density = px.density_mapbox(
                masked,
                lat="Latitude",
                lon="Longitude",
                z="Abs_Error",
                radius=20,
                center=dict(lat=25.276987, lon=55.296249),
                zoom=10,
                mapbox_style="carto-positron",
                height=500
            )
            st.plotly_chart(fig_density, use_container_width=True)

    else:
        # Cluster Aggregates: group by Geo_Cluster and show centroids + stats
        st.write("Showing cluster aggregates (grouped by `Geo_Cluster`).")
        agg = masked.groupby("Geo_Cluster").agg(
            Count=("Rent", "count"),
            Avg_Rent=("Rent", "mean"),
            Avg_Predicted=("Predicted_Rent", "mean"),
            Avg_Error=("Error", "mean"),
            Avg_Abs_Error=("Abs_Error", "mean"),
            Lat=("Latitude", "mean"),
            Lon=("Longitude", "mean"),
            Overpriced_Count=("Price_Status", lambda s: (s=="Overpriced").sum()),
            Underpriced_Count=("Price_Status", lambda s: (s=="Underpriced").sum()),
            Fair_Count=("Price_Status", lambda s: (s=="Fair").sum())
        ).reset_index()

        # cluster color by Avg_Error sign and magnitude
        agg["Color"] = np.where(agg["Avg_Error"]>0, "red", np.where(agg["Avg_Error"]<0, "green", "blue"))
        # size by Count (scale)
        sizes = np.interp(agg["Count"], (agg["Count"].min(), agg["Count"].max()), (8, 35))

        fig_agg = px.scatter_mapbox(
            agg,
            lat="Lat",
            lon="Lon",
            size="Count",
            size_max=40,
            color="Avg_Error",
            color_continuous_scale=px.colors.diverging.RdYlGn[::-1],  # red = positive error, green = negative
            hover_data=["Count","Avg_Rent","Avg_Predicted","Avg_Error","Avg_Abs_Error","Overpriced_Count","Underpriced_Count","Fair_Count"],
            center=dict(lat=25.276987, lon=55.296249),
            zoom=9,
            height=700
        )
        fig_agg.update_layout(mapbox_style="carto-positron", margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_agg, use_container_width=True)

# -----------------------
# PLOTS TAB (improved visuals)
# -----------------------
with tab_plots:
    st.subheader("Distributions & Comparisons")

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("**Rent distribution by Price Status**")
        fig_rent = px.histogram(
            masked,
            x="Rent",
            color="Price_Status",
            nbins=60,
            marginal="box",
            color_discrete_map={"Fair":"blue","Overpriced":"red","Underpriced":"green"}
        )
        fig_rent.update_layout(barmode="overlay")
        st.plotly_chart(fig_rent, use_container_width=True)

        st.markdown("**Predicted vs Actual (scatter)**")
        fig_sc = px.scatter(masked, x="Predicted_Rent", y="Rent", color="Price_Status",
                            hover_data=["Address"], color_discrete_map={"Fair":"blue","Overpriced":"red","Underpriced":"green"})
        fig_sc.add_shape(type="line", x0=masked["Predicted_Rent"].min(), x1=masked["Predicted_Rent"].max(),
                         y0=masked["Predicted_Rent"].min(), y1=masked["Predicted_Rent"].max(),
                         line=dict(dash="dash"))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        st.markdown("**Error % by Property Type (boxplot)**")
        if "Type" in masked.columns:
            fig_box = px.box(masked, x="Type", y="Error_Percent", color="Type", points="outliers")
            fig_box.update_layout(showlegend=False, height=700)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No 'Type' column available for the boxplot.")

# -----------------------
# AGGREGATIONS TAB (city + cluster-level)
# -----------------------
with tab_aggregations:
    st.subheader("Aggregations: City and Geo_Cluster")

    # City-level
    if "City" in masked.columns:
        city_stats = masked.groupby("City").agg(
            Avg_Rent=("Rent","mean"),
            Avg_Pred=("Predicted_Rent","mean"),
            Count=("Rent","count"),
            Mean_Error_Percent=("Error_Percent","mean")
        ).reset_index().sort_values("Avg_Rent", ascending=False)

        st.markdown("**Average Rent vs Predicted by City**")
        fig_city = go.Figure(data=[
            go.Bar(name="Actual Avg Rent", x=city_stats["City"], y=city_stats["Avg_Rent"]),
            go.Bar(name="Predicted Avg Rent", x=city_stats["City"], y=city_stats["Avg_Pred"])
        ])
        fig_city.update_layout(barmode="group", xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_city, use_container_width=True)

        st.markdown("**Price Status breakdown (counts) by City**")
        status_counts = masked.groupby(["City","Price_Status"]).size().reset_index(name="Count")
        fig_status = px.bar(status_counts, x="City", y="Count", color="Price_Status", barmode="stack",
                            color_discrete_map={"Fair":"blue","Overpriced":"red","Underpriced":"green"})
        st.plotly_chart(fig_status, use_container_width=True)

    # Cluster-level summary table (top clusters by count)
    st.markdown("**Top Geo_Clusters (by listing count)**")
    cluster_tbl = masked.groupby("Geo_Cluster").agg(
        Count=("Rent","count"),
        Avg_Rent=("Rent","mean"),
        Avg_Pred=("Predicted_Rent","mean"),
        Avg_Error=("Error","mean"),
        Avg_Abs_Error=("Abs_Error","mean")
    ).reset_index().sort_values("Count", ascending=False).head(15)
    st.dataframe(cluster_tbl.style.format({"Avg_Rent":"{:.0f}","Avg_Pred":"{:.0f}","Avg_Error":"{:.0f}","Avg_Abs_Error":"{:.0f}"}))

# -----------------------
# INSIGHTS TAB (over/under stats & KPIs)
# -----------------------
with tab_insights:
    st.subheader("Overpricing / Underpricing Statistics & KPIs")

    total = len(masked)
    cnt_over = (masked["Price_Status"] == "Overpriced").sum()
    cnt_under = (masked["Price_Status"] == "Underpriced").sum()
    cnt_fair = (masked["Price_Status"] == "Fair").sum()
    pct_over = cnt_over / total * 100
    pct_under = cnt_under / total * 100
    pct_fair = cnt_fair / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total listings (filtered)", total)
    k2.metric("Overpriced (count)", f"{cnt_over:,}", f"{pct_over:.1f}%")
    k3.metric("Underpriced (count)", f"{cnt_under:,}", f"{pct_under:.1f}%")
    k4.metric("Fair (count)", f"{cnt_fair:,}", f"{pct_fair:.1f}%")

    # Error metrics
    st.markdown("### Error metrics (filtered subset)")
    st.write(f"RMSE: {global_metrics.get('RMSE', np.nan):,.0f} AED")
    st.write(f"MAE: {global_metrics.get('MAE', np.nan):,.0f} AED")
    st.write(f"Mean Abs Error: {global_metrics.get('Mean_Abs_Error_AED', np.nan):,.0f} AED")
    st.write(f"Mean Error %: {global_metrics.get('Mean_Error_%', np.nan):.2f}%")

    # Top deals (tables)
    st.markdown("### Top underpriced deals (largest negative Error)")
    st.dataframe(masked.nsmallest(15, "Error")[["Address","Type","Rent_fmt","Predicted_Rent_fmt","Error","Error_Percent","Price_Status"]].style.format({"Error":"{:,}","Error_Percent":"{:.1f}%"}))

    st.markdown("### Top overpriced listings (largest positive Error)")
    st.dataframe(masked.nlargest(15, "Error")[["Address","Type","Rent_fmt","Predicted_Rent_fmt","Error","Error_Percent","Price_Status"]].style.format({"Error":"{:,}","Error_Percent":"{:.1f}%"}))

# -----------------------
# Bottom: download filtered data
# -----------------------
st.sidebar.markdown("---")
st.sidebar.download_button("Download filtered CSV", data=masked.to_csv(index=False).encode("utf-8"), file_name="filtered_listings.csv", mime="text/csv")

