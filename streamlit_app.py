# app.py
import os
# avoid inotify crash in some hosted environments
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"

# streamlit_app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Optional: check for statsmodels for trendline
try:
    import statsmodels.api as sm  # noqa: F401
    TRENDLINE_AVAILABLE = True
except Exception:
    TRENDLINE_AVAILABLE = False

st.set_page_config(layout="wide", page_title="UAE Rent Prediction Analysis")

# -------------------------
# Helpers: robust column resolution
# -------------------------
def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Load and normalize data
# -------------------------
@st.cache_data
def load_and_prepare(csv_path: str = "dubai_rent_predictions_with_status.csv"):
    df = pd.read_csv(csv_path)

    # make a copy to avoid SettingWithCopy warnings later
    df = df.copy()

    # map columns robustly to canonical names we use below
    col_map = {}
    col_map["Rent"] = choose_col(df, ["Rent", "rent", "Actual_Rent", "actual_rent"])
    col_map["Predicted_Rent"] = choose_col(df, ["Predicted_Rent", "predicted_rent", "Predicted rent", "predicted rent"])
    col_map["Error"] = choose_col(df, ["Error", "error"])
    col_map["Error_Percent"] = choose_col(df, ["Error_Percent", "Error%","%Error", "Error_Percentage", "Error_Percent"])
    col_map["Abs_Error"] = choose_col(df, ["Abs_Error", "AbsError", "Absolute_Error"])
    col_map["Over_Under"] = choose_col(df, ["Over_Under", "Over_Under_Value", "OverUnder"])
    col_map["Price_Status"] = choose_col(df, ["Price_Status", "Price Status", "status", "PriceStatus"])
    col_map["Area"] = choose_col(df, ["Area_in_sqft", "Area", "area","area_in_sqft"])
    col_map["Beds"] = choose_col(df, ["Beds", "beds", "bedrooms"])
    col_map["Baths"] = choose_col(df, ["Baths", "baths", "bathrooms"])
    col_map["Rent_per_sqft"] = choose_col(df, ["Rent_per_sqft", "rent_per_sqft"])
    col_map["Location"] = choose_col(df, ["Location", "location", "community", "Community"])
    col_map["City"] = choose_col(df, ["City", "city"])
    col_map["Latitude"] = choose_col(df, ["Latitude", "latitude", "lat"])
    col_map["Longitude"] = choose_col(df, ["Longitude", "longitude", "lon", "lng"])
    col_map["Geo_Cluster"] = choose_col(df, ["Geo_Cluster", "GeoCluster", "geo_cluster"])
    col_map["Address"] = choose_col(df, ["Address", "address"])
    col_map["Posted_date"] = choose_col(df, ["Posted_date", "posted_date", "Posted Date"])

    # create canonical columns used by app
    # Rent & Predicted_Rent are required; raise if missing
    if col_map["Rent"] is None or col_map["Predicted_Rent"] is None:
        raise ValueError("CSV must contain Rent and Predicted_Rent columns (or synonyms).")

    df["Rent"] = pd.to_numeric(df[col_map["Rent"]], errors="coerce")
    df["Predicted_Rent"] = pd.to_numeric(df[col_map["Predicted_Rent"]], errors="coerce")

    # Error: if provided use it; otherwise compute Predicted - Actual
    if col_map["Error"]:
        df["Error"] = pd.to_numeric(df[col_map["Error"]], errors="coerce")
    else:
        df["Error"] = df["Predicted_Rent"] - df["Rent"]

    # Abs error
    if col_map["Abs_Error"]:
        df["Abs_Error"] = pd.to_numeric(df[col_map["Abs_Error"]], errors="coerce")
    else:
        df["Abs_Error"] = df["Error"].abs()

    # Over/Under value (signed)
    if col_map["Over_Under"]:
        df["Over_Under"] = pd.to_numeric(df[col_map["Over_Under"]], errors="coerce")
    else:
        df["Over_Under"] = df["Error"]  # default

    # Error percent: detect format. If provided, use it; else compute
    if col_map["Error_Percent"]:
        ep = pd.to_numeric(df[col_map["Error_Percent"]], errors="coerce")
        if ep.abs().max(skipna=True) > 1.5:
            df["Error_Percent"] = ep / 100.0
        else:
            df["Error_Percent"] = ep.fillna(0.0)
    else:
        # Avoid division by zero
        valid_rent = df["Rent"].replace({0: np.nan})
        df["Error_Percent"] = df["Error"] / valid_rent
        df["Error_Percent"] = df["Error_Percent"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Price status
    if col_map["Price_Status"]:
        df["Price_Status"] = df[col_map["Price_Status"]].astype(str)
    else:
        # default classification (can be overridden)
        def classify_pct(x, thresh=0.05):
            if abs(x) <= thresh:
                return "Fair"
            return "Overpriced" if x > thresh else "Underpriced"
        df["Price_Status"] = df["Error_Percent"].apply(classify_pct)

    # Area
    if col_map["Area"]:
        df["Area_in_sqft"] = pd.to_numeric(df[col_map["Area"]], errors="coerce")
    else:
        df["Area_in_sqft"] = np.nan

    # Beds/Baths
    if col_map["Beds"]:
        df["Beds"] = pd.to_numeric(df[col_map["Beds"]], errors="coerce")
    if col_map["Baths"]:
        df["Baths"] = pd.to_numeric(df[col_map["Baths"]], errors="coerce")

    # rent_per_sqft
    if col_map["Rent_per_sqft"]:
        df["Rent_per_sqft"] = pd.to_numeric(df[col_map["Rent_per_sqft"]], errors="coerce")
    else:
        # compute if possible
        df["Rent_per_sqft"] = (df["Rent"] / df["Area_in_sqft"]).replace([np.inf, -np.inf], np.nan)

    # location/city/coords
    if col_map["Location"]:
        df["Location"] = df[col_map["Location"]].astype(str)
    else:
        df["Location"] = df[col_map["Address"]] if col_map["Address"] else "Unknown"

    if col_map["City"]:
        df["City"] = df[col_map["City"]].astype(str)
    else:
        df["City"] = "Unknown"

    df["Latitude"] = pd.to_numeric(df[col_map["Latitude"]], errors="coerce") if col_map["Latitude"] else np.nan
    df["Longitude"] = pd.to_numeric(df[col_map["Longitude"]], errors="coerce") if col_map["Longitude"] else np.nan

    # Geo cluster
    if col_map["Geo_Cluster"]:
        df["Geo_Cluster"] = df[col_map["Geo_Cluster"]]
    else:
        df["Geo_Cluster"] = np.nan

    # Address
    df["Address"] = df[col_map["Address"]] if col_map["Address"] else df["Location"]

    # pretty formatted columns for table display
    df["Rent_fmt"] = df["Rent"].apply(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    df["Predicted_Rent_fmt"] = df["Predicted_Rent"].apply(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    df["Error_fmt"] = df["Error"].apply(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    df["Error_Percent_display"] = df["Error_Percent"].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")

    # final cleanup: drop rows without essential coords/rent if needed
    # (we keep them for non-geo visuals)
    return df

# load
CSV = "dubai_rent_predictions_with_status.csv"
if not Path(CSV).exists():
    st.error(f"CSV not found: {CSV}. Put the file in the app folder and re-run.")
    st.stop()

df = load_and_prepare(CSV)


# Sidebar filters (global)

st.sidebar.header("Filters (global)")
type_options = sorted(df["Type"].dropna().unique().tolist())
sel_types = st.sidebar.multiselect("Property Type", options=type_options, default=type_options)

city_options = sorted(df["City"].dropna().unique().tolist())
sel_cities = st.sidebar.multiselect("City", options=city_options, default=city_options)

rent_min = int(np.nanmin(df["Rent"].fillna(0)))
rent_max = int(np.nanmax(df["Rent"].fillna(0)))
sel_rent = st.sidebar.slider("Rent (AED) range", rent_min, rent_max, (rent_min, rent_max), step=1000)

area_min = int(np.nanmin(df["Area_in_sqft"].fillna(0)))
area_max = int(np.nanmax(df["Area_in_sqft"].fillna(0)))
sel_area = st.sidebar.slider("Area (sqft) range", area_min, area_max, (area_min, area_max), step=50)

# filtered df for most charts
df_filtered = df[
    (df["Type"].isin(sel_types)) &
    (df["City"].isin(sel_cities)) &
    (df["Rent"].between(sel_rent[0], sel_rent[1])) &
    (df["Area_in_sqft"].between(sel_area[0], sel_area[1]))
].copy()

st.title("🏙️ UAE Rent Prediction Analysis Dashboard")
st.markdown("Comprehensive dashboard: predicted vs actual rents, over/under pricing hotspots, and feature-driven error analysis — designed for hiring managers and real-estate stakeholders.")


# Top KPIs

st.header("Key metrics")
mae = np.nanmean(np.abs(df_filtered["Predicted_Rent"] - df_filtered["Rent"]))
# Error_Percent is fractional (0.05 == 5%)
valid_errs = df_filtered["Error_Percent"].replace([np.inf, -np.inf], np.nan).dropna()
mape = np.mean(np.abs(valid_errs)) * 100
overpriced_pct = 100.0 * (df_filtered["Price_Status"] == "Overpriced").sum() / max(1, len(df_filtered))
underpriced_pct = 100.0 * (df_filtered["Price_Status"] == "Underpriced").sum() / max(1, len(df_filtered))
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"AED {mae:,.0f}")
col2.metric("MAPE", f"{mape:.2f}%")
col3.metric("Overpriced", f"{overpriced_pct:.1f}%")
col4.metric("Underpriced", f"{underpriced_pct:.1f}%")


# Tabs layout
tabs = st.tabs(["Heatmaps (geo)", "Predicted vs Actual", "Distributions & Boxplots", "Top 10 & Lookup", "Cluster / City Insights"])


# Tab: Heatmaps (geospatial)

with tabs[0]:
    st.subheader("Geo Heatmaps (UAE)")

    # Rent per sqft heatmap (density)
    st.markdown("**Rent per sqft — spatial intensity** (clipped for visualization).")
    # clip rent_per_sqft to reasonable range for visualization
    df_filtered["rent_per_sqft_clip"] = df_filtered["Rent_per_sqft"].clip(lower=0, upper=500)
    fig1 = px.density_mapbox(
        df_filtered.dropna(subset=["Latitude", "Longitude"]),
        lat="Latitude", lon="Longitude", z="rent_per_sqft_clip",
        radius=20, center={"lat": df_filtered["Latitude"].mean(), "lon": df_filtered["Longitude"].mean()},
        zoom=9, mapbox_style="carto-positron",
        color_continuous_scale="Viridis",
        title="Rent per sqft density (UAE)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Over/Underpricing heatmap: use normalized / clipped Over_Under for color
    st.markdown("**Over / Under pricing intensity** (normalized & clipped). Red = overpriced, Blue = underpriced.")
    # normalize and clip to percentile-based bounds to avoid outliers dominating
    ou = df_filtered["Over_Under"].fillna(0)
    lower = np.nanpercentile(ou, 2)
    upper = np.nanpercentile(ou, 98)
    df_filtered["ou_clip"] = ou.clip(lower=lower, upper=upper)
    # create sign-aware intensity by shifting to positive for density z (use absolute) but color using original sign on separate panel
    fig2 = px.density_mapbox(
        df_filtered.dropna(subset=["Latitude", "Longitude"]),
        lat="Latitude", lon="Longitude", z="ou_clip",
        radius=20, center={"lat": df_filtered["Latitude"].mean(), "lon": df_filtered["Longitude"].mean()},
        zoom=9, mapbox_style="carto-positron",
        color_continuous_scale="RdBu_r",
        title=f"Over/Underpricing intensity (clipped to {int(lower):,} / {int(upper):,})"
    )
    st.plotly_chart(fig2, use_container_width=True)


# Tab: Predicted vs Actual (scatter & trendline)

with tabs[1]:
    st.subheader("Predicted vs Actual Rent (scaled, UAE)")
    st.markdown("Scatter with an ideal-fit reference line (y=x). Use the trendline if statsmodels is installed.")

    # scatter with marginal histograms (fast enough)
    # axis scaling: show up to 5M for clarity, but if dataset max < 5M use dataset max *1.05
    max_r = max(5_000_000, int(df_filtered["Rent"].max() * 1.05))
    # base scatter
    fig_sc = px.scatter(
        df_filtered,
        x="Rent",
        y="Predicted_Rent",
        color="Price_Status",
        hover_data=["Address", "Location", "City", "Type", "Beds", "Area_in_sqft", "Abs_Error", "Error_Percent"],
        labels={"Rent": "Actual Rent (AED)", "Predicted_Rent": "Predicted Rent (AED)"},
        title="Predicted vs Actual Rent (with reference line)"
    )
    # add reference diagonal y = x
    fig_sc.add_shape(
        type="line",
        x0=0, y0=0,
        x1=max_r, y1=max_r,
        line=dict(color="black", dash="dash")
    )
    fig_sc.update_xaxes(range=[0, max_r])
    fig_sc.update_yaxes(range=[0, max_r])
    st.plotly_chart(fig_sc, use_container_width=True)

    # If statsmodels available, show scatter + OLS trendline
    if TRENDLINE_AVAILABLE:
        st.markdown("**Scatter with OLS trendline (statsmodels available)**")
        fig_tr = px.scatter(
            df_filtered,
            x="Rent",
            y="Predicted_Rent",
            color="Price_Status",
            trendline="ols",
            trendline_scope="overall",
            hover_data=["Address", "Location", "City", "Type"],
            labels={"Rent": "Actual Rent (AED)", "Predicted_Rent": "Predicted Rent (AED)"},
            title="Predicted vs Actual Rent with OLS trendline"
        )
        fig_tr.add_shape(type="line", x0=0, y0=0, x1=max_r, y1=max_r, line=dict(color="black", dash="dash"))
        fig_tr.update_xaxes(range=[0, max_r])
        fig_tr.update_yaxes(range=[0, max_r])
        st.plotly_chart(fig_tr, use_container_width=True)
    else:
        st.info("Trendline not shown — install `statsmodels` (add to requirements) to enable OLS trendline.")


# Tab: Distributions & Boxplots
with tabs[2]:
    st.subheader("Distributions & Error Breakdown")

    # Histogram of Rent (capped at 0-5M for readability)
    st.markdown("**Actual Rent distribution (capped at 0–5M AED for visualization).**")
    rent_cap = df_filtered["Rent"].clip(0, 5_000_000)
    fig_hist_rent = px.histogram(
        df_filtered.assign(Rent_vis=rent_cap),
        x="Rent_vis",
        nbins=80,
        color="Price_Status",
        title="Actual Rent Distribution (visual cap 0–5M AED)",
        labels={"Rent_vis": "Rent (AED)"}
    )
    fig_hist_rent.update_xaxes(range=[0, 5_000_000])
    st.plotly_chart(fig_hist_rent, use_container_width=True)

    # Error percent violin by Type (if too many types, pick top N)
    st.markdown("**% Error distribution by Property Type**")
    top_types = df_filtered["Type"].value_counts().nlargest(12).index.tolist()
    df_v = df_filtered[df_filtered["Type"].isin(top_types)]
    fig_violin = px.violin(
        df_v,
        x="Type",
        y="Error_Percent",
        color="Price_Status",
        box=True,
        points="outliers",
        hover_data=["Address", "Rent", "Predicted_Rent"],
        title="% Error by Property Type (top types shown)"
    )
    fig_violin.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig_violin, use_container_width=True)

    # Area vs Rent colored by price status (bubble size by Rent_per_sqft)
    st.markdown("**Area vs Rent (bubble size ~ rent per sqft)**")
    fig_area = px.scatter(
        df_filtered,
        x="Area_in_sqft",
        y="Rent",
        color="Price_Status",
        size="Rent_per_sqft",
        hover_data=["Address", "Type", "Predicted_Rent", "Abs_Error"],
        title="Area vs Rent (bubble size = rent_per_sqft)"
    )
    st.plotly_chart(fig_area, use_container_width=True)


# Tab: Top 10 & Lookup
with tabs[3]:
    st.subheader("Top properties & lookup")

    # Top 10 overpriced and underpriced by Abs_Error
    st.markdown("**Top 10 Overpriced (highest positive Abs_Error)**")
    top_over = df_filtered.nlargest(10, "Abs_Error")[["Address","Location","City","Type","Area_in_sqft","Rent","Predicted_Rent","Abs_Error","Error_Percent","Price_Status"]]
    # format numbers for display
    top_over_display = top_over.copy()
    top_over_display["Rent"] = top_over_display["Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_over_display["Predicted_Rent"] = top_over_display["Predicted_Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_over_display["Abs_Error"] = top_over_display["Abs_Error"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_over_display["Error_Percent"] = top_over_display["Error_Percent"].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
    st.dataframe(top_over_display, use_container_width=True)

    st.markdown("**Top 10 Underpriced (most negative Abs_Error)**")
    top_under = df_filtered.nsmallest(10, "Abs_Error")[["Address","Location","City","Type","Area_in_sqft","Rent","Predicted_Rent","Abs_Error","Error_Percent","Price_Status"]]
    top_under_display = top_under.copy()
    top_under_display["Rent"] = top_under_display["Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_under_display["Predicted_Rent"] = top_under_display["Predicted_Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_under_display["Abs_Error"] = top_under_display["Abs_Error"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    top_under_display["Error_Percent"] = top_under_display["Error_Percent"].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
    st.dataframe(top_under_display, use_container_width=True)

    # Lookup table: allow search by address substring
    st.markdown("**Lookup — search by address / location**")
    query = st.text_input("Address contains (case-insensitive):")
    if query:
        lookup = df_filtered[df_filtered["Address"].str.contains(query, case=False, na=False)]
    else:
        lookup = df_filtered.sample(min(200, len(df_filtered)))  # show a sample if no query
    lookup_display = lookup[["Address","Location","City","Type","Beds","Area_in_sqft","Rent","Predicted_Rent","Abs_Error","Error_Percent","Price_Status"]].copy()
    lookup_display["Rent"] = lookup_display["Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    lookup_display["Predicted_Rent"] = lookup_display["Predicted_Rent"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    lookup_display["Abs_Error"] = lookup_display["Abs_Error"].map(lambda x: f"AED {int(x):,}" if pd.notnull(x) else "")
    lookup_display["Error_Percent"] = lookup_display["Error_Percent"].map(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "")
    st.dataframe(lookup_display, use_container_width=True)


# Tab: Cluster / City Insights
with tabs[4]:
    st.subheader("Cluster & City level insights")

    # Average % Error per Geo_Cluster (if present)
    if "Geo_Cluster" in df_filtered.columns and df_filtered["Geo_Cluster"].notna().any():
        cluster_stats = df_filtered.groupby("Geo_Cluster").agg(
            count=("Rent","count"),
            avg_error_pct=("Error_Percent","mean")
        ).reset_index().sort_values("avg_error_pct", ascending=False)
        fig_cluster = px.bar(cluster_stats, x="Geo_Cluster", y="avg_error_pct", color="avg_error_pct", color_continuous_scale="RdBu_r", title="Avg % Error per Geo Cluster")
        fig_cluster.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("No Geo_Cluster column present or no cluster data available.")

    # Median Rent vs Area per Type
    st.subheader("Median Rent vs Median Area (by Type)")
    med = df_filtered.groupby("Type").agg(median_rent=("Rent","median"), median_area=("Area_in_sqft","median"), count=("Rent","count")).reset_index()
    fig_med = px.scatter(med[med["count"]>=10], x="median_area", y="median_rent", size="count", color="Type", hover_data=["Type","count"], title="Median Rent vs Median Area (types with >=10 listings)")
    st.plotly_chart(fig_med, use_container_width=True)

    # Price Status counts per city
    st.subheader("Price Status breakdown by City")
    city_status = df_filtered.groupby(["City","Price_Status"]).size().reset_index(name="count")
    fig_city = px.bar(city_status, x="City", y="count", color="Price_Status", barmode="stack", title="Price Status counts per City")
    st.plotly_chart(fig_city, use_container_width=True)

    # Choropleth attempts: use Location (community) if a geojson exists
    geojson_path = Path("uae_geo.json")
    if geojson_path.exists():
        st.subheader("Choropleth: Average price per sqft by Location")
        agg_loc = df_filtered.groupby("Location").agg(avg_ppsqft=("Rent_per_sqft","mean")).reset_index()
        # attempt a choropleth by Location feature; requires geojson 'properties' to contain matching name keys
        try:
            fig_choro = px.choropleth(agg_loc, geojson=str(geojson_path), locations="Location", color="avg_ppsqft",
                                     featureidkey="properties.community", color_continuous_scale="Viridis", title="Avg price per sqft")
            fig_choro.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_choro, use_container_width=True)
        except Exception as e:
            st.warning("Failed to create choropleth from geojson: " + str(e))
            st.write(agg_loc.sort_values("avg_ppsqft", ascending=False).head(20))
    else:
        st.info("No 'uae_geo.json' found — showing top Locations by avg price per sqft instead.")
        agg_loc = df_filtered.groupby("Location").agg(avg_ppsqft=("Rent_per_sqft","mean"), count=("Rent","count")).reset_index().sort_values("avg_ppsqft", ascending=False)
        st.dataframe(agg_loc.head(30), use_container_width=True)


st.markdown("---")
st.caption("CSV used: dubai_rent_predictions_with_status.csv — Fields used: Rent, Predicted_Rent, Error, Error_Percent, Abs_Error, Over_Under, Price_Status, Rent_per_sqft, Area_in_sqft, Beds, Location, City, Latitude, Longitude, Geo_Cluster.")


