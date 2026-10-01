import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from chatbot import render_chat_tab


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Zomato Restaurant Intelligence",
    page_icon="🍽️",
    layout="wide"
)

SEGMENT_ORDER = [
    "Top Performers",
    "Above Average",
    "Average",
    "Underperformers"
]

SEGMENT_COLORS = {
    "Top Performers": "#2E86DE",
    "Above Average": "#F1C40F",
    "Average": "#1ABC9C",
    "Underperformers": "#8E44AD"
}


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("zomato_performence_segmented.csv")

    # The CSV has no "quality_segment" column (that was the KeyError).
    # Build it from quality_cluster: rank clusters by average rating
    # (highest -> lowest) and give them the business-style names.
    cluster_rank = (
        df.groupby("quality_cluster")["rating_clean"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    label_map = dict(zip(cluster_rank, SEGMENT_ORDER))
    df["quality_segment"] = df["quality_cluster"].map(label_map)

    return df


df = load_data()


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🍽️ Zomato Restaurant Intelligence")
st.write(
    "Restaurant quality segmentation, exploration and location analysis. "
    "Use the filters on the left and every chart updates instantly."
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("🔎 Filters")

if st.sidebar.button("🔄 Reset filters"):
    st.session_state.clear()
    st.rerun()

search_text = st.sidebar.text_input(
    "Search restaurant name",
    key="search"
)

selected_cities = st.sidebar.multiselect(
    "City",
    sorted(df["city"].dropna().unique().tolist()),
    key="cities",
    placeholder="All cities"
)

selected_segments = st.sidebar.multiselect(
    "Quality Segment",
    SEGMENT_ORDER,
    key="segments",
    placeholder="All segments"
)

selected_prices = st.sidebar.multiselect(
    "Price Range (1 = cheapest, 4 = costliest)",
    sorted(df["price_range"].dropna().unique().tolist()),
    key="prices",
    placeholder="All price ranges"
)

selected_cuisines = st.sidebar.multiselect(
    "Primary Cuisine",
    sorted(df["primary_cuisine"].dropna().unique().tolist()),
    key="cuisines",
    placeholder="All cuisines"
)

rating_range = st.sidebar.slider(
    "Rating range",
    min_value=0.0,
    max_value=5.0,
    value=(0.0, 5.0),
    step=0.1,
    key="rating"
)

max_votes = int(df["votes"].max())
votes_range = st.sidebar.slider(
    "Minimum votes",
    min_value=0,
    max_value=max_votes,
    value=0,
    key="votes"
)

st.sidebar.subheader("Services")
only_delivery = st.sidebar.checkbox("Online delivery available", key="delivery")
only_booking = st.sidebar.checkbox("Table booking available", key="booking")


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df.copy()

if search_text:
    filtered_df = filtered_df[
        filtered_df["restaurant_name"].str.contains(
            search_text, case=False, na=False
        )
    ]

if selected_cities:
    filtered_df = filtered_df[filtered_df["city"].isin(selected_cities)]

if selected_segments:
    filtered_df = filtered_df[
        filtered_df["quality_segment"].isin(selected_segments)
    ]

if selected_prices:
    filtered_df = filtered_df[
        filtered_df["price_range"].isin(selected_prices)
    ]

if selected_cuisines:
    filtered_df = filtered_df[
        filtered_df["primary_cuisine"].isin(selected_cuisines)
    ]

filtered_df = filtered_df[
    filtered_df["rating_clean"].between(rating_range[0], rating_range[1])
    & (filtered_df["votes"] >= votes_range)
]

if only_delivery:
    filtered_df = filtered_df[filtered_df["has_online_delivery"] == 1]

if only_booking:
    filtered_df = filtered_df[filtered_df["has_table_booking"] == 1]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing **{len(filtered_df):,}** of {len(df):,} restaurants"
)

if filtered_df.empty:
    st.warning("No restaurants match these filters. Try widening them.")
    st.stop()


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.header("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Restaurants", f"{len(filtered_df):,}")
col2.metric("Average Rating", round(filtered_df["rating_clean"].mean(), 2))
col3.metric("Average Votes", f"{filtered_df['votes'].mean():,.0f}")
col4.metric("Cities", filtered_df["city"].nunique())


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab_chat, tab_quality, tab_explore, tab_city, tab_map = st.tabs(
    [
        "💬 Ask AI",
        "🏆 Quality Segmentation",
        "🍴 Restaurant Explorer",
        "🏙️ City & Cuisine",
        "📍 Map",
    ]
)


# ---------------- CHATBOT ----------------

with tab_chat:
    render_chat_tab(df, filtered_df)


# ---------------- QUALITY SEGMENTATION ----------------

with tab_quality:

    quality_summary = (
        filtered_df.groupby("quality_segment")
        .agg(
            Restaurants=("restaurant_id", "count"),
            Average_Rating=("rating_clean", "mean"),
            Average_Votes=("votes", "mean"),
        )
        .reindex(SEGMENT_ORDER)
        .dropna()
        .reset_index()
    )
    quality_summary["Average_Rating"] = quality_summary["Average_Rating"].round(2)
    quality_summary["Average_Votes"] = quality_summary["Average_Votes"].round(0)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Segment summary")
        st.dataframe(
            quality_summary,
            use_container_width=True,
            hide_index=True
        )

    with right:
        st.subheader("Restaurants per segment")
        fig_count = px.bar(
            quality_summary,
            x="quality_segment",
            y="Restaurants",
            color="quality_segment",
            color_discrete_map=SEGMENT_COLORS,
            text="Restaurants",
            category_orders={"quality_segment": SEGMENT_ORDER},
            labels={"quality_segment": "Quality Segment"},
        )
        fig_count.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_count, use_container_width=True)

    st.subheader("Rating vs customer engagement")

    c1, c2 = st.columns([1, 1])
    y_choice = c1.radio(
        "Y-axis",
        ["Log Votes", "Raw Votes"],
        horizontal=True
    )
    size_by_votes = c2.checkbox("Size dots by votes", value=False)

    y_col = "log_votes" if y_choice == "Log Votes" else "votes"

    # Plotting thousands of points is slow, so offer a sample option
    sample_all = st.checkbox(
        "Show all points (slower with large selections)",
        value=len(filtered_df) <= 3000
    )
    plot_df = filtered_df if sample_all else filtered_df.sample(
        min(3000, len(filtered_df)), random_state=1
    )

    fig = px.scatter(
        plot_df,
        x="rating_clean",
        y=y_col,
        color="quality_segment",
        size="votes" if size_by_votes else None,
        color_discrete_map=SEGMENT_COLORS,
        category_orders={"quality_segment": SEGMENT_ORDER},
        hover_name="restaurant_name",
        hover_data={
            "city": True,
            "cuisines": True,
            "price_range": True,
            "rating_clean": True,
            "votes": True,
            "quality_segment": False,
        },
        opacity=0.7,
        title="Restaurant Quality: Rating vs Customer Engagement",
        labels={
            "rating_clean": "Restaurant Rating",
            "log_votes": "Log Votes",
            "votes": "Votes",
            "quality_segment": "Quality Segment",
        },
    )
    fig.update_layout(height=600)
    st.caption("Tip: click a segment in the legend to hide or show it, drag to zoom, double-click to reset.")
    st.plotly_chart(fig, use_container_width=True)


# ---------------- RESTAURANT EXPLORER ----------------

with tab_explore:

    st.subheader("Top restaurants")

    e1, e2, e3 = st.columns(3)
    sort_col = e1.selectbox(
        "Sort by",
        ["rating_clean", "votes", "average_cost_for_two"],
        format_func=lambda c: {
            "rating_clean": "Rating",
            "votes": "Votes",
            "average_cost_for_two": "Cost for two",
        }[c],
    )
    sort_order = e2.radio("Order", ["Highest first", "Lowest first"], horizontal=True)
    n_rows = e3.slider("Rows to show", 10, 200, 50, step=10)

    display_columns = [
        "restaurant_name",
        "city",
        "cuisines",
        "price_range",
        "rating_clean",
        "votes",
        "quality_segment",
    ]

    table_df = filtered_df.sort_values(
        sort_col, ascending=(sort_order == "Lowest first")
    ).head(n_rows)[display_columns]

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "restaurant_name": "Restaurant",
            "city": "City",
            "cuisines": "Cuisines",
            "price_range": "Price",
            "rating_clean": st.column_config.ProgressColumn(
                "Rating", min_value=0, max_value=5, format="%.1f"
            ),
            "votes": st.column_config.NumberColumn("Votes", format="%d"),
            "quality_segment": "Segment",
        },
    )

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=filtered_df.drop(columns=["quality_segment"]).assign(
            quality_segment=filtered_df["quality_segment"]
        ).to_csv(index=False).encode("utf-8"),
        file_name="zomato_filtered.csv",
        mime="text/csv",
    )

    st.subheader("Restaurant details")
    chosen = st.selectbox(
        "Pick a restaurant to see its details",
        table_df["restaurant_name"].tolist(),
    )
    row = filtered_df[filtered_df["restaurant_name"] == chosen].iloc[0]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rating", row["rating_clean"])
    d2.metric("Votes", f"{int(row['votes']):,}")
    d3.metric("Segment", row["quality_segment"])
    d4.metric("Cost for two", f"{row['average_cost_for_two']} ({row['currency']})")
    st.write(f"**Cuisines:** {row['cuisines']}")
    st.write(f"**Address:** {row['address']}")
    st.write(
        f"**Online delivery:** {'Yes' if row['has_online_delivery'] == 1 else 'No'}  |  "
        f"**Table booking:** {'Yes' if row['has_table_booking'] == 1 else 'No'}"
    )


# ---------------- CITY & CUISINE ----------------

with tab_city:

    a, b = st.columns(2)

    with a:
        st.subheader("Top cities")
        city_metric = st.selectbox(
            "Rank cities by",
            ["Number of restaurants", "Average rating"],
        )
        min_count = st.slider("Minimum restaurants per city", 1, 50, 5)

        city_df = (
            filtered_df.groupby("city")
            .agg(
                Restaurants=("restaurant_id", "count"),
                Average_Rating=("rating_clean", "mean"),
            )
            .reset_index()
        )
        city_df = city_df[city_df["Restaurants"] >= min_count]
        sort_key = "Restaurants" if city_metric == "Number of restaurants" else "Average_Rating"
        city_df = city_df.sort_values(sort_key, ascending=False).head(15)
        city_df["Average_Rating"] = city_df["Average_Rating"].round(2)

        fig_city = px.bar(
            city_df.sort_values(sort_key),
            x=sort_key,
            y="city",
            orientation="h",
            color=sort_key,
            color_continuous_scale="Blues",
            hover_data=["Restaurants", "Average_Rating"],
            labels={"city": "City"},
        )
        fig_city.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

    with b:
        st.subheader("Cuisines")
        cuisine_metric = st.selectbox(
            "Rank cuisines by",
            ["Number of restaurants", "Average rating"],
        )
        min_cuisine = st.slider("Minimum restaurants per cuisine", 1, 50, 10)

        cuisine_df = (
            filtered_df.groupby("primary_cuisine")
            .agg(
                Restaurants=("restaurant_id", "count"),
                Average_Rating=("rating_clean", "mean"),
            )
            .reset_index()
        )
        cuisine_df = cuisine_df[cuisine_df["Restaurants"] >= min_cuisine]
        ckey = "Restaurants" if cuisine_metric == "Number of restaurants" else "Average_Rating"
        cuisine_df = cuisine_df.sort_values(ckey, ascending=False).head(15)
        cuisine_df["Average_Rating"] = cuisine_df["Average_Rating"].round(2)

        fig_cuisine = px.bar(
            cuisine_df.sort_values(ckey),
            x=ckey,
            y="primary_cuisine",
            orientation="h",
            color=ckey,
            color_continuous_scale="Oranges",
            hover_data=["Restaurants", "Average_Rating"],
            labels={"primary_cuisine": "Cuisine"},
        )
        fig_cuisine.update_layout(height=500, coloraxis_showscale=False)
        st.plotly_chart(fig_cuisine, use_container_width=True)

    st.subheader("Segment mix by price range")
    mix = (
        filtered_df.groupby(["price_range", "quality_segment"])
        .size()
        .reset_index(name="Restaurants")
    )
    fig_mix = px.bar(
        mix,
        x="price_range",
        y="Restaurants",
        color="quality_segment",
        color_discrete_map=SEGMENT_COLORS,
        category_orders={"quality_segment": SEGMENT_ORDER},
        barmode="stack",
        labels={"price_range": "Price Range", "quality_segment": "Quality Segment"},
    )
    fig_mix.update_xaxes(type="category")
    st.plotly_chart(fig_mix, use_container_width=True)


# ---------------- MAP ----------------

with tab_map:

    st.subheader("Restaurant locations")

    location_df = filtered_df[
        [
            "restaurant_name",
            "city",
            "latitude",
            "longitude",
            "rating_clean",
            "votes",
            "quality_segment",
        ]
    ].dropna(subset=["latitude", "longitude"])

    # drop invalid (0,0) coordinates that stretch the map
    location_df = location_df[
        (location_df["latitude"] != 0) & (location_df["longitude"] != 0)
    ]

    m1, m2 = st.columns(2)
    zoom_city = m1.selectbox(
        "Zoom to city",
        ["All (world view)"] + sorted(location_df["city"].unique().tolist()),
    )
    map_style = m2.selectbox(
        "Map style",
        ["open-street-map", "carto-positron", "carto-darkmatter"],
    )

    if zoom_city != "All (world view)":
        location_df = location_df[location_df["city"] == zoom_city]

    if location_df.empty:
        st.info("No location data for this selection.")
    else:
        map_args = dict(
            data_frame=location_df,
            lat="latitude",
            lon="longitude",
            color="quality_segment",
            color_discrete_map=SEGMENT_COLORS,
            category_orders={"quality_segment": SEGMENT_ORDER},
            hover_name="restaurant_name",
            hover_data={
                "city": True,
                "rating_clean": True,
                "votes": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=10 if zoom_city != "All (world view)" else 1,
            height=650,
        )
        # scatter_map is the newer plotly name; fall back for older versions
        if hasattr(px, "scatter_map"):
            fig_map = px.scatter_map(**map_args, map_style=map_style)
        else:
            fig_map = px.scatter_mapbox(**map_args, mapbox_style=map_style)

        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)