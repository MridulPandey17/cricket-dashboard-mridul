import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/total_data.csv")

df = load_data()

# Sidebar selectors
st.sidebar.header("Select Options")
player = st.sidebar.selectbox("Select Player", df["name"].unique())
match_format = st.sidebar.selectbox("Select Format", df["match_type"].unique())

# Filter based on selections
player_df = df[(df["name"] == player) & (df["match_type"] == match_format)]

# Prepare map data
map_df = player_df.groupby("opposition_team").agg({
    "runs_scored": "sum",
    "balls_faced": "sum",
    "player_out": "sum",
    "match_id": "nunique"
}).reset_index()

map_df["country"] = map_df["opposition_team"]  # Make sure this matches ISO names if needed

# Map Display
st.subheader("🌍 Click a Country to View Player Stats")

fig = px.choropleth(
    map_df,
    locations="country",
    locationmode="country names",
    color="runs_scored",
    hover_name="country",
    title=f"{player}'s Performance against Teams in {match_format}",
    color_continuous_scale="Blues"
)

fig.update_layout(clickmode='event+select')

click = st.plotly_chart(fig, use_container_width=True)

# Capture Click
st.info("Click on a country in the map to view detailed stats.")
clicked_country = st.session_state.get("clicked_country", None)

# Use plotly_events for interactivity
from streamlit_plotly_events import plotly_events
selected_points = plotly_events(fig, click_event=True, select_event=True)

if selected_points:
    clicked_country = selected_points[0]["hovertext"]
    st.session_state["clicked_country"] = clicked_country

# Show stats
if clicked_country:
    st.success(f"Showing stats for {player} vs {clicked_country}")

    vs_team_df = player_df[player_df["opposition_team"] == clicked_country]

    if not vs_team_df.empty:
        st.write("### 🧮 Stats Breakdown")

        st.markdown(f"- **Innings Played:** {vs_team_df['match_id'].nunique()}")
        st.markdown(f"- **Total Runs:** {vs_team_df['runs_scored'].sum()}")
        st.markdown(f"- **Batting Average:** {vs_team_df['runs_scored'].sum() / vs_team_df['player_out'].sum():.2f}")
        st.markdown(f"- **Batting Strike Rate:** {(vs_team_df['runs_scored'].sum() / vs_team_df['balls_faced'].sum()) * 100:.2f}")
        st.markdown(f"- **Avg Balls Faced per Innings:** {vs_team_df['balls_faced'].sum() / vs_team_df['match_id'].nunique():.2f}")

        if vs_team_df['balls_bowled'].sum() > 0:
            st.markdown(f"- **Bowling Average:** {vs_team_df['runs_conceded'].sum() / vs_team_df['wickets_taken'].sum():.2f}")
            st.markdown(f"- **Bowling Strike Rate:** {vs_team_df['balls_bowled'].sum() / vs_team_df['wickets_taken'].sum():.2f}")
        else:
            st.markdown(f"- ❌ Player did not bowl vs {clicked_country}")

        # Top dismissal modes
        top_dismissals = (
            vs_team_df["out_kind"]
            .value_counts()
            .dropna()
            .head(3)
        )

        st.write("### ⚰️ Top 3 Dismissal Modes")
        for mode, count in top_dismissals.items():
            st.markdown(f"- **{mode.capitalize()}**: {count} times")

        # Most common way of taking a wicket
        top_wicket_method = (
            vs_team_df[vs_team_df["wickets_taken"] > 0][["bowled_done", "lbw_done"]]
            .sum()
            .sort_values(ascending=False)
        )

        if not top_wicket_method.empty:
            st.write("### 🏏 Most Common Wicket Method (as Bowler)")
            st.markdown(f"- **{top_wicket_method.idxmax().replace('_done', '').upper()}**: {int(top_wicket_method.max())} times")
    else:
        st.warning("No data available for this country.")
