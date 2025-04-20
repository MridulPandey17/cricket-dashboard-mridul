import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events

# Constants for session state keys
S_KEY_PLAYER = "selected_player"
S_KEY_FORMAT = "selected_format"
S_KEY_OPPOSITION = "selected_opposition"

# Load your CSV
df_full = pd.read_csv("total_data.csv")

# Unique values for dropdowns
players = df_full["name"].dropna().unique()
formats = df_full["match_type"].dropna().unique()

# Title and Sidebar
st.set_page_config(layout="wide")
st.title("🏏 Player Stats Explorer")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    player = st.selectbox("Select Player", sorted(players))
    match_format = st.selectbox("Select Format", sorted(formats))

# Save selections to session state
st.session_state[S_KEY_PLAYER] = player
st.session_state[S_KEY_FORMAT] = match_format

# Filter data for map
df_filtered = df_full[(df_full["name"] == player) & (df_full["match_type"] == match_format)]
df_country_stats = df_filtered.groupby("opposition").agg({
    "runs_scored": "sum",
    "balls_faced": "sum",
    "player_out": "sum",
    "wickets_taken": "sum",
    "balls_bowled": "sum",
    "runs_conceded": "sum"
}).reset_index()

# Map opposition to ISO country codes and flags
country_to_iso = {
    "India": "IND", "Australia": "AUS", "England": "ENG", "Pakistan": "PAK", "New Zealand": "NZL",
    "South Africa": "ZAF", "Sri Lanka": "LKA", "Bangladesh": "BGD", "West Indies": "WI", "Afghanistan": "AFG",
    "Zimbabwe": "ZWE", "Ireland": "IRL", "Scotland": "SCO", "Netherlands": "NLD", "Namibia": "NAM",
    "U.A.E.": "ARE", "Kenya": "KEN", "Canada": "CAN", "Bermuda": "BMU", "Nepal": "NPL", "Oman": "OMN",
    "Hong Kong": "HKG", "U.S.A.": "USA", "Papua New Guinea": "PNG"
}

country_to_flag = {
    "India": "🇮🇳", "Australia": "🇦🇺", "England": "🏴", "Pakistan": "🇵🇰", "New Zealand": "🇳🇿",
    "South Africa": "🇿🇦", "Sri Lanka": "🇱🇰", "Bangladesh": "🇧🇩", "West Indies": "🌴", "Afghanistan": "🇦🇫",
    "Zimbabwe": "🇿🇼", "Ireland": "🇮🇪", "Scotland": "🏴", "Netherlands": "🇳🇱", "Namibia": "🇳🇦",
    "U.A.E.": "🇦🇪", "Kenya": "🇰🇪", "Canada": "🇨🇦", "Bermuda": "🇧🇲", "Nepal": "🇳🇵", "Oman": "🇴🇲",
    "Hong Kong": "🇭🇰", "U.S.A.": "🇺🇸", "Papua New Guinea": "🇵🇬"
}

df_country_stats["iso_code"] = df_country_stats["opposition"].map(country_to_iso)
df_country_stats["flag"] = df_country_stats["opposition"].map(country_to_flag)

# Build Choropleth Map
fig = px.choropleth(
    df_country_stats,
    locations="iso_code",
    color="runs_scored",
    hover_name="opposition",
    color_continuous_scale="OrRd",
    title="Runs Scored vs Countries"
)
fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

# Display clickable map
st.subheader("🌍 Click on a Country to View Detailed Stats")
iso_to_opposition_map = {v: k for k, v in country_to_iso.items()}

click_data = plotly_events(
    fig,
    click_event=True,
    hover_event=False,
    select_event=False,
    override_height=600,
    override_width="100%"
)

if click_data:
    clicked_iso = click_data[0].get("location")
    if clicked_iso in iso_to_opposition_map:
        st.session_state[S_KEY_OPPOSITION] = iso_to_opposition_map[clicked_iso]
        st.rerun()

# Add fallback buttons
st.subheader("📌 Or Choose an Opponent")
cols = st.columns(4)
for idx, row in enumerate(df_country_stats.itertuples()):
    with cols[idx % 4]:
        if st.button(f"{row.flag or ''} {row.opposition}"):
            st.session_state[S_KEY_OPPOSITION] = row.opposition
            st.rerun()

# Stats Section
current_selection_for_stats = st.session_state.get(S_KEY_OPPOSITION)

if current_selection_for_stats:
    st.subheader(f"📈 Stats Against {current_selection_for_stats}")
    player_data = df_filtered[df_filtered["opposition"] == current_selection_for_stats]

    runs = player_data["runs_scored"].sum()
    outs = player_data["player_out"].sum()
    balls_faced = player_data["balls_faced"].sum()
    batting_avg = runs / outs if outs > 0 else float('inf')
    batting_strike_rate = (runs / balls_faced) * 100 if balls_faced > 0 else 0.0

    wickets = player_data["wickets_taken"].sum()
    runs_conceded = player_data["runs_conceded"].sum()
    balls_bowled = player_data["balls_bowled"].sum()
    bowling_avg = runs_conceded / wickets if wickets > 0 else float('inf')
    bowling_economy = (runs_conceded / balls_bowled) * 6 if balls_bowled > 0 else 0.0

    st.markdown(f"**Total Runs:** {runs}")
    st.markdown(f"**Batting Average:** {batting_avg:.2f}")
    st.markdown(f"**Strike Rate:** {batting_strike_rate:.2f}")
    st.markdown("---")
    st.markdown(f"**Wickets Taken:** {wickets}")
    st.markdown(f"**Bowling Average:** {bowling_avg:.2f}")
    st.markdown(f"**Economy Rate:** {bowling_economy:.2f}")

    # --- Career Stats ---
    career_df = df_full[(df_full["name"] == player) & (df_full["match_type"] == match_format)]
    career_runs = career_df["runs_scored"].sum()
    career_outs = career_df["player_out"].sum()
    career_balls_faced = career_df["balls_faced"].sum()
    career_avg = career_runs / career_outs if career_outs > 0 else float('inf')
    career_sr = (career_runs / career_balls_faced) * 100 if career_balls_faced > 0 else 0.0
    career_wickets = career_df["wickets_taken"].sum()
    career_runs_conceded = career_df["runs_conceded"].sum()
    career_balls_bowled = career_df["balls_bowled"].sum()
    career_bowling_avg = career_runs_conceded / career_wickets if career_wickets > 0 else float('inf')
    career_econ = (career_runs_conceded / career_balls_bowled) * 6 if career_balls_bowled > 0 else 0.0

    # --- Comparison Bar Charts ---
    st.markdown("### 🔍 Career vs Country Comparison")

    comp_data = {
        "Metric": ["Batting Avg", "Strike Rate", "Bowling Avg", "Economy"],
        "Career": [career_avg, career_sr, career_bowling_avg, career_econ],
        f"vs {current_selection_for_stats}": [batting_avg, batting_strike_rate, bowling_avg, bowling_economy],
    }
    comp_df = pd.DataFrame(comp_data)

    fig_comp = px.bar(
        comp_df.melt(id_vars="Metric", var_name="Type", value_name="Value"),
        x="Metric", y="Value", color="Type", barmode="group", text_auto=".2s",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_comp.update_layout(height=400)
    st.plotly_chart(fig_comp, use_container_width=True)
