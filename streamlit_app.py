import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_plotly_events import plotly_events
import pycountry

# Load Data
@st.cache_data
def load_total_data():
    return pd.read_csv('data/total_data.csv')

df = load_total_data()

st.title("🏏 Player Performance Dashboard")

# --- Player and Format Selection ---
players = df['name'].dropna().unique()
selected_player = st.selectbox("Select Player", sorted(players))

format_options = {
    "T20I": "T20I",
    "ODI": "ODI",
    "Test": "Test"
}
selected_format = st.radio("Select Match Format", list(format_options.keys()))
filtered_df = df[(df['name'] == selected_player) & (df['match_type'] == format_options[selected_format])]

if filtered_df.empty:
    st.warning("No data found for this player and format.")
    st.stop()

# --- Get Country ISO Codes ---
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# --- Aggregate Data by Country ---
country_df = (
    filtered_df
    .groupby('opposition_team', as_index=False)
    .agg(
        matches=('match_id', 'nunique'),
        total_runs=('runs_scored', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        dismissals=('player_out', 'sum'),
        wickets=('wickets_taken', 'sum'),
        balls_bowled=('balls_bowled', 'sum'),
        runs_conceded=('runs_conceded', 'sum')
    )
)
country_df['iso_alpha'] = country_df['opposition_team'].apply(get_country_code)
country_df.dropna(subset=['iso_alpha'], inplace=True)

# --- World Map Visualization ---
fig = px.choropleth(
    country_df,
    locations="iso_alpha",
    color="total_runs",
    hover_name="opposition_team",
    hover_data={
        "matches": True,
        "total_runs": True,
        "balls_faced": True,
        "dismissals": True,
        "iso_alpha": False
    },
    title=f"{selected_player}'s Performance against Teams in {selected_format}",
    color_continuous_scale="Oranges"
)
fig.update_geos(projection_type="natural earth")

st.subheader("🌍 Click a Country to View Player Stats")
click = plotly_events(fig, click_event=True, select_event=False)

if click:
    iso_clicked = click[0]['location']
    country_row = country_df[country_df['iso_alpha'] == iso_clicked]
    if not country_row.empty:
        country_name = country_row['opposition_team'].values[0]
        st.markdown(f"## 📊 Stats vs {country_name}")

        df_vs_country = filtered_df[filtered_df['opposition_team'] == country_name]

        # Batting stats
        total_runs = df_vs_country['runs_scored'].sum()
        innings = df_vs_country.shape[0]
        balls_faced = df_vs_country['balls_faced'].sum()
        dismissals = df_vs_country['player_out'].sum()
        avg_runs = total_runs / dismissals if dismissals > 0 else total_runs
        avg_balls = balls_faced / innings if innings > 0 else 0
        strike_rate = (total_runs / balls_faced * 100) if balls_faced > 0 else 0
        top_outs = df_vs_country['out_kind'].value_counts().head(3)

        # Bowling stats
        balls_bowled = df_vs_country['balls_bowled'].sum()
        wickets = df_vs_country['wickets_taken'].sum()
        runs_conceded = df_vs_country['runs_conceded'].sum()
        bowling_avg = (runs_conceded / wickets) if wickets > 0 else 0
        bowling_sr = (balls_bowled / wickets) if wickets > 0 else 0
        top_dismissals = df_vs_country[['bowled_done', 'lbw_done']].sum().to_dict()
        most_common_dismissal = max(top_dismissals, key=top_dismissals.get)

        # --- Display Stats ---
        st.markdown(f"""
        ### 🏏 Batting Stats
        - 🧮 Matches Played: **{innings}**
        - 🥇 Total Runs: **{total_runs}**
        - 📊 Average Runs per Dismissal: **{avg_runs:.2f}**
        - 🎯 Batting Strike Rate: **{strike_rate:.2f}**
        - ⏱️ Avg Balls Faced per Innings: **{avg_balls:.2f}**

        **Top 3 Modes of Dismissal:**
        """)

        for kind, count in top_outs.items():
            st.markdown(f"- {kind}: {count} times")

        st.markdown(f"""
        ### 🎯 Bowling Stats
        - 🥎 Total Balls Bowled: **{balls_bowled}**
        - 🧷 Wickets Taken: **{wickets}**
        - 🧮 Bowling Average: **{bowling_avg:.2f}**
        - 🔥 Bowling Strike Rate: **{bowling_sr:.2f}**
        - ✅ Most Common Wicket Type: **{most_common_dismissal}**
        """)

else:
    st.info("Click on a country in the map to view detailed stats.")
