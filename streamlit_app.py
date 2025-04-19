import streamlit as st
import pandas as pd
import plotly.express as px
import pycountry

# Load Data
@st.cache_data
def load_total_data():
    return pd.read_csv('data/total_data.csv')

df = load_total_data()

st.title("🏏 Player Performance Dashboard")

# Step 1: Player Selection
players = df['name'].dropna().unique()
selected_player = st.selectbox("Select Player", sorted(players))

# Step 2: Format Selection
format_map = {
    "T20I": "T20I",
    "ODI": "ODI",
    "Test": "Test"
}
selected_format = st.radio("Select Match Format", list(format_map.keys()))

# Step 3: Filtered Data
filtered_df = df[(df['name'] == selected_player) & (df['match_type'] == format_map[selected_format])]

if filtered_df.empty:
    st.warning("No data found for this player and format.")
else:
    st.subheader(f"🌍 Performance Map: {selected_player} in {selected_format}")

    # Get country codes using pycountry
    def get_country_code(name):
        try:
            return pycountry.countries.lookup(name).alpha_3
        except:
            return None

    country_df = (
        filtered_df
        .groupby('opposition_team', as_index=False)
        .agg(
            matches=('match_id', 'nunique'),
            total_runs=('runs_scored', 'sum'),
            average_runs=('runs_scored', 'mean'),
            dismissals=('player_out', 'sum')
        )
    )
    country_df['iso_alpha'] = country_df['opposition_team'].apply(get_country_code)
    country_df.dropna(subset=['iso_alpha'], inplace=True)

    # Step 4: Plot World Map
    fig = px.choropleth(
        country_df,
        locations="iso_alpha",
        color="total_runs",
        hover_name="opposition_team",
        hover_data={
            "matches": True,
            "total_runs": True,
            "average_runs": True,
            "dismissals": True,
            "iso_alpha": False
        },
        title=f"{selected_player}'s Performance against Teams in {selected_format}",
        color_continuous_scale="Oranges"
    )
    fig.update_geos(projection_type="natural earth")

    st.plotly_chart(fig, use_container_width=True)

    # Step 5: Optional - Country click simulation via selectbox
    st.subheader("📊 Detailed Stats")
    selected_country = st.selectbox("Select Country to View Detailed Stats", country_df['opposition_team'])
    stats = country_df[country_df['opposition_team'] == selected_country].iloc[0]

    st.markdown(f"""
    ### Stats vs {selected_country}
    - 🏏 Matches Played: {stats['matches']}
    - 🥇 Total Runs Scored: {stats['total_runs']}
    - 📊 Average Runs: {stats['average_runs']:.2f}
    - ❌ Dismissals: {int(stats['dismissals'])}
    """)
