import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. Load Data ---
@st.cache_data
def load_data():
    total = pd.read_csv('data/total_data.csv')
    pw_profiles = pd.read_csv('data/mw_pw_profiles.csv')
    overall = pd.read_csv('data/mw_overall.csv')
    style = pd.read_csv('data/style_based_features.csv')
    return total, pw_profiles, overall, style

total_df, pw_df, overall_df, style_df = load_data()

# --- 2. Sidebar Filters ---
st.sidebar.title("Filters")

grounds = total_df['ground'].unique()
selected_grounds = st.sidebar.multiselect("Select Ground(s)", options=grounds, default=grounds[:3])
innings_opts = total_df['innings'].unique()
selected_innings = st.sidebar.multiselect("Select Innings", options=innings_opts, default=innings_opts.tolist())

players = pw_df['player_name'].unique()
selected_player = st.sidebar.selectbox("Player for Player vs Team", players)
teams = pw_df['opponent_team'].unique()
selected_opponent = st.sidebar.selectbox("Opponent Team", teams)

teamA = st.sidebar.selectbox("Team A (Head‑to‑Head)", overall_df['teamA'].unique())
teamB = st.sidebar.selectbox("Team B (Head‑to‑Head)", overall_df['teamB'].unique())

tournaments = total_df['tournament'].dropna().unique()
selected_tourney = st.sidebar.selectbox("Tournament Trends", tournaments)

formats = total_df['format'].unique()
selected_format = st.sidebar.selectbox("Format for Dismissals", formats)

st.title("🏏 Cricket Analytics Dashboard")

# --- 3. Average Innings Score per Ground ---
st.header("1. Average Innings Score per Ground")
df1 = (
    total_df
    .query("ground in @selected_grounds and innings in @selected_innings")
    .groupby(['ground','innings'])
    .agg(avg_score=('score','mean'))
    .reset_index()
)
fig1 = px.bar(
    df1,
    x='ground', y='avg_score',
    color='innings',
    barmode='group',
    labels={'avg_score':'Avg Score'},
    title="Average Innings Score per Ground"
)
st.plotly_chart(fig1, use_container_width=True)

# --- 4. Player Performance vs Specific Team ---
st.header("2. Player Performance vs Specific Team")
df2 = pw_df.query("player_name == @selected_player and opponent_team == @selected_opponent")
if df2.empty:
    st.write("No data for this combination.")
else:
    fig2 = px.line(
        df2,
        x='match_date', y='runs_scored',
        markers=True,
        labels={'runs_scored':'Runs'},
        title=f"{selected_player} Runs vs {selected_opponent}"
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- 5. Head-to-Head Team Comparison ---
st.header("3. Head‑to‑Head Team Comparison")
df3 = overall_df.query("teamA == @teamA and teamB == @teamB")
if df3.empty:
    st.write("No head‑to‑head data for this pair.")
else:
    fig3 = px.bar(
        df3,
        x='series', y=['winsA','winsB'],
        labels={'value':'Wins','series':'Series'},
        title=f"{teamA} vs {teamB} Head‑to‑Head",
        barmode='group'
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- 6. Tournament-Based Trends ---
st.header("4. Tournament-Based Trends")
df4 = total_df.query("tournament == @selected_tourney")
df4_agg = (
    df4
    .groupby('match_date')
    .agg(runs=('score','sum'), wickets=('wickets','sum'))
    .reset_index()
)
fig4 = px.line(
    df4_agg,
    x='match_date', y=['runs','wickets'],
    labels={'value':'Count','match_date':'Date'},
    title=f"{selected_tourney} Trends"
)
st.plotly_chart(fig4, use_container_width=True)

# --- 7. Dismissal Type Analysis ---
st.header("5. Dismissal Type Breakdown")
df5 = total_df.query("format == @selected_format")
dismissal_counts = df5['dismissal_type'].value_counts().reset_index()
dismissal_counts.columns = ['dismissal_type','count']
fig5 = px.pie(
    dismissal_counts,
    names='dismissal_type', values='count',
    title=f"Dismissal Types in {selected_format}"
)
st.plotly_chart(fig5, use_container_width=True)
