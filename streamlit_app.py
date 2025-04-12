import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. Load Only the Available Data ---
@st.cache_data
def load_data():
    overall = pd.read_csv('data/mw_overall.csv')
    style   = pd.read_csv('data/style_based_features.csv')
    return overall, style

overall_df, style_df = load_data()

# --- 2. Sidebar Controls ---
st.sidebar.title("Filters")

# Head‑to‑Head filters
teamA = st.sidebar.selectbox("Team A", overall_df['teamA'].unique())
teamB = st.sidebar.selectbox("Team B", overall_df['teamB'].unique())

# Style‑features filters
players = style_df['player_name'].unique()
selected_player = st.sidebar.selectbox("Player (Style Analysis)", players)

# Dynamically pick style columns (exclude identifying columns)
style_cols = [c for c in style_df.columns if c not in ['player_name']]
selected_features = st.sidebar.multiselect(
    "Style Features to Plot",
    options=style_cols,
    default=style_cols[:3]
)

st.title("🏏 Cricket Analytics (Limited Data)")

# --- 3. Head‑to‑Head Team Comparison ---
st.header("1. Head‑to‑Head: Wins by Series")
df_h2h = overall_df.query("teamA == @teamA and teamB == @teamB")
if df_h2h.empty:
    st.write("No head‑to‑head data for this pair.")
else:
    fig_h2h = px.bar(
        df_h2h,
        x='series',
        y=['winsA', 'winsB'],
        barmode='group',
        labels={'value':'Wins','series':'Series'},
        title=f"{teamA} vs {teamB}"
    )
    st.plotly_chart(fig_h2h, use_container_width=True)

# --- 4. Style‑Based Feature Analysis ---
st.header("2. Player Style Features")
df_style = style_df[style_df['player_name'] == selected_player]
if df_style.empty:
    st.write("No style data for this player.")
else:
    # Melt to long form for plotting
    df_plot = df_style.melt(
        id_vars=['player_name'],
        value_vars=selected_features,
        var_name='feature',
        value_name='value'
    )
    fig_style = px.bar(
        df_plot,
        x='feature',
        y='value',
        labels={'value':'Score','feature':'Style Feature'},
        title=f"{selected_player} Style Profile"
    )
    st.plotly_chart(fig_style, use_container_width=True)
