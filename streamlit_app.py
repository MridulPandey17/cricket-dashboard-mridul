import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- DEBUG: show filesystem ---
st.sidebar.title("🛠️ Debug Info")
st.sidebar.write("**CWD:**", os.getcwd())
st.sidebar.write("**Files:**", os.listdir("."))

# --- 1. Load Only the Available Data with Error Reporting ---
@st.cache_data
def load_data():
    dfs = {}
    for name, path in {
        'overall_df': 'data/mw_overall.csv',
        'style_df':   'data/style_based_features.csv'
    }.items():
        try:
            dfs[name] = pd.read_csv(path)
            st.sidebar.write(f"Loaded `{path}`: shape={dfs[name].shape}")
        except Exception as e:
            st.sidebar.error(f"Error loading `{path}`:\n{e}")
            dfs[name] = pd.DataFrame()  # empty
    return dfs['overall_df'], dfs['style_df']

overall_df, style_df = load_data()

# --- Bail early if nothing loaded ---
if overall_df.empty and style_df.empty:
    st.error("No data loaded. Check that your `data/` folder is in the repo and contains the two CSVs.")
    st.stop()

# --- 2. Sidebar Controls ---
st.sidebar.header("Filters")
teamA = st.sidebar.selectbox("Team A", overall_df['teamA'].unique() if not overall_df.empty else [])
teamB = st.sidebar.selectbox("Team B", overall_df['teamB'].unique() if not overall_df.empty else [])
players = style_df['player_name'].unique() if not style_df.empty else []
selected_player = st.sidebar.selectbox("Player (Style Analysis)", players)

st.title("🏏 Cricket Analytics (Limited Data)")

# --- 3. Head‑to‑Head Team Comparison ---
st.header("1. Head‑to‑Head: Wins by Series")
if overall_df.empty:
    st.write("No head‑to‑head data available.")
else:
    df_h2h = overall_df.query("teamA == @teamA and teamB == @teamB")
    if df_h2h.empty:
        st.write("No head‑to‑head rows for this pair.")
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
if style_df.empty:
    st.write("No style‑feature data available.")
else:
    df_style = style_df[style_df['player_name'] == selected_player]
    if df_style.empty:
        st.write("No style data for this player.")
    else:
        # Choose some columns automatically if user hasn't
        style_cols = [c for c in style_df.columns if c not in ['player_name']]
        selected_features = st.multiselect(
            "Features to Plot",
            options=style_cols,
            default=style_cols[:5]
        )
        if not selected_features:
            st.write("Pick at least one feature.")
        else:
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
