import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- 1. Debug Info ---
st.sidebar.title("🛠️ Debug Info")
st.sidebar.write("**CWD:**", os.getcwd())
st.sidebar.write("**Files:**", os.listdir("."))
st.sidebar.write("**Data Folder:**", os.listdir("data"))

# --- 2. Load Data ---
@st.cache_data
def load_data():
    overall = pd.read_csv('data/mw_overall.csv')
    style = pd.read_csv('data/style_based_features.csv')
    return overall, style

overall_df, style_df = load_data()
st.sidebar.success(f"Loaded mw_overall.csv: shape = {overall_df.shape}")
st.sidebar.success(f"Loaded style_based_features.csv: shape = {style_df.shape}")

st.sidebar.subheader("🧾 Columns in mw_overall.csv")
st.sidebar.write(overall_df.columns.tolist())

st.sidebar.subheader("🧾 Columns in style_based_features.csv")
st.sidebar.write(style_df.columns.tolist())

st.title("🏏 Cricket Analytics (Limited Data)")

# --- 3. Average Total Runs per Innings ---
st.header("1. Average Total Runs per Innings")
if 'innings' in overall_df.columns and 'total_runs' in overall_df.columns:
    df_inn = (
        overall_df
        .groupby('innings', as_index=False)
        .agg(avg_runs=('total_runs', 'mean'))
    )
    fig1 = px.bar(
        df_inn,
        x='innings', y='avg_runs',
        labels={'innings': 'Innings', 'avg_runs': 'Avg Runs'},
        title="Average Total Runs per Innings"
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Columns 'innings' or 'total_runs' not found in mw_overall.csv")

# --- 4. Top 10 Teams by Avg Total Runs ---
st.header("2. Top 10 Teams by Avg Total Runs")
if 'team' in overall_df.columns and 'total_runs' in overall_df.columns:
    df_team = (
        overall_df
        .groupby('team', as_index=False)
        .agg(avg_runs=('total_runs', 'mean'))
        .sort_values('avg_runs', ascending=False)
        .head(10)
    )
    fig2 = px.bar(
        df_team,
        x='team', y='avg_runs',
        labels={'team': 'Team', 'avg_runs': 'Avg Runs'},
        title="Top 10 Teams by Avg Total Runs"
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("Columns 'team' or 'total_runs' not found in mw_overall.csv")

# --- 5. Style-Based Feature Analysis ---
st.header("3. Player Style Profile")
if 'name' in style_df.columns:
    players = style_df['name'].unique()
    selected_player = st.selectbox("Select Player", players)

    df_player = style_df[style_df['name'] == selected_player]
    if df_player.empty:
        st.write("No style data for this player.")
    else:
        feature_cols = [c for c in df_player.columns if c not in ['match_id', 'name']]
        selected_feats = st.multiselect("Features to Plot", options=feature_cols, default=feature_cols[:5])
        if selected_feats:
            df_melt = df_player.melt(
                id_vars='name',
                value_vars=selected_feats,
                var_name='feature',
                value_name='value'
            )
            fig3 = px.bar(
                df_melt,
                x='feature', y='value',
                labels={'feature': 'Style Feature', 'value': 'Value'},
                title=f"{selected_player} Style Features"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Please select at least one feature.")
else:
    st.warning("Column 'name' not found in style_based_features.csv")
