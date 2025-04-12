import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. Load Data ---
@st.cache_data
def load_data():
    overall = pd.read_csv('data/mw_overall.csv')
    style   = pd.read_csv('data/style_based_features.csv')
    return overall, style

overall_df, style_df = load_data()

# Bail if load failed
if overall_df.empty and style_df.empty:
    st.error("No data loaded. Check that data/mw_overall.csv and data/style_based_features.csv exist.")
    st.stop()

st.title("🏏 Cricket Analytics (Limited Data)")

# --- 2. Overall Batting Statistics ---
st.header("1. Avg Runs Off Bat by Innings")
df_inn = (
    overall_df
    .groupby('innings', as_index=False)
    .agg(avg_runs=('runs_off_bat','mean'))
)
fig1 = px.bar(
    df_inn,
    x='innings', y='avg_runs',
    labels={'innings':'Innings','avg_runs':'Avg Runs'},
    title="Average Runs Off Bat per Innings"
)
st.plotly_chart(fig1, use_container_width=True)

st.header("2. Top 10 Teams by Avg Runs Off Bat")
df_team = (
    overall_df
    .groupby('batting_team', as_index=False)
    .agg(avg_runs=('runs_off_bat','mean'))
    .sort_values('avg_runs', ascending=False)
    .head(10)
)
fig2 = px.bar(
    df_team,
    x='batting_team', y='avg_runs',
    labels={'batting_team':'Team','avg_runs':'Avg Runs'},
    title="Top 10 Teams by Avg Runs Off Bat"
)
st.plotly_chart(fig2, use_container_width=True)

# --- 3. Style‑Based Feature Analysis ---
st.header("3. Player Style Profile")
players = style_df['name'].unique()
selected_player = st.selectbox("Select Player", players)

df_player = style_df[style_df['name']==selected_player]
if df_player.empty:
    st.write("No style data for this player.")
else:
    # pick numeric feature columns
    feat_cols = [c for c in df_player.columns if c not in ['match_id','name']]
    selected_feats = st.multiselect(
        "Features to plot", options=feat_cols, default=feat_cols[:5]
    )
    if not selected_feats:
        st.info("Please select at least one feature.")
    else:
        df_melt = df_player.melt(
            id_vars=['name'],
            value_vars=selected_feats,
            var_name='feature',
            value_name='value'
        )
        fig3 = px.bar(
            df_melt,
            x='feature', y='value',
            labels={'feature':'Style Feature','value':'Value'},
            title=f"{selected_player} Style Features"
        )
        st.plotly_chart(fig3, use_container_width=True)
