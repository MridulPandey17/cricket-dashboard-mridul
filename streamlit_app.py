import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# --- Helper Functions ---

@st.cache_data
def load_data():
    df = pd.read_csv("data/total_data.csv")
    return df

def get_country_code(name):
    overrides = {
        "U.A.E.": "AE", "UAE": "AE", "Scotland": "GB", "USA": "US",
        "Netherlands": "NL", "England": "GB", "Ireland": "IE",
        "West Indies": "JM", "Hong Kong": "HK"
    }
    try:
        return pycountry.countries.lookup(overrides.get(name, name)).alpha_2
    except:
        return None

def get_flag_emoji(country_code):
    if country_code is None:
        return ""
    return chr(127397 + ord(country_code[0])) + chr(127397 + ord(country_code[1]))

def export_pdf(stats_text, filename="player_stats.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text = c.beginText(40, 750)
    for line in stats_text.split("\n"):
        text.textLine(line)
    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

# --- Main App ---
st.set_page_config(page_title="Cricket Player Stats", layout="wide")
st.title("🌍 Click a Country to View Player Stats")

# Load Data
df = load_data()

# UI: Player and Format selection
players = df["name"].unique()
player = st.selectbox("Choose a Player", sorted(players))
formats = ["ODI", "T20I", "Test"]
selected_format = st.radio("Select Format", formats, horizontal=True)

# Filter by player and format
player_df = df[(df["name"] == player) & (df["match_type"] == selected_format)]

# Map opposition teams to ISO codes
player_df["iso_code"] = player_df["opposition_team"].apply(get_country_code)

# World Map Visualization
map_df = player_df.groupby(["opposition_team", "iso_code"], as_index=False)["runs_scored"].sum()
fig = px.choropleth(
    map_df,
    locations="iso_code",
    color_discrete_sequence=["#83c5be"],
    hover_name="opposition_team",
    locationmode="ISO-3",  # ISO-3 to allow better coverage
)
fig.update_traces(marker_line_width=0.5)
fig.update_layout(coloraxis_showscale=False)

st.subheader(f"{player}'s Performance against Teams in {selected_format}")
selected = st.plotly_chart(fig, use_container_width=True)

# Get clicked country from map
clicked_country = st.session_state.get("selected_country", None)

# Workaround: Let user select country via dropdown until map-click handling improves
all_opponents = sorted(player_df["opposition_team"].unique())
selected_country = st.selectbox("Or select country manually:", all_opponents)

# Filter stats by selected country
country_stats = player_df[player_df["opposition_team"] == selected_country]

if not country_stats.empty:
    team_code = get_country_code(country_stats["player_team"].iloc[0])
    opp_code = get_country_code(selected_country)
    team_flag = get_flag_emoji(team_code)
    opp_flag = get_flag_emoji(opp_code)

    st.markdown(f"### {player} ({team_flag} {country_stats['player_team'].iloc[0]}) vs {opp_flag} {selected_country}")

    total_runs = country_stats["runs_scored"].sum()
    innings = country_stats.shape[0]
    avg_balls_faced = country_stats["balls_faced"].mean()
    batting_strike_rate = (country_stats["runs_scored"].sum() / country_stats["balls_faced"].sum()) * 100 if country_stats["balls_faced"].sum() > 0 else 0

    total_wickets = country_stats["wickets_taken"].sum()
    total_balls_bowled = country_stats["balls_bowled"].sum()
    bowling_strike_rate = (total_balls_bowled / total_wickets) if total_wickets > 0 else 0
    bowling_avg = (country_stats["runs_conceded"].sum() / total_wickets) if total_wickets > 0 else 0

    # Dismissals
    top_dismissals = (
        country_stats["out_kind"]
        .value_counts()
        .head(3)
        .to_frame("Count")
        .reset_index()
        .rename(columns={"index": "Dismissal Type"})
    )

    # Most common bowling mode
    dismissal_cols = ["bowled_done", "lbw_done"]
    bowling_modes = country_stats[dismissal_cols].sum().sort_values(ascending=False)
    top_bowling_mode = bowling_modes.idxmax().replace("_done", "").upper()

    # Display stats
    st.markdown(f"""
    **Innings Played:** {innings}  
    **Total Runs Scored:** {total_runs}  
    **Average Balls Faced:** {avg_balls_faced:.2f}  
    **Batting Strike Rate:** {batting_strike_rate:.2f}  
    **Wickets Taken:** {total_wickets}  
    **Bowling Strike Rate:** {bowling_strike_rate:.2f}  
    **Bowling Average:** {bowling_avg:.2f}  
    **Most Common Bowling Dismissal:** {top_bowling_mode}  
    """)

    st.subheader("Top 3 Dismissal Modes")
    st.dataframe(top_dismissals, use_container_width=True)

    # Export Buttons
    csv = country_stats.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name=f"{player}_{selected_country}.csv", mime="text/csv")

    # PDF Export
    pdf_text = f"""{player} vs {selected_country} ({selected_format}):
Innings: {innings}
Total Runs: {total_runs}
Avg Balls Faced: {avg_balls_faced:.2f}
Batting SR: {batting_strike_rate:.2f}
Wickets Taken: {total_wickets}
Bowling SR: {bowling_strike_rate:.2f}
Bowling Avg: {bowling_avg:.2f}
Top Bowling Dismissal: {top_bowling_mode}
"""
    pdf_bytes = export_pdf(pdf_text)
    b64 = base64.b64encode(pdf_bytes.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{player}_{selected_country}.pdf">📄 Download PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.warning(f"No stats found for {player} against {selected_country} in {selected_format}.")

