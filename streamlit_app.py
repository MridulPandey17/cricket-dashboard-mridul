import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go
# Remove: from streamlit_plotly_events import plotly_events (No longer needed)
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import warnings
import math # For calculating button columns

# --- Configuration ---
warnings.filterwarnings("ignore", message=".*Country name specification contains tokens.*")
st.set_page_config(page_title="Cricket Player Stats", layout="wide")

# --- Constants ---
ALLOWED_FORMATS = ["ODI", "T20I", "Test"]
COLOR_PLAYED = "#008080" # Teal
COLOR_NOT_PLAYED = "#E0E0E0" # Lighter Grey
# Session State Keys
S_KEY_OPPOSITION = "selected_opposition"
S_KEY_PLAYER = "player"
S_KEY_FORMAT = "format"
# Remove: S_KEY_MAP_DATA (No longer needed for interaction logic)
BUTTON_COLS = 5 # Number of columns for opponent buttons

# --- Helper Functions --- (Keep these as they are needed for map visualization)

# @st.cache_data # Re-enable caching later
def load_data():
    """Loads, cleans, and pre-filters the cricket data."""
    try:
        df = pd.read_csv("data/total_data.csv")
        numeric_cols = ['runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled', 'runs_conceded', 'bowled_done', 'lbw_done', 'player_out', 'fours_scored', 'sixes_scored']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['out_kind'] = df['out_kind'].fillna('not out')
        df_filtered = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
        if df_filtered.empty: return None
        # Standardize opposition names
        df_filtered['opposition_team'] = df_filtered['opposition_team'].replace({
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies', 'England Lions': None, 'Ireland A': None,
        }).str.strip()
        df_filtered.dropna(subset=['opposition_team'], inplace=True)
        return df_filtered
    except FileNotFoundError:
        st.error("Fatal Error: `data/total_data.csv` not found."); return None
    except Exception as e:
        st.error(f"Fatal Error during data loading: {e}"); return None

def get_country_code(name, code_type='alpha_3'):
    """Gets the ISO 3166-1 alpha-3 code."""
    overrides_alpha3 = { # Use standardized names
        "United Arab Emirates": "ARE", "Scotland": "GBR", "United States of America": "USA",
        "Netherlands": "NLD", "England": "GBR", "Ireland": "IRL", "West Indies": "JAM",
        "Hong Kong": "HKG", "Papua New Guinea": "PNG", "Bermuda": "BMU", "Afghanistan": "AFG",
        "Bangladesh": "BGD", "India": "IND", "Pakistan": "PAK", "Sri Lanka": "LKA", "Australia": "AUS",
        "New Zealand": "NZL", "South Africa": "ZAF", "Zimbabwe": "ZWE", "Kenya": "KEN",
        "Canada": "CAN", "Namibia": "NAM", "Nepal": "NPL", "Oman": "OMN",
        "ICC World XI": None, "Asia XI": None, "Africa XI": None, "East Africa": None,
    }
    std_name = name.strip() if isinstance(name, str) else name
    if not std_name: return None
    if std_name in overrides_alpha3: return overrides_alpha3[std_name]
    try:
        country = pycountry.countries.lookup(std_name)
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        try:
            results = pycountry.countries.search_fuzzy(std_name)
            if results: return results[0].alpha_3 if code_type == 'alpha_3' else results[0].alpha_2
            return None
        except LookupError: return None
    except Exception: return None

# @st.cache_data # Re-enable caching later
def get_all_country_iso3():
    return {country.alpha_3 for country in pycountry.countries}

def export_pdf(stats_text, filename="player_stats.pdf"):
    buffer = BytesIO(); c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter; text = c.beginText(inch, height - inch); text.setFont("Helvetica", 10)
    for line in stats_text.split("\n"): text.textLine(line)
    c.drawText(text); c.save(); buffer.seek(0)
    return buffer

# --- Main App Logic ---
st.title("🏏 International Cricket Player Performance Analyzer")

# Initialize Session State
if S_KEY_PLAYER not in st.session_state: st.session_state[S_KEY_PLAYER] = None
if S_KEY_FORMAT not in st.session_state: st.session_state[S_KEY_FORMAT] = None
if S_KEY_OPPOSITION not in st.session_state: st.session_state[S_KEY_OPPOSITION] = None

# --- Load Data ---
df_full = load_data()
if df_full is None or df_full.empty:
    st.error("Application cannot run without valid data."); st.stop()

# --- Sidebar Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df_full["name"].unique())
player_idx = players.index(st.session_state[S_KEY_PLAYER]) if st.session_state[S_KEY_PLAYER] in players else 0
format_idx = ALLOWED_FORMATS.index(st.session_state[S_KEY_FORMAT]) if st.session_state[S_KEY_FORMAT] in ALLOWED_FORMATS else 0
_player = st.sidebar.selectbox("Choose a Player", players, index=player_idx)
_format = st.sidebar.radio("Select Format", ALLOWED_FORMATS, index=format_idx)

# --- Detect Change & Update State ---
if _player != st.session_state[S_KEY_PLAYER] or _format != st.session_state[S_KEY_FORMAT]:
    st.session_state[S_KEY_PLAYER] = _player
    st.session_state[S_KEY_FORMAT] = _format
    st.session_state[S_KEY_OPPOSITION] = None # Reset opposition on player/format change
    st.rerun() # Rerun immediately to reflect changes

# --- Filter Data ---
if not st.session_state[S_KEY_PLAYER] or not st.session_state[S_KEY_FORMAT]:
     st.info("Please select a player and format in the sidebar."); st.stop()

player_df = df_full[(df_full["name"] == st.session_state[S_KEY_PLAYER]) &
                      (df_full["match_type"] == st.session_state[S_KEY_FORMAT])].copy()

if player_df.empty:
    st.warning(f"No data found for {st.session_state[S_KEY_PLAYER]} in {st.session_state[S_KEY_FORMAT]}.")
    st.session_state[S_KEY_OPPOSITION] = None; st.stop()

# --- Add ISO Code for Map Visualization ---
player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))

# --- Prepare Data for Static Map Visualization ---
iso_to_opposition_map = player_df.dropna(subset=['iso_code']).set_index('iso_code')['opposition_team'].to_dict()
all_iso3_codes = get_all_country_iso3()
world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])
valid_map_isos = list(iso_to_opposition_map.keys())
agg_data = player_df[player_df['iso_code'].isin(valid_map_isos)].groupby('iso_code', as_index=False).agg(
    Innings=('match_id', 'nunique'), TotalRuns=('runs_scored', 'sum')
)
world_df = world_df.merge(agg_data, on='iso_code', how='left')
world_df['PlayedAgainst'] = world_df['TotalRuns'].notna()
world_df['HoverName'] = world_df['iso_code'].map(iso_to_opposition_map).fillna('N/A')
world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

# --- Display Static Map ---
st.subheader(f"World Map: {st.session_state[S_KEY_PLAYER]}'s Opponents in {st.session_state[S_KEY_FORMAT]}")
fig = px.choropleth(
    world_df, locations="iso_code", locationmode="ISO-3", color="PlayedAgainst",
    color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED}, hover_name="HoverName",
    hover_data={"iso_code": False, "PlayedAgainst": False, "TotalRuns": ':,.0f', "Innings": True},
)
fig.update_layout(showlegend=False, geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth', bgcolor='rgba(0,0,0,0)', landcolor=COLOR_NOT_PLAYED, subunitcolor='rgba(255,255,255,0.5)'), margin={"r":0,"t":10,"l":0,"b":0}, coloraxis_showscale=False)
fig.update_traces(marker_line_width=0.5, marker_line_color='white', selector=dict(type='choropleth'))
# Display the plotly chart statically
st.plotly_chart(fig, use_container_width=True) # Removed plotly_events

# --- Generate Buttons for Opponents ---
st.markdown("---")
st.subheader("Select Opponent to View Stats:")

# Get unique, sorted list of opponents the player *actually* played against
opponents_list = sorted(player_df['opposition_team'].unique())

if not opponents_list:
    st.warning("No opponents found in the data for this player/format combination.")
else:
    # Create columns for buttons
    cols = st.columns(BUTTON_COLS)
    for i, opponent_name in enumerate(opponents_list):
        col_index = i % BUTTON_COLS
        # Place button in the current column
        # Use opponent name for button label and unique key
        button_key = f"btn_{st.session_state[S_KEY_PLAYER]}_{st.session_state[S_KEY_FORMAT]}_{opponent_name.replace(' ', '_')}"
        if cols[col_index].button(opponent_name, key=button_key, use_container_width=True):
            # If a button is clicked, update the session state
            st.session_state[S_KEY_OPPOSITION] = opponent_name
            # No explicit rerun needed here, button click triggers it automatically

# --- Display Stats Based on Button Selection (Session State) ---
current_selection_for_stats = st.session_state[S_KEY_OPPOSITION]

if current_selection_for_stats:
    st.markdown("---") # Separator before stats
    st.subheader(f"Detailed Stats vs: {current_selection_for_stats}")
    # Filter player_df using the name from session state
    country_stats_df = player_df[player_df["opposition_team"] == current_selection_for_stats]

    if not country_stats_df.empty:
        player_team = country_stats_df["player_team"].iloc[0] if "player_team" in country_stats_df.columns else "N/A"
        st.markdown(f"#### {st.session_state[S_KEY_PLAYER]} ({player_team}) vs {current_selection_for_stats} ({st.session_state[S_KEY_FORMAT]})")

        # --- Calculate Stats --- (Same as before)
        innings=country_stats_df.shape[0]; total_runs=int(country_stats_df["runs_scored"].sum()); total_balls_faced=int(country_stats_df["balls_faced"].sum()); outs=int(country_stats_df["player_out"].sum())
        batting_avg=total_runs/outs if outs > 0 else float('inf') if total_runs > 0 else 0.0; batting_strike_rate=(total_runs/total_balls_faced)*100 if total_balls_faced > 0 else 0.0
        fours=int(country_stats_df["fours_scored"].sum()); sixes=int(country_stats_df["sixes_scored"].sum()); total_wickets=int(country_stats_df["wickets_taken"].sum()); total_balls_bowled=int(country_stats_df["balls_bowled"].sum())
        total_runs_conceded=int(country_stats_df["runs_conceded"].sum()); bowling_avg=total_runs_conceded/total_wickets if total_wickets > 0 else float('inf') if total_runs_conceded > 0 else 0.0
        bowling_strike_rate=total_balls_bowled/total_wickets if total_wickets > 0 else float('inf') if total_balls_bowled > 0 else 0.0; bowling_economy=(total_runs_conceded/total_balls_bowled)*6 if total_balls_bowled > 0 else 0.0
        dismissal_counts=country_stats_df[country_stats_df["player_out"] == 1]["out_kind"].value_counts().reset_index(); dismissal_counts.columns=["Dismissal Type","Count"]
        bowling_modes={}; bowled=int(country_stats_df['bowled_done'].sum()); lbw=int(country_stats_df['lbw_done'].sum())
        if bowled > 0: bowling_modes['Bowled']=bowled;
        if lbw > 0: bowling_modes['LBW']=lbw
        top_bowling_mode=max(bowling_modes,key=bowling_modes.get) if bowling_modes else "N/A"; most_common_bowling_count=bowling_modes.get(top_bowling_mode,0)

        # --- Display Stats Columns --- (Same display)
        col1, col2 = st.columns(2)
        with col1: st.subheader("Batting"); st.metric("Inn", innings); st.metric("Runs", total_runs); st.metric("Avg", f"{batting_avg:.2f}" if batting_avg!=float('inf') else "N/A"); st.metric("SR", f"{batting_strike_rate:.2f}"); st.metric("4s", fours); st.metric("6s", sixes)
        with col2: st.subheader("Bowling"); st.metric("Wkts", total_wickets); st.metric("Runs Conceded", total_runs_conceded); st.metric("Avg", f"{bowling_avg:.2f}" if bowling_avg!=float('inf') else "N/A"); st.metric("SR", f"{bowling_strike_rate:.2f}" if bowling_strike_rate!=float('inf') else "N/A"); st.metric("Econ", f"{bowling_economy:.2f}"); st.metric("Top Dismissal", f"{top_bowling_mode} ({most_common_bowling_count}x)" if top_bowling_mode != "N/A" else "N/A")
        st.subheader("Dismissals (Batting)"); st.dataframe(dismissal_counts.style.format({"Count": "{:,}"}), use_container_width=True) if not dismissal_counts.empty else st.info(f"Not dismissed vs {current_selection_for_stats}.")

        # --- Export Buttons --- (Same export)
        st.subheader("Download"); col_btn3, col_btn4 = st.columns(2)
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8"); csv_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.csv".replace(" ", "_")
        with col_btn3: st.download_button("⬇️ CSV", csv_data, file_name=csv_filename, mime="text/csv")
        pdf_text = f"""Player: {st.session_state[S_KEY_PLAYER]} vs {current_selection_for_stats} ({st.session_state[S_KEY_FORMAT]})\nRuns: {total_runs}, Bat Avg: {f"{batting_avg:.2f}" if batting_avg!=float('inf') else "N/A"}, Bat SR: {batting_strike_rate:.2f}\nWkts: {total_wickets}, Bowl Avg: {f"{bowling_avg:.2f}" if bowling_avg!=float('inf') else "N/A"}, Bowl Econ: {bowling_economy:.2f}\n""" # Simplified
        pdf_bytes = export_pdf(pdf_text); b64 = base64.b64encode(pdf_bytes.read()).decode(); pdf_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.pdf".replace(" ", "_"); href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" style="display: inline-block; padding: 0.375rem 0.75rem; font-size: 1rem; font-weight: 400; line-height: 1.5; color: #fff; background-color: #0069d9; border-color: #0062cc; text-align: center; vertical-align: middle; border: 1px solid transparent; border-radius: 0.25rem; text-decoration: none;">📄 PDF</a>'
        with col_btn4: st.markdown(href, unsafe_allow_html=True)

    else:
        st.error(f"Internal inconsistency: Could not filter stats for '{current_selection_for_stats}'.")
else:
    # Message shown when no button has been clicked yet for the current player/format
    st.info("Select an opponent from the buttons above to view detailed statistics.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Data source: `data/total_data.csv`")
# with st.sidebar.expander("Current Session State"): st.json(st.session_state.to_dict())
