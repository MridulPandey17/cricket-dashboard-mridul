import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import warnings
import json # For printing session state nicely

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
S_KEY_MAP_DATA = "map_iso_to_name" # Store the mapping relevant to the current view

# --- Helper Functions ---

# @st.cache_data # Re-enable caching once interaction is confirmed working
def load_data():
    """Loads, cleans, and pre-filters the cricket data."""
    # st.write("DEBUG: Running load_data()")
    try:
        df = pd.read_csv("data/total_data.csv")
        # Basic Cleaning
        numeric_cols = ['runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled', 'runs_conceded', 'bowled_done', 'lbw_done', 'player_out', 'fours_scored', 'sixes_scored']
        for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['out_kind'] = df['out_kind'].fillna('not out')

        # Filter for allowed formats
        df_filtered = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
        if df_filtered.empty:
             st.warning(f"No data found for allowed formats: {', '.join(ALLOWED_FORMATS)}")
             return None

        # Standardize known tricky opposition names (CRITICAL STEP)
        # Add ALL variations found in your data
        df_filtered['opposition_team'] = df_filtered['opposition_team'].replace({
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea',
            'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies', # Example
            'England Lions': None, # Example: Exclude non-international teams if needed
            'Ireland A': None,
            # Add more specific replacements as needed
        }).str.strip()

        # Remove rows where standardization resulted in None (e.g., non-international teams)
        df_filtered.dropna(subset=['opposition_team'], inplace=True)

        # st.write(f"DEBUG: load_data() finished, shape: {df_filtered.shape}")
        return df_filtered
    except FileNotFoundError:
        st.error("Fatal Error: `data/total_data.csv` not found. Place it in the 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"Fatal Error during data loading/processing: {e}")
        return None

def get_country_code(name, code_type='alpha_3'):
    """Gets the ISO 3166-1 alpha-3 code for a country name. More robust."""
    # Prioritize known overrides and handle edge cases
    overrides_alpha3 = {
        # Use standardized names from load_data() as keys
        "United Arab Emirates": "ARE", "Scotland": "GBR", "United States of America": "USA",
        "Netherlands": "NLD", "England": "GBR", "Ireland": "IRL", "West Indies": "JAM", # Jamaica proxy
        "Hong Kong": "HKG", "Papua New Guinea": "PNG", "Bermuda": "BMU", "Afghanistan": "AFG",
        "Bangladesh": "BGD", "India": "IND", "Pakistan": "PAK", "Sri Lanka": "LKA", "Australia": "AUS",
        "New Zealand": "NZL", "South Africa": "ZAF", "Zimbabwe": "ZWE", "Kenya": "KEN",
        "Canada": "CAN", "Namibia": "NAM", "Nepal": "NPL", "Oman": "OMN",
        # Teams without standard ISO codes - explicitly map to None
        "ICC World XI": None, "Asia XI": None, "Africa XI": None, "East Africa": None,
    }
    std_name = name.strip() if isinstance(name, str) else name
    if not std_name: return None

    # 1. Check direct override
    if std_name in overrides_alpha3:
        # print(f"DEBUG get_code: Override found for '{std_name}' -> {overrides_alpha3[std_name]}")
        return overrides_alpha3[std_name]

    # 2. Try pycountry lookup
    try:
        country = pycountry.countries.lookup(std_name)
        # print(f"DEBUG get_code: pycountry lookup for '{std_name}' -> {country.alpha_3}")
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        # 3. Try pycountry fuzzy search as last resort
        try:
            results = pycountry.countries.search_fuzzy(std_name)
            if results:
                # print(f"DEBUG get_code: pycountry fuzzy for '{std_name}' -> {results[0].alpha_3}")
                return results[0].alpha_3 if code_type == 'alpha_3' else results[0].alpha_2
            # print(f"DEBUG get_code: No code found for '{std_name}' after all attempts.")
            return None # Explicitly return None if nothing found
        except LookupError:
            return None # Fuzzy search itself failed
    except Exception: # Catch other potential errors during lookup
         return None


# @st.cache_data # Re-enable caching later
def get_all_country_iso3():
    """Returns a set of all official ISO 3166-1 alpha-3 codes."""
    # st.write("DEBUG: Running get_all_country_iso3()")
    return {country.alpha_3 for country in pycountry.countries}

def export_pdf(stats_text, filename="player_stats.pdf"):
    """Exports the provided text statistics to a PDF file."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text = c.beginText(inch, height - inch); text.setFont("Helvetica", 10)
    for line in stats_text.split("\n"): text.textLine(line)
    c.drawText(text); c.save(); buffer.seek(0)
    return buffer

# --- Main App Logic ---
st.title("🏏 International Cricket Player Performance Analyzer")

# Initialize Session State (Robustly)
if S_KEY_PLAYER not in st.session_state: st.session_state[S_KEY_PLAYER] = None
if S_KEY_FORMAT not in st.session_state: st.session_state[S_KEY_FORMAT] = None
if S_KEY_OPPOSITION not in st.session_state: st.session_state[S_KEY_OPPOSITION] = None
if S_KEY_MAP_DATA not in st.session_state: st.session_state[S_KEY_MAP_DATA] = {} # Initialize map data store

# --- Load Data ---
df_full = load_data()
if df_full is None or df_full.empty:
    st.error("Application cannot run without valid data.")
    st.stop()

# --- Sidebar Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df_full["name"].unique())
player_idx = players.index(st.session_state[S_KEY_PLAYER]) if st.session_state[S_KEY_PLAYER] in players else 0
format_idx = ALLOWED_FORMATS.index(st.session_state[S_KEY_FORMAT]) if st.session_state[S_KEY_FORMAT] in ALLOWED_FORMATS else 0

# Use temporary variables for widget values to detect changes
_player = st.sidebar.selectbox("Choose a Player", players, index=player_idx)
_format = st.sidebar.radio("Select Format", ALLOWED_FORMATS, index=format_idx)

# --- Detect Change & Update State ---
# Compare widget values with session state. If changed, update state and reset dependent state.
rerun_for_sidebar_change = False
if _player != st.session_state[S_KEY_PLAYER]:
    st.session_state[S_KEY_PLAYER] = _player
    st.session_state[S_KEY_OPPOSITION] = None # Reset opposition
    st.session_state[S_KEY_MAP_DATA] = {} # Reset map data
    # st.write(f"DEBUG: Player changed to {st.session_state[S_KEY_PLAYER]}, reset opposition & map.")
    rerun_for_sidebar_change = True

if _format != st.session_state[S_KEY_FORMAT]:
    st.session_state[S_KEY_FORMAT] = _format
    st.session_state[S_KEY_OPPOSITION] = None # Reset opposition
    st.session_state[S_KEY_MAP_DATA] = {} # Reset map data
    # st.write(f"DEBUG: Format changed to {st.session_state[S_KEY_FORMAT]}, reset opposition & map.")
    rerun_for_sidebar_change = True

# Force a rerun immediately if sidebar changed to ensure map/filters use the new values
if rerun_for_sidebar_change:
    st.rerun()

# --- Filter Data (Use current state values) ---
# Proceed only if player and format are selected
if not st.session_state[S_KEY_PLAYER] or not st.session_state[S_KEY_FORMAT]:
     st.info("Please select a player and format in the sidebar.")
     st.stop()

player_df = df_full[(df_full["name"] == st.session_state[S_KEY_PLAYER]) &
                      (df_full["match_type"] == st.session_state[S_KEY_FORMAT])].copy()

# If filtering results in empty DF (no data for this combo), stop and inform user
if player_df.empty:
    st.warning(f"No data found for {st.session_state[S_KEY_PLAYER]} in {st.session_state[S_KEY_FORMAT]}.")
    st.session_state[S_KEY_OPPOSITION] = None # Ensure selection is cleared
    st.session_state[S_KEY_MAP_DATA] = {}
    st.stop()

# --- Prepare Map Data (Only if needed or changed) ---
# Generate mapping only once per player/format selection
if not st.session_state[S_KEY_MAP_DATA]: # Check if map data needs generation
    # st.write("DEBUG: Generating map data...")
    player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))
    # **Critical:** Filter out rows where ISO code lookup failed BEFORE grouping
    opp_stats = player_df.dropna(subset=['iso_code']).groupby(
        ["opposition_team", "iso_code"], as_index=False
    ).agg(Innings=('match_id', 'nunique'), TotalRuns=('runs_scored', 'sum'))

    # Store the valid mapping in session state
    st.session_state[S_KEY_MAP_DATA] = opp_stats.set_index('iso_code')['opposition_team'].to_dict()
    # st.sidebar.expander("DEBUG: Map ISO->Name").json(st.session_state[S_KEY_MAP_DATA]) # Debug


# Use the stored mapping
iso_to_opposition_map = st.session_state[S_KEY_MAP_DATA]

# Create world map base DataFrame
all_iso3_codes = get_all_country_iso3()
world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])

# Merge aggregate stats for played countries based on current map data
# Need to recreate opp_stats df based on iso_to_opposition_map keys if needed or pass it
temp_opp_stats = pd.DataFrame({
    'iso_code': iso_to_opposition_map.keys(),
    'opposition_team': iso_to_opposition_map.values()
})
# Need to re-aggregate or pass the aggregated data too
# Let's re-filter and aggregate here to ensure consistency
valid_opp_df = player_df[player_df['iso_code'].isin(iso_to_opposition_map.keys())]
agg_data = valid_opp_df.groupby('iso_code', as_index=False).agg(
    Innings=('match_id', 'nunique'),
    TotalRuns=('runs_scored', 'sum')
)

world_df = world_df.merge(agg_data, on='iso_code', how='left')
world_df['PlayedAgainst'] = world_df['TotalRuns'].notna()
world_df['HoverName'] = world_df['iso_code'].map(iso_to_opposition_map).fillna('N/A')
world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

# --- Create and Display Interactive Map ---
st.subheader(f"World Map: {st.session_state[S_KEY_PLAYER]}'s Opponents in {st.session_state[S_KEY_FORMAT]}")
st.markdown("Click on a highlighted country on the map to view detailed stats below.")

fig = px.choropleth(
    world_df, locations="iso_code", locationmode="ISO-3", color="PlayedAgainst",
    color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED}, hover_name="HoverName",
    hover_data={"iso_code": False, "PlayedAgainst": False, "TotalRuns": ':,.0f', "Innings": True},
    # title=f"Click Map: Opponents for {st.session_state[S_KEY_PLAYER]} ({st.session_state[S_KEY_FORMAT]})", # Title inside map redundant
)
fig.update_layout(showlegend=False, geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth', bgcolor='rgba(0,0,0,0)', landcolor=COLOR_NOT_PLAYED, subunitcolor='rgba(255,255,255,0.5)'), margin={"r":0,"t":10,"l":0,"b":0}, coloraxis_showscale=False) # Reduced top margin
fig.update_traces(marker_line_width=0.5, marker_line_color='white', selector=dict(type='choropleth'))

# Use plotly_events - Key must be stable for a given player/format view
map_key = f"map_click_{st.session_state[S_KEY_PLAYER]}_{st.session_state[S_KEY_FORMAT]}"
selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=map_key)

# --- Process Click Result and Update State ---
# This logic runs *after* the component potentially triggered a rerun
# st.write(f"DEBUG: Before click processing. Current Opposition State: '{st.session_state[S_KEY_OPPOSITION]}'")
# st.write(f"DEBUG: plotly_events returned: {selected_points}")

rerun_needed_after_click = False
if selected_points:
    clicked_iso = selected_points[0].get('location')
    # st.write(f"DEBUG: Click detected! ISO: '{clicked_iso}'")

    if clicked_iso and isinstance(clicked_iso, str):
        # Use the mapping stored in state
        current_map = st.session_state.get(S_KEY_MAP_DATA, {})
        if clicked_iso in current_map:
            clicked_opposition_name = current_map[clicked_iso]
            # st.write(f"DEBUG: Mapped ISO '{clicked_iso}' to Opposition: '{clicked_opposition_name}'")

            # Update state only if the selection has actually changed
            if st.session_state[S_KEY_OPPOSITION] != clicked_opposition_name:
                st.session_state[S_KEY_OPPOSITION] = clicked_opposition_name
                # st.write(f"DEBUG: Session state '{S_KEY_OPPOSITION}' UPDATED to: '{st.session_state[S_KEY_OPPOSITION]}'")
                rerun_needed_after_click = True # Set flag to rerun
            # else: st.write(f"DEBUG: Clicked same country ('{clicked_opposition_name}'), no state change/rerun needed.")
        else:
            # Clicked on a grey area or an unmapped country for this player/format
            # st.write(f"DEBUG: Clicked ISO '{clicked_iso}' not in current map data for this player/format.")
            if st.session_state[S_KEY_OPPOSITION] is not None:
                 st.session_state[S_KEY_OPPOSITION] = None # Reset if clicking outside mapped area
                 rerun_needed_after_click = True
                 # st.write(f"DEBUG: Clicked non-opponent, reset opposition state.")
    # else: st.write("DEBUG: Click event had no valid ISO location.")

# Trigger rerun IF the click processing resulted in a state change
if rerun_needed_after_click:
    # st.write("DEBUG: Rerun triggered after click processing updated state.")
    st.rerun()

# --- Display Stats Based on Current Session State ---
st.markdown("---") # Visual separator
current_selection_for_stats = st.session_state[S_KEY_OPPOSITION]
# st.write(f"DEBUG: Rendering stats section. Current opposition in state: '{current_selection_for_stats}'")

if current_selection_for_stats:
    st.subheader(f"Detailed Stats vs: {current_selection_for_stats}")
    # Filter the already filtered player_df using the name from session state
    country_stats_df = player_df[player_df["opposition_team"] == current_selection_for_stats]

    if not country_stats_df.empty:
        # Extract player's team name from the filtered data
        player_team = country_stats_df["player_team"].iloc[0] if "player_team" in country_stats_df.columns else "N/A"
        st.markdown(f"#### {st.session_state[S_KEY_PLAYER]} ({player_team}) vs {current_selection_for_stats} ({st.session_state[S_KEY_FORMAT]})")

        # --- Calculate Stats --- (Keep calculations concise)
        innings = country_stats_df.shape[0]; total_runs = int(country_stats_df["runs_scored"].sum());
        total_balls_faced = int(country_stats_df["balls_faced"].sum()); outs = int(country_stats_df["player_out"].sum())
        batting_avg = total_runs / outs if outs > 0 else float('inf') if total_runs > 0 else 0.0
        batting_strike_rate = (total_runs / total_balls_faced) * 100 if total_balls_faced > 0 else 0.0
        fours = int(country_stats_df["fours_scored"].sum()); sixes = int(country_stats_df["sixes_scored"].sum())
        total_wickets = int(country_stats_df["wickets_taken"].sum()); total_balls_bowled = int(country_stats_df["balls_bowled"].sum())
        total_runs_conceded = int(country_stats_df["runs_conceded"].sum())
        bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else float('inf') if total_runs_conceded > 0 else 0.0
        bowling_strike_rate = total_balls_bowled / total_wickets if total_wickets > 0 else float('inf') if total_balls_bowled > 0 else 0.0
        bowling_economy = (total_runs_conceded / total_balls_bowled) * 6 if total_balls_bowled > 0 else 0.0
        dismissal_counts = country_stats_df[country_stats_df["player_out"] == 1]["out_kind"].value_counts().reset_index(); dismissal_counts.columns = ["Dismissal Type", "Count"]
        bowling_modes = {}; bowled = int(country_stats_df['bowled_done'].sum()); lbw = int(country_stats_df['lbw_done'].sum())
        if bowled > 0: bowling_modes['Bowled'] = bowled
        if lbw > 0: bowling_modes['LBW'] = lbw
        top_bowling_mode = max(bowling_modes, key=bowling_modes.get) if bowling_modes else "N/A"; most_common_bowling_count = bowling_modes.get(top_bowling_mode, 0)

        # --- Display Stats Columns ---
        col1, col2 = st.columns(2)
        with col1: st.subheader("Batting"); st.metric("Inn", innings); st.metric("Runs", total_runs); st.metric("Avg", f"{batting_avg:.2f}" if batting_avg!=float('inf') else "N/A"); st.metric("SR", f"{batting_strike_rate:.2f}"); st.metric("4s", fours); st.metric("6s", sixes)
        with col2: st.subheader("Bowling"); st.metric("Wkts", total_wickets); st.metric("Runs Conceded", total_runs_conceded); st.metric("Avg", f"{bowling_avg:.2f}" if bowling_avg!=float('inf') else "N/A"); st.metric("SR", f"{bowling_strike_rate:.2f}" if bowling_strike_rate!=float('inf') else "N/A"); st.metric("Econ", f"{bowling_economy:.2f}"); st.metric("Top Dismissal", f"{top_bowling_mode} ({most_common_bowling_count}x)" if top_bowling_mode != "N/A" else "N/A")
        st.subheader("Dismissals (Batting)"); st.dataframe(dismissal_counts.style.format({"Count": "{:,}"}), use_container_width=True) if not dismissal_counts.empty else st.info(f"Not dismissed vs {current_selection_for_stats}.")

        # --- Export Buttons ---
        st.subheader("Download"); col_btn3, col_btn4 = st.columns(2) # Use different variable names
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8"); csv_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.csv".replace(" ", "_")
        with col_btn3: st.download_button("⬇️ CSV", csv_data, file_name=csv_filename, mime="text/csv")
        # Simplified PDF text generation for brevity
        pdf_text = f"""Player: {st.session_state[S_KEY_PLAYER]} vs {current_selection_for_stats} ({st.session_state[S_KEY_FORMAT]})\nRuns: {total_runs}, Bat Avg: {f"{batting_avg:.2f}" if batting_avg!=float('inf') else "N/A"}, Bat SR: {batting_strike_rate:.2f}\nWkts: {total_wickets}, Bowl Avg: {f"{bowling_avg:.2f}" if bowling_avg!=float('inf') else "N/A"}, Bowl Econ: {bowling_economy:.2f}\n"""
        pdf_bytes = export_pdf(pdf_text); b64 = base64.b64encode(pdf_bytes.read()).decode(); pdf_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.pdf".replace(" ", "_"); href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" style="display: inline-block; padding: 0.375rem 0.75rem; font-size: 1rem; font-weight: 400; line-height: 1.5; color: #fff; background-color: #0069d9; border-color: #0062cc; text-align: center; vertical-align: middle; border: 1px solid transparent; border-radius: 0.25rem; text-decoration: none;">📄 PDF</a>'
        with col_btn4: st.markdown(href, unsafe_allow_html=True)

    else:
        # This case implies the opposition name IS in the state, but filtering player_df failed
        st.error(f"Internal inconsistency: Could not filter stats for '{current_selection_for_stats}'. Check data integrity for player '{st.session_state[S_KEY_PLAYER]}'.")
else:
    # Message when no country is selected (initial state or after reset)
    st.info("⬅️ Select player/format, then click a highlighted country on the map for stats.")


# --- Footer / Debug ---
st.sidebar.markdown("---")
st.sidebar.info("Data: `total_data.csv` | Map: `streamlit-plotly-events`")
# Provide easy access to current state for debugging
# with st.sidebar.expander("Current Session State"):
#    st.json(st.session_state.to_dict())
