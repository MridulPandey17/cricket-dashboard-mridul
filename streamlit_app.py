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

# Ignore specific warnings if needed
warnings.filterwarnings("ignore", message=".*Country name specification contains tokens.*")

# --- Constants ---
ALLOWED_FORMATS = ["ODI", "T20I", "Test"]
COLOR_PLAYED = "#008080"
COLOR_NOT_PLAYED = "#D3D3D3"
# Using simpler keys for session state
S_KEY_OPPOSITION = "selected_opposition"
S_KEY_PLAYER = "player"
S_KEY_FORMAT = "format"

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Loads and pre-filters the cricket data."""
    try:
        df = pd.read_csv("data/total_data.csv")
        numeric_cols = ['runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled',
                        'runs_conceded', 'bowled_done', 'lbw_done', 'player_out',
                        'fours_scored', 'sixes_scored']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['out_kind'] = df['out_kind'].fillna('not out')
        df_filtered = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
        if df_filtered.empty:
             st.warning(f"No data found for the allowed formats: {', '.join(ALLOWED_FORMATS)}")
             return None
        # Standardize known tricky opposition names
        df_filtered['opposition_team'] = df_filtered['opposition_team'].replace({
            'U.A.E.': 'United Arab Emirates', 'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea', 'USA': 'United States of America',
            'West Indies Cricket Board': 'West Indies' # Add specific examples if needed
        }).str.strip() # Remove leading/trailing whitespace
        return df_filtered
    except FileNotFoundError:
        st.error("Error: `data/total_data.csv` not found. Please ensure it's in a 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading/processing data: {e}")
        return None

def get_country_code(name, code_type='alpha_3'):
    """Gets the ISO 3166-1 alpha-3 or alpha-2 code for a country name."""
    overrides_alpha3 = {
        "United Arab Emirates": "ARE", "Scotland": "GBR", "United States of America": "USA",
        "Netherlands": "NLD", "England": "GBR", "Ireland": "IRL", "West Indies": "JAM", # Jamaica proxy
        "Hong Kong": "HKG", "Papua New Guinea": "PNG", "Bermuda": "BMU", "Afghanistan": "AFG",
        "Bangladesh": "BGD", "India": "IND", "Pakistan": "PAK", "Sri Lanka": "LKA", "Australia": "AUS",
        "New Zealand": "NZL", "South Africa": "ZAF", "Zimbabwe": "ZWE", "Kenya": "KEN",
        "Canada": "CAN", "Namibia": "NAM", "Nepal": "NPL", "Oman": "OMN",
        "ICC World XI": None, "Asia XI": None, "Africa XI": None, "East Africa": None,
    }
    # Handle direct name match first (case-insensitive comparison might be needed if data varies)
    std_name = name.strip() # Ensure no whitespace issues
    lookup_name = overrides_alpha3.get(std_name, std_name)

    if lookup_name is None: return None
    try:
        # Prioritize lookup for common names
        country = pycountry.countries.lookup(lookup_name)
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        # Fallback to fuzzy search if direct lookup fails
        try:
            results = pycountry.countries.search_fuzzy(lookup_name)
            if results: return results[0].alpha_3 if code_type == 'alpha_3' else results[0].alpha_2
            # else: st.warning(f"Code not found for: {name} (Lookup: {lookup_name})")
            return None
        except LookupError:
            # st.warning(f"Fuzzy search failed for: {name}")
            return None
    except Exception:
        # st.warning(f"Error during code lookup for {name}: {e}")
        return None

@st.cache_data
def get_all_country_iso3():
    """Returns a set of all official ISO 3166-1 alpha-3 codes."""
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

# --- Main App ---
st.set_page_config(page_title="Cricket Player Stats", layout="wide")
st.title("🏏 International Cricket Player Performance Analyzer")

# Load Data (cached)
df_full = load_data()
if df_full is None or df_full.empty:
    st.error("Failed to load data for analysis.")
    st.stop()

# --- Initialize Session State ---
if S_KEY_PLAYER not in st.session_state: st.session_state[S_KEY_PLAYER] = None
if S_KEY_FORMAT not in st.session_state: st.session_state[S_KEY_FORMAT] = None
if S_KEY_OPPOSITION not in st.session_state: st.session_state[S_KEY_OPPOSITION] = None

# --- Sidebar Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df_full["name"].unique())
# Use session state value if available and valid, otherwise default
player_idx = players.index(st.session_state[S_KEY_PLAYER]) if st.session_state[S_KEY_PLAYER] in players else 0
format_idx = ALLOWED_FORMATS.index(st.session_state[S_KEY_FORMAT]) if st.session_state[S_KEY_FORMAT] in ALLOWED_FORMATS else 0

selected_player = st.sidebar.selectbox("Choose a Player", players, index=player_idx)
selected_format = st.sidebar.radio("Select Format", ALLOWED_FORMATS, index=format_idx)

# --- Detect Change & Update State ---
# Check if selections changed compared to state, if so, reset opposition and update state
if selected_player != st.session_state[S_KEY_PLAYER] or selected_format != st.session_state[S_KEY_FORMAT]:
    # print(f"Change detected: Player {st.session_state[S_KEY_PLAYER]}->{selected_player}, Format {st.session_state[S_KEY_FORMAT]}->{selected_format}. Resetting opposition.") # Debug
    st.session_state[S_KEY_PLAYER] = selected_player
    st.session_state[S_KEY_FORMAT] = selected_format
    st.session_state[S_KEY_OPPOSITION] = None # Reset opposition selection

# --- Filter Data for Current Player/Format ---
player_df = df_full[(df_full["name"] == st.session_state[S_KEY_PLAYER]) &
                      (df_full["match_type"] == st.session_state[S_KEY_FORMAT])].copy()

if player_df.empty:
    st.warning(f"No data found for {st.session_state[S_KEY_PLAYER]} in {st.session_state[S_KEY_FORMAT]} format.")
    st.session_state[S_KEY_OPPOSITION] = None # Ensure reset if no data
    st.stop() # Stop if no data for this combo

# --- Prepare Map Data ---
st.subheader(f"World Map: {st.session_state[S_KEY_PLAYER]}'s Opponents in {st.session_state[S_KEY_FORMAT]}")
st.markdown("Click on a highlighted country on the map to view detailed stats below.")

# Generate ISO codes and map
player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))
opp_stats = player_df.dropna(subset=['iso_code']).groupby(
    ["opposition_team", "iso_code"], as_index=False
).agg(
    Innings=('match_id', 'nunique'),
    TotalRuns=('runs_scored', 'sum')
)
iso_to_opposition_map = opp_stats.set_index('iso_code')['opposition_team'].to_dict()
# print("ISO to Opposition Map:", iso_to_opposition_map) # Debug: Check the mapping

# Create world map base
all_iso3_codes = get_all_country_iso3()
world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])
world_df = world_df.merge(opp_stats[['iso_code', 'TotalRuns', 'Innings']], on='iso_code', how='left')
world_df['PlayedAgainst'] = world_df['TotalRuns'].notna()
world_df['HoverName'] = world_df['iso_code'].map(iso_to_opposition_map).fillna('N/A') # Show mapped name or N/A
world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

# --- Create and Display Interactive Map ---
fig = px.choropleth(
    world_df, locations="iso_code", locationmode="ISO-3",
    color="PlayedAgainst", color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED},
    hover_name="HoverName",
    hover_data={"iso_code": False, "PlayedAgainst": False, "TotalRuns": ':,.0f', "Innings": True},
    title=f"Clickable Map: Opponents for {st.session_state[S_KEY_PLAYER]} in {st.session_state[S_KEY_FORMAT]}",
)
fig.update_layout(showlegend=False, geo=dict(showframe=False, showcoastlines=False, projection_type='natural earth', bgcolor='rgba(0,0,0,0)', landcolor=COLOR_NOT_PLAYED, subunitcolor='white'), margin={"r":0,"t":40,"l":0,"b":0}, coloraxis_showscale=False)
fig.update_traces(marker_line_width=0.5, marker_line_color='white', selector=dict(type='choropleth'))

# Use plotly_events, ensuring key changes when player/format changes
map_key = f"map_click_{st.session_state[S_KEY_PLAYER]}_{st.session_state[S_KEY_FORMAT]}"
selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=map_key)

# --- Process Click Result and Update State ---
# This part executes *after* the click happened in the *previous* run
clicked_opposition_name = None
if selected_points:
    clicked_iso = selected_points[0].get('location')
    # print(f"Click detected in run. ISO: {clicked_iso}. Point data: {selected_points[0]}") # Debug
    if clicked_iso and clicked_iso in iso_to_opposition_map:
        # Get the name corresponding to the clicked ISO code
        clicked_opposition_name = iso_to_opposition_map[clicked_iso]
        # print(f"Mapped to: {clicked_opposition_name}") # Debug
        # **Crucially, update the session state HERE if the click was valid**
        # This ensures the state reflects the click *before* the stats display logic runs below
        if st.session_state[S_KEY_OPPOSITION] != clicked_opposition_name:
            st.session_state[S_KEY_OPPOSITION] = clicked_opposition_name
            # print(f"Session state '{S_KEY_OPPOSITION}' updated to: {st.session_state[S_KEY_OPPOSITION]}") # Debug
            # **Force a rerun AFTER state update to guarantee display refresh**
            st.rerun() # <--- Add this rerun

# --- Display Stats Based on Current Session State ---
st.markdown("---")
current_selection_for_stats = st.session_state[S_KEY_OPPOSITION]
# print(f"Rendering stats section. Current selection in state: {current_selection_for_stats}") # Debug

if current_selection_for_stats:
    st.subheader(f"Detailed Stats vs: {current_selection_for_stats}")
    # Filter using the name reliably stored in session state
    country_stats_df = player_df[player_df["opposition_team"] == current_selection_for_stats]

    if not country_stats_df.empty:
        player_team = country_stats_df["player_team"].iloc[0]
        st.markdown(f"#### {st.session_state[S_KEY_PLAYER]} ({player_team}) vs {current_selection_for_stats} ({st.session_state[S_KEY_FORMAT]})")

        # --- Calculate Stats --- (Identical calculations)
        innings = country_stats_df.shape[0]; total_runs = int(country_stats_df["runs_scored"].sum())
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
        bowling_modes = {}; bowled = int(country_stats_df['bowled_done'].sum()); lbw = int(country_stats_df['lbw_done'].sum()) # Example
        if bowled > 0: bowling_modes['Bowled'] = bowled
        if lbw > 0: bowling_modes['LBW'] = lbw
        top_bowling_mode = max(bowling_modes, key=bowling_modes.get) if bowling_modes else "N/A"; most_common_bowling_count = bowling_modes.get(top_bowling_mode, 0)

        # --- Display Stats Columns --- (Identical display)
        col1, col2 = st.columns(2)
        with col1: st.subheader("Batting Summary"); st.metric("Innings", innings); st.metric("Runs", total_runs); st.metric("Average", f"{batting_avg:.2f}" if batting_avg != float('inf') else "N/A"); st.metric("Strike Rate", f"{batting_strike_rate:.2f}"); st.metric("Fours", fours); st.metric("Sixes", sixes)
        with col2: st.subheader("Bowling Summary"); st.metric("Wickets", total_wickets); st.metric("Runs Conceded", total_runs_conceded); st.metric("Average", f"{bowling_avg:.2f}" if bowling_avg != float('inf') else "N/A"); st.metric("Strike Rate", f"{bowling_strike_rate:.2f}" if bowling_strike_rate != float('inf') else "N/A"); st.metric("Economy", f"{bowling_economy:.2f}"); st.metric(f"Top Dismissal", f"{top_bowling_mode} ({most_common_bowling_count} times)" if top_bowling_mode != "N/A" else "N/A")
        st.subheader("Dismissal Analysis (Batting)"); st.dataframe(dismissal_counts.style.format({"Count": "{:,}"}), use_container_width=True) if not dismissal_counts.empty else st.info(f"Not dismissed against {current_selection_for_stats}.")

        # --- Export Buttons --- (Identical export)
        st.subheader("Download Data"); col_btn1, col_btn2 = st.columns(2)
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8"); csv_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.csv".replace(" ", "_")
        with col_btn1: st.download_button("⬇️ Download CSV", csv_data, file_name=csv_filename, mime="text/csv")
        pdf_text = f"""... (rest of PDF text generation using calculated stats) ...""" # Same as before
        pdf_bytes = export_pdf(pdf_text); b64 = base64.b64encode(pdf_bytes.read()).decode(); pdf_filename = f"{st.session_state[S_KEY_PLAYER]}_vs_{current_selection_for_stats}_{st.session_state[S_KEY_FORMAT]}_stats.pdf".replace(" ", "_"); href = f'<a ...>📄 Download PDF</a>' # Same as before
        with col_btn2: st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning(f"Could not retrieve detailed stats for {st.session_state[S_KEY_PLAYER]} against {current_selection_for_stats}. Check data consistency.")
else:
    st.info("Select a player and format, then click a highlighted country on the map to view detailed statistics.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Data source: `data/total_data.csv`.\nMap Interaction powered by `streamlit-plotly-events`.")

# Optional: Add debug view for session state
# with st.sidebar.expander("Debug Info"):
#    st.write(st.session_state)
