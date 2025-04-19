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
SESSION_STATE_KEY_OPPOSITION = "selected_opposition_name" # More specific key
SESSION_STATE_KEY_PLAYER = "current_player"
SESSION_STATE_KEY_FORMAT = "current_format"

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
        # *** Standardize known tricky opposition names early ***
        df_filtered['opposition_team'] = df_filtered['opposition_team'].replace({
            'U.A.E.': 'United Arab Emirates',
            'UAE': 'United Arab Emirates',
            'P.N.G.': 'Papua New Guinea',
            'USA': 'United States of America' # Use full name if pycountry prefers it
            # Add more standardizations if needed based on your data
        })
        return df_filtered
    except FileNotFoundError:
        st.error("Error: `data/total_data.csv` not found. Please ensure it's in a 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading/processing data: {e}")
        return None

def get_country_code(name, code_type='alpha_3'):
    """Gets the ISO 3166-1 alpha-3 or alpha-2 code for a country name."""
    # Using full names often works better with pycountry
    overrides_alpha3 = {
        "United Arab Emirates": "ARE",
        "Scotland": "GBR", # Representing Scotland under GBR
        "United States of America": "USA",
        "Netherlands": "NLD",
        "England": "GBR", # Representing England under GBR
        "Ireland": "IRL", # Republic of Ireland
        "West Indies": "JAM", # Using Jamaica as proxy
        "Hong Kong": "HKG",
        "Papua New Guinea": "PNG",
        "Bermuda": "BMU",
        "Afghanistan": "AFG",
        "Bangladesh": "BGD",
        "India": "IND",
        "Pakistan": "PAK",
        "Sri Lanka": "LKA",
        "Australia": "AUS",
        "New Zealand": "NZL",
        "South Africa": "ZAF",
        "Zimbabwe": "ZWE",
        "Kenya": "KEN",
        "Canada": "CAN",
        "Namibia": "NAM",
        "Nepal": "NPL",
        "Oman": "OMN",
        # Teams without standard ISO codes
        "ICC World XI": None, "Asia XI": None, "Africa XI": None,
        "East Africa": None,
    }
    lookup_name = overrides_alpha3.get(name, name)
    if lookup_name is None: return None
    try:
        country = pycountry.countries.lookup(lookup_name)
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        try:
            results = pycountry.countries.search_fuzzy(lookup_name)
            if results: return results[0].alpha_3 if code_type == 'alpha_3' else results[0].alpha_2
            #else: st.warning(f"Could not find country code for: {name} (original: {lookup_name})")
            return None
        except LookupError:
            #st.warning(f"Could not find country code (fuzzy failed) for: {name}")
            return None
    except Exception: # Catch any other pycountry error
        #st.warning(f"Error looking up country code for {name}: {e}")
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
    text = c.beginText(inch, height - inch)
    text.setFont("Helvetica", 10)
    for line in stats_text.split("\n"):
        text.textLine(line)
    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

# --- Main App ---
st.set_page_config(page_title="Cricket Player Stats", layout="wide")
st.title("🏏 International Cricket Player Performance Analyzer")

# Load and Filter Data
df_full = load_data()

if df_full is None or df_full.empty:
    st.error("Failed to load data for allowed formats. Please check the data file.")
    st.stop()

# --- Initialize Session State ---
# Initialize keys if they don't exist
if SESSION_STATE_KEY_OPPOSITION not in st.session_state:
    st.session_state[SESSION_STATE_KEY_OPPOSITION] = None
if SESSION_STATE_KEY_PLAYER not in st.session_state:
    st.session_state[SESSION_STATE_KEY_PLAYER] = None
if SESSION_STATE_KEY_FORMAT not in st.session_state:
    st.session_state[SESSION_STATE_KEY_FORMAT] = None

# --- User Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df_full["name"].unique())
# Get index of last selected player, default to 0 if not found or first time
default_player_index = 0
if st.session_state[SESSION_STATE_KEY_PLAYER] in players:
    default_player_index = players.index(st.session_state[SESSION_STATE_KEY_PLAYER])
player = st.sidebar.selectbox(
    "Choose a Player",
    players,
    index=default_player_index,
    key='player_select' # Add a key to the widget
)

# Get index of last selected format
default_format_index = 0
if st.session_state[SESSION_STATE_KEY_FORMAT] in ALLOWED_FORMATS:
    default_format_index = ALLOWED_FORMATS.index(st.session_state[SESSION_STATE_KEY_FORMAT])
selected_format = st.sidebar.radio(
    "Select Format",
    ALLOWED_FORMATS,
    index=default_format_index,
    key='format_select' # Add a key to the widget
)

# --- Detect Changes and Reset Selection ---
# If player or format has changed from what's stored, reset the selected opposition
if player != st.session_state[SESSION_STATE_KEY_PLAYER] or selected_format != st.session_state[SESSION_STATE_KEY_FORMAT]:
    #st.write("Player/Format changed, resetting selection.") # Debugging line
    st.session_state[SESSION_STATE_KEY_OPPOSITION] = None
    st.session_state[SESSION_STATE_KEY_PLAYER] = player
    st.session_state[SESSION_STATE_KEY_FORMAT] = selected_format

# --- Data Filtering based on Current Player/Format Selection ---
player_df = df_full[(df_full["name"] == player) & (df_full["match_type"] == selected_format)].copy()

if player_df.empty:
    st.warning(f"No data found for {player} in {selected_format} format.")
    # Ensure selection is cleared if no data exists for this combo
    st.session_state[SESSION_STATE_KEY_OPPOSITION] = None
    st.stop()

# --- Prepare Data for Map ---
st.subheader(f"World Map: {player}'s Opponents in {selected_format}")
st.markdown("Click on a highlighted country on the map to view detailed stats below.")

# 1. Get ISO codes, handling potential errors
player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))

# 2. Aggregate stats per opponent (only include valid ISO codes)
opp_stats = player_df.dropna(subset=['iso_code']).groupby(
    ["opposition_team", "iso_code"], as_index=False
).agg(
    Innings=('match_id', 'nunique'),
    TotalRuns=('runs_scored', 'sum')
)

# 3. Create mapping from ISO code back to the *standardized* opposition team name
iso_to_opposition_map = opp_stats.set_index('iso_code')['opposition_team'].to_dict()

# 4. Create base DataFrame with ALL countries
all_iso3_codes = get_all_country_iso3()
world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])

# 5. Merge opponent stats onto the world map data
world_df = world_df.merge(opp_stats[['iso_code', 'TotalRuns', 'Innings']], on='iso_code', how='left')
world_df['PlayedAgainst'] = world_df['TotalRuns'].notna()
# Use the mapped name for hover, fallback to ISO if no match (shouldn't happen for played)
world_df['HoverName'] = world_df['iso_code'].map(iso_to_opposition_map)
# Fill HoverName for non-played countries (optional, could show ISO or name)
world_df['HoverName'] = world_df.apply(
    lambda row: get_country_code(row['iso_code'], code_type='alpha_3') if pd.isna(row['HoverName']) else row['HoverName'],
    axis=1
) # Attempt to get a name for non-played countries too
world_df['HoverName'] = world_df['HoverName'].fillna(world_df['iso_code']) # Fallback to ISO code if name lookup fails

world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

# --- Create Interactive World Map ---
fig = px.choropleth(
    world_df,
    locations="iso_code",
    locationmode="ISO-3",
    color="PlayedAgainst",
    color_discrete_map={True: COLOR_PLAYED, False: COLOR_NOT_PLAYED},
    hover_name="HoverName",
    hover_data={
        "iso_code": False,
        "PlayedAgainst": False,
        "TotalRuns": ':,.0f',
        "Innings": True
    },
    title=f"Clickable Map: Opponents for {player} in {selected_format}",
)
fig.update_layout(
    showlegend=False,
    geo=dict(
        showframe=False, showcoastlines=False,
        projection_type='natural earth',
        bgcolor='rgba(0,0,0,0)',
        landcolor=COLOR_NOT_PLAYED, # Base color for non-played
        subunitcolor='white'
    ),
    margin={"r":0,"t":40,"l":0,"b":0},
    coloraxis_showscale=False
)
fig.update_traces(
     marker_line_width=0.5,
     marker_line_color='white',
     selector=dict(type='choropleth')
)

# --- Use streamlit-plotly-events to capture clicks ---
# Use a consistent key that changes ONLY when the underlying data changes
map_key = f"map_click_{player}_{selected_format}"
selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=map_key)

# --- Process Click Event and Update Session State ---
# This block runs *after* plotly_events returns data from a click in the *previous* run
if selected_points:
    clicked_iso = selected_points[0].get('location')
    # st.write(f"Click detected! Point data: {selected_points[0]}") # Debugging
    if clicked_iso and clicked_iso in iso_to_opposition_map:
        # IMPORTANT: Update session state *here* based on the click
        new_selection = iso_to_opposition_map[clicked_iso]
        # Only update if the selection actually changed
        if st.session_state[SESSION_STATE_KEY_OPPOSITION] != new_selection:
             st.session_state[SESSION_STATE_KEY_OPPOSITION] = new_selection
             #st.write(f"Session state updated with: {new_selection}") # Debugging
             # *** Force a rerun after updating state ***
             # This ensures the rest of the script uses the *new* state value
             # However, plotly_events usually triggers this automatically.
             # Uncommenting st.rerun() might cause infinite loops if not careful.
             # st.rerun()
    # else:
        # st.write(f"Clicked ISO '{clicked_iso}' not found in mapping or is invalid.") # Debugging

# --- Display Detailed Stats Based on Session State ---
# Retrieve the selected country *from session state* for display logic
current_selected_opposition = st.session_state[SESSION_STATE_KEY_OPPOSITION]

st.markdown("---") # Separator

if current_selected_opposition:
    st.subheader(f"Detailed Stats vs: {current_selected_opposition}")
    # Filter using the name stored in session state
    country_stats_df = player_df[player_df["opposition_team"] == current_selected_opposition]

    if not country_stats_df.empty:
        player_team = country_stats_df["player_team"].iloc[0]
        st.markdown(f"#### {player} ({player_team}) vs {current_selected_opposition} ({selected_format})")

        # --- Calculate Stats --- (Identical calculations as before)
        innings = country_stats_df.shape[0]
        total_runs = int(country_stats_df["runs_scored"].sum())
        total_balls_faced = int(country_stats_df["balls_faced"].sum())
        outs = int(country_stats_df["player_out"].sum())
        batting_avg = total_runs / outs if outs > 0 else float('inf') if total_runs > 0 else 0.0
        batting_strike_rate = (total_runs / total_balls_faced) * 100 if total_balls_faced > 0 else 0.0
        fours = int(country_stats_df["fours_scored"].sum())
        sixes = int(country_stats_df["sixes_scored"].sum())

        total_wickets = int(country_stats_df["wickets_taken"].sum())
        total_balls_bowled = int(country_stats_df["balls_bowled"].sum())
        total_runs_conceded = int(country_stats_df["runs_conceded"].sum())
        bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else float('inf') if total_runs_conceded > 0 else 0.0
        bowling_strike_rate = total_balls_bowled / total_wickets if total_wickets > 0 else float('inf') if total_balls_bowled > 0 else 0.0
        bowling_economy = (total_runs_conceded / total_balls_bowled) * 6 if total_balls_bowled > 0 else 0.0

        dismissal_counts = (
            country_stats_df[country_stats_df["player_out"] == 1]["out_kind"]
            .value_counts()
            .reset_index()
        )
        dismissal_counts.columns = ["Dismissal Type", "Count"]

        bowling_modes = {}
        if 'bowled_done' in country_stats_df.columns: bowling_modes['Bowled'] = int(country_stats_df['bowled_done'].sum())
        if 'lbw_done' in country_stats_df.columns: bowling_modes['LBW'] = int(country_stats_df['lbw_done'].sum())
        # Add others like 'caught_bowler', 'stumped_bowler' if columns exist

        top_bowling_mode = max(bowling_modes, key=bowling_modes.get) if bowling_modes and sum(bowling_modes.values()) > 0 else "N/A"
        most_common_bowling_count = bowling_modes.get(top_bowling_mode, 0)

        # --- Display Stats using Columns --- (Identical display as before)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Batting Summary")
            st.metric("Innings Played", innings)
            st.metric("Total Runs Scored", total_runs)
            st.metric("Batting Average", f"{batting_avg:.2f}" if batting_avg != float('inf') else "N/A")
            st.metric("Batting Strike Rate", f"{batting_strike_rate:.2f}")
            st.metric("Fours", fours)
            st.metric("Sixes", sixes)
        with col2:
             st.subheader("Bowling Summary")
             st.metric("Wickets Taken", total_wickets)
             st.metric("Runs Conceded", total_runs_conceded)
             st.metric("Bowling Average", f"{bowling_avg:.2f}" if bowling_avg != float('inf') else "N/A")
             st.metric("Bowling Strike Rate", f"{bowling_strike_rate:.2f}" if bowling_strike_rate != float('inf') else "N/A")
             st.metric("Bowling Economy Rate", f"{bowling_economy:.2f}")
             if bowling_modes and sum(bowling_modes.values()) > 0:
                 st.metric(f"Top Bowling Dismissal", f"{top_bowling_mode} ({most_common_bowling_count} times)")
             else:
                  st.metric("Top Bowling Dismissal", "N/A")

        st.subheader("How Player Got Out (Batting)")
        if not dismissal_counts.empty:
            st.dataframe(dismissal_counts.style.format({"Count": "{:,}"}), use_container_width=True)
        else:
            st.info(f"{player} was not dismissed in any innings against {current_selected_opposition} in {selected_format}.")

        # --- Export Buttons --- (Identical export logic as before)
        st.subheader("Download Data")
        col_btn1, col_btn2 = st.columns(2)
        # CSV Export
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8")
        csv_filename = f"{player}_vs_{current_selected_opposition}_{selected_format}_stats.csv".replace(" ", "_")
        with col_btn1:
            st.download_button(
                label="⬇️ Download Stats as CSV", data=csv_data,
                file_name=csv_filename, mime="text/csv",
            )
        # PDF Export
        pdf_text = f"""
        Player Statistics Report
        --------------------------
        Player: {player} ({player_team})
        Opposition: {current_selected_opposition}
        Format: {selected_format}
        --------------------------

        Batting Summary:
        Innings: {innings} | Runs Scored: {total_runs} | Times Out: {outs}
        Average: {f"{batting_avg:.2f}" if batting_avg != float('inf') else "N/A"} | SR: {batting_strike_rate:.2f}
        Balls Faced: {total_balls_faced} | Fours: {fours} | Sixes: {sixes}

        Bowling Summary:
        Wickets: {total_wickets} | Runs Conceded: {total_runs_conceded} | Balls Bowled: {total_balls_bowled}
        Average: {f"{bowling_avg:.2f}" if bowling_avg != float('inf') else "N/A"} | SR: {f"{bowling_strike_rate:.2f}" if bowling_strike_rate != float('inf') else "N/A"} | Econ: {bowling_economy:.2f}
        Top Bowling Dismissal: {f"{top_bowling_mode} ({most_common_bowling_count} times)" if top_bowling_mode != "N/A" else "N/A"}

        Dismissal Types (When Batting):
        """
        if not dismissal_counts.empty:
             for _, row in dismissal_counts.iterrows(): pdf_text += f"- {row['Dismissal Type']}: {row['Count']}\n"
        else: pdf_text += "- Not Dismissed\n"
        pdf_bytes = export_pdf(pdf_text)
        b64 = base64.b64encode(pdf_bytes.read()).decode()
        pdf_filename = f"{player}_vs_{current_selected_opposition}_{selected_format}_stats.pdf".replace(" ", "_")
        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" style="display: inline-block; padding: 0.375rem 0.75rem; font-size: 1rem; font-weight: 400; line-height: 1.5; color: #fff; background-color: #0069d9; border-color: #0062cc; text-align: center; vertical-align: middle; border: 1px solid transparent; border-radius: 0.25rem; text-decoration: none;">📄 Download Stats as PDF</a>'
        with col_btn2: st.markdown(href, unsafe_allow_html=True)

    else:
        # This might happen briefly if the state updates but filtering hasn't caught up
        st.warning(f"No detailed stats found for {player} against {current_selected_opposition} in {selected_format}. Data might be missing or name mismatch.")
else:
    # Display message only if no opposition is selected in the state
    st.info("Select a player and format, then click a highlighted country on the map above to view detailed statistics.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Data source: `data/total_data.csv`.\nMap Interaction powered by `streamlit-plotly-events`.")

# Optional: Add debug view for session state
# with st.sidebar.expander("Debug Info"):
#    st.write(st.session_state)
