import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
import plotly.graph_objects as go # Needed for more map control potentially
from streamlit_plotly_events import plotly_events # Import the component
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import warnings

# Ignore specific warnings if needed (e.g., from pycountry fuzzy search)
warnings.filterwarnings("ignore", message=".*Country name specification contains tokens.*")

# --- Constants ---
ALLOWED_FORMATS = ["ODI", "T20I", "Test"]
COLOR_PLAYED = "#008080"  # Teal for countries played against
COLOR_NOT_PLAYED = "#D3D3D3" # Light grey for others

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Loads and pre-filters the cricket data."""
    try:
        df = pd.read_csv("data/total_data.csv")
        # Basic data cleaning
        numeric_cols = ['runs_scored', 'balls_faced', 'wickets_taken', 'balls_bowled',
                        'runs_conceded', 'bowled_done', 'lbw_done', 'player_out',
                        'fours_scored', 'sixes_scored']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['out_kind'] = df['out_kind'].fillna('not out')

        # *** Filter for allowed international formats ONLY ***
        df_filtered = df[df["match_type"].isin(ALLOWED_FORMATS)].copy()
        if df_filtered.empty:
             st.warning(f"No data found for the allowed formats: {', '.join(ALLOWED_FORMATS)}")
             return None
        return df_filtered
    except FileNotFoundError:
        st.error("Error: `data/total_data.csv` not found. Please make sure the file exists in a 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading or processing the data: {e}")
        return None

def get_country_code(name, code_type='alpha_3'):
    """Gets the ISO 3166-1 alpha-3 or alpha-2 code for a country name."""
    # Expanded overrides for common cricket team names to ISO-3
    overrides_alpha3 = {
        "U.A.E.": "ARE", "UAE": "ARE", "United Arab Emirates": "ARE",
        "Scotland": "GBR", # Representing Scotland under GBR for mapping
        "USA": "USA", "United States of America": "USA",
        "Netherlands": "NLD",
        "England": "GBR", # Representing England under GBR
        "Ireland": "IRL", # Republic of Ireland
        "West Indies": "JAM", # Using Jamaica as a representative proxy
        "Hong Kong": "HKG",
        "Papua New Guinea": "PNG",
        "P.N.G.": "PNG",
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
        # Teams without standard ISO codes - map to None
        "ICC World XI": None, "Asia XI": None, "Africa XI": None,
        "East Africa": None, # Handle historical/combined teams if necessary
    }

    lookup_name = overrides_alpha3.get(name, name)
    if lookup_name is None:
        return None

    try:
        country = pycountry.countries.get(alpha_3=lookup_name) # Try direct alpha_3 override
        if country: return country.alpha_3
        country = pycountry.countries.lookup(lookup_name) # Fallback to lookup
        return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
    except LookupError:
        try:
            results = pycountry.countries.search_fuzzy(lookup_name)
            if results:
                country = results[0]
                # Double check if fuzzy result makes sense (optional)
                # print(f"Fuzzy match for '{name}' ('{lookup_name}'): {country.name} ({country.alpha_3})")
                return country.alpha_3 if code_type == 'alpha_3' else country.alpha_2
            else:
                 # st.warning(f"Could not find country code for: {name}")
                 return None # Explicitly return None if no match found
        except LookupError:
             # st.warning(f"Could not find country code (fuzzy failed) for: {name}")
             return None
    except Exception as e:
        # st.warning(f"Error looking up country code for {name}: {e}")
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
    st.error("Failed to load or process data. Please check the data file and format.")
    st.stop()

# --- User Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df_full["name"].unique())
player = st.sidebar.selectbox("Choose a Player", players)

# Use the predefined ALLOWED_FORMATS
selected_format = st.sidebar.radio("Select Format", ALLOWED_FORMATS, index=0) # Default to ODI

# --- Data Filtering based on Player/Format Selection ---
player_df = df_full[(df_full["name"] == player) & (df_full["match_type"] == selected_format)].copy()

if player_df.empty:
    st.warning(f"No data found for {player} in {selected_format} format.")
    st.stop()

# --- Prepare Data for Map ---
st.subheader(f"World Map: {player}'s Opponents in {selected_format}")
st.markdown("Click on a highlighted country on the map to view detailed stats.")

# 1. Get ISO codes for opposition teams played against
player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))

# 2. Aggregate stats per opponent (ensure iso_code is valid)
opp_stats = player_df.dropna(subset=['iso_code']).groupby(
    ["opposition_team", "iso_code"], as_index=False
).agg(
    Innings=('match_id', 'nunique'),
    TotalRuns=('runs_scored', 'sum')
)

# 3. Create a mapping from valid ISO code back to the opposition team name
iso_to_opposition_map = opp_stats.set_index('iso_code')['opposition_team'].to_dict()

# 4. Create a base DataFrame with ALL countries
all_iso3_codes = get_all_country_iso3()
world_df = pd.DataFrame(list(all_iso3_codes), columns=['iso_code'])

# 5. Merge opponent stats onto the world map data
world_df = world_df.merge(opp_stats[['iso_code', 'TotalRuns', 'Innings']], on='iso_code', how='left')
world_df['PlayedAgainst'] = world_df['TotalRuns'].notna() # Mark countries played against
world_df['HoverName'] = world_df['iso_code'].map(iso_to_opposition_map).fillna('N/A') # Map name for hover
world_df['TotalRuns'] = world_df['TotalRuns'].fillna(0).astype(int)
world_df['Innings'] = world_df['Innings'].fillna(0).astype(int)

# --- Create Interactive World Map ---
fig = px.choropleth(
    world_df,
    locations="iso_code",
    locationmode="ISO-3",
    color="PlayedAgainst",  # Color based on whether the player played against them
    color_discrete_map={  # Define colors
        True: COLOR_PLAYED,
        False: COLOR_NOT_PLAYED
    },
    hover_name="HoverName", # Show opposition name on hover
    hover_data={ # Customize hover data
        "iso_code": False, # Don't show ISO code directly
        "PlayedAgainst": False, # Don't show True/False
        "TotalRuns": ':,.0f', # Show runs if played against
        "Innings": True
    },
    # scope="world", # Ensure the whole world is attempted to be shown
    title=f"Clickable Map: Opponents for {player} in {selected_format}",
)

fig.update_layout(
    showlegend=False, # Hide the True/False legend
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='natural earth', #'mercator' is also common
        bgcolor='rgba(0,0,0,0)', # Transparent background
        landcolor=COLOR_NOT_PLAYED, # Base land color
        subunitcolor='white' # Borders between countries
    ),
    margin={"r":0,"t":40,"l":0,"b":0},
    coloraxis_showscale=False
)
# Override the color for the countries played against explicitly in the trace
# This seems more reliable than relying solely on color_discrete_map sometimes
fig.update_traces(
     marker_line_width=0.5,
     marker_line_color='white',
     selector=dict(type='choropleth') # Apply to the choropleth trace
)

# --- Use streamlit-plotly-events to capture clicks ---
# The key ensures session state is managed correctly for this element
selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key=f"map_click_{player}_{selected_format}")

# --- Display Detailed Stats Based on Click ---
selected_country_name = None
clicked_iso = None

# Check if a point was clicked
if selected_points:
    # plotly_events returns a list of dicts, get the first click
    clicked_iso = selected_points[0].get('location')
    if clicked_iso in iso_to_opposition_map:
        # Map the clicked ISO code back to the opposition team name used in the dataframe
        selected_country_name = iso_to_opposition_map[clicked_iso]
        # Store in session state to remember the selection across reruns
        st.session_state['selected_opposition'] = selected_country_name
    else:
        # Clicked on a country not played against or an invalid area
        # Keep the previously selected country if any, otherwise clear
        selected_country_name = st.session_state.get('selected_opposition', None)
        # Optionally provide feedback:
        # st.info("You clicked on an area or country not played against in this format.")

# Retrieve the selected country from session state if not just clicked
if not selected_country_name:
     selected_country_name = st.session_state.get('selected_opposition', None)

# Clear selection if player or format changes
if 'current_player' not in st.session_state or st.session_state['current_player'] != player or \
   'current_format' not in st.session_state or st.session_state['current_format'] != selected_format:
    st.session_state['selected_opposition'] = None
    selected_country_name = None
    st.session_state['current_player'] = player
    st.session_state['current_format'] = selected_format


st.markdown("---") # Separator

if selected_country_name:
    st.subheader(f"Detailed Stats vs: {selected_country_name}")
    country_stats_df = player_df[player_df["opposition_team"] == selected_country_name]

    if not country_stats_df.empty:
        player_team = country_stats_df["player_team"].iloc[0]

        st.markdown(f"#### {player} ({player_team}) vs {selected_country_name} ({selected_format})")

        # --- Calculate Stats --- (Same calculations as before)
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
        if 'bowled_done' in country_stats_df.columns:
            bowling_modes['Bowled'] = int(country_stats_df['bowled_done'].sum())
        if 'lbw_done' in country_stats_df.columns:
             bowling_modes['LBW'] = int(country_stats_df['lbw_done'].sum())
        # Add other modes if columns exist (e.g., caught_bowler, stumped_bowler)

        top_bowling_mode = max(bowling_modes, key=bowling_modes.get) if bowling_modes and sum(bowling_modes.values()) > 0 else "N/A"
        most_common_bowling_count = bowling_modes.get(top_bowling_mode, 0)


        # --- Display Stats using Columns ---
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
            # Improve display of dismissal types
            st.dataframe(dismissal_counts.style.format({"Count": "{:,}"}), use_container_width=True)
        else:
            st.info(f"{player} was not dismissed in any innings against {selected_country_name} in {selected_format}.")

        # --- Export Buttons ---
        st.subheader("Download Data")
        col_btn1, col_btn2 = st.columns(2)

        # CSV Export
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8")
        csv_filename = f"{player}_vs_{selected_country_name}_{selected_format}_stats.csv".replace(" ", "_")
        with col_btn1:
            st.download_button(
                label="⬇️ Download Stats as CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
            )

        # PDF Export (Ensure text uses calculated values)
        pdf_text = f"""
        Player Statistics Report
        --------------------------
        Player: {player} ({player_team})
        Opposition: {selected_country_name}
        Format: {selected_format}
        --------------------------

        Batting Summary:
        Innings: {innings}
        Runs Scored: {total_runs}
        Times Out: {outs}
        Average: {f"{batting_avg:.2f}" if batting_avg != float('inf') else "N/A"}
        Strike Rate: {batting_strike_rate:.2f}
        Balls Faced: {total_balls_faced}
        Fours: {fours}
        Sixes: {sixes}

        Bowling Summary:
        Wickets Taken: {total_wickets}
        Runs Conceded: {total_runs_conceded}
        Balls Bowled: {total_balls_bowled}
        Average: {f"{bowling_avg:.2f}" if bowling_avg != float('inf') else "N/A"}
        Strike Rate: {f"{bowling_strike_rate:.2f}" if bowling_strike_rate != float('inf') else "N/A"}
        Economy Rate: {bowling_economy:.2f}
        Top Bowling Dismissal: {f"{top_bowling_mode} ({most_common_bowling_count} times)" if top_bowling_mode != "N/A" else "N/A"}

        Dismissal Types (When Batting):
        """
        if not dismissal_counts.empty:
             for _, row in dismissal_counts.iterrows():
                 pdf_text += f"- {row['Dismissal Type']}: {row['Count']}\n"
        else:
             pdf_text += "- Not Dismissed\n"

        pdf_bytes = export_pdf(pdf_text)
        b64 = base64.b64encode(pdf_bytes.read()).decode()
        pdf_filename = f"{player}_vs_{selected_country_name}_{selected_format}_stats.pdf".replace(" ", "_")
        # Slightly improved button styling using markdown link
        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" style="display: inline-block; padding: 0.375rem 0.75rem; font-size: 1rem; font-weight: 400; line-height: 1.5; color: #fff; background-color: #0069d9; border-color: #0062cc; text-align: center; vertical-align: middle; border: 1px solid transparent; border-radius: 0.25rem; text-decoration: none;">📄 Download Stats as PDF</a>'

        with col_btn2:
             st.markdown(href, unsafe_allow_html=True) # Use markdown for better button look

    else:
        # This case shouldn't happen if selected_country_name is derived correctly, but good to have
        st.warning(f"Internal check failed: No detailed stats found for {player} against {selected_country_name} in {selected_format}, though country was selected.")
else:
    st.info("Select a player and format, then click a highlighted country on the map above to view detailed statistics.")


# Add some footer or information
st.sidebar.markdown("---")
st.sidebar.info("Data source: `data/total_data.csv`.\nMap Interaction powered by `streamlit-plotly-events`.")
