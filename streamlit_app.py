import streamlit as st
import pandas as pd
import pycountry
import plotly.express as px
from io import BytesIO
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

# --- Helper Functions ---

@st.cache_data
def load_data():
    """Loads the cricket data from the CSV file."""
    try:
        df = pd.read_csv("data/total_data.csv")
        # Basic data cleaning (optional but recommended)
        df['runs_scored'] = pd.to_numeric(df['runs_scored'], errors='coerce').fillna(0)
        df['balls_faced'] = pd.to_numeric(df['balls_faced'], errors='coerce').fillna(0)
        df['wickets_taken'] = pd.to_numeric(df['wickets_taken'], errors='coerce').fillna(0)
        df['balls_bowled'] = pd.to_numeric(df['balls_bowled'], errors='coerce').fillna(0)
        df['runs_conceded'] = pd.to_numeric(df['runs_conceded'], errors='coerce').fillna(0)
        df['bowled_done'] = pd.to_numeric(df['bowled_done'], errors='coerce').fillna(0)
        df['lbw_done'] = pd.to_numeric(df['lbw_done'], errors='coerce').fillna(0)
        df['out_kind'] = df['out_kind'].fillna('not out') # Handle NaN dismissal types
        return df
    except FileNotFoundError:
        st.error("Error: `data/total_data.csv` not found. Please make sure the file exists in a 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading the data: {e}")
        return None

def get_country_code(name, code_type='alpha_3'):
    """
    Gets the ISO 3166-1 alpha-3 or alpha-2 code for a country name.
    Includes overrides for common cricket team names.
    """
    overrides = {
        # Alpha-3 Overrides
        "U.A.E.": "ARE", "UAE": "ARE", "Scotland": "GBR", "USA": "USA",
        "Netherlands": "NLD", "England": "GBR", "Ireland": "IRL",
        "West Indies": "JAM", # Using Jamaica as a proxy, no official WI code
        "Hong Kong": "HKG",
        "ICC World XI": None, # Or handle specific non-country teams
        "Asia XI": None,
        "Africa XI": None,
        # Add more specific overrides if needed based on your data
    }
    # Alpha-2 Overrides (if needed, map overrides dict accordingly)
    overrides_alpha2 = {
        "U.A.E.": "AE", "UAE": "AE", "Scotland": "GB", "USA": "US",
        "Netherlands": "NL", "England": "GB", "Ireland": "IE",
        "West Indies": "JM", "Hong Kong": "HK" ,
        "ICC World XI": None, "Asia XI": None, "Africa XI": None,
    }

    lookup_name = name
    if code_type == 'alpha_3':
        lookup_name = overrides.get(name, name)
    elif code_type == 'alpha_2':
         lookup_name = overrides_alpha2.get(name, name)

    if lookup_name is None: # Handle explicitly mapped None cases
         return None

    try:
        country = pycountry.countries.lookup(lookup_name)
        if code_type == 'alpha_3':
            return country.alpha_3
        elif code_type == 'alpha_2':
            return country.alpha_2
        else:
            return None # Should not happen with current logic
    except LookupError:
        # Attempt fuzzy search if direct lookup fails
        try:
            results = pycountry.countries.search_fuzzy(lookup_name)
            if results:
                country = results[0]
                if code_type == 'alpha_3':
                    return country.alpha_3
                elif code_type == 'alpha_2':
                    return country.alpha_2
            else:
                 # Fallback: Try searching within overrides again if fuzzy fails
                 if name in overrides:
                     return overrides[name] if code_type == 'alpha_3' else overrides_alpha2.get(name)
                 st.warning(f"Could not find country code for: {name}")
                 return None
        except LookupError:
             st.warning(f"Could not find country code for: {name}")
             return None
    except Exception as e:
        st.warning(f"Error looking up country code for {name}: {e}")
        return None


def export_pdf(stats_text, filename="player_stats.pdf"):
    """Exports the provided text statistics to a PDF file."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter # Get page dimensions

    # Set up text object
    text = c.beginText()
    text.setTextOrigin(inch, height - inch) # Start near top-left corner
    text.setFont("Helvetica", 10) # Set font and size

    # Add text line by line
    for line in stats_text.split("\n"):
        text.textLine(line)

    c.drawText(text)
    c.save()
    buffer.seek(0)
    return buffer

# --- Main App ---
st.set_page_config(page_title="Cricket Player Stats", layout="wide")
st.title("🏏 Cricket Player Performance Analyzer")

# Load Data
df = load_data()

if df is None:
    st.stop() # Stop execution if data loading failed

# --- User Selections ---
st.sidebar.header("Select Player and Format")
players = sorted(df["name"].unique())
player = st.sidebar.selectbox("Choose a Player", players)

formats = sorted(df["match_type"].unique()) # Get available formats from data
# Provide default formats if needed, but using unique values is better
# formats = ["ODI", "T20I", "Test", "MDM"] # Add 'MDM' if present or others
selected_format = st.sidebar.radio("Select Format", formats, index=formats.index("ODI") if "ODI" in formats else 0) # Default to ODI if available

# --- Data Filtering based on Selection ---
player_df = df[(df["name"] == player) & (df["match_type"] == selected_format)].copy() # Use .copy() to avoid SettingWithCopyWarning

if player_df.empty:
    st.warning(f"No data found for {player} in {selected_format} format.")
    st.stop()

# --- Map Visualization ---
st.subheader(f"{player}'s Performance Map ({selected_format})")
st.markdown("Shows countries played against in this format.")

# Get ISO codes for opposition teams for the map
player_df["iso_code"] = player_df["opposition_team"].apply(lambda x: get_country_code(x, 'alpha_3'))

# Prepare data for the map (aggregate something simple like innings count or total runs)
map_df = player_df.groupby(["opposition_team", "iso_code"], as_index=False).agg(
    Innings=('match_id', 'nunique'), # Count unique matches as innings approximation
    TotalRuns=('runs_scored', 'sum')
).dropna(subset=['iso_code']) # Remove rows where ISO code lookup failed

if not map_df.empty:
    fig = px.choropleth(
        map_df,
        locations="iso_code",
        locationmode="ISO-3",
        color_discrete_sequence=["#008080"], # Teal color
        hover_name="opposition_team",
        hover_data={"iso_code": False, 'Innings': True, 'TotalRuns': True}, # Show innings/runs on hover
        title=f"Countries {player} Played Against in {selected_format}",
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='mercator' # Common map projection
        ),
        margin={"r":0,"t":40,"l":0,"b":0}, # Reduce margins
        coloraxis_showscale=False # Hide color scale as it's single color
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color='white')

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"{player} has no recorded opposition data with valid country codes for mapping in {selected_format}.")


# --- Country Selection for Detailed Stats ---
st.subheader("Select Opposition for Detailed Stats")
all_opponents = sorted(player_df["opposition_team"].unique())

if not all_opponents:
     st.warning("No opponents found for this player and format combination.")
     st.stop()

selected_country = st.selectbox("Select Opposition Country:", all_opponents)

# --- Detailed Stats Display ---
if selected_country:
    country_stats_df = player_df[player_df["opposition_team"] == selected_country]

    if not country_stats_df.empty:
        player_team = country_stats_df["player_team"].iloc[0] # Get player's team

        st.markdown(f"### Detailed Stats: {player} ({player_team}) vs {selected_country} ({selected_format})")

        # --- Calculate Stats ---
        innings = country_stats_df.shape[0] # Each row represents an innings/match appearance
        total_runs = int(country_stats_df["runs_scored"].sum())
        total_balls_faced = int(country_stats_df["balls_faced"].sum())
        outs = int(country_stats_df["player_out"].sum())

        # Batting Avg: Runs / Outs (handle division by zero)
        batting_avg = total_runs / outs if outs > 0 else float('inf') if total_runs > 0 else 0.0

        # Batting SR: (Runs / Balls Faced) * 100 (handle division by zero)
        batting_strike_rate = (total_runs / total_balls_faced) * 100 if total_balls_faced > 0 else 0.0

        fours = int(country_stats_df["fours_scored"].sum())
        sixes = int(country_stats_df["sixes_scored"].sum())

        # Bowling Stats
        total_wickets = int(country_stats_df["wickets_taken"].sum())
        total_balls_bowled = int(country_stats_df["balls_bowled"].sum())
        total_runs_conceded = int(country_stats_df["runs_conceded"].sum())

        # Bowling Avg: Runs Conceded / Wickets (handle division by zero)
        bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else float('inf') if total_runs_conceded > 0 else 0.0

        # Bowling SR: Balls Bowled / Wickets (handle division by zero)
        bowling_strike_rate = total_balls_bowled / total_wickets if total_wickets > 0 else float('inf') if total_balls_bowled > 0 else 0.0

        # Bowling Economy: (Runs Conceded / Balls Bowled) * 6 (handle division by zero)
        bowling_economy = (total_runs_conceded / total_balls_bowled) * 6 if total_balls_bowled > 0 else 0.0

        # Dismissals (Batting)
        dismissal_counts = (
            country_stats_df[country_stats_df["player_out"] == 1]["out_kind"]
            .value_counts()
            .reset_index()
        )
        dismissal_counts.columns = ["Dismissal Type", "Count"]

        # Bowling Dismissal Modes (if data available)
        bowling_modes = {}
        if 'bowled_done' in country_stats_df.columns:
            bowling_modes['Bowled'] = int(country_stats_df['bowled_done'].sum())
        if 'lbw_done' in country_stats_df.columns:
             bowling_modes['LBW'] = int(country_stats_df['lbw_done'].sum())
        # Add others like 'caught_bowler', 'stumped_bowler' if columns exist

        top_bowling_mode = max(bowling_modes, key=bowling_modes.get) if bowling_modes else "N/A"
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
             if bowling_modes:
                 st.metric(f"Top Bowling Dismissal", f"{top_bowling_mode} ({most_common_bowling_count} times)")


        st.subheader("How Player Got Out (Batting)")
        if not dismissal_counts.empty:
            st.dataframe(dismissal_counts, use_container_width=True)
        else:
            st.info(f"{player} was not dismissed in any innings against {selected_country} in {selected_format}.")

        # --- Export Buttons ---
        st.subheader("Download Data")
        col_btn1, col_btn2 = st.columns(2)

        # CSV Export
        csv_data = country_stats_df.to_csv(index=False).encode("utf-8")
        with col_btn1:
            st.download_button(
                label="⬇️ Download Stats as CSV",
                data=csv_data,
                file_name=f"{player}_vs_{selected_country}_{selected_format}_stats.csv",
                mime="text/csv",
            )

        # PDF Export
        pdf_text = f"""
        Player Statistics Report
        --------------------------
        Player: {player} ({player_team})
        Opposition: {selected_country}
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
        Top Bowling Dismissal: {top_bowling_mode} ({most_common_bowling_count} times) if bowling_modes else "N/A"

        Dismissal Types (When Batting):
        """
        if not dismissal_counts.empty:
             for _, row in dismissal_counts.iterrows():
                 pdf_text += f"- {row['Dismissal Type']}: {row['Count']}\n"
        else:
             pdf_text += "- Not Dismissed\n"

        pdf_bytes = export_pdf(pdf_text)
        b64 = base64.b64encode(pdf_bytes.read()).decode()
        pdf_filename = f"{player}_vs_{selected_country}_{selected_format}_stats.pdf"
        href = f'<a href="data:application/pdf;base64,{b64}" download="{pdf_filename}" style="display: inline-block; padding: 0.25em 0.6em; font-size: 0.875rem; font-weight: 400; line-height: 1.5; color: #000; background-color: #f0f2f6; border: 1px solid #f0f2f6; border-radius: 0.25rem; text-decoration: none;">📄 Download Stats as PDF</a>'

        with col_btn2:
            st.markdown(href, unsafe_allow_html=True)

    else:
        st.warning(f"No detailed stats found for {player} against {selected_country} in {selected_format}.")

# Add some footer or information
st.sidebar.markdown("---")
st.sidebar.info("Data source: `total_data.csv`")
