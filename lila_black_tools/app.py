import os
import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="LILA BLACK - Match Explorer", layout="wide")
st.title("🗺️ LILA BLACK - Match Explorer")

# Resolve paths dynamically relative to where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Check both inside lila_black_tools/ and at the repository root
DATA_DIR = os.path.join(BASE_DIR, "player_data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(REPO_ROOT, "player_data")

MAPS_DIR = os.path.join(BASE_DIR, "minimaps")
if not os.path.exists(MAPS_DIR):
    MAPS_DIR = os.path.join(REPO_ROOT, "minimaps")

MAP_CONFIG = {
    "AmbroseValley": {"scale": 900, "origin_x": -370, "origin_z": -473, "img": "AmbroseValley_Minimap.png"},
    "GrandRift": {"scale": 581, "origin_x": -290, "origin_z": -290, "img": "GrandRift_Minimap.png"},
    "Lockdown": {"scale": 1000, "origin_x": -500, "origin_z": -500, "img": "Lockdown_Minimap.jpg"}
}

EVENT_COLORS = {
    "Position": "blue",
    "BotPosition": "lightblue",
    "Kill": "red",
    "BotKill": "orange",
    "Killed": "darkred",
    "BotKilled": "darkorange",
    "Loot": "green",
    "KilledByStorm": "purple"
}

# --- DATA PROCESSING ---
@st.cache_data(show_spinner="Loading and crunching Parquet data...")
def load_and_process_data(data_dir="player_data"):
    all_frames = []
    
    if not os.path.exists(data_dir):
        st.warning(f"Data directory '{data_dir}' not found. Please ensure it exists.")
        return pd.DataFrame()

    # Traverse folders (February_10, etc.)
    for root, _, files in os.walk(data_dir):
        date_folder = os.path.basename(root)
        for file in files:
            if file == "README.md":
                continue
                
            filepath = os.path.join(root, file)
            try:
                # Read Parquet
                df = pq.read_table(filepath).to_pandas()
                df['date_folder'] = date_folder
                all_frames.append(df)
            except Exception as e:
                pass # Skip unreadable files
                
    if not all_frames:
        return pd.DataFrame()
        
    master_df = pd.concat(all_frames, ignore_index=True)
    
    # 1. Decode event bytes
    master_df['event'] = master_df['event'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
    )
    
    # 2. Identify bots vs humans (Humans have '-' in UUID, bots are numeric)
    master_df['is_bot'] = ~master_df['user_id'].astype(str).str.contains('-')
    master_df['player_type'] = master_df['is_bot'].map({True: 'Bot', False: 'Human'})
    
    # 3. Calculate Map Coordinates
    # Convert x, z to pixel_x, pixel_y based on map_id
    def map_to_pixels(row):
        map_id = row['map_id']
        if map_id not in MAP_CONFIG:
            return pd.Series({"pixel_x": None, "pixel_y": None})
            
        cfg = MAP_CONFIG[map_id]
        u = (row['x'] - cfg['origin_x']) / cfg['scale']
        v = (row['z'] - cfg['origin_z']) / cfg['scale']
        
        pixel_x = u * 1024
        pixel_y = (1 - v) * 1024 # Y is flipped for images
        return pd.Series({"pixel_x": pixel_x, "pixel_y": pixel_y})

    # Apply vectorized math for performance
    map_configs_df = pd.DataFrame(MAP_CONFIG).T.reset_index().rename(columns={'index': 'map_id'})
    merged = master_df.merge(map_configs_df, on='map_id', how='left')
    
    master_df['pixel_x'] = ((merged['x'] - merged['origin_x']) / merged['scale']) * 1024
    master_df['pixel_y'] = (1 - ((merged['z'] - merged['origin_z']) / merged['scale'])) * 1024
    
    # Convert timestamps
    master_df['ts'] = pd.to_datetime(master_df['ts'], unit='ms')
    
    return master_df

# Load the data
df = load_and_process_data()

if df.empty:
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")

selected_date = st.sidebar.selectbox("Date", sorted(df['date_folder'].unique()))
date_filtered = df[df['date_folder'] == selected_date]

selected_map = st.sidebar.selectbox("Map", sorted(date_filtered['map_id'].dropna().unique()))
map_filtered = date_filtered[date_filtered['map_id'] == selected_map]

selected_match = st.sidebar.selectbox("Match ID", sorted(map_filtered['match_id'].unique()))
match_data = map_filtered[map_filtered['match_id'] == selected_match]

view_mode = st.sidebar.radio("View Mode", ["Player Journeys (Timeline)", "Heatmap (Aggregated)"])

# --- VISUALIZATION ---
st.subheader(f"Map: {selected_map} | Match: {selected_match}")

# Load Background Image
img_path = os.path.join("minimaps", MAP_CONFIG[selected_map]["img"])
try:
    bg_img = Image.open(img_path)
except FileNotFoundError:
    st.error(f"Minimap image not found at {img_path}")
    st.stop()

if view_mode == "Player Journeys (Timeline)":
    # Timeline Slider
    min_time = match_data['ts'].min()
    max_time = match_data['ts'].max()
    
    if min_time != max_time:
        selected_time = st.slider(
            "Match Timeline", 
            min_value=min_time.to_pydatetime(), 
            max_value=max_time.to_pydatetime(), 
            value=max_time.to_pydatetime(),
            format="HH:mm:ss"
        )
        plot_data = match_data[match_data['ts'] <= selected_time]
    else:
        plot_data = match_data

    # Create Plotly figure
    fig = go.Figure()
    
    # Add image background
    fig.add_layout_image(
        dict(
            source=bg_img,
            xref="x", yref="y",
            x=0, y=0, # Top-left corner of the image in the plot
            sizex=1024, sizey=1024,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Plot Journeys (Lines)
    for user_id, group in plot_data.groupby('user_id'):
        is_bot = group['is_bot'].iloc[0]
        color = "lightblue" if is_bot else "blue"
        opacity = 0.4 if is_bot else 0.8
        
        fig.add_trace(go.Scatter(
            x=group['pixel_x'], y=group['pixel_y'],
            mode='lines',
            line=dict(color=color, width=2),
            opacity=opacity,
            name=f"{'Bot' if is_bot else 'Human'} Path",
            hoverinfo='skip',
            showlegend=False
        ))

    # Plot Discrete Events (Scatter Points)
    events_only = plot_data[~plot_data['event'].isin(['Position', 'BotPosition'])]
    if not events_only.empty:
        for event_type, group in events_only.groupby('event'):
            fig.add_trace(go.Scatter(
                x=group['pixel_x'], y=group['pixel_y'],
                mode='markers',
                marker=dict(
                    size=10, 
                    color=EVENT_COLORS.get(event_type, "gray"),
                    line=dict(width=1, color='white')
                ),
                name=event_type,
                text=group['player_type'] + " - " + group['event'],
                hoverinfo='text'
            ))

    # Configure axes to match 1024x1024 image coordinate system
    fig.update_layout(
        xaxis=dict(range=[0, 1024], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[1024, 0], showgrid=False, zeroline=False, visible=False), # Y is flipped!
        width=800, height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Heatmap (Aggregated)":
    st.write("View dense activity zones. Best used across the whole match.")
    heatmap_metric = st.selectbox("Heatmap Target", ["All Movement", "Kills/Deaths", "Looting"])
    
    if heatmap_metric == "All Movement":
        hm_data = match_data
    elif heatmap_metric == "Kills/Deaths":
        hm_data = match_data[match_data['event'].str.contains('Kill', na=False)]
    else:
        hm_data = match_data[match_data['event'] == 'Loot']

    if hm_data.empty:
        st.info("No data available for this metric in the current match.")
    else:
        fig = go.Figure()
        
        # Add image background
        fig.add_layout_image(
            dict(
                source=bg_img,
                xref="x", yref="y",
                x=0, y=0,
                sizex=1024, sizey=1024,
                sizing="stretch",
                opacity=0.8,
                layer="below"
            )
        )
        
        # 2D Histogram Contour (Heatmap)
        fig.add_trace(go.Histogram2dContour(
            x=hm_data['pixel_x'], 
            y=hm_data['pixel_y'],
            colorscale="YlOrRd",
            reversescale=False,
            opacity=0.6,
            showscale=False,
            ncontours=15
        ))
        
        fig.update_layout(
            xaxis=dict(range=[0, 1024], showgrid=False, zeroline=False, visible=False),
            yaxis=dict(range=[1024, 0], showgrid=False, zeroline=False, visible=False),
            width=800, height=800,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)