import os
import streamlit as st
import pandas as pd
import pyarrow.parquet as pq
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title="LILA BLACK - Match Explorer", layout="wide")
st.title("🗺️ LILA BLACK - Match Explorer")

# --- PATH RESOLUTION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Resolve player_data folder
DATA_DIR = os.path.join(BASE_DIR, "player_data")
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(REPO_ROOT, "player_data")

# Resolve minimaps folder
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

# --- STEP 1: FIND AVAILABLE DATES ---
@st.cache_data
def get_available_dates(target_data_dir):
    if not os.path.exists(target_data_dir):
        return []
    return sorted([
        d for d in os.listdir(target_data_dir)
        if os.path.isdir(os.path.join(target_data_dir, d)) and not d.startswith('.')
    ])

available_dates = get_available_dates(DATA_DIR)

if not available_dates:
    st.error(
        f"Data directory not found or empty.\n\n"
        f"Searched paths:\n- {os.path.join(BASE_DIR, 'player_data')}\n- {os.path.join(REPO_ROOT, 'player_data')}"
    )
    st.stop()

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filters")
selected_date = st.sidebar.selectbox("Date", available_dates)

# --- STEP 2: LOAD SELECTED DAY ONLY (PREVENTS CPU THROTTLING) ---
@st.cache_data(show_spinner=f"Loading matches for {selected_date}...")
def load_day_data(target_data_dir, date_folder):
    day_folder = os.path.join(target_data_dir, date_folder)
    frames = []

    for file_name in os.listdir(day_folder):
        if file_name.endswith('.md') or file_name.startswith('.'):
            continue
        filepath = os.path.join(day_folder, file_name)
        try:
            table = pq.read_table(filepath)
            frames.append(table.to_pandas())
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    master_df = pd.concat(frames, ignore_index=True)

    # 1. Decode event bytes
    master_df['event'] = master_df['event'].apply(
        lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x)
    )

    # 2. Identify bots vs humans (UUIDs have hyphens, bots are numeric)
    master_df['is_bot'] = ~master_df['user_id'].astype(str).str.contains('-')
    master_df['player_type'] = master_df['is_bot'].map({True: 'Bot', False: 'Human'})

    # 3. Vectorized world-to-minimap coordinate conversion
    map_configs_df = pd.DataFrame(MAP_CONFIG).T.reset_index().rename(columns={'index': 'map_id'})
    merged = master_df.merge(map_configs_df, on='map_id', how='left')

    master_df['pixel_x'] = ((merged['x'] - merged['origin_x']) / merged['scale']) * 1024
    master_df['pixel_y'] = (1 - ((merged['z'] - merged['origin_z']) / merged['scale'])) * 1024

    # 4. Parse timestamp
    master_df['ts'] = pd.to_datetime(master_df['ts'], unit='ms')

    return master_df

df = load_day_data(DATA_DIR, selected_date)

if df.empty:
    st.warning("No valid event data found for the selected date.")
    st.stop()

# Additional Filters
available_maps = sorted(df['map_id'].dropna().unique())
selected_map = st.sidebar.selectbox("Map", available_maps)
map_filtered = df[df['map_id'] == selected_map]

available_matches = sorted(map_filtered['match_id'].unique())
selected_match = st.sidebar.selectbox("Match ID", available_matches)
match_data = map_filtered[map_filtered['match_id'] == selected_match]

view_mode = st.sidebar.radio("View Mode", ["Player Journeys (Timeline)", "Heatmap (Aggregated)"])

# --- VISUALIZATION ---
st.subheader(f"Map: {selected_map} | Match: {selected_match}")

# Load Background Minimap Image
img_filename = MAP_CONFIG.get(selected_map, {}).get("img")
img_path = os.path.join(MAPS_DIR, img_filename) if img_filename else ""

if not os.path.exists(img_path):
    st.error(f"Minimap image not found at: {img_path}")
    st.stop()

bg_img = Image.open(img_path)

if view_mode == "Player Journeys (Timeline)":
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

    fig = go.Figure()

    fig.add_layout_image(
        dict(
            source=bg_img,
            xref="x", yref="y",
            x=0, y=0,
            sizex=1024, sizey=1024,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Plot Journeys
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

    # Plot Events
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

    fig.update_layout(
        xaxis=dict(range=[0, 1024], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[1024, 0], showgrid=False, zeroline=False, visible=False),
        width=800, height=800,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255, 255, 255, 0.8)")
    )

    st.plotly_chart(fig, use_container_width=True)

elif view_mode == "Heatmap (Aggregated)":
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