import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import numpy as np
import os

st.set_page_config(layout="wide")
st.title("Australia Temperature Anomaly Heatmaps (BOM Style)")

# Load dataset
df = pd.read_csv("Final_Data.csv")

# Create output dir for maps if not exist
os.makedirs("map_thumbnails", exist_ok=True)

# Get unique years
years = sorted(df['year'].unique())

# Use a slider instead of a dropdown
selected_year = st.slider(
    "📅 Select a year to enlarge",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=int(min(years)),
    step=1
)

# Function to plot interpolated map
def plot_map(df_year, year, save_path=None, size=(6, 4)):
    lats = df_year['latitude'].values
    lons = df_year['longitude'].values
    vals = df_year['anomaly'].values

    lon_grid, lat_grid = np.meshgrid(
        np.linspace(110, 155, 300),
        np.linspace(-45, -10, 300)
    )

    grid_vals = griddata((lons, lats), vals, (lon_grid, lat_grid), method='cubic')
    grid_vals = np.ma.masked_invalid(grid_vals)  # Masking invalid extrapolated areas

    fig = plt.figure(figsize=size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([110, 155, -45, -10], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.set_title(f"{year} - Temperature Anomaly (°C)")

    c = ax.contourf(
        lon_grid, lat_grid, grid_vals,
        cmap="RdYlGn_r",
        transform=ccrs.PlateCarree()
    )

    if size[0] > 5:
        plt.colorbar(c, ax=ax, orientation="vertical", label="Anomaly (°C)")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        st.pyplot(fig)


# Show enlarged selected map
df_selected = df[df['year'] == selected_year]
st.subheader(f"Enlarged Heatmap for {selected_year}")
plot_map(df_selected, selected_year, size=(8, 6))

# Grid of thumbnails
st.subheader("📅 Yearly Heatmap Gallery")
thumb_cols = st.columns(6)

# Optional: preprocess to avoid slicing every time in loop
year_dfs = {year: df[df['year'] == year] for year in years}

for idx, year in enumerate(years):
    col = thumb_cols[idx % 6]
    thumbnail_path = f"map_thumbnails/{year}.png"

    # Generate thumbnail if not already saved
    if not os.path.exists(thumbnail_path):
        plot_map(year_dfs[year], year, save_path=thumbnail_path, size=(2.5, 2))

    # Display in grid
    with col:
        st.image(thumbnail_path, caption=str(year), use_column_width=True)
