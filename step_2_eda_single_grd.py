import os
import geopandas as gpd
import imdlib as imd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

def visualize_grid_with_labels():
    data_dir = './data'
    eda_dir = './eda'
    os.makedirs(eda_dir, exist_ok=True)
    
    print("1. Loading the IMD Shapefile cookie-cutter...")
    shapefile_path = os.path.join(data_dir, 'sd_boundary.json')
    gdf = gpd.read_file(shapefile_path)
    
    # Apply correct native projection and convert to Lat/Lon
    gdf.set_crs(epsg=32644, allow_override=True, inplace=True)
    gdf = gdf.to_crs(epsg=4326)
    
    print("2. Loading 2024 Grid Data...")
    data = imd.open_data('rain', 2024, 2024, 'yearwise', data_dir)
    ds = data.get_xarray().isel(time=200) 
    
    date_str = str(ds.time.values)[:10]
    
    print("3. Generating Labeled Visualization...")
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Plot 1: The Base Dough (The raw IMD rainfall grid)
    ds['rain'].plot(ax=ax, cmap='Blues', add_colorbar=True, 
                    cbar_kwargs={'label': 'Rainfall (mm)'})
    
    # Plot 2: The Cookie Cutter 
    # Made the boundary lines black and slightly thicker so they don't get lost
    gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2)
    
    # Label each subdivision at its geographic center
    for idx, row in gdf.iterrows():
        if row['geometry']:
            centroid = row['geometry'].centroid
            ax.annotate(
                text=row['subdivisio'], 
                xy=(centroid.x, centroid.y),
                horizontalalignment='center', 
                verticalalignment='center',
                fontsize=7,
                color='maroon',
                weight='bold',
                # Add a tiny white background to the text so it's readable over the blue rain
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1)
            )
    
    plt.title(f'IMD Labeled Subdivisions over Gridded Rainfall\nDate: {date_str}', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Zoom in specifically on India's mainland
    plt.xlim(67, 98)
    plt.ylim(7, 38)
    
    output_path = os.path.join(eda_dir, 'step_2_grid_labeled.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! Labeled overlay saved to: {output_path}")

if __name__ == "__main__":
    visualize_grid_with_labels()

# python step_2_eda_single_grd.py