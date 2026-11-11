import os
import geopandas as gpd
import regionmask
import imdlib as imd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def verify_correct_projection():
    data_dir = './data'
    shapefile_path = os.path.join(data_dir, 'sd_boundary.json')

    # 1. Load the official IMD Subdivisions JSON
    gdf = gpd.read_file(shapefile_path)
    
    # 2. THE REAL FIX: Set the exact native projection IMD used (UTM Zone 44N)
    gdf.set_crs(epsg=32644, allow_override=True, inplace=True)
    
    # 3. Convert to standard Lat/Lon (EPSG:4326) to match the .grd files
    gdf = gdf.to_crs(epsg=4326)

    # 4. Load Day 1 of 2024 to test overlap
    data = imd.open_data('rain', 2024, 2024, 'yearwise', data_dir)
    ds = data.get_xarray().isel(time=0) 

    # 5. Create Mask (wrap_lon=False prevents regionmask from crashing)
    mask = regionmask.mask_geopandas(gdf, ds.lon, ds.lat, wrap_lon=False)
    
    mask_values = mask.values
    rain_values = ds['rain'].values

    print("\n--- PIXEL OVERLAP DIAGNOSTIC (OFFICIAL IMD SUBDIVISIONS) ---")
    print(f"{'Subdivision':<30} | {'Total Pixels':<12} | {'Valid (Non-NaN) Pixels'}")
    print("-" * 75)

    for idx, row in gdf.iterrows():
        sub_name = row['subdivisio']
        
        pixel_coords = np.where(mask_values == idx)
        total_pixels = len(pixel_coords[0])
        
        valid_pixels = 0
        if total_pixels > 0:
            sub_rain_data = rain_values[pixel_coords]
            valid_pixels = np.sum(~np.isnan(sub_rain_data))
            
        print(f"{str(sub_name)[:30]:<30} | {total_pixels:<12} | {valid_pixels}")

if __name__ == "__main__":
    verify_correct_projection()

# python 08_fix_imd_shapefile.py