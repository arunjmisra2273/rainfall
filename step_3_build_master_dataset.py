import os
import geopandas as gpd
import regionmask
import imdlib as imd
import pandas as pd
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

def build_master_dataset(start_yr=1901, end_yr=2025):
    data_dir = './data'
    shapefile_path = os.path.join(data_dir, 'sd_boundary.json')

    print("1. Loading and projecting the official IMD shapefile...")
    gdf = gpd.read_file(shapefile_path)
    # Applying the correct native projection discovered in Step 2
    gdf.set_crs(epsg=32644, allow_override=True, inplace=True)
    gdf = gdf.to_crs(epsg=4326)
    
    name_col = 'subdivisio'
    region_mapping = dict(enumerate(gdf[name_col]))
    
    all_years_data = []

    print(f"\n2. Processing years {start_yr} to {end_yr}...")
    print("This will open each .grd file one by one to keep RAM usage low.")
    
    for year in tqdm(range(start_yr, end_yr + 1), desc="Processing Years"):
        try:
            # Load single year
            data = imd.open_data('rain', year, year, 'yearwise', data_dir)
            ds = data.get_xarray()
            
            # Create the cookie-cutter mask
            subdivision_mask = regionmask.mask_geopandas(
                gdf, ds.lon, ds.lat, wrap_lon=False
            )
            subdivision_mask.name = 'mask'
            
            # Average the pixels inside each subdivision for every day
            daily_sub_rain = ds['rain'].groupby(subdivision_mask).mean('stacked_lat_lon')
            
            df = daily_sub_rain.to_dataframe().reset_index()
            
            # Normalize column names 
            for col in ['mask', 'region', 'group']:
                if col in df.columns:
                    df = df.rename(columns={col: 'Subdivision'})
                    break
                    
            df = df.rename(columns={'time': 'Date', 'rain': 'Rainfall_mm'})
            df['Subdivision'] = df['Subdivision'].map(region_mapping)
            
            # Drop NaN rows (pixels that fall into the ocean or outside India)
            df = df.dropna(subset=['Subdivision'])
            
            all_years_data.append(df)
            
        except Exception as e:
            print(f"\nError processing year {year}: {e}")
    
    print("\n3. Combining and saving master dataset...")
    final_df = pd.concat(all_years_data, ignore_index=True)
    
    # Sort chronologically by subdivision
    final_df = final_df.sort_values(by=['Subdivision', 'Date']).reset_index(drop=True)
    
    output_path = os.path.join(data_dir, 'master_daily_rainfall_1901_2025.csv')
    final_df.to_csv(output_path, index=False)
    
    print(f"\nSuccess! Master dataset saved to: {output_path}")
    print(f"Total Rows: {len(final_df):,}")
    print("\nPreview of the final data:")
    print(final_df.head())

if __name__ == "__main__":
    build_master_dataset()

# python step_3_build_master_dataset.py