import imdlib as imd
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def inspect_raw_grd():
    data_dir = './data'
    year = 2024

    print(f"Loading raw IMD data for {year}...")
    data = imd.open_data('rain', year, year, 'yearwise', data_dir)
    ds = data.get_xarray()

    print("\n--- 1. The 3D Grid Structure ---")
    print(f"Longitude range: {ds.lon.min().values} to {ds.lon.max().values} (Resolution: 0.25°)")
    print(f"Latitude range:  {ds.lat.min().values} to {ds.lat.max().values} (Resolution: 0.25°)")
    print(f"Time steps:      {len(ds.time)} days")

    print("\n--- 2. Flattening Day 1 into a Table ---")
    # Select just the very first day (index 0) to keep it fast
    day1_ds = ds.isel(time=0)
    
    # Convert the 2D spatial slice into a flat Pandas DataFrame
    df = day1_ds.to_dataframe().reset_index()
    
    # The grid covers a massive square over South Asia. 
    # IMD sets pixels over the ocean or outside India to NaN. 
    # Let's filter those out to see the real data.
    df_valid = df.dropna(subset=['rain'])
    
    print(f"Total grid points in the square: {len(df):,}")
    print(f"Valid points (inside India's border): {len(df_valid):,}")
    
    print("\n--- 3. Sample Data (First 10 Valid Points) ---")
    print(df_valid.head(10))

if __name__ == "__main__":
    inspect_raw_grd()

# python 07_inspect_grd.py