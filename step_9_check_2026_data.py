import imdlib as imd
import os

def check_live_data(year=2026):
    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"Attempting to download IMD gridded rainfall data for {year}...")
    try:
        # imd.get_data actively reaches out to the IMD Pune servers to fetch the file
        data = imd.get_data('rain', year, year, fn_format='yearwise', file_dir=data_dir)
        
        # If it succeeds, let's peek at the most recent day it recorded
        ds = data.get_xarray()
        latest_date = str(ds.time.values[-1])[:10]
        
        print(f"\nSuccess! 2026 data is available.")
        print(f"The most recent day recorded in the file is: {latest_date}")
        
    except Exception as e:
        print(f"\nFailed to retrieve 2026 data. The IMD server might not have bundled it yet.")
        print(f"Error details: {e}")

if __name__ == "__main__":
    check_live_data(2026)

# python step_9_check_2026_data.py