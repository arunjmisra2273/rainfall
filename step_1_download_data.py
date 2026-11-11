# Step 1: Download the IMD Dataset

import imdlib as imd
import os

def download_imd_rainfall():
    start_yr = 1901
    end_yr = 2025
    variable = 'rain' # IMD variable name for rainfall
    data_dir = './data'

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    print(f"Initiating download for IMD {variable} data from {start_yr} to {end_yr}...")
    print("Note: This will download year-wise files. It may take a while depending on your internet connection.")
    
    # get_data downloads the raw files year by year into the specified directory
    data = imd.get_data(variable, start_yr, end_yr, fn_format='yearwise', file_dir=data_dir)
    
    print("\nDownload complete!")
    print("Converting downloaded data to a single NetCDF file for easier processing...")
    
    # Convert the year-wise data into a single NetCDF file
    nc_path = os.path.join(data_dir, 'imd_rainfall_1901_2025.nc')
    data.to_netcdf(nc_path)
    
    print(f"Success! Data saved to: {nc_path}")

if __name__ == "__main__":
    download_imd_rainfall()

# python 01_download_data.py