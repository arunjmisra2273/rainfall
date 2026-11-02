import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("data/RF25_ind1951_rfp25.nc")
ds = ds.rename({"TIME": "time", "LATITUDE": "lat", "LONGITUDE": "lon"})
rain = ds["RAINFALL"]

print("Sizes     :", dict(ds.sizes))
print("Time range:", str(rain.time[0].values)[:10], "->", str(rain.time[-1].values)[:10])

# Check 1: is the NaN pattern constant across days? (stable mask = ocean, spikes = trouble)
nan_by_day = rain.isnull().mean(dim=("lat", "lon"))
print("NaN fraction per day: min %.4f  max %.4f" % (nan_by_day.min(), nan_by_day.max()))

# Check 2: land mask = cells with data on >90% of days
land = rain.notnull().mean("time") > 0.9
india = rain.where(land)
print("Land grid points:", int(land.sum()), "of", land.size)
print("NaN fraction within land:", float(india.isnull().mean()))

# Check 3: physically plausible values over land
print("Land min: %.2f  max: %.2f mm/day" % (india.min(), india.max()))
print("Negative values in land:", int((india < 0).sum()))

# Check 4: JJAS mean should be of order 6-9 mm/day over Indian land
jjas = india.sel(time=india.time.dt.month.isin([6, 7, 8, 9]))
print("JJAS land mean: %.2f mm/day" % jjas.mean())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
rain.sel(time="1951-07-15").plot(ax=ax[0], cmap="Blues", vmin=0, vmax=40)
ax[0].set_title("Rainfall, 15 July 1951 (mm)")
india.mean(dim=("lat", "lon")).plot(ax=ax[1])
ax[1].set_title("All-India daily mean rain, 1951 (mm/day)")
plt.tight_layout()
plt.savefig("inspection.png", dpi=150)
print("Saved inspection.png")