
import os
import glob
import geopandas as gpd
import rasterio
import rioxarray
import xarray as xr
from rasterio.errors import RasterioIOError

def get_raster_info(path):
    try:
        with rasterio.open(path) as src:
            res = src.res
            crs = src.crs
            width = src.width
            height = src.height
            # Convert resolution to km if possible
            # If CRS is degrees, approximate 1 deg ~ 111 km
            if crs and hasattr(crs, 'is_geographic') and crs.is_geographic:
                res_km = (round(res[0]*111, 3), round(res[1]*111, 3))
            else:
                res_km = (round(res[0]/1000, 3), round(res[1]/1000, 3))
            return crs, res_km, width, height
    except RasterioIOError:
        return None

def get_gpkg_info(path):
    try:
        gdf = gpd.read_file(path)
        crs = gdf.crs
        bounds = gdf.total_bounds
        num_features = len(gdf)
        # Estimate average feature spacing as a proxy for resolution (very rough)
        # Only if there are enough features
        if num_features > 1:
            x_res = (bounds[2] - bounds[0]) / (num_features ** 0.5)
            y_res = (bounds[3] - bounds[1]) / (num_features ** 0.5)
            # If CRS is degrees, convert to km
            if crs and hasattr(crs, 'is_geographic') and crs.is_geographic:
                x_res_km = round(x_res * 111, 3)
                y_res_km = round(y_res * 111, 3)
            else:
                x_res_km = round(x_res / 1000, 3)
                y_res_km = round(y_res / 1000, 3)
            res_km = (x_res_km, y_res_km)
        else:
            res_km = (None, None)
        return crs, bounds, num_features, res_km
    except Exception:
        return None

def get_netcdf_info(path):
    try:
        ds = rioxarray.open_rasterio(path)
        crs = ds.rio.crs
        res = ds.rio.resolution()
        width = ds.rio.width
        height = ds.rio.height
        return [(os.path.basename(path), crs, res, width, height)]
    except Exception:
        try:
            ds = xr.open_dataset(path)
            info = []
            for var in ds.data_vars:
                da = ds[var]
                if hasattr(da, 'rio') and hasattr(da.rio, 'crs'):
                    crs = da.rio.crs
                    res = da.rio.resolution()
                    width = da.rio.width
                    height = da.rio.height
                else:
                    crs = None
                    res = None
                    width = None
                    height = None
                # Try to get lat/lon info if available
                # Always try to output numpy float values for lat/lon info, or empty tuple if not available
                if 'latitude' in da.coords and 'longitude' in da.coords:
                    lat = da.coords['latitude'].values
                    lon = da.coords['longitude'].values
                    lat_min = lat.min() if len(lat) > 0 else ''
                    lat_max = lat.max() if len(lat) > 0 else ''
                    lat_step = lat[1] - lat[0] if len(lat) > 1 else ''
                    lon_min = lon.min() if len(lon) > 0 else ''
                    lon_max = lon.max() if len(lon) > 0 else ''
                    lon_step = lon[1] - lon[0] if len(lon) > 1 else ''
                    lat_info = f"(np.float64({lat_min}), np.float64({lat_max}), np.float64({lat_step}))"
                    lon_info = f"(np.float64({lon_min}), np.float64({lon_max}), np.float64({lon_step}))"
                else:
                    lat_info = lon_info = "()"
                info.append((var, crs, res_km, width, height, lat_info, lon_info, da.shape))
            ds.close()
            return info
        except Exception:
            return None

import json
output = {
    "gpkg": [],
    "sentinel": [],
    "era5": []
}

# GPKG section
for file in glob.glob("data/*.gpkg"):
    info = get_gpkg_info(file)
    if info:
        crs, bounds, num_features, res_km = info
        output["gpkg"].append({
            "file": file,
            "crs": str(crs),
            "bounds": bounds.tolist() if hasattr(bounds, 'tolist') else bounds,
            "num_features": num_features,
            "resolution_km": res_km
        })
    else:
        output["gpkg"].append({"file": file, "error": True})

# Sentinel-2 and Sentinel-3 raster files (GeoTIFF)
for file in glob.glob("data/sentinel2_ndvi/*.tif"):
    info = get_raster_info(file)
    if info:
        crs, res, width, height = info
        output["sentinel"].append({
            "file": file,
            "crs": str(crs),
            "resolution": res,
            "width": width,
            "height": height,
            "satellite": "sentinel2"
        })
    else:
        output["sentinel"].append({"file": file, "error": True, "satellite": "sentinel2"})
for file in glob.glob("data/sentinel3-olci-ndvi/*.tif"):
    info = get_raster_info(file)
    if info:
        crs, res, width, height = info
        output["sentinel"].append({
            "file": file,
            "crs": str(crs),
            "resolution": res,
            "width": width,
            "height": height,
            "satellite": "sentinel3"
        })
    else:
        output["sentinel"].append({"file": file, "error": True, "satellite": "sentinel3"})

# ERA5 NetCDF files
for file in glob.glob("data/derived-era5-*-daily-statistics/*.nc"):
    info = get_netcdf_info(file)
    if info:
        for entry in info:
            if len(entry) == 5:
                var, crs, res, width, height = entry
                output["era5"].append({
                    "file": file,
                    "variable": var,
                    "crs": str(crs),
                    "resolution": (res[0]/1000, res[1]/1000) if res else None, #because in meters
                    "width": width,
                    "height": height
                })
            else:
                var, crs, res, width, height, lat_info, lon_info, shape = entry
                output["era5"].append({
                    "file": file,
                    "variable": var,
                    "crs": str(crs),
                    "resolution": (res[0]/1000, res[1]/1000) if res else None,
                    "width": width,
                    "height": height,
                    "latitude_info": lat_info,
                    "longitude_info": lon_info,
                    "shape": shape
                })
    else:
        output["era5"].append({"file": file, "error": True})

with open("data_resolutions.json", "w") as f:
    json.dump(output, f, indent=2)
