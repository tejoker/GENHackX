import os
import glob
import numpy as np
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from skimage.measure import block_reduce
from esda.moran import Moran
from libpysal.weights import lat2W

# Helper: Resample raster to new resolution using block_reduce
# Only for demonstration, not for production (no nodata handling, etc.)
def resample_raster(data, factor, method=np.nanmean):
    if factor == 1:
        return data
    return block_reduce(data, block_size=(factor, factor), func=method)

# Helper: Compute variance explained
# orig: original data, agg: aggregated data (upsampled to original shape)
def variance_explained(orig, agg):
    orig_flat = orig.flatten()
    agg_flat = agg.flatten()
    # Filter out NaN values
    valid_mask = ~(np.isnan(orig_flat) | np.isnan(agg_flat))
    orig_valid = orig_flat[valid_mask]
    agg_valid = agg_flat[valid_mask]
    if len(orig_valid) == 0:
        return 0
    var_orig = np.var(orig_valid)
    var_diff = np.var(orig_valid - agg_valid)
    return 1 - var_diff / var_orig if var_orig > 0 else 0

# Helper: Compute Moran's I for spatial autocorrelation
# Only works for 2D arrays
# Uses queen contiguity weights
# Returns Moran's I value
# Requires esda and libpysal

def morans_i(data):
    # Filter out NaN values for Moran's I calculation
    # Note: This is a simplified approach; proper spatial analysis would preserve spatial structure
    data_clean = np.where(np.isnan(data), np.nanmean(data), data)
    nrows, ncols = data_clean.shape
    w = lat2W(nrows, ncols)
    mi = Moran(data_clean.flatten(), w)
    return mi.I

# Main function: test resolutions for a raster dataset
# data_path: path to raster file
# res_factors: list of integer factors (e.g., [1, 2, 5, 10])
# max_size: maximum dimension size to read (to avoid memory issues)
def test_resolutions_raster(data_path, res_factors, max_size=2000):
    with rasterio.open(data_path) as src:
        # For large files, read a subset to avoid memory issues
        height, width = src.shape

        # Calculate downsampling factor if needed
        downsample = max(1, max(height, width) // max_size)

        # Read with downsampling using out_shape parameter
        out_shape = (height // downsample, width // downsample)

        print(f"Original size: {height}x{width}, Reading at: {out_shape[0]}x{out_shape[1]}")

        # Read as masked array with downsampling and convert to float before filling with NaN
        masked_arr = src.read(1,
                             out_shape=out_shape,
                             resampling=Resampling.average,
                             masked=True)
        arr = masked_arr.astype(np.float32).filled(np.nan)

        results = []
        for f in res_factors:
            agg = resample_raster(arr, f)
            # Upsample back to original shape for fair comparison
            upsampled = np.repeat(np.repeat(agg, f, axis=0), f, axis=1)
            upsampled = upsampled[:arr.shape[0], :arr.shape[1]]
            var_exp = variance_explained(arr, upsampled)
            mi = morans_i(agg)
            results.append({'factor': f, 'variance_explained': var_exp, 'morans_i': mi})
        return results

# Helper: Test resolutions for ERA5 NetCDF data (handles 3D time series)
def test_resolutions_era5(data_path, res_factors, max_size=2000):
    with xr.open_dataset(data_path) as ds:
        var = list(ds.data_vars)[0]
        arr = ds[var].values

        # If 3D (time, lat, lon), take temporal mean
        if arr.ndim == 3:
            print(f"Taking temporal mean of {arr.shape[0]} time steps")
            arr = np.nanmean(arr, axis=0)

        print(f"Original size: {arr.shape[0]}x{arr.shape[1]}")

        # Downsample if needed
        downsample = max(1, max(arr.shape) // max_size)
        if downsample > 1:
            new_shape = (arr.shape[0] // downsample, arr.shape[1] // downsample)
            arr = block_reduce(arr, block_size=(downsample, downsample), func=np.nanmean)
            print(f"Reading at: {arr.shape[0]}x{arr.shape[1]}")

        results = []
        for f in res_factors:
            agg = resample_raster(arr, f)
            upsampled = np.repeat(np.repeat(agg, f, axis=0), f, axis=1)
            upsampled = upsampled[:arr.shape[0], :arr.shape[1]]
            var_exp = variance_explained(arr, upsampled)
            mi = morans_i(agg)
            results.append({'factor': f, 'variance_explained': var_exp, 'morans_i': mi})
        return results

# Helper: Test resolutions for GPKG vector data (rasterizes boundaries)
def test_resolutions_gpkg(data_path, res_factors, target_resolution=0.1, max_size=2000):
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Read vector data
    gdf = gpd.read_file(data_path)
    print(f"Loaded {len(gdf)} features")

    # Get bounds
    bounds = gdf.total_bounds

    # Calculate raster dimensions based on target resolution
    width = int((bounds[2] - bounds[0]) / target_resolution)
    height = int((bounds[3] - bounds[1]) / target_resolution)

    # Downsample if too large
    downsample = max(1, max(width, height) // max_size)
    width = width // downsample
    height = height // downsample

    print(f"Rasterizing to {height}x{width}")

    # Create transform
    transform = from_bounds(*bounds, width, height)

    # Rasterize (assign unique ID to each feature)
    shapes = [(geom, idx) for idx, geom in enumerate(gdf.geometry)]
    arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype=np.float32)

    # Replace 0 with NaN for consistency
    arr[arr == 0] = np.nan

    results = []
    for f in res_factors:
        agg = resample_raster(arr, f)
        upsampled = np.repeat(np.repeat(agg, f, axis=0), f, axis=1)
        upsampled = upsampled[:arr.shape[0], :arr.shape[1]]
        var_exp = variance_explained(arr, upsampled)
        mi = morans_i(agg)
        results.append({'factor': f, 'variance_explained': var_exp, 'morans_i': mi})
    return results

# Example usage for Sentinel-2, Sentinel-3, ERA5, and GPKG
if __name__ == "__main__":
    # Example: test on one Sentinel-2 file
    s2_files = glob.glob("data/sentinel2_ndvi/*.tif")
    if s2_files:
        print("Sentinel-2 NDVI:")
        print(test_resolutions_raster(s2_files[0], [1, 2, 5, 10]))
        print()

    # Example: test on one Sentinel-3 file
    s3_files = glob.glob("data/sentinel3-olci-ndvi/*.tif")
    if s3_files:
        print("Sentinel-3 NDVI:")
        print(test_resolutions_raster(s3_files[0], [1, 2, 5, 10]))
        print()

    # Example: test on one ERA5 file
    era5_files = glob.glob("data/derived-era5-*-daily-statistics/*.nc")
    if era5_files:
        print(f"ERA5 ({os.path.basename(era5_files[0])}):")
        print(test_resolutions_era5(era5_files[0], [1, 2, 5, 10]))
        print()

    # Example: test on GPKG (administrative boundaries)
    gpkg_files = glob.glob("data/*europe*.gpkg")
    if gpkg_files:
        print(f"GPKG ({os.path.basename(gpkg_files[0])}):")
        print(test_resolutions_gpkg(gpkg_files[0], [1, 2, 5, 10], target_resolution=0.05))
        print()

# Note: This script is a framework. You can adapt it for batch processing, plotting, or dichotomy search as needed.
