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

# Automatic optimizer: Find optimal resolution factor using binary search
def find_optimal_factor(data_path, data_type='raster', min_variance=0.90,
                       max_factor=50, target_resolution=0.1, max_size=2000):
    """
    Find the highest resolution factor that maintains at least min_variance.

    Args:
        data_path: Path to data file
        data_type: 'raster', 'era5', or 'gpkg'
        min_variance: Minimum acceptable variance_explained (0-1)
        max_factor: Maximum factor to test
        target_resolution: For GPKG only
        max_size: Maximum dimension size

    Returns:
        dict with optimal_factor, variance_explained, and all tested results
    """
    print(f"Searching for optimal factor with variance >= {min_variance}...")

    # Binary search
    low, high = 1, max_factor
    best_factor = 1
    best_variance = 1.0
    all_results = []

    # Test function based on data type
    if data_type == 'raster':
        test_func = lambda f: test_resolutions_raster(data_path, [f], max_size)
    elif data_type == 'era5':
        test_func = lambda f: test_resolutions_era5(data_path, [f], max_size)
    elif data_type == 'gpkg':
        test_func = lambda f: test_resolutions_gpkg(data_path, [f], target_resolution, max_size)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    while low <= high:
        mid = (low + high) // 2
        print(f"Testing factor {mid}...", end=' ')

        result = test_func(mid)[0]
        all_results.append(result)
        variance = result['variance_explained']

        print(f"variance={variance:.4f}")

        if variance >= min_variance:
            # This factor is acceptable, try higher
            best_factor = mid
            best_variance = variance
            low = mid + 1
        else:
            # Factor too high, try lower
            high = mid - 1

    print(f"\nOptimal factor: {best_factor} (variance={best_variance:.4f})")

    return {
        'optimal_factor': best_factor,
        'variance_explained': best_variance,
        'all_results': sorted(all_results, key=lambda x: x['factor'])
    }

# Helper: Find optimal factor from existing results
def find_optimal_from_results(results, min_variance=0.90):
    """
    Find the highest resolution factor from existing results that maintains min_variance.

    Args:
        results: List of dicts with 'factor', 'variance_explained', 'morans_i'
        min_variance: Minimum acceptable variance_explained (0-1)

    Returns:
        dict with optimal_factor and variance_explained
    """
    best_factor = 1
    best_variance = 1.0

    # Sort by factor in descending order to find the highest acceptable factor
    sorted_results = sorted(results, key=lambda x: x['factor'], reverse=True)

    for r in sorted_results:
        if r['variance_explained'] >= min_variance:
            best_factor = r['factor']
            best_variance = r['variance_explained']
            break

    return {
        'optimal_factor': best_factor,
        'variance_explained': best_variance
    }

# Visualization: Plot resolution analysis
def plot_resolution_analysis(results, title="Resolution Analysis", save_path=None):
    """
    Plot variance_explained and Moran's I vs resolution factor.

    Args:
        results: List of dicts with 'factor', 'variance_explained', 'morans_i'
        title: Plot title
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt

    factors = [r['factor'] for r in results]
    variances = [r['variance_explained'] for r in results]
    morans = [r['morans_i'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Variance explained
    ax1.plot(factors, variances, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
    ax1.axhline(y=0.95, color='orange', linestyle='--', label='95% threshold')
    ax1.set_xlabel('Resolution Factor (coarser →)', fontsize=12)
    ax1.set_ylabel('Variance Explained', fontsize=12)
    ax1.set_title('Information Retention', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([min(variances) - 0.05, 1.02])

    # Plot 2: Moran's I
    ax2.plot(factors, morans, 's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Resolution Factor (coarser →)', fontsize=12)
    ax2.set_ylabel("Moran's I", fontsize=12)
    ax2.set_title('Spatial Autocorrelation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([min(morans) - 0.05, max(morans) + 0.05])

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    return fig

# Example usage for Sentinel-2, Sentinel-3, ERA5, and GPKG
if __name__ == "__main__":
    import sys

    # Check command line arguments for mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "basic"

    if mode == "basic":
        print("=" * 60)
        print("BASIC MODE: Testing fixed resolution factors [1, 2, 5, 10]")
        print("=" * 60 + "\n")

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

    elif mode == "optimize":
        print("=" * 60)
        print("OPTIMIZER MODE: Finding optimal resolution factors")
        print("=" * 60 + "\n")

        # Find optimal factor for each dataset type
        s2_files = glob.glob("data/sentinel2_ndvi/*.tif")
        if s2_files:
            print("\n" + "=" * 60)
            print("Sentinel-2 NDVI - Finding optimal factor (variance >= 0.90)")
            print("=" * 60)
            result = find_optimal_factor(s2_files[0], 'raster', min_variance=0.90, max_factor=30)
            print(f"\nRecommendation: Use factor {result['optimal_factor']} for 90% variance retention")

        s3_files = glob.glob("data/sentinel3-olci-ndvi/*.tif")
        if s3_files:
            print("\n" + "=" * 60)
            print("Sentinel-3 NDVI - Finding optimal factor (variance >= 0.90)")
            print("=" * 60)
            result = find_optimal_factor(s3_files[0], 'raster', min_variance=0.90, max_factor=30)
            print(f"\nRecommendation: Use factor {result['optimal_factor']} for 90% variance retention")

        era5_files = glob.glob("data/derived-era5-*-daily-statistics/*.nc")
        if era5_files:
            print("\n" + "=" * 60)
            print(f"ERA5 ({os.path.basename(era5_files[0])}) - Finding optimal factor (variance >= 0.90)")
            print("=" * 60)
            result = find_optimal_factor(era5_files[0], 'era5', min_variance=0.90, max_factor=30)
            print(f"\nRecommendation: Use factor {result['optimal_factor']} for 90% variance retention")

        gpkg_files = glob.glob("data/*europe*.gpkg")
        if gpkg_files:
            print("\n" + "=" * 60)
            print(f"GPKG ({os.path.basename(gpkg_files[0])}) - Finding optimal factor (variance >= 0.90)")
            print("=" * 60)
            result = find_optimal_factor(gpkg_files[0], 'gpkg', min_variance=0.90,
                                        max_factor=30, target_resolution=0.05)
            print(f"\nRecommendation: Use factor {result['optimal_factor']} for 90% variance retention")

    elif mode == "visualize":
        print("=" * 60)
        print("VISUALIZATION MODE: Plotting resolution analysis curves")
        print("=" * 60 + "\n")

        # Test a range of factors and create plots
        test_factors = [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30]

        s2_files = glob.glob("data/sentinel2_ndvi/*.tif")
        if s2_files:
            print("Analyzing Sentinel-2 NDVI...")
            results = test_resolutions_raster(s2_files[0], test_factors)
            plot_resolution_analysis(results,
                                   title="Sentinel-2 NDVI - Resolution Analysis",
                                   save_path="sentinel2_resolution_analysis.png")
            # Find and print optimal factor
            optimal = find_optimal_from_results(results, min_variance=0.90)
            print(f"Optimal factor for 90% variance: {optimal['optimal_factor']} (variance={optimal['variance_explained']:.4f})")
            optimal_95 = find_optimal_from_results(results, min_variance=0.95)
            print(f"Optimal factor for 95% variance: {optimal_95['optimal_factor']} (variance={optimal_95['variance_explained']:.4f})")
            print()

        s3_files = glob.glob("data/sentinel3-olci-ndvi/*.tif")
        if s3_files:
            print("Analyzing Sentinel-3 NDVI...")
            results = test_resolutions_raster(s3_files[0], test_factors)
            plot_resolution_analysis(results,
                                   title="Sentinel-3 NDVI - Resolution Analysis",
                                   save_path="sentinel3_resolution_analysis.png")
            # Find and print optimal factor
            optimal = find_optimal_from_results(results, min_variance=0.90)
            print(f"Optimal factor for 90% variance: {optimal['optimal_factor']} (variance={optimal['variance_explained']:.4f})")
            optimal_95 = find_optimal_from_results(results, min_variance=0.95)
            print(f"Optimal factor for 95% variance: {optimal_95['optimal_factor']} (variance={optimal_95['variance_explained']:.4f})")
            print()

        era5_files = glob.glob("data/derived-era5-*-daily-statistics/*.nc")
        if era5_files:
            print(f"Analyzing ERA5 ({os.path.basename(era5_files[0])})...")
            results = test_resolutions_era5(era5_files[0], test_factors)
            plot_resolution_analysis(results,
                                   title=f"ERA5 - Resolution Analysis",
                                   save_path="era5_resolution_analysis.png")
            # Find and print optimal factor
            optimal = find_optimal_from_results(results, min_variance=0.90)
            print(f"Optimal factor for 90% variance: {optimal['optimal_factor']} (variance={optimal['variance_explained']:.4f})")
            optimal_95 = find_optimal_from_results(results, min_variance=0.95)
            print(f"Optimal factor for 95% variance: {optimal_95['optimal_factor']} (variance={optimal_95['variance_explained']:.4f})")
            print()

        gpkg_files = glob.glob("data/*europe*.gpkg")
        if gpkg_files:
            print(f"Analyzing GPKG ({os.path.basename(gpkg_files[0])})...")
            results = test_resolutions_gpkg(gpkg_files[0], test_factors, target_resolution=0.05)
            plot_resolution_analysis(results,
                                   title="GPKG Administrative Boundaries - Resolution Analysis",
                                   save_path="gpkg_resolution_analysis.png")
            # Find and print optimal factor
            optimal = find_optimal_from_results(results, min_variance=0.90)
            print(f"Optimal factor for 90% variance: {optimal['optimal_factor']} (variance={optimal['variance_explained']:.4f})")
            optimal_95 = find_optimal_from_results(results, min_variance=0.95)
            print(f"Optimal factor for 95% variance: {optimal_95['optimal_factor']} (variance={optimal_95['variance_explained']:.4f})")
            print()

    else:
        print("Usage:")
        print("  python test_resolution_framework.py              # Basic mode (default)")
        print("  python test_resolution_framework.py optimize     # Find optimal factors")
        print("  python test_resolution_framework.py visualize    # Create plots")

# Note: This script is a framework. You can adapt it for batch processing, plotting, or dichotomy search as needed.
