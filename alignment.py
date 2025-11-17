import rasterio
from rasterio.warp import reproject, Resampling
import glob
import os

# Set reference raster (e.g., first ERA5 file)
ref_files = glob.glob("data/derived-era5-land-daily-statistics/*.nc")
if not ref_files:
    raise FileNotFoundError("No reference ERA5 NetCDF files found.")
ref_file = ref_files[0]

# Open reference raster (assume variable 't2m' for temperature, update as needed)
with rasterio.open(f'NETCDF:"{ref_file}":t2m') as ref:
    ref_crs = ref.crs
    ref_transform = ref.transform
    ref_width = ref.width
    ref_height = ref.height
    ref_profile = ref.profile.copy()

# List all raster datasets to align (GeoTIFFs, NetCDFs, etc.)
raster_patterns = [
    "data/sentinel2_ndvi/*.tif",
    "data/sentinel3-olci-ndvi/*.tif",
    "data/derived-era5-land-daily-statistics/*.nc",
    "data/derived-era5-single-levels-daily-statistics/*.nc"
]
raster_files = []
for pattern in raster_patterns:
    raster_files.extend(glob.glob(pattern))

# Output directory
os.makedirs("data/aligned", exist_ok=True)

for raster_path in raster_files:
    # Skip the reference file itself
    if raster_path == ref_file:
        continue
    # Determine file type and open accordingly
    if raster_path.endswith('.tif'):
        with rasterio.open(raster_path) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_count = src.count
            # Resample and reproject
            aligned_data = src.read(
                out_shape=(src_count, ref_height, ref_width),
                resampling=Resampling.average
            )
            if src_crs != ref_crs:
                aligned_data_reproj = aligned_data.copy()
                for i in range(src_count):
                    reproject(
                        source=aligned_data[i],
                        destination=aligned_data_reproj[i],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.average
                    )
                aligned_data = aligned_data_reproj
            out_profile = ref_profile.copy()
            out_profile.update(dtype=aligned_data.dtype, count=src_count)
            out_path = os.path.join("data/aligned", os.path.basename(raster_path))
            with rasterio.open(out_path, 'w', **out_profile) as dst:
                dst.write(aligned_data)
            print(f"Aligned and saved: {out_path}")
    elif raster_path.endswith('.nc'):
        # For NetCDF, align only the first variable (update as needed)
        with rasterio.open(f'NETCDF:"{raster_path}":t2m') as src:
            src_crs = src.crs
            src_transform = src.transform
            src_count = src.count
            aligned_data = src.read(
                out_shape=(src_count, ref_height, ref_width),
                resampling=Resampling.average
            )
            if src_crs != ref_crs:
                aligned_data_reproj = aligned_data.copy()
                for i in range(src_count):
                    reproject(
                        source=aligned_data[i],
                        destination=aligned_data_reproj[i],
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.average
                    )
                aligned_data = aligned_data_reproj
            out_profile = ref_profile.copy()
            out_profile.update(dtype=aligned_data.dtype, count=src_count)
            out_name = os.path.splitext(os.path.basename(raster_path))[0] + "_aligned.tif"
            out_path = os.path.join("data/aligned", out_name)
            with rasterio.open(out_path, 'w', **out_profile) as dst:
                dst.write(aligned_data)
            print(f"Aligned and saved: {out_path}")
    else:
        print(f"Unsupported file type: {raster_path}")
