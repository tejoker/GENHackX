import geopandas as gpd
import glob
import pyogrio

def inspect_gpkg_files():
    gpkg_files = glob.glob("gadm_410*.gpkg")
    if not gpkg_files:
        print("No gadm_410*.gpkg files found.")
        return
    for file in gpkg_files:
        print(f"\n--- {file} ---")
        try:
            layers = pyogrio.list_layers(file)
            print("Layers:", [l[0] for l in layers])
            if len(layers) > 0:
                # Read only the first 5 rows for memory efficiency
                gdf = gpd.read_file(file, layer=layers[0][0], rows=5)
                print("Columns:", gdf.columns.tolist())
                print(gdf)
            else:
                print("No layers found in this file.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

if __name__ == "__main__":
    inspect_gpkg_files()