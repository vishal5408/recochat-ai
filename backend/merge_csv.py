import pandas as pd
import glob
import os

# Folder where your CSVs are extracted
path = "data/"

# Get all CSV files in that folder
all_files = glob.glob(os.path.join(path, "*.csv"))

dfs = []

for file in all_files:
    try:
        df = pd.read_csv(file, encoding='latin1')
        print(f"Loaded: {os.path.basename(file)} with {len(df)} rows")
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ Error reading {file}: {e}")

# Only merge if at least one file loaded
if dfs:
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv("data/amazon_products.csv", index=False)
    print(f"✅ Merged successfully! Total rows: {len(merged_df)}")
else:
    print("❌ No CSV files were merged. Check your data folder path.")

