import os
import pandas as pd
from config import DATA_RAW, ensure_dirs

def preview_csv_folder(folder, n_files=5, n_rows=5):
    folder = str(folder)
    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
    files.sort()
    print(f"\nFolder: {folder}")
    print(f"Found {len(files)} CSV files")
    for f in files[:n_files]:
        fp = os.path.join(folder, f)
        df = pd.read_csv(fp, nrows=n_rows)
        print(f"\n--- {f} ---")
        print(df.head(n_rows))
        print(f"Columns ({len(df.columns)}): {list(df.columns)[:20]}{'...' if len(df.columns)>20 else ''}")

if __name__ == "__main__":
    ensure_dirs()
    preview_csv_folder(DATA_RAW / "cicids2017")
    preview_csv_folder(DATA_RAW / "bot_iot")
