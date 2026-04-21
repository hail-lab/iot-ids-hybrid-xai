"""Download BoT-IoT 5% subset (Training + Testing CSVs, 10-best feature version).

Uses the public Research Data Australia / Kaggle mirrors. On Colab, prefers
the Kaggle API if credentials are configured; otherwise falls back to direct
HTTPS download from a hosted mirror.

Outputs to data/raw/bot_iot/*.csv
"""
from __future__ import annotations
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_RAW, ensure_dirs

BOT_DIR = DATA_RAW / "bot_iot"

# Mirror-hosted copies of the 10-best-feature 5% subset (Koroniotis et al. 2019).
# These files contain the canonical 10 "best" features used in most BoT-IoT papers.
MIRROR_URLS = {
    "UNSW_2018_IoT_Botnet_Final_10_best_Training.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download",
    "UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv":
        "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download",
}


def download_via_kaggle() -> bool:
    """Try Kaggle API (works if ~/.kaggle/kaggle.json is configured)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        BOT_DIR.mkdir(parents=True, exist_ok=True)
        print("  downloading via Kaggle API (vigneshvenkateswaran/bot-iot-dataset)...")
        api.dataset_download_files(
            "vigneshvenkateswaran/bot-iot-dataset",
            path=str(BOT_DIR),
            unzip=True,
        )
        csvs = list(BOT_DIR.rglob("*.csv"))
        print(f"  downloaded {len(csvs)} CSV files")
        return len(csvs) > 0
    except Exception as e:
        print(f"  Kaggle download failed: {e}")
        return False


def download_via_https() -> bool:
    """Direct HTTPS fallback. Replace MIRROR_URLS with a working host if Kaggle fails."""
    BOT_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fname, url in MIRROR_URLS.items():
        dst = BOT_DIR / fname
        if dst.exists() and dst.stat().st_size > 1_000_000:
            print(f"  {fname} already present ({dst.stat().st_size / 1e6:.1f} MB)")
            ok += 1
            continue
        try:
            print(f"  downloading {fname}...")
            urllib.request.urlretrieve(url, dst)
            ok += 1
        except Exception as e:
            print(f"  failed {fname}: {e}")
    return ok >= 1


def main():
    ensure_dirs()
    if any(BOT_DIR.glob("*.csv")):
        print(f"BoT-IoT files already present in {BOT_DIR}")
        for p in sorted(BOT_DIR.glob("*.csv")):
            print(f"  {p.name}: {p.stat().st_size / 1e6:.1f} MB")
        return

    print("Attempting BoT-IoT download...")
    if download_via_kaggle():
        print("✓ downloaded via Kaggle")
        return
    print("  falling back to HTTPS mirror...")
    if download_via_https():
        print("✓ downloaded via HTTPS mirror")
        return

    print("\n" + "=" * 60)
    print("AUTOMATED DOWNLOAD FAILED")
    print("=" * 60)
    print("Please manually download BoT-IoT 5% subset from:")
    print("  https://research.unsw.edu.au/projects/bot-iot-dataset")
    print(f"Extract the zip and copy the CSV files from \"5%/10-best features/10-best Training-Testing split/\" to: {BOT_DIR}")
    sys.exit(1)


if __name__ == "__main__":
    main()
