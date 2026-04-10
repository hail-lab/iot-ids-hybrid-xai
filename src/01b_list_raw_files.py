from pathlib import Path
from config import DATA_RAW

def list_all(path: Path):
    print(f"\nListing: {path}")
    if not path.exists():
        print("Path does not exist.")
        return
    items = sorted(path.rglob("*"))
    print(f"Total files (any type): {sum(1 for p in items if p.is_file())}")
    for p in items[:50]:
        if p.is_file():
            print(p.relative_to(DATA_RAW))

if __name__ == "__main__":
    list_all(DATA_RAW / "cicids2017")
    list_all(DATA_RAW / "bot_iot")