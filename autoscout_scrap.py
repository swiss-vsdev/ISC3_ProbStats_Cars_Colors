from autoscout24_scraper import scrape
import requests
import os
from pathlib import Path
from PIL import Image

# --- CONFIG ---
OUTPUT_DIR = "autoscout24_images"

MAKE = "aixam"
MODEL = "crossline"

IMG_URL = "https://listing-images.autoscout24.ch/"

# --- CRÉER DOSSIER ---
output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

# --- SCRAPER ---
result = scrape(make=MAKE, model=MODEL, category='car')

for row in result.rows:
    images_str = row["images"]
    img = images_str.split(";")[0]

    id_nbr = row["id"]
    year = row["firstRegistrationYear"]
    
    if isinstance(year, str) and "/" in year:
        year = year.split("/")[-1]
    
    print(id_nbr, year, img)
    
    img_data = requests.get(f"{IMG_URL}{img}").content
    
    filepath = output_path / f"{year}_{id_nbr}.jpg"
    with open(filepath, 'wb') as handler:
        handler.write(img_data)


def compress_folder(src, quality=65):

    src_path = Path(src)
    max_size = 50 * 1024

    # 1. Delete tiny files (uses real paths, not the list object)
    for f in src_path.iterdir():
        if f.is_file() and f.stat().st_size < max_size:
            f.unlink()

    # 2. Compress remaining images
    for f in src_path.rglob("*"):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            continue

        img = Image.open(f)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        if f.suffix.lower() in {".jpg", ".jpeg"}:
            out = f
        else:
            out = f.with_suffix(".jpg")
            i = 1
            while out.exists():
                out = f.with_name(f"{f.stem}_{i}.jpg")
                i += 1

        img.save(out, "JPEG", quality=quality, optimize=True)
        img.close()

        if out != f:
            f.unlink()


compress_folder(OUTPUT_DIR, quality=50)
result.to_csv(output_path / f"{MAKE}_{MODEL}.csv")


print(f"\nTerminé: {len(result.rows)} images ajoutées dans {OUTPUT_DIR}")