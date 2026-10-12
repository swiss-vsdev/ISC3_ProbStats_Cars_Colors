from autoscout24_scraper import scrape, ScrapeResult
from make_model_library import MAKE_MODEL_PAIRS
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin
from io import BytesIO
from pathlib import Path
import threading
import requests
import csv
import time
import random
import os
import tempfile
from PIL import Image, UnidentifiedImageError

# --- CONFIG ---
OUTPUT_DIR = "autoscout24_images"
IMG_URL = "https://listing-images.autoscout24.ch/"
CSV_PATH = Path(OUTPUT_DIR) / "all_cars.csv"
MAX_WORKERS = 12            # was 24; fewer + Session is faster & kinder to the CDN
IMG_QUALITY = 50
MIN_SIZE = 50 * 1024        # skip files smaller than this
TIMEOUT = 15
RETRIES = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

# --- Thread-local HTTP session (reuses TCP/TLS per worker) ---
_thread_local = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; autoscout-scraper/1.0)"})
        _thread_local.session = s
    return s


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
    """Return a path that doesn't collide with an existing file."""
    dest = directory / f"{stem}{suffix}"
    if not dest.exists():
        return dest
    i = 1
    while True:
        dest = directory / f"{stem}_{i}{suffix}"
        if not dest.exists():
            return dest
        i += 1


def sort_by_year(src, subfolder=None):
    """Move images into subfolders named after their year (from '{year}_{id}')."""
    src_path = Path(src)
    target_root = src_path / subfolder if subfolder else src_path
    target_root.mkdir(exist_ok=True)

    for f in list(src_path.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue

        stem = f.stem
        if "_" not in stem:
            continue

        year = stem.split("_", 1)[0]
        if not (year.isdigit() and 1900 <= int(year) <= 2100):
            continue

        year_dir = target_root / year
        year_dir.mkdir(exist_ok=True)

        dest = _unique_path(year_dir, stem, f.suffix)
        f.replace(dest)


def download_and_compress(img_name, year, id_nbr):
    """Download one image, compress in memory, write atomically. Returns path or None."""
    # Guard against bad identifiers that would produce garbage filenames
    if not id_nbr:
        return None
    year_part = str(year) if year and str(year).isdigit() else "unknown"
    stem = f"{year_part}_{id_nbr}"

    url = urljoin(IMG_URL, img_name)
    session = _get_session()

    last_err = None
    for attempt in range(RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.content
            break
        except (requests.RequestException, OSError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"[ERREUR image] {img_name} : {last_err}")
                return None

    if len(data) < MIN_SIZE:
        return None

    try:
        with Image.open(BytesIO(data)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # Resolve final collision-free path
            final_path = _unique_path(output_path, stem, ".jpg")

            # Write atomically: temp file in the same dir, then os.replace
            fd, tmp_name = tempfile.mkstemp(dir=output_path, suffix=".jpg.tmp")
            os.close(fd)
            tmp_path = Path(tmp_name)
            try:
                img.save(tmp_path, "JPEG", quality=IMG_QUALITY, optimize=True)
                os.replace(tmp_path, final_path)
            except Exception:
                tmp_path.unlink(missing_ok=True)
                raise

        return final_path
    except (UnidentifiedImageError, OSError, ValueError) as e:
        print(f"[ERREUR image] {img_name} : {e}")
        return None


# --- SCRAPER ---
def main():
    all_count = 0

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = None

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for MAKE, MODEL in MAKE_MODEL_PAIRS:
                try:
                    result = scrape(make=MAKE, model=MODEL, category="car")
                except Exception as e:
                    print(f"[ERREUR scrape] {MAKE} - {MODEL} : {e}")
                    time.sleep(random.uniform(2, 5))
                    continue

                rows = list(result.rows)  # materialize once (in case it's a generator)
                if not rows:
                    print(f"{MAKE} - {MODEL}, vide")
                    continue

                # --- CSV: initialize writer once, using the first row's schema ---
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                    csv_writer.writeheader()

                # --- Build download tasks ---
                tasks = []
                for row in rows:
                    try:
                        images_str = row.get("images")
                        if not images_str:
                            continue
                        img = images_str.split(";")[0]

                        id_nbr = row.get("id")
                        year = row.get("firstRegistrationYear")
                        if isinstance(year, str) and "/" in year:
                            year = year.split("/")[-1]

                        tasks.append((img, year, id_nbr))
                    except Exception as e:
                        print(f"[ERREUR parsing] {MAKE} - {MODEL} : {e}")

                # --- Parallel download + compress ---
                futures = [
                    pool.submit(download_and_compress, img, year, id_nbr)
                    for img, year, id_nbr in tasks
                ]
                for fut in as_completed(futures):
                    try:
                        if fut.result() is not None:
                            all_count += 1
                    except Exception as e:
                        print(f"[ERREUR futur] {MAKE} - {MODEL} : {e}")

                # --- Write CSV for this make/model ---
                try:
                    for row in rows:
                        csv_writer.writerow(row)
                    csv_file.flush()
                except Exception as e:
                    print(f"[ERREUR CSV] {MAKE} - {MODEL} : {e}")

                print(f"{MAKE} - {MODEL}, Fait ({all_count} images)")

                time.sleep(random.uniform(1, 3))
    finally:
        csv_file.close()

    # --- Sort by year at the very end ---
    try:
        sort_by_year(OUTPUT_DIR)
    except Exception as e:
        print(f"[ERREUR tri par année] {e}")

    print(f"\nTerminé: {all_count} images dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()