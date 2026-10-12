from autoscout24_scraper import scrape
from make_model_library import MAKE_MODEL_PAIRS
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin
from queue import Queue
from io import BytesIO
from pathlib import Path
from threading import Thread, Event
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

SCRAPE_WORKERS = 4          # parallel scrape() calls (be polite to the site)
IMG_WORKERS = 32            # parallel image downloads (Session reuse makes this cheap)
IMG_QUALITY = 50
MIN_SIZE = 50 * 1024
TIMEOUT = 15
RETRIES = 2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

output_path = Path(OUTPUT_DIR)
output_path.mkdir(exist_ok=True)

# --- Thread-local HTTP session ---
_thread_local = threading.local()

def _get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; autoscout-scraper/1.0)"})
        _thread_local.session = s
    return s


def _unique_path(directory: Path, stem: str, suffix: str) -> Path:
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
    """Download one image and write atomically. Returns path or None.

    Skips re-download if the target file (in root or any year subfolder) already exists.
    """
    if not id_nbr:
        return None

    year_part = str(year) if year and str(year).isdigit() else "unknown"
    stem = f"{year_part}_{id_nbr}"

    # Already downloaded? Check both root and year subfolder (post-sort layout).
    if (output_path / f"{stem}.jpg").exists():
        return output_path / f"{stem}.jpg"
    if year_part != "unknown":
        existing = output_path / year_part / f"{stem}.jpg"
        if existing.exists():
            return existing

    url = urljoin(IMG_URL, img_name)
    session = _get_session()

    last_err = None
    data = None
    for attempt in range(RETRIES + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.content
            break
        except (requests.RequestException, OSError) as e:
            last_err = e
            if attempt < RETRIES:
                time.sleep(0.4 * (attempt + 1))

    if data is None:
        print(f"[ERREUR image] {img_name} : {last_err}")
        return None
    if len(data) < MIN_SIZE:
        return None

    try:
        with Image.open(BytesIO(data)) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            final_path = _unique_path(output_path, stem, ".jpg")
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


# --- PIPELINE ---
# scrape() runs in a small pool, producing (make, model, rows) onto a queue.
# A single writer thread consumes rows -> CSV, and submits images to the download pool.
# This way scraping of the next make/model overlaps with downloading the previous one.

def _scraper_worker(make, model, result_q: Queue):
    try:
        result = scrape(make=make, model=model, category="car")
        result_q.put((make, model, list(result.rows), None))
    except Exception as e:
        result_q.put((make, model, None, e))
    finally:
        # tiny jitter so all scrapers don't fire simultaneously
        time.sleep(random.uniform(0.2, 1.0))


def main():
    all_count = 0
    total_targets = len(MAKE_MODEL_PAIRS)
    done_targets = 0

    csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
    csv_writer = None

    result_q: Queue = Queue()

    # Download pool lives for the whole run
    img_pool = ThreadPoolExecutor(max_workers=IMG_WORKERS)
    img_futures = []  # track so we can wait at the end

    def submit_downloads(rows, make, model):
        nonlocal all_count
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
                fut = img_pool.submit(download_and_compress, img, year, id_nbr)
                img_futures.append(fut)
            except Exception as e:
                print(f"[ERREUR parsing] {make} - {model} : {e}")

    try:
        # Start the scraper pool
        with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as scrape_pool:
            for make, model in MAKE_MODEL_PAIRS:
                scrape_pool.submit(_scraper_worker, make, model, result_q)

            # Consume results as they arrive
            for _ in range(total_targets):
                make, model, rows, err = result_q.get()
                done_targets += 1

                if err is not None:
                    print(f"[ERREUR scrape] {make} - {model} : {err}")
                    continue
                if not rows:
                    print(f"[{done_targets}/{total_targets}] {make} - {model} : vide")
                    continue

                # CSV: init writer once
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                    csv_writer.writeheader()

                # Fire off image downloads (non-blocking — pool handles them)
                submit_downloads(rows, make, model)

                # Write CSV rows
                try:
                    for row in rows:
                        csv_writer.writerow(row)
                    csv_file.flush()
                except Exception as e:
                    print(f"[ERREUR CSV] {make} - {model} : {e}")

                print(f"[{done_targets}/{total_targets}] {make} - {model} "
                      f"({len(rows)} annonces, {len(img_futures)} images en cours)")

        # Scraping done. Wait for all image downloads.
        print(f"\nScraping terminé. Attente de {len(img_futures)} téléchargements...")
        for fut in img_futures:
            try:
                if fut.result() is not None:
                    all_count += 1
            except Exception as e:
                print(f"[ERREUR futur] {e}")

    finally:
        csv_file.close()
        img_pool.shutdown(wait=True)

    try:
        sort_by_year(OUTPUT_DIR)
    except Exception as e:
        print(f"[ERREUR tri par année] {e}")

    print(f"\nTerminé: {all_count} images dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()