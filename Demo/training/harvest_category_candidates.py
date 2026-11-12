"""Targeted harvester for authentic Vietnam motorcycle photos from Wikimedia Commons categories.
Queries category members from subcategories of 'Motorcycles in Vietnam' and downloads candidates politely.
"""
import urllib.request
import urllib.parse
import json
import time
import os
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
ACQ_DIR = TRAINING_DIR / "acquisition_5000"
DOWNLOAD_DIR = ACQ_DIR / "download"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "HelmetResearchCodex5000/1.0 (https://github.com/caindandut/CV; caindandut@users.noreply.github.com)"
}

TARGET_CATEGORIES = [
    "Category:People with motorcycles in Vietnam",
    "Category:Groups of motorcycles in Vietnam",
    "Category:Motor scooters in Vietnam",
    "Category:Motorcycle helmets in Vietnam",
    "Category:Parked motorcycles in Vietnam",
    "Category:Transport motorcycles in Vietnam",
    "Category:Motorcycle taxis in Vietnam",
    "Category:Motorcycles in Vietnam"
]

def get_category_files(cat_title, limit=50):
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "categorymembers",
        "cmtitle": cat_title,
        "cmtype": "file",
        "cmlimit": str(limit),
        "format": "json"
    })
    req = urllib.request.Request(f"{API}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("query", {}).get("categorymembers", [])
    except Exception as e:
        print(f"Error querying {cat_title}: {e}")
        return []

def get_image_info_batch(pageids):
    if not pageids:
        return {}
    params = urllib.parse.urlencode({
        "action": "query",
        "pageids": "|".join(str(p) for p in pageids),
        "prop": "imageinfo",
        "iiprop": "url|size|mime|extmetadata|sha1",
        "format": "json"
    })
    req = urllib.request.Request(f"{API}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("query", {}).get("pages", {})
    except Exception as e:
        print(f"Error querying imageinfo: {e}")
        return {}

def main():
    print("=== Harvesting High-Quality Vietnam Motorcycle Candidates ===")
    
    # 1. Load manifest used pageids
    manifest_path = TRAINING_DIR / "dataset_v224" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    used_pageids = {r.get("pageid") for r in manifest if r.get("pageid")}
    print(f"Already used pageids in dataset: {len(used_pageids)}")

    # 2. Load existing pool
    pool_path = ACQ_DIR / "candidates_pool.json"
    if pool_path.is_file():
        with open(pool_path, "r", encoding="utf-8") as f:
            pool = json.load(f)
    else:
        pool = []
    pool_by_id = {c["pageid"]: c for c in pool}
    print(f"Existing candidates in pool: {len(pool)}")

    # 3. Harvest file lists across categories
    discovered_files = {}
    for cat in TARGET_CATEGORIES:
        print(f"Querying {cat}...")
        members = get_category_files(cat, limit=100)
        for m in members:
            pid = m["pageid"]
            if pid not in used_pageids:
                discovered_files[pid] = m["title"]
        time.sleep(1.0)
    print(f"Total candidate files discovered: {len(discovered_files)}")

    # 4. Batch query imageinfo for candidates not yet in pool
    to_query = [pid for pid in discovered_files.keys() if pid not in pool_by_id]
    print(f"Candidates needing metadata query: {len(to_query)}")

    new_pool_entries = []
    chunk_size = 25
    for i in range(0, min(100, len(to_query)), chunk_size):
        chunk = to_query[i:i+chunk_size]
        pages = get_image_info_batch(chunk)
        for pid_str, pdata in pages.items():
            pid = int(pid_str)
            info_list = pdata.get("imageinfo", [])
            if not info_list:
                continue
            info = info_list[0]
            mime = info.get("mime", "")
            if mime not in ("image/jpeg", "image/png"):
                continue
            width = info.get("width", 0)
            height = info.get("height", 0)
            if width < 1000 or height < 700:
                continue
            meta = info.get("extmetadata", {})
            lic = meta.get("LicenseShortName", {}).get("value", "")
            if not any(k in lic for k in ["CC BY", "CC-BY", "CC0", "Public domain", "PD"]):
                continue

            artist = meta.get("Artist", {}).get("value", "Unknown")
            desc = meta.get("ImageDescription", {}).get("value", "")
            cats = meta.get("Categories", {}).get("value", "")

            entry = {
                "pageid": pid,
                "title": pdata.get("title", ""),
                "url": info.get("url", ""),
                "width": width,
                "height": height,
                "mime": mime,
                "sha1": info.get("sha1", ""),
                "license": lic,
                "artist": artist,
                "description": desc,
                "categories": cats
            }
            new_pool_entries.append(entry)
            pool_by_id[pid] = entry
        time.sleep(1.2)

    print(f"New valid candidates added to pool: {len(new_pool_entries)}")
    full_pool = list(pool_by_id.values())
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(full_pool, f, ensure_ascii=False, indent=2)

    # 5. Download the top candidates that aren't downloaded yet
    existing_downloads = {int(p.stem.replace("commons_", "")) for p in DOWNLOAD_DIR.glob("commons_*.jpg")}
    candidates_to_download = [
        c for c in full_pool 
        if c["pageid"] not in used_pageids and c["pageid"] not in existing_downloads
        and not any(neg in c["title"].lower() for neg in ["signal", "sign", "map", "flag", "icon", "vector", "melbourne", "australia"])
        and not any(neg in c.get("description", "").lower() for neg in ["melbourne", "australia"])
    ]

    print(f"Candidates ready for download: {len(candidates_to_download)}")
    downloaded_count = 0
    for c in candidates_to_download[:8]:
        pid = c["pageid"]
        out_file = DOWNLOAD_DIR / f"commons_{pid}.jpg"
        print(f"Downloading [{c['license']}] {c['title']} -> {out_file.name}...")
        req = urllib.request.Request(c["url"], headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_file, "wb") as out:
                shutil.copyfileobj(resp, out)
            downloaded_count += 1
            print(f"  Successfully saved {out_file.name} ({out_file.stat().st_size} bytes)")
        except Exception as e:
            print(f"  Download error: {e}")
        time.sleep(1.5)

    print(f"=== Harvester completed. Downloaded {downloaded_count} new candidates! ===")

if __name__ == "__main__":
    main()
