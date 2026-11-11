"""Discover and fetch candidate metadata from Wikimedia Commons for Vietnam motorcycle traffic.
Ensures open licenses, Vietnam location context, minimum resolution, and excludes existing manifest items.
"""
import urllib.request
import urllib.parse
import json
import hashlib
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "acquisition_5000"
OUT_DIR.mkdir(exist_ok=True)

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "HelmetResearchCodex5000/1.0 (academic research; vietnam traffic detection)"}

SEARCH_TERMS = [
    "motorcycle vietnam", "motorbike vietnam", "traffic hanoi", "traffic saigon",
    "scooter vietnam", "xe may vietnam", "traffic ho chi minh city",
    "hanoi motorcycle", "saigon motorcycle", "danang motorcycle",
    "vietnam traffic jam", "vietnam street motorbike", "vietnam moped",
    "vietnam road traffic", "da nang traffic", "hue traffic", "can tho traffic",
    "hai phong traffic", "Category:Motorcycles in Vietnam",
    "Category:Motorcycle taxis in Vietnam"
]

def query_search(term):
    pages = {}
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srnamespace": "6",
        "srlimit": "200",
        "format": "json"
    })
    req = urllib.request.Request(f"{API}?{params}", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for h in data.get("query", {}).get("search", []):
                pages[h["pageid"]] = h["title"]
    except Exception as e:
        print(f"Search error for {term}: {e}")
    return pages

def main():
    # 1. Load existing manifest to avoid duplicates
    manifest_path = ROOT / "dataset_v203" / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    existing_pageids = set()
    for r in manifest:
        if "pageid" in r:
            existing_pageids.add(int(r["pageid"]))
        if "original_pageid" in r:
            existing_pageids.add(int(r["original_pageid"]))
        rid = r.get("id", "")
        if "commons_" in rid:
            try:
                pid = int(rid.split("commons_")[-1].split(".")[0].split("_")[0])
                existing_pageids.add(pid)
            except Exception:
                pass
    print(f"Existing pageids in manifest: {len(existing_pageids)}")

    # 2. Gather candidates
    all_pages = {}
    for term in SEARCH_TERMS:
        hits = query_search(term)
        all_pages.update(hits)
        time.sleep(0.2)
    print(f"Total discovered pages: {len(all_pages)}")

    candidate_ids = [pid for pid in all_pages.keys() if pid not in existing_pageids]
    print(f"Candidate pageids not in existing manifest: {len(candidate_ids)}")

    # 3. Batch query imageinfo
    results = []
    chunk_size = 40
    for i in range(0, len(candidate_ids), chunk_size):
        chunk = candidate_ids[i:i+chunk_size]
        pageids_str = "|".join(str(p) for p in chunk)
        params = urllib.parse.urlencode({
            "action": "query",
            "pageids": pageids_str,
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|sha1",
            "format": "json"
        })
        req = urllib.request.Request(f"{API}?{params}", headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid_str, pdata in pages.items():
                    info_list = pdata.get("imageinfo", [])
                    if not info_list:
                        continue
                    info = info_list[0]
                    mime = info.get("mime", "")
                    if not (mime.startswith("image/jpeg") or mime.startswith("image/png") or mime.startswith("image/webp")):
                        continue
                    w = info.get("width", 0)
                    h = info.get("height", 0)
                    if w < 400 or h < 300:
                        continue
                    
                    meta = info.get("extmetadata", {})
                    license_short = meta.get("LicenseShortName", {}).get("value", "")
                    artist = meta.get("Artist", {}).get("value", "")
                    desc = meta.get("ImageDescription", {}).get("value", "")
                    categories = meta.get("Categories", {}).get("value", "")
                    
                    # Verify open license
                    lic_upper = license_short.upper()
                    is_open = any(k in lic_upper for k in ["CC BY", "CC-BY", "CC0", "PUBLIC DOMAIN", "PD", "GPL"])
                    if not is_open:
                        continue
                    
                    # Verify Vietnam context evidence
                    text_corpus = f"{pdata.get('title', '')} {desc} {categories}".lower()
                    vn_markers = ["vietnam", "viet nam", "việt nam", "hanoi", "ha noi", "hà nội", 
                                  "saigon", "sài gòn", "ho chi minh", "hồ chí minh", "da nang", 
                                  "đà nẵng", "hue", "huế", "can tho", "cần thơ", "hai phong", 
                                  "hải phòng", "nha trang", "dalat", "đà lạt"]
                    has_vn_context = any(m in text_corpus for m in vn_markers)
                    if not has_vn_context:
                        continue
                    
                    results.append({
                        "pageid": int(pid_str),
                        "title": pdata.get("title", ""),
                        "url": info.get("url", ""),
                        "descriptionurl": info.get("descriptionurl", ""),
                        "width": w,
                        "height": h,
                        "mime": mime,
                        "sha1": info.get("sha1", ""),
                        "license": license_short,
                        "artist": artist,
                        "description": desc,
                        "categories": categories
                    })
        except Exception as e:
            print(f"Error fetching chunk {i}: {e}")
        time.sleep(0.2)
        if (i // chunk_size) % 5 == 0:
            print(f"Processed {i}/{len(candidate_ids)} candidates, {len(results)} valid candidates so far...")

    out_file = OUT_DIR / "commons_candidates.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} valid candidates to {out_file}")

if __name__ == "__main__":
    main()
