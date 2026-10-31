"""Download licensed Vietnam candidates, never automatically accept training labels.

Resume category/API caches and byte downloads. Every source still needs visual
location and object review. Category membership is evidence to inspect, not GT.
"""

import argparse
import hashlib
import html
import json
import re
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "acquisition_2000"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {
    "User-Agent": "HelmetSCConvDatasetResearch/1.0 (academic image annotation; Wikimedia Commons)"
}
START_CATEGORIES = [
    "Category:Motorcycles in Vietnam",
    "Category:Road traffic in Vietnam",
    "Category:Road transport in Hanoi",
    "Category:Road transport in Ho Chi Minh City",
]
QUERIES = [
    "Vietnam motorcycle",
    "Vietnam motorbike",
    "Vietnam scooter",
    "Hanoi traffic",
    "Saigon traffic",
    "Vietnam motorcyclist",
    "Vietnam motorcycles street",
    "Ho Chi Minh City motorcycles",
]
PAUSE = threading.Event()


class RateLimitPause(Exception):
    def __init__(self, retry_after=300):
        self.retry_after = retry_after
        super().__init__(
            f"Wikimedia rate limit: pause at least {retry_after} seconds; no further requests in this batch."
        )


def rate_limit(response):
    value = response.headers.get("Retry-After", "")
    retry_after = int(value) if value.isdigit() else 300
    PAUSE.set()
    save(
        OUT / "server_pause.json",
        {
            "http_status": 429,
            "retry_after_seconds": retry_after,
            "not_before_utc": (
                datetime.now(UTC) + timedelta(seconds=retry_after)
            ).isoformat(),
            "reason": "Server requested less disruptive access; stop this batch, retain cache, use one request at a time.",
        },
    )
    raise RateLimitPause(retry_after)


def plain(value):
    return html.unescape(re.sub("<[^>]+>", " ", value)).strip()


def safe_image_url(url):
    parsed = urlparse(url)
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"upload.wikimedia.org", "thumb.wikimedia.org"}
        and not parsed.username
    )


def admissible_metadata(info, provenance):
    meta = info.get("extmetadata", {})
    license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
    artist = plain(meta.get("Artist", {}).get("value", ""))
    # A search for Vietnam can return photographs taken abroad. Search terms
    # themselves are not location evidence; inspect titles/descriptions/categories.
    description = plain(meta.get("ImageDescription", {}).get("value", "")).casefold()
    evidence = (
        " ".join(p for p in provenance if not p.startswith("search:"))
        + " "
        + description
        + " "
        + info.get("title", "")
    ).casefold()
    return (
        info.get("mime") in {"image/jpeg", "image/png"}
        and min(info.get("width", 0), info.get("height", 0)) >= 480
        and license_name.startswith(("CC BY", "CC0", "Public domain"))
        and bool(artist)
        and any(
            place in evidence for place in ["vietnam", "hanoi", "saigon", "ho chi minh"]
        )
    )


def candidate_priority(row):
    title = row["title"].casefold()
    evidence = " ".join(row["provenance"]).casefold()
    if any(word in title for word in ["without helmet", "no helmet", "bareheaded"]):
        priority = 0
    elif any(
        word in evidence for word in ["people with motorcycles", "motorcycle taxis"]
    ):
        priority = 1
    elif any(
        word in title for word in ["motorcycle", "motorbike", "scooter", "motorcyclist"]
    ):
        priority = 2
    elif any(word in evidence for word in ["motorcycles", "motor scooters"]):
        priority = 3
    elif any(word in title for word in ["traffic", "riding"]):
        priority = 4
    else:
        priority = 5
    return priority, row["pageid"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(10):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def api(params, cache_name):
    cached = OUT / "api_cache" / cache_name
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    for attempt in range(4):
        response = httpx.get(
            API, params={"format": "json", **params}, headers=HEADERS, timeout=40
        )
        if response.status_code == 429:
            rate_limit(response)
        if response.status_code in {502, 503, 504}:
            time.sleep(min(2**attempt, 8))
            continue
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(payload["error"])
        save(cached, payload)
        return payload
    raise RuntimeError(f"API temporarily unavailable: {cache_name}")


def discover(max_categories):
    state_path = OUT / "discovery.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {
            "files": {},
            "visited": [],
            "queue": [[c, 0] for c in START_CATEGORIES],
            "searches": [],
        }
    )
    queue = deque(state["queue"])
    visited = set(state["visited"])
    while queue and len(visited) < max_categories:
        category, depth = queue.popleft()
        if category in visited:
            continue
        continuation = {}
        while True:
            cache = (
                hashlib.sha256(
                    (category + json.dumps(continuation, sort_keys=True)).encode()
                ).hexdigest()
                + ".json"
            )
            try:
                result = api(
                    dict(
                        action="query",
                        list="categorymembers",
                        cmtitle=category,
                        cmlimit=500,
                        **continuation,
                    ),
                    cache,
                )
            except (httpx.HTTPError, RuntimeError) as exc:
                state.setdefault("discovery_errors", []).append(
                    {"category": category, "error": str(exc)}
                )
                print(f"Category unavailable: {category}", flush=True)
                break
            for item in result.get("query", {}).get("categorymembers", []):
                if item["ns"] == 6:
                    state["files"].setdefault(item["title"], []).append(category)
                elif (
                    item["ns"] == 14
                    and depth < 3
                    and any(
                        s in item["title"].casefold()
                        for s in ["vietnam", "hanoi", "ho chi minh", "saigon"]
                    )
                ):
                    # Stay inside the Vietnam branch; avoid linked vehicle topics worldwide.
                    queue.append([item["title"], depth + 1])
            continuation = result.get("continue", {})
            if not continuation:
                break
        visited.add(category)
        state.update(visited=sorted(visited), queue=list(queue))
        save(state_path, state)
        print(f"categories={len(visited)} candidates={len(state['files'])}", flush=True)
    for index, query in enumerate(QUERIES):
        if query in state["searches"]:
            continue
        try:
            result = api(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srnamespace": 6,
                    "srlimit": 100,
                },
                f"search_{index}.json",
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            state.setdefault("search_errors", []).append(
                {"query": query, "error": str(exc)}
            )
            save(state_path, state)
            continue
        for item in result.get("query", {}).get("search", []):
            state["files"].setdefault(item["title"], []).append("search:" + query)
        state["searches"].append(query)
        save(state_path, state)
    return state["files"]


def metadata(files):
    path = OUT / "sources.json"
    sources = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    known = {r["title"] for r in sources}
    missing = sorted(set(files) - known)
    for start in range(0, len(missing), 10):
        titles = missing[start : start + 10]
        key = "info_" + hashlib.sha256("|".join(titles).encode()).hexdigest() + ".json"
        try:
            payload = api(
                {
                    "action": "query",
                    "titles": "|".join(titles),
                    "prop": "imageinfo",
                    "iiprop": "url|extmetadata|sha1|size|timestamp|mime",
                    "iiurlwidth": 1280,
                },
                key,
            )
        except (httpx.HTTPError, RuntimeError) as exc:
            failures_path = OUT / "metadata_failures.json"
            failures = (
                json.loads(failures_path.read_text(encoding="utf-8"))
                if failures_path.exists()
                else []
            )
            failures.append({"titles": titles, "error": str(exc)})
            save(failures_path, failures)
            print(f"Metadata unavailable: {start}..{start + len(titles)}", flush=True)
            continue
        for page in payload.get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo", [None])[0]
            if info:
                sources.append(
                    dict(
                        title=page["title"],
                        pageid=page["pageid"],
                        **info,
                        provenance=files[page["title"]],
                        review_status="unreviewed",
                    )
                )
        save(path, sources)
        print(f"metadata={len(sources)}", flush=True)
    return sources


def download(row):
    if PAUSE.is_set():
        raise RateLimitPause()
    url = row.get("thumburl", row["url"])
    if not safe_image_url(url):
        raise ValueError("Download host not allowed")
    target = OUT / "images" / f"commons_{row['pageid']}.jpg"
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        tmp = target.with_suffix(".part")
        try:
            with httpx.stream(
                "GET", url, headers=HEADERS, timeout=40, follow_redirects=False
            ) as response:
                if response.status_code == 429:
                    rate_limit(response)
                response.raise_for_status()
                total = 0
                with tmp.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        total += len(chunk)
                        if total > 20 * 1024 * 1024:
                            raise ValueError("Image larger than 20 MB")
                        stream.write(chunk)
            with Image.open(tmp) as image:
                image.load()
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Unexpected image format")
            tmp.replace(target)
        finally:
            tmp.unlink(missing_ok=True)
    with Image.open(target) as image:
        image.load()
        row["download_size"] = list(image.size)
    row.update(
        local_image=target.relative_to(OUT).as_posix(),
        download_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        downloaded_at_utc=datetime.now(UTC).isoformat(),
    )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-categories", type=int, default=100)
    parser.add_argument("--download-limit", type=int, default=2000)
    parser.add_argument(
        "--render-cached-only",
        action="store_true",
        help="No network requests; render only verified cached bytes",
    )
    parser.add_argument(
        "--download-cached-only",
        action="store_true",
        help="Download an independent partial review batch while discovery continues",
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    pause_path = OUT / "server_pause.json"
    if pause_path.exists() and not args.render_cached_only:
        pause = json.loads(pause_path.read_text(encoding="utf-8"))
        if datetime.now(UTC) < datetime.fromisoformat(pause["not_before_utc"]):
            raise RateLimitPause(pause["retry_after_seconds"])
    sources = (
        json.loads((OUT / "sources.json").read_text(encoding="utf-8"))
        if args.download_cached_only or args.render_cached_only
        else metadata(discover(args.max_categories))
    )
    metadata_path = OUT / (
        "downloaded_partial.json" if args.download_cached_only else "sources.json"
    )
    sheet_prefix = "partial_sheet" if args.download_cached_only else "sheet"
    old_ids = {
        r["pageid"]
        for r in json.loads(
            (ROOT / "external_commons/sources.json").read_text(encoding="utf-8")
        )
        if "download_sha256" in r
    }
    selected = sorted(
        [
            r
            for r in sources
            if admissible_metadata(r, r["provenance"]) and r["pageid"] not in old_ids
        ],
        key=candidate_priority,
    )[: args.download_limit]
    errors = []
    if not args.render_cached_only:
        for count, row in enumerate(selected, 1):
            try:
                downloaded = (OUT / "images" / f"commons_{row['pageid']}.jpg").exists()
                download(row)
                row.pop("download_error", None)
                if not downloaded:
                    time.sleep(1)
            except RateLimitPause as exc:
                row["download_error"] = str(exc)
                errors.append({"pageid": row["pageid"], "error": str(exc)})
                print(str(exc), flush=True)
                break
            except (
                httpx.HTTPError,
                OSError,
                ValueError,
                Image.DecompressionBombError,
            ) as exc:
                row["download_error"] = str(exc)
                errors.append({"pageid": row["pageid"], "error": str(exc)})
            if count % 20 == 0:
                save(metadata_path, sources)
                print(
                    f"downloads attempted={count}/{len(selected)} failures={len(errors)}",
                    flush=True,
                )
    save(metadata_path, sources)
    save(
        OUT
        / (
            "partial_download_failures.json"
            if args.download_cached_only
            else "download_failures.json"
        ),
        errors,
    )
    valid = sorted(
        [r for r in selected if "download_sha256" in r], key=lambda r: r["pageid"]
    )
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 18)
    for start in range(0, len(valid), 12):
        sheet = Image.new("RGB", (1920, 1200), "#172029")
        for offset, row in enumerate(valid[start : start + 12]):
            im = Image.open(OUT / row["local_image"]).convert("RGB")
            im.thumbnail((630, 250))
            x, y = (offset % 3) * 640, (offset // 3) * 300
            sheet.paste(im, (x, y))
            ImageDraw.Draw(sheet).text(
                (x + 3, y + 252),
                f"{start + offset:04d} | {row['pageid']} | {row['title'][5:32]}",
                font=font,
                fill="white",
            )
        sheet.save(OUT / f"{sheet_prefix}_{start // 12:03d}.jpg", quality=95)
    save(
        OUT
        / (
            "review_index_partial.json"
            if args.download_cached_only
            else "review_index.json"
        ),
        valid,
    )
    print(
        json.dumps(
            {
                "downloaded": len(valid),
                "candidates": len(sources),
                "accepted": 0,
                "note": "Downloaded candidates require visual review, not clean training images.",
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except RateLimitPause as exc:
        print(str(exc), flush=True)
