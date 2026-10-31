"""Cache Commons photographs with their original license/provenance, without labels.

The category is a discovery hint only; every admitted annotation needs visual review.
"""
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'external_vietnam_candidates'
API = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'HelmetSCConvDatasetResearch/1.0 (academic image annotation; Wikimedia Commons)'}
CATEGORIES = [
    'Motorcycles in Vietnam',
]


def get(url, **kwargs):
    for attempt in range(4):
        response = httpx.get(url, headers=HEADERS, timeout=60, follow_redirects=True, **kwargs)
        if response.status_code not in {429, 502, 503, 504}:
            response.raise_for_status()
            return response
        time.sleep(2 ** attempt)
    response.raise_for_status()


def main():
    OUT.mkdir(exist_ok=True)
    metadata = OUT / 'sources.json'
    if metadata.exists() and metadata.read_text('utf-8') != '[]':
        rows = json.loads(metadata.read_text('utf-8'))
    else:
        titles = set()
        for category in CATEGORIES:
            result = get(API, params=dict(action='query', list='categorymembers',
                         cmtitle='Category:' + category, cmtype='file', cmlimit=200, format='json')).json()
            titles.update(r['title'] for r in result['query']['categorymembers'])
        rows = []
        ordered = sorted(titles)
        for start in range(0, len(ordered), 40):
            result = get(API, params=dict(action='query', titles='|'.join(ordered[start:start+40]),
                         prop='imageinfo', iiprop='url|extmetadata|sha1|size|timestamp|mime',
                         iiurlwidth=1280, format='json')).json()
            for page in result['query']['pages'].values():
                info = page.get('imageinfo', [None])[0]
                if info and info.get('mime', '').startswith('image/'):
                    rows.append(dict(title=page['title'], pageid=page['pageid'], **info))
        rows.sort(key=lambda row: row['title'])
        for i, row in enumerate(rows):
            row['index'] = i
            row['local_image'] = f'images/commons_{row["pageid"]}.jpg'
        metadata.write_text(json.dumps(rows, ensure_ascii=False, indent=2), 'utf-8')

    def download(row):
        target = OUT / row['local_image']
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            data = get(row.get('thumburl', row['url'])).content
            target.write_bytes(data)
        with Image.open(target) as im:
            im.load()
        row['download_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        row['downloaded_at_utc'] = '2026-10-13'
        return row['index']

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(download, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append(dict(index=futures[future]['index'], error=str(exc)))
    metadata.write_text(json.dumps(rows, ensure_ascii=False, indent=2), 'utf-8')
    (OUT/'download_failures.json').write_text(json.dumps(failures, indent=2), 'utf-8')
    font = ImageFont.truetype(str(ROOT.parent/'assets/fonts/DejaVuSans.ttf'), 16)
    valid = [r for r in rows if 'download_sha256' in r]
    for start in range(0, len(valid), 20):
        canvas = Image.new('RGB', (1600, 1250), '#172029')
        for offset, row in enumerate(valid[start:start+20]):
            im = Image.open(OUT/row['local_image']).convert('RGB')
            im.thumbnail((396, 210))
            x, y = (offset % 4)*400, (offset//4)*250
            canvas.paste(im, (x, y))
            ImageDraw.Draw(canvas).text((x+3, y+214), f'{row["index"]:03d} {row["title"][5:40]}', font=font, fill='white')
        canvas.save(OUT/f'sheet_{start//20:02d}.jpg', quality=93)
    print(json.dumps(dict(downloaded=len(valid), failed=len(failures), total=len(rows))))


if __name__ == '__main__':
    main()
