"""Read-only source inventory; reviewed, deduplicated and group-split copies."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import yaml

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1] / 'DataSet2-20260927T023713Z-1-001' / 'DataSet2'
NAMES = ['motorcycle', 'rider_with_helmet', 'rider_without_helmet']
COLORS = ['#1689ff', '#18b84b', '#fa3030']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_labels(text):
    labels = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] not in {'0', '1', '2'}:
            raise ValueError(f'Invalid label: {line}')
        box = list(map(float, parts[1:]))
        if not all(math.isfinite(v) for v in box):
            raise ValueError(f'Nonfinite label: {line}')
        x, y, w, h = box
        if w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1.00001:
            raise ValueError(f'Out of bounds: {line}')
        labels.append([int(parts[0]), *box])
    return labels


def source_key(name):
    stem = re.sub(r'\.rf\..*', '', name, flags=re.I)
    video = re.match(r'(.+?)[_-](?:mov|mp4)[_-]', stem, re.I)
    if video:
        return 'video:' + video.group(1).casefold()
    stem = re.sub(r'_(?:jpg|jpeg|png)$', '', stem, flags=re.I)
    return 'image:' + stem.casefold()


def perceptual_hash(image):
    gray = np.asarray(ImageOps.grayscale(image).resize((32, 32)), dtype=np.float32)
    low = cv2.dct(gray)[:8, :8].reshape(-1)
    bits = low > np.median(low[1:])
    return sum(int(bit) << i for i, bit in enumerate(bits))


def inventory(source):
    rows = []
    for split in ['train', 'valid', 'test']:
        for path in sorted((source / split / 'images').iterdir()):
            if not path.is_file():
                continue
            label = source / split / 'labels' / (path.stem + '.txt')
            text = label.read_text(encoding='utf-8')
            with Image.open(path) as im:
                im.load()
                width, height = im.size
                phash = perceptual_hash(im)
            rows.append(dict(id=f'{split}/{path.name}', image=str(path.resolve()), label=str(label.resolve()),
                             sha256=digest(path), label_sha256=digest(label), labels=parse_labels(text),
                             width=width, height=height, phash=f'{phash:016x}', source=source_key(path.name)))
    return rows


def group_rows(rows):
    """Union source, byte duplicates and conservative perceptually similar pairs."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    seen = {}
    links = []
    for i, row in enumerate(rows):
        for key in [('source', row['source']), ('sha256', row['sha256'])]:
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i
    for i, a in enumerate(rows):
        ah = int(a['phash'], 16)
        for j in range(i):
            b = rows[j]
            if a['source'] == b['source'] or a['sha256'] == b['sha256']:
                continue
            if abs(a['width']/a['height'] - b['width']/b['height']) > .05:
                continue
            distance = (ah ^ int(b['phash'], 16)).bit_count()
            if distance <= 6:
                union(i, j)
                links.append({'a': a['id'], 'b': b['id'], 'phash_distance': distance})
    for i, row in enumerate(rows):
        row['group'] = f'group_{find(i):04d}'
    return links


def apply_review(row, review):
    row = dict(row)
    row['review_status'] = 'pending'
    row['reason'] = 'Chưa được rà bằng mắt; không tự coi nhãn là đúng.'
    row['source_verified'] = False
    if review is None:
        return row
    if review.get('label_sha256') != row['label_sha256']:
        raise ValueError(f'Original label changed: {row["id"]}')
    status = review['status']
    if status not in {'accepted', 'corrected', 'pending'} or not review.get('reason'):
        raise ValueError(f'Invalid review: {row["id"]}')
    row.update(review_status=status, reason=review['reason'], source_verified=review.get('source_verified', False))
    if status == 'corrected' or 'labels' in review:
        new = review['labels']
        row['labels'] = parse_labels('\n'.join(' '.join(map(str, box)) for box in new))
    return row


def assign_splits(rows, seed=0, group_assignments=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row['group']].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    target = dict(zip(['train', 'val', 'test'], [len(rows)*.8, len(rows)*.1, len(rows)*.1]))
    counts = Counter()
    assignment = {}
    pinned = group_assignments or {}
    if any(split not in {'train', 'val', 'test'} for split in pinned.values()):
        raise ValueError('Invalid inherited split')
    for key in keys:
        if key in pinned:
            split = pinned[key]
            if split != 'train' and any(r.get('train_only', False) for r in groups[key]):
                raise ValueError(f'Training-only source in inherited holdout: {key}')
            if split != 'train' and not all(r['source_verified'] for r in groups[key]):
                raise ValueError(f'Unverified source in inherited holdout: {key}')
            counts[split] += len(groups[key])
            assignment.update({r['id']: split for r in groups[key]})
    for key in keys:
        if key in pinned:
            continue
        group = groups[key]
        eligible = ['train', 'val', 'test'] if all(r['source_verified'] for r in group) and not any(r.get('train_only', False) for r in group) else ['train']
        split = max(eligible, key=lambda s: (target[s]-counts[s])/max(target[s], 1))
        counts[split] += len(group)
        assignment.update({r['id']: split for r in group})
    # With very few scenes, a ratio-only split can leave training with a single background.
    # Keep at least two training scenes if this can preserve nonempty independent val/test.
    group_split = {key: assignment[groups[key][0]['id']] for key in keys}
    if len(groups)>=4 and sum(s=='train' for s in group_split.values())<2:
        candidates = [key for key in keys if key not in pinned and group_split[key]!='train'
                      and sum(s==group_split[key] for s in group_split.values())>1]
        if candidates:
            key=min(candidates,key=lambda k:(len(groups[k]),k))
            assignment.update({r['id']:'train' for r in groups[key]})
    return assignment


def render_review(rows, out, per_sheet=24):
    out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(ROOT.parent / 'assets' / 'fonts' / 'DejaVuSans.ttf'), 15)
    index = []
    for start in range(0, len(rows), per_sheet):
        canvas = Image.new('RGB', (1800, math.ceil(per_sheet/4)*260), '#171c22')
        for offset, row in enumerate(rows[start:start+per_sheet]):
            image = Image.open(row['image']).convert('RGB')
            image.thumbnail((448, 225))
            draw = ImageDraw.Draw(image)
            w, h = image.size
            for k, (cls, x, y, bw, bh) in enumerate(row['labels']):
                draw.rectangle(((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h), outline=COLORS[cls], width=2)
                draw.text(((x-bw/2)*w, max(0, (y-bh/2)*h-12)), f'{cls}:{k}', fill=COLORS[cls], font=font)
            x0, y0 = offset%4*450, offset//4*260
            canvas.paste(image, (x0, y0))
            ImageDraw.Draw(canvas).text((x0+3, y0+227), f'{start+offset:04d} {row["id"][:42]}', font=font, fill='white')
            index.append({'index': start+offset, 'id': row['id'], 'sheet': f'sheet_{start//per_sheet:03d}.jpg'})
        canvas.save(out / f'sheet_{start//per_sheet:03d}.jpg', quality=92)
    write_json(out / 'index.json', index)


def export(rows, reviews, output, inherit_manifest=None):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite frozen dataset: {output}')
    reviewed = [apply_review(r, reviews.get(r['id'])) for r in rows]
    by_hash = defaultdict(list)
    for row in reviewed:
        by_hash[row['sha256']].append(row)
    kept, duplicates = [], []
    for group in by_hash.values():
        variants = {json.dumps(r['labels']) for r in group}
        if len(variants) > 1:
            for row in group:
                row.update(review_status='pending', reason='Bản sao có nhãn sửa không thống nhất; cần duyệt.')
        candidates = sorted(group, key=lambda r: (r['review_status'] == 'pending', r['id']))
        kept.append(candidates[0])
        duplicates.extend({'id': r['id'], 'keep': candidates[0]['id'], 'sha256': r['sha256']} for r in candidates[1:])
    active = [r for r in kept if r['review_status'] != 'pending']
    pinned = {}
    if inherit_manifest is not None:
        parent = json.loads(Path(inherit_manifest).read_text(encoding='utf-8'))
        current = {r['id']: r for r in rows}
        # Reuse current union groups: new similarity links may reveal leakage in an old split.
        for row in parent:
            if row['new_split'] == 'pending':
                continue
            if row['id'] not in current:
                raise ValueError(f'Inherited image missing: {row["id"]}')
            group = current[row['id']]['group']
            if group in pinned and pinned[group] != row['new_split']:
                raise ValueError(f'Inherited split conflict after grouping: {group}')
            pinned[group] = row['new_split']
    assignments = assign_splits(active, group_assignments=pinned)
    for row in kept:
        split = assignments.get(row['id'], 'pending')
        row['new_split'] = split
        name = row['id'].replace('/', '__')
        image_out = output / split / 'images' / name
        label_out = output / split / 'labels' / (Path(name).stem + '.txt')
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        if digest(row['image']) != row['sha256'] or digest(row['label']) != row['label_sha256']:
            raise ValueError(f'Source changed during preparation: {row["id"]}')
        shutil.copy2(row['image'], image_out)
        label_out.write_text(''.join(' '.join(map(str, b))+'\n' for b in row['labels']), encoding='utf-8')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'manifest.json', kept)
    write_json(output / 'duplicates.json', duplicates)
    write_json(output / 'changes.json', [dict(id=r['id'], before=next(x['labels'] for x in rows if x['id']==r['id']),
                                            after=r['labels'], reason=r['reason'], label_sha256=r['label_sha256'])
                                           for r in kept if r['labels'] != next(x['labels'] for x in rows if x['id']==r['id'])])
    stats = {}
    for split in ['train', 'val', 'test', 'pending']:
        subset = [r for r in kept if r['new_split']==split]
        stats[split] = dict(images=len(subset), classes=dict(Counter(str(b[0]) for r in subset for b in r['labels'])),
                            groups=len({r['group'] for r in subset}))
    write_json(output / 'statistics.json', stats)
    (output / 'data.yaml').write_text(yaml.safe_dump(dict(train='train/images', val='val/images',
                                                         test='test/images', nc=3, names=NAMES)), encoding='utf-8')
    lock = {'manifest_sha256': digest(output/'manifest.json'),
            'test_ids': sorted(r['id'] for r in kept if r['new_split']=='test'), 'seed': 0,
            'note': 'Freeze before training; pending excluded. No claims of independent camera split without provenance.'}
    if inherit_manifest is not None:
        lock.update(parent_manifest_sha256=digest(inherit_manifest), inherited_group_splits=pinned,
                    note='Scene split inherited; additional reviewed images stay in their original scene split. Parent version unchanged.')
    write_json(output / 'split_lock.json', lock)
    print(json.dumps(stats, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['inventory', 'export'])
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=ROOT/'dataset_v1')
    p.add_argument('--reviews', type=Path, default=ROOT/'reviews.json')
    p.add_argument('--inherit-splits', type=Path, help='Parent manifest.json; preserve scene assignments')
    args = p.parse_args()
    if args.command == 'inventory':
        rows = inventory(args.source)
        links = group_rows(rows)
        write_json(ROOT/'inventory.json', rows)
        write_json(ROOT/'near_duplicate_links.json', links)
        render_review(rows, ROOT/'review_sheets')
        print(f'{len(rows)} images, {len({r["group"] for r in rows})} groups, {len(links)} similarity links')
    else:
        rows = json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
        reviews = json.loads(args.reviews.read_text(encoding='utf-8'))
        export(rows, reviews, args.output, inherit_manifest=args.inherit_splits)


if __name__ == '__main__':
    main()
