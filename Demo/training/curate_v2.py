"""Export the explicitly reviewed VN subset; never promote unreviewed proposals."""
from __future__ import annotations

import argparse
import copy
import html
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from prepare_dataset import ROOT, apply_review, digest, export, group_rows, parse_labels, perceptual_hash, write_json


def normalize_boxes(boxes, size):
    width, height = size
    result = []
    for cls, x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'Invalid manual box: {(cls, x1, y1, x2, y2)} / {size}')
        result.append([cls, round((x1+x2)/2/width, 8), round((y1+y2)/2/height, 8),
                       round((x2-x1)/width, 8), round((y2-y1)/height, 8)])
    return parse_labels('\n'.join(' '.join(map(str, b)) for b in result))


def validate_external(record):
    if (record.get('country') != 'VN' or not record.get('location_evidence')
            or record.get('visual_reviewed') is not True or 'boxes' not in record):
        raise ValueError('External image needs explicit Vietnam provenance and visual review')


def plain(value):
    return html.unescape(re.sub('<[^>]*>', '', value)).strip()


def main(preview_only=False):
    output = ROOT/'dataset_v2'
    if output.exists():
        raise FileExistsError('dataset_v2 is frozen; use a new version for subsequent changes')
    decisions = json.loads((ROOT/'annotations_v2.json').read_text(encoding='utf-8'))
    rows = json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    old_reviews = json.loads((ROOT/'reviews.json').read_text(encoding='utf-8'))
    reviews = {}
    for row in rows:
        reviews[row['id']] = dict(status='pending', label_sha256=row['label_sha256'],
            source_verified=True, reason='Chưa hoàn tất rà chi tiết lượt 2; giữ nguyên cả ảnh ngoài tập train/val/test.')
    for entry in decisions['original']:
        row = rows[entry['index']]
        if entry['status'] == 'pending':
            reviews[row['id']]['reason'] = entry['reason']
            continue
        if entry.get('visual_reviewed') is not True:
            raise ValueError('Manual review is required for every admitted original')
        labels = copy.deepcopy(old_reviews.get(row['id'], {}).get('labels', row['labels']))
        if 'boxes' in entry:
            labels = normalize_boxes(entry['boxes'], entry.get('canvas', [row['width'], row['height']]))
        else:
            labels = [b for i, b in enumerate(labels) if i not in entry.get('drop', [])]
            # Replacements refer to original review indices, so do them before drops in decisions.
            if entry.get('replace') and entry.get('drop'):
                raise ValueError('Use full replacement boxes for mixed drop/replace decisions')
            for index, box in entry.get('replace', {}).items():
                labels[int(index)] = normalize_boxes([box], entry.get('canvas', [1280, 720]))[0]
            for index, cls in entry.get('reclass', {}).items():
                labels[int(index)][0] = cls
            labels += normalize_boxes(entry.get('append', []), entry.get('canvas', [1280, 720]))
        reviews[row['id']] = dict(status='corrected' if labels != row['labels'] else 'accepted',
            label_sha256=row['label_sha256'], labels=labels, source_verified=True, reason=entry['reason'])
        row['review_evidence'] = entry.get('evidence', 'review_round3: full frame and annotation overlay')
        row['review_method'] = 'visual_manual_review'

    commons = ROOT/'external_commons'
    sources = {r['index']: r for r in json.loads((commons/'sources.json').read_text(encoding='utf-8'))}
    label_root = commons/'unlabeled_baseline'
    label_root.mkdir(exist_ok=True)
    attribution = []
    for entry in decisions['external']:
        validate_external(entry)
        source = sources[entry['index']]
        meta = source['extmetadata']
        license_name = plain(meta.get('LicenseShortName', {}).get('value', ''))
        if not license_name.startswith(('CC BY', 'CC0', 'Public domain')):
            raise ValueError(f'Unreviewed license: {license_name}')
        im_path = commons/source['local_image']
        if digest(im_path) != source['download_sha256']:
            raise ValueError('Downloaded image changed')
        label = label_root/f"commons_{source['pageid']}.txt"
        if not label.exists():
            label.write_text('', encoding='utf-8')
        if label.read_bytes():
            raise ValueError('Unlabeled baseline changed')
        with Image.open(im_path) as im:
            width, height = im.size
            phash = f'{perceptual_hash(im):016x}'
        artist = plain(meta.get('Artist', {}).get('value', ''))
        if not artist:
            raise ValueError('Attribution missing')
        row = dict(id=f"commons/commons_{source['pageid']}.jpg", image=str(im_path.resolve()),
            label=str(label.resolve()), sha256=digest(im_path), label_sha256=digest(label), labels=[],
            width=width, height=height, phash=phash, source='commons-author:'+entry.get('source_group', artist),
            country='VN', location_evidence=entry['location_evidence'], review_method='visual_manual_review',
            review_evidence=f"external_commons/full_review/{entry['index']:03d}.jpg",
            license=license_name, source_url=source['descriptionurl'])
        rows.append(row)
        reviews[row['id']] = dict(status='corrected', label_sha256=row['label_sha256'], source_verified=True,
            labels=normalize_boxes(entry['boxes'], entry['canvas']), reason=entry['reason'])
        attribution.append(dict(id=row['id'], title=source['title'], artist=artist, license=license_name,
            license_url=meta.get('LicenseUrl', {}).get('value', ''), page_url=source['descriptionurl'],
            download_url=source.get('thumburl', source['url']), original_url=source['url'],
            sha256=row['sha256'], country='VN', location_evidence=entry['location_evidence'],
            modifications='Bản thumbnail do Commons cung cấp; ảnh giữ nguyên byte. Nhãn YOLO mới được gán và rà bằng mắt.',
            annotation_license='CC BY-SA 4.0'))
    links = group_rows(rows)
    write_json(ROOT/'inventory_v2.json', rows)
    write_json(ROOT/'reviews_v2.json', reviews)
    write_json(ROOT/'near_duplicate_links_v2.json', links)
    if preview_only:
        preview = ROOT/'review_round3/preview'
        preview.mkdir(exist_ok=True)
        reviewed = [apply_review(r, reviews[r['id']]) for r in rows]
        for row in reviewed:
            row['new_split'] = 'pending' if row['review_status']=='pending' else 'review'
        write_json(preview/'manifest.json', reviewed)
        make_gallery(preview)
        return
    # Pin new independent sources before training; old scene split is inherited unchanged.
    parent = json.loads((ROOT/'dataset_v1/manifest.json').read_text(encoding='utf-8'))
    for entry in decisions['external']:
        if entry.get('split'):
            source = sources[entry['index']]
            parent.append(dict(id=f"commons/commons_{source['pageid']}.jpg", new_split=entry['split']))
    inherited = ROOT/'review_round3/inherited_splits_v2.json'
    write_json(inherited, parent)
    export(rows, reviews, output, inherit_manifest=inherited)
    write_json(output/'attribution.json', attribution)
    write_json(output/'external_candidates_excluded.json', [dict(index=i, title=r['title'],
        reason='Không nằm trong danh sách VN đã xác minh và gán đủ nhãn; không đưa vào ZIP train.')
        for i, r in sources.items() if i not in {e['index'] for e in decisions['external']}])
    (output/'ATTRIBUTION.md').write_text('# Nguồn ảnh bổ sung tại Việt Nam\n\n'
        'Ảnh giữ giấy phép riêng từng tác giả; nhãn bổ sung được chia sẻ theo CC BY-SA 4.0. '
        'Không gán giấy phép Commons cho dataset gốc do người dùng cung cấp.\n\n'+ '\n\n'.join(
        f"- [{a['title']}]({a['page_url']}) — {a['artist']} — [{a['license']}]({a['license_url'] or a['page_url']}). {a['modifications']}"
        for a in attribution), encoding='utf-8')
    make_gallery(output)


def make_gallery(output, version='v2', inventory_path=None):
    rows = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    before = {r['id']: r['labels'] for r in json.loads((inventory_path or ROOT/'inventory_v2.json').read_text(encoding='utf-8'))}
    assets = ROOT/f'review_{version}_assets'
    assets.mkdir(exist_ok=True)
    font = ImageFont.truetype(str(ROOT.parent/'assets/fonts/DejaVuSans.ttf'), 18)
    cards = []
    for i, row in enumerate(r for r in rows if r['new_split'] != 'pending'):
        for side, boxes in [('before', before[row['id']]), ('after', row['labels'])]:
            im = Image.open(row['image']).convert('RGB')
            im.thumbnail((1280, 960))
            width, height = im.size
            draw = ImageDraw.Draw(im)
            for k, (cls, x, y, w, h) in enumerate(boxes):
                color = ['#1689ff', '#18b84b', '#fa3030'][cls]
                draw.rectangle(((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height), outline=color, width=2)
                draw.text(((x-w/2)*width, max(0, (y-h/2)*height-20)), f'{cls}:{k}', fill=color, font=font)
            im.save(assets/f'{i:03d}-{side}.jpg', quality=95)
        cards.append(f'<article><h2>{i:03d} · {html.escape(row["id"])} · {row["new_split"]}</h2>'
            f'<p>{html.escape(row["reason"])}</p><div><img loading="lazy" src="review_v2_assets/{i:03d}-before.jpg" alt="Trước sửa">'
            f'<img loading="lazy" src="review_v2_assets/{i:03d}-after.jpg" alt="Sau sửa"></div></article>')
    (ROOT/f'review_{version}.html').write_text(('<!doctype html><html lang="vi"><meta charset="utf-8">'
        '<title>Rà nhãn dataset v2</title><style>body{font:16px system-ui;margin:24px}h2{font-size:17px;overflow-wrap:anywhere}'
        'article{margin:30px 0}article div{display:flex;gap:12px}img{width:49%;object-fit:contain}@media(max-width:800px){article div{display:block}img{width:100%}}</style>'
        '<h1>Dataset v2 — đối chiếu trước / sau</h1><p>0: xe máy (xanh dương); 1: người đội mũ (xanh lá); '
        '2: người không đội mũ (đỏ). Ảnh Commons phía trước chưa có nhãn, không phải ảnh nền âm tính.</p>'
        + ''.join(cards)+'</html>').replace('review_v2_assets', f'review_{version}_assets').replace('Dataset v2', f'Dataset {version}').replace('dataset v2', f'dataset {version}'), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preview-only', action='store_true')
    main(parser.parse_args().preview_only)
