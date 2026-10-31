"""Verify immutable originals, label geometry, split inheritance and source leakage."""
import json
from itertools import combinations

from prepare_dataset import ROOT, digest, parse_labels, write_json
from train_colab import verify_dataset


def main():
    data = ROOT/'dataset_v3'
    verify_dataset(data/'data.yaml')
    originals = json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    for row in originals:
        assert digest(row['image']) == row['sha256'], row['id']
        assert digest(row['label']) == row['label_sha256'], row['id']
    rows = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    active = [r for r in rows if r['new_split'] != 'pending']
    for row in active:
        assert parse_labels('\n'.join(' '.join(map(str, b)) for b in row['labels'])) == row['labels']
    for a, b in combinations(active, 2):
        if a['new_split'] == b['new_split']:
            continue
        assert a['source'] != b['source']
        assert a['sha256'] != b['sha256']
        if abs(a['width']/a['height'] - b['width']/b['height']) <= .05:
            assert (int(a['phash'],16)^int(b['phash'],16)).bit_count() > 6
    parent = json.loads((ROOT/'dataset_v2/manifest.json').read_text(encoding='utf-8'))
    lookup = {r['id']: r for r in active}
    for old in parent:
        if old['new_split'] != 'pending':
            assert lookup[old['id']]['new_split'] == old['new_split']
            assert lookup[old['id']]['labels'] == old['labels']
    assert {r['id'] for r in active if r['new_split']=='test'} == {r['id'] for r in parent if r['new_split']=='test'}
    external = json.loads((data/'attribution.json').read_text(encoding='utf-8'))
    assert len(external)==12 and all(r['country']=='VN' and r['location_evidence'] for r in external)
    result = dict(passed=True, original_images_and_labels_unchanged=len(originals),
        active_images=len(active), external_vietnam_images=len(external),
        exact_duplicate_and_group_leakage=False, cross_split_phash_distance_at_most_6=False,
        test_unchanged_from_v2=True, label_geometry_valid=True,
        limitation='Heuristic near-duplicate detection cannot prove all unknown provenance is independent.')
    write_json(ROOT/'verification/dataset_v3.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
