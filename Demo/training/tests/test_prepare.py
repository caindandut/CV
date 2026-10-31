import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_dataset import assign_splits, apply_review, parse_labels, source_key, export, digest


def test_invalid_label_is_rejected():
    with pytest.raises(ValueError):
        parse_labels('3 .5 .5 .2 .2')
    with pytest.raises(ValueError):
        parse_labels('1 .99 .5 .2 .2')


def test_video_and_augmentation_variants_share_source():
    assert source_key('IMG_5030-2_MOV-4_jpg.rf.abc.jpg') == source_key('IMG_5030-2_MOV-99_jpg.rf.def.jpg')
    assert source_key('image_75_jpg.rf.abc.jpg') == source_key('image_75_jpg.rf.def.jpg')


def test_review_guard_and_pending_image():
    row = {'id': 'train/a.jpg', 'label_sha256': 'original', 'labels': [[0, .5, .5, .2, .2]]}
    with pytest.raises(ValueError):
        apply_review(row, {'status': 'corrected', 'label_sha256': 'wrong', 'labels': [], 'reason': 'test'})
    assert apply_review(row, None)['review_status'] == 'pending'
    assert apply_review(row, {'status': 'pending', 'label_sha256': 'original', 'reason': 'occluded'})['review_status'] == 'pending'


def test_group_integrity_and_unknown_train_only():
    rows = [{'id': str(i), 'group': f'g{i // 3}', 'source_verified': i < 90, 'review_status': 'accepted',
             'labels': [[i % 3, .5, .5, .1, .1]]} for i in range(99)]
    a = assign_splits(rows, seed=0)
    assert a == assign_splits(list(reversed(rows)), seed=0)
    for group in {r['group'] for r in rows}:
        assert len({a[r['id']] for r in rows if r['group'] == group}) == 1
    assert all(a[r['id']] == 'train' for r in rows if not r['source_verified'])
    assert {'train', 'val', 'test'} == set(a.values())


def test_new_review_keeps_existing_scene_splits_even_when_counts_change():
    rows = [dict(id=str(i), group='old_test', source_verified=True) for i in range(30)]
    rows += [dict(id='train', group='old_train', source_verified=True)]
    rows += [dict(id='new', group='new_source', source_verified=False)]
    result = assign_splits(rows, group_assignments={'old_test': 'test', 'old_train': 'train'})
    assert all(result[str(i)] == 'test' for i in range(30))
    assert result['train'] == result['new'] == 'train'


def test_unverified_source_cannot_be_pinned_to_holdout():
    with pytest.raises(ValueError, match='Unverified'):
        assign_splits([dict(id='a', group='unknown', source_verified=False)],
                      group_assignments={'unknown': 'test'})


def test_exact_duplicates_with_conflicting_labels_are_quarantined(tmp_path):
    import json
    rows=[]
    reviews={}
    for i,cls in enumerate([1,2]):
        image=tmp_path/f'{i}.jpg'
        label=tmp_path/f'{i}.txt'
        image.write_bytes(b'byte-identical-image')
        label.write_text(f'{cls} .5 .5 .2 .2\n')
        row=dict(id=f'train/{i}.jpg',image=str(image),label=str(label),sha256=digest(image),
                 label_sha256=digest(label),labels=parse_labels(label.read_text()),group='same',source_verified=True)
        rows.append(row)
        reviews[row['id']]=dict(status='accepted',label_sha256=row['label_sha256'],reason='test',source_verified=True)
    target=tmp_path/'export'
    export(rows,reviews,target)
    result=json.loads((target/'manifest.json').read_text())
    assert len(result)==1 and result[0]['new_split']=='pending'
    assert len(json.loads((target/'duplicates.json').read_text()))==1
    with pytest.raises(FileExistsError):
        export(rows,reviews,target)
