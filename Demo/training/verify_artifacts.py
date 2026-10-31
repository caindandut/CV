"""Verify exported images/labels, source immutability and all grouping edges."""
import json
from pathlib import Path
from collections import Counter
from prepare_dataset import ROOT, digest, parse_labels, write_json
from train_colab import verify_dataset


def main():
    rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'dataset_v1/manifest.json').read_text(encoding='utf-8'))
    original_by_id={r['id']:r for r in rows}
    for row in rows:
        assert digest(row['image'])==row['sha256']
        assert digest(row['label'])==row['label_sha256']
    for row in manifest:
        name=row['id'].replace('/','__')
        out=ROOT/'dataset_v1'/row['new_split']
        assert digest(out/'images'/name)==row['sha256']
        assert parse_labels((out/'labels'/(Path(name).stem+'.txt')).read_text())==row['labels']
    links=json.loads((ROOT/'near_duplicate_links.json').read_text())
    for link in links:
        assert original_by_id[link['a']]['group']==original_by_id[link['b']]['group']
    verify_dataset(ROOT/'dataset_v1/data.yaml')
    result=dict(passed=True,source_images_and_labels_unchanged=len(rows),copied_images=len(manifest),
                removed_exact_duplicates=len(rows)-len(manifest),near_duplicate_edges=len(links),
                first_pass_reviews=len(rows),review_note='Contact sheet first pass; pending excluded, not certified exhaustive relabeling.',
                split_images=dict(Counter(r['new_split'] for r in manifest)),
                limitations=['Very few scene groups; predominantly one actor and background in parking frames.',
                             'No class-2 ground truth in independent test; cannot estimate class-2 test accuracy.',
                             'CUDA preflight and actual Colab Drive persistence not executed on this CPU machine.'])
    write_json(ROOT/'verification/dataset.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
