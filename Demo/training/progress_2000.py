"""Report clean progress from the exported manifest, separately from downloads."""

import json
from datetime import UTC, datetime

from kaggle_reviews import count_kaggle_candidates
from ncpedia_reviews import is_external
from prepare_dataset import ROOT, write_json


def source_frame_count(rows):
    """Count exported frames after conflict handling and duplicate removal."""
    return len({row["id"].split(".rf.")[0].split("/", 1)[-1] for row in rows})


def main():
    folder = ROOT / "acquisition_2000"
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    target = progress.get("target_clean_images", 2000)
    if type(target) is not int or target <= 0:
        raise ValueError("target_clean_images must be a positive integer")
    manifest = json.loads(
        (ROOT / progress["frozen_dataset"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    active = [r for r in manifest if r["new_split"] != "pending"]
    sources = folder / "sources.json"
    metadata = (
        json.loads(sources.read_text(encoding="utf-8")) if sources.exists() else []
    )
    discovery = json.loads((folder / "discovery.json").read_text(encoding="utf-8"))
    reviewed = []
    for path in folder.glob("original_reviews_batch*.json"):
        reviewed.extend(json.loads(path.read_text(encoding="utf-8")))
    progress.update(
        updated_at_utc=datetime.now(UTC).isoformat(),
        accepted_clean_images=len(active),
        unique_source_frames=source_frame_count(active),
        completion_percent=round(len(active) / target * 100, 2),
        remaining_clean_images=max(0, target - len(active)),
        target_achieved=len(active) >= target,
        metadata_candidates=len(metadata),
        discovered_candidate_files=len(discovery["files"]),
        downloaded_candidates=len(list((folder / "images").glob("*.jpg"))),
        original_accepted=sum(not is_external(r) for r in active),
        external_accepted=sum(is_external(r) for r in active),
        commons_accepted=sum(r["id"].startswith("commons/") for r in active),
        ncpedia_accepted=sum(r["id"].startswith("ncpedia/") for r in active),
        kaggle_accepted=sum(
            r["id"].startswith(("kaggle-maitam-v8/", "kaggle-huan-v2/"))
            for r in active
        ),
        downloaded_kaggle_candidates=count_kaggle_candidates(ROOT),
        zenodo_accepted=sum(r["id"].startswith("zenodo-vntraffic/") for r in active),
        downloaded_ncpedia_candidates=len(
            list((folder / "research_ncpedia").glob("vietnam_*_original_candidate.jpg"))
        ),
        next_original_review_proposal=max(
            (r.get("proposal", -1) for r in reviewed), default=-1
        )
        + 1,
    )
    progress["total_downloaded_candidates"] = (
        progress["downloaded_candidates"] + progress["downloaded_ncpedia_candidates"]
        + progress["downloaded_kaggle_candidates"]
    )
    write_json(folder / "progress.json", progress)
    print(json.dumps(progress))


if __name__ == "__main__":
    main()
