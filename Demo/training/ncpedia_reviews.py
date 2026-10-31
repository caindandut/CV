"""Import explicit NCpedia reviews without dropping source or NC-SA conditions."""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from curate_v2 import normalize_boxes
from PIL import Image
from prepare_dataset import digest, parse_labels, perceptual_hash
from training_crops import prepare_review_image

LICENSE = "CC BY-NC-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def is_external(row):
    return row["id"].startswith(("commons/", "ncpedia/", "kaggle-maitam-v8/", "kaggle-huan-v2/", "zenodo-vntraffic/"))


def _local(root, value):
    acquisition = Path(root) / "acquisition_2000"
    path = (acquisition / value).resolve()
    if not path.is_relative_to((acquisition / "research_ncpedia").resolve()):
        raise ValueError("NCpedia evidence path outside its research folder")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_evidence(entry, root):
    match = re.fullmatch(r"ncpedia-vietnam-(\d+)", entry["source_id"])
    if not match:
        raise ValueError("Invalid NCpedia source ID")
    number = int(match[1])
    asset_number = match[1]
    url = urlparse(entry["page_url"])
    if (
        url.scheme != "https"
        or url.netloc != "www.ncpedia.org"
        or not re.fullmatch(r"/media/[a-z0-9-]+", url.path)
        or url.query
        or url.fragment
        or entry["image_url"]
        != f"https://www.ncpedia.org/sites/default/files/vietnam_{asset_number}.jpg"
    ):
        raise ValueError("Unverified NCpedia source URL")
    if entry["license"] != LICENSE or entry["license_url"] != LICENSE_URL:
        raise ValueError("Unreviewed or incompatible NCpedia license")
    if (
        entry["country"] != "VN"
        or not entry["location_evidence"].strip()
        or entry["source_group"] != "ncpedia-vietnam-collection"
        or entry["credit_provider"] != "NCpedia"
        or entry["photographer_named"] is not False
    ):
        raise ValueError("Unverified location, collection, or supplied credit")
    source = _local(root, entry["source_image"])
    page = _local(root, entry["provider_page_snapshot"])
    if (
        digest(source) != entry["source_sha256"]
        or digest(page) != entry["provider_page_sha256"]
    ):
        raise ValueError("Source image or page snapshot changed")
    html = page.read_text(encoding="utf-8")
    if (
        LICENSE_URL not in html
        or f"vietnam_{asset_number}.jpg" not in html
        or not entry["supplied_copyright_field"]
        or entry["supplied_copyright_field"] not in html
    ):
        raise ValueError(
            "Snapshot does not support image, license, and supplied notice"
        )
    return source, number


def load_ncpedia_reviews(root):
    root = Path(root)
    ledger = root / "acquisition_2000/ncpedia_reviews.json"
    if not ledger.exists():
        return [], {}, [], []
    entries = json.loads(ledger.read_text(encoding="utf-8"))
    rows, reviews, attrs, audit = [], {}, [], []
    seen = set()
    for entry in entries:
        source, number = _source_evidence(entry, root)
        if number in seen:
            raise ValueError("Duplicate NCpedia source photograph")
        seen.add(number)
        if (
            entry["status"] != "accepted"
            or entry["visual_reviewed"] is not True
            or entry["source_verified"] is not True
            or not entry["reason"].strip()
            or not entry["title"].strip()
        ):
            raise ValueError("Only explicit, complete manual reviews may be exported")
        labels = normalize_boxes(entry["boxes"], entry["canvas"])
        parse_labels("\n".join(" ".join(map(str, box)) for box in labels))
        image, lineage = prepare_review_image(
            source,
            entry,
            root / "acquisition_2000/curated_ncpedia_crops",
            number,
            provider="ncpedia",
        )
        baseline = (
            root / "acquisition_2000/unlabeled_baseline" / f"ncpedia_{number}.txt"
        )
        baseline.parent.mkdir(parents=True, exist_ok=True)
        if not baseline.exists():
            baseline.write_text("", encoding="utf-8")
        if baseline.read_bytes():
            raise ValueError("Unlabeled NCpedia baseline changed")
        with Image.open(image) as im:
            width, height = im.size
            phash = f"{perceptual_hash(im):016x}"
        identifier = f"ncpedia/ncpedia_{number}{image.suffix}"
        evidence = {
            k: entry[k]
            for k in (
                "source_id",
                "provider_page_snapshot",
                "provider_page_sha256",
                "page_url",
                "image_url",
                "country",
                "location_evidence",
                "source_group",
                "credit_provider",
                "photographer_named",
                "supplied_copyright_field",
                "license",
                "license_url",
            )
        }
        row = dict(
            id=identifier,
            image=str(image.resolve()),
            label=str(baseline.resolve()),
            labels=[],
            sha256=digest(image),
            label_sha256=digest(baseline),
            width=width,
            height=height,
            phash=phash,
            source="ncpedia-collection:vietnam",
            source_url=entry["page_url"],
            review_method="visual_manual_review",
            review_evidence=str(image.resolve()),
            creator=None,
            creator_status="not_named_by_provider",
            noncommercial_only=True,
            **evidence,
            **lineage,
        )
        rows.append(row)
        reviews[identifier] = {
            "status": "corrected",
            "label_sha256": row["label_sha256"],
            "labels": labels,
            "source_verified": True,
            "reason": entry["reason"],
        }
        attrs.append(
            dict(
                id=identifier,
                title=entry["title"],
                artist="NCpedia (nhiếp ảnh gia không được nêu tên)",
                creator=None,
                noncommercial_only=True,
                annotation_license=LICENSE,
                download_url=entry["image_url"],
                original_url=entry["image_url"],
                sha256=row["sha256"],
                modifications=(
                    f"Cắt vùng {entry['crop']}, giữ nguyên pixel PNG; nhãn thủ công."
                    if "crop" in entry
                    else "Giữ nguyên ảnh nguồn; nhãn thủ công."
                ),
                **evidence,
                **lineage,
            )
        )
        audit.append(dict(batch=ledger.name, id=identifier, **entry))
    return rows, reviews, attrs, audit


def verify_ncpedia_attribution(rows, attrs, root):
    selected = {r["id"]: r for r in rows if r["id"].startswith("ncpedia/")}
    credits = [a for a in attrs if a["id"].startswith("ncpedia/")]
    if len(credits) != len(selected) or {a["id"] for a in credits} != selected.keys():
        raise ValueError("NCpedia attribution coverage differs from exported images")
    for attr in credits:
        row = selected[attr["id"]]
        _, number = _source_evidence(row, root)
        if (
            row["source"] != "ncpedia-collection:vietnam"
            or row["id"] != f"ncpedia/ncpedia_{number}{Path(row['image']).suffix}"
        ):
            raise ValueError("NCpedia source identity or collection group changed")
        if (
            attr["annotation_license"] != LICENSE
            or attr["license"] != row["license"]
            or attr["credit_provider"] != "NCpedia"
            or attr["creator"] is not None
            or row["creator_status"] != "not_named_by_provider"
            or attr["noncommercial_only"] is not True
            or row["noncommercial_only"] is not True
            or attr["sha256"] != row["sha256"]
        ):
            raise ValueError("NCpedia attribution or NC-SA conditions changed")
        for key in (
            "source_id",
            "page_url",
            "image_url",
            "supplied_copyright_field",
            "provider_page_snapshot",
            "provider_page_sha256",
            "country",
            "location_evidence",
            "source_sha256",
            "license_url",
            "source_group",
            "photographer_named",
        ):
            if attr[key] != row[key]:
                raise ValueError(f"NCpedia attribution differs: {key}")
