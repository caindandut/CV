"""Import reviewed Kaggle images with source-specific provenance and country gates."""

import hashlib
import atexit
import json
import re
import zipfile
from pathlib import Path

from curate_v2 import normalize_boxes
from PIL import Image, ImageChops
from prepare_dataset import digest, perceptual_hash
from training_crops import prepare_review_image

DATASET = "maitam/vietnamese-traffic-signs"
GROUP = f"kaggle:{DATASET}"
PAGE = f"https://www.kaggle.com/datasets/{DATASET}"
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
PROVENANCE_KEYS = (
    "provider",
    "dataset_ref",
    "version",
    "image_id",
    "archive_entry",
    "archive_file",
    "archive_sha256",
    "metadata_file",
    "metadata_sha256",
    "source_sha256",
    "source_group",
    "page_url",
    "license",
    "license_url",
    "creator",
    "maintainer",
    "country",
    "location_evidence",
)
CAMERA_KEYS = ("country_gate_file", "country_gate_sha256", "camera_prefix")

# These curation scripts repeatedly read a few immutable source archives. Reuse
# only their parsed central directory; every provenance/member/pixel check below
# remains unchanged. Handles close at process exit and reopen if file stats change.
_ARCHIVES = {}


def _archive_reader(path):
    path = Path(path).resolve()
    stat = path.stat()
    identity = (stat.st_size, stat.st_mtime_ns)
    previous = _ARCHIVES.get(path)
    if previous and previous[0] == identity and previous[1].fp is not None:
        return previous[1]
    if previous:
        previous[1].close()
    reader = zipfile.ZipFile(path)
    _ARCHIVES[path] = (identity, reader)
    return reader


@atexit.register
def _close_archives():
    for _, reader in _ARCHIVES.values():
        reader.close()
    _ARCHIVES.clear()


def _profile(entry):
    dataset = entry["dataset_ref"]
    if dataset == DATASET:
        return {
            "dataset": DATASET,
            "version": 8,
            "prefix": "kaggle-maitam-v8",
            "member": f"archive/images/{entry['image_id']}.jpg",
            "pattern": r"\d{4}",
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "creator": "Data Palpating Team",
            "maintainer": "Mai Tam (maitam)",
            "artist": "Data Palpating Team; Kaggle maintainer Mai Tam (maitam)",
            "destination": "curated_kaggle_maitam_v8",
            "name": entry["image_id"],
        }
    if dataset == "huanhuyle/vietnam-traffic-detection":
        camera = entry.get("camera_prefix", "day_cam_24")
        if camera not in {"day_cam_10", "day_cam_11", "day_cam_23", "day_cam_24"}:
            raise ValueError("Camera subset has not been enabled for manual review")
        return {
            "dataset": dataset,
            "version": 2,
            "prefix": "kaggle-huan-v2",
            "member": f"images/train/{camera}_{entry['image_id']}.jpg",
            "pattern": r"\d{5}",
            "license": "Apache 2.0",
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
            "creator": "Original video author not supplied",
            "maintainer": "huan huy le (huanhuyle)",
            "artist": "Kaggle publisher huan huy le (huanhuyle); original video author not supplied",
            "destination": f"curated_kaggle_huan_v2_cam{camera.rsplit('_', 1)[1]}",
            "name": f"{camera}_{entry['image_id']}",
        }
    raise ValueError("Unsupported Kaggle source")


def _provenance_keys(profile):
    return PROVENANCE_KEYS + (
        CAMERA_KEYS if profile["prefix"] == "kaggle-huan-v2" else ()
    )


def _camera_country_evidence(entry, root, archive):
    camera = entry.get("camera_prefix", "day_cam_24")
    gate_file = _local(root, entry["country_gate_file"])
    if digest(gate_file) != entry["country_gate_sha256"]:
        raise ValueError("Camera country review changed")
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    if camera == "day_cam_23":
        # This rainy view is admitted only through its reviewed scene match to
        # camera24, whose independent country proof is revalidated in full.
        if (
            gate.get("reference_camera") != "day_cam_24"
            or gate.get("scene_match_reviewed") is not True
            or not gate.get("reference_country_gate_file")
            or not gate.get("reference_country_gate_sha256")
        ):
            raise ValueError("Camera23 requires a reviewed camera24 reference chain")
        reference_entry = dict(
            entry,
            camera_prefix="day_cam_24",
            country_gate_file=gate["reference_country_gate_file"],
            country_gate_sha256=gate["reference_country_gate_sha256"],
        )
        _camera_country_evidence(reference_entry, root, archive)
    proof = gate["country_evidence"]
    if (
        gate["dataset"] != entry["dataset_ref"]
        or gate["version"] != 2
        or gate["camera_prefix"] != camera
        or gate["country"] != "VN"
        or gate["country_verified_by"] != "manual native-image visual review"
        or gate["metadata_sha256"] != entry["metadata_sha256"]
        or len(proof["visible_features"]) < 2
        or not re.fullmatch(
            rf"images/(train|val)/{re.escape(camera)}_\d{{5}}\.jpg",
            proof["archive_entry"],
        )
    ):
        raise ValueError("Country evidence does not establish selected camera subset")
    source = _local(root, proof["source_file"])
    crop = _local(root, proof["crop_file"])
    z = _archive_reader(archive)
    member = z.getinfo(proof["archive_entry"])
    if member.file_size > 20 * 1024 * 1024:
        raise ValueError("Unexpectedly large country evidence member")
    member_hash = hashlib.sha256(z.read(member)).hexdigest()
    if (
        digest(source) != member_hash
        or member_hash != proof["source_sha256"]
        or digest(crop) != proof["crop_sha256"]
    ):
        raise ValueError("Country evidence pixels changed")
    with Image.open(source) as raw, Image.open(crop) as saved:
        expected = raw.convert("RGB").crop(proof["crop"])
        if (
            saved.size != expected.size
            or ImageChops.difference(expected, saved.convert("RGB")).getbbox()
        ):
            raise ValueError("Country evidence crop differs from archive pixels")


def count_kaggle_candidates(root):
    folder = Path(root) / "acquisition_2000/research_kaggle"
    report = folder / "maitam_vnts_v8_acquisition01.json"
    if not report.exists():
        return 0
    acquisition = json.loads(report.read_text(encoding="utf-8"))
    if acquisition["status"] != "downloaded_crc_verified_candidates_only":
        return 0
    archive = folder / "maitam_vnts_v8.zip"
    if digest(archive) != acquisition["sha256"]:
        raise ValueError("Completed Kaggle candidate archive changed")
    with zipfile.ZipFile(archive) as z:
        return len(
            {
                name
                for name in z.namelist()
                if re.fullmatch(r"archive/images/\d{4}\.jpg", name)
            }
        )


def _local(root, value):
    base = (Path(root) / "acquisition_2000").resolve()
    path = (base / value).resolve()
    if not path.is_relative_to((base / "research_kaggle").resolve()):
        raise ValueError("Kaggle source/evidence escapes research folder")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _source_evidence(entry, root, archive_cache):
    profile = _profile(entry)
    page = f"https://www.kaggle.com/datasets/{profile['dataset']}"
    member = profile["member"]
    # Provider val folders do not determine our split, but retain exact archive identity.
    if profile["prefix"] == "kaggle-huan-v2" and entry["archive_entry"].startswith(
        "images/val/"
    ):
        member = member.replace("images/train/", "images/val/", 1)
    if (
        entry["provider"] != "kaggle"
        or entry["version"] != profile["version"]
        or entry["source_group"] != f"kaggle:{profile['dataset']}"
        or entry["page_url"] != page
        or entry["country"] != "VN"
        or not entry["location_evidence"].strip()
        or entry["license"] != profile["license"]
        or entry["license_url"] != profile["license_url"]
        or entry["creator"] != profile["creator"]
        or entry["maintainer"] != profile["maintainer"]
        or not re.fullmatch(profile["pattern"], entry["image_id"])
        or entry["archive_entry"] != member
    ):
        raise ValueError(
            "Unverified Kaggle identity, source group, country, or license"
        )
    metadata = _local(root, entry["metadata_file"])
    if digest(metadata) != entry["metadata_sha256"]:
        raise ValueError("Kaggle metadata snapshot changed")
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    if (
        meta["ref"] != profile["dataset"]
        or meta["currentVersionNumber"] != profile["version"]
        or meta["licenseName"] != profile["license"]
    ):
        raise ValueError(
            "Metadata does not support dataset, version, credit, or license"
        )
    if profile["prefix"] == "kaggle-maitam-v8":
        if (
            "Data Palpating Team" not in meta["description"]
            or "Vietnam" not in meta["description"]
        ):
            raise ValueError("VNTS metadata does not establish country and credit")
    elif meta["ownerRef"] != "huanhuyle" or meta["creatorName"] != "huan huy le":
        raise ValueError("Camera dataset publisher changed")
    archive = _local(root, entry["archive_file"])
    cache_key = (archive, entry["archive_sha256"])
    if cache_key not in archive_cache:
        if digest(archive) != entry["archive_sha256"]:
            raise ValueError("Kaggle archive changed")
        archive_cache.add(cache_key)
    if profile["prefix"] == "kaggle-huan-v2":
        _camera_country_evidence(entry, root, archive)
    source = _local(root, entry["source_image"])
    z = _archive_reader(archive)
    member = z.getinfo(entry["archive_entry"])
    if member.file_size > 20 * 1024 * 1024:
        raise ValueError("Unexpectedly large image member")
    member_hash = hashlib.sha256(z.read(member)).hexdigest()
    if member_hash != entry["source_sha256"] or digest(source) != member_hash:
        raise ValueError("Source image differs from reviewed archive member")
    return source


def load_kaggle_reviews(root):
    root = Path(root)
    rows, reviews, attrs, audit = [], {}, [], []
    seen, cache = set(), set()
    for ledger in sorted(
        (root / "acquisition_2000").glob("kaggle_reviews_batch*.json")
    ):
        for entry in json.loads(ledger.read_text(encoding="utf-8")):
            if (
                entry["status"] != "corrected"
                or entry["visual_reviewed"] is not True
                or entry.get("source_verified") is not True
                or not entry["reason"].strip()
            ):
                raise ValueError(
                    "Only explicit complete manual Kaggle reviews may export"
                )
            source = _source_evidence(entry, root, cache)
            profile = _profile(entry)
            if entry["source_sha256"] in seen:
                raise ValueError("Duplicate Kaggle source photograph or crop")
            seen.add(entry["source_sha256"])
            evidence = (root / entry["evidence"]).resolve()
            if not evidence.is_relative_to(root.resolve()) or not evidence.is_file():
                raise ValueError("Missing or invalid manual review evidence")
            with Image.open(source) as raw:
                if list(raw.size) != entry["source_canvas"]:
                    raise ValueError("Source dimensions differ from reviewed canvas")
            labels = normalize_boxes(entry["boxes"], entry["canvas"])
            image, lineage = prepare_review_image(
                source,
                entry,
                root / "acquisition_2000" / profile["destination"],
                int(entry["image_id"]),
                provider="kaggle",
            )
            baseline = (
                root
                / "acquisition_2000/unlabeled_baseline"
                / f"{profile['prefix'].replace('-', '_')}_{profile['name']}.txt"
            )
            baseline.parent.mkdir(parents=True, exist_ok=True)
            if not baseline.exists():
                baseline.write_text("", encoding="utf-8")
            if baseline.read_bytes():
                raise ValueError("Unlabeled Kaggle baseline changed")
            identifier = f"{profile['prefix']}/{profile['name']}{image.suffix}"
            provenance = {
                key: entry.get("camera_prefix", "day_cam_24")
                if key == "camera_prefix"
                else entry[key]
                for key in _provenance_keys(profile)
            }
            provenance.update(lineage)
            with Image.open(image) as im:
                width, height = im.size
                phash = f"{perceptual_hash(im):016x}"
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
                source=entry["source_group"],
                source_url=entry["page_url"],
                review_method="visual_manual_review",
                review_evidence=entry["evidence"],
                train_only=profile["prefix"] == "kaggle-huan-v2",
                **provenance,
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
                    title=f"{profile['dataset']} v{profile['version']} archive member {entry['archive_entry']}",
                    artist=profile["artist"],
                    annotation_license=profile["license"],
                    sha256=row["sha256"],
                    download_url=f"https://www.kaggle.com/api/v1/datasets/download/{profile['dataset']}?datasetVersionNumber={profile['version']}",
                    original_url=entry["page_url"],
                    modifications=f"Native PNG crop {entry.get('crop')}; manual rider/motorcycle labels.",
                    **provenance,
                )
            )
            audit.append(dict(batch=ledger.name, id=identifier, **entry))
    return rows, reviews, attrs, audit


def verify_kaggle_attribution(rows, attrs, root):
    prefixes = ("kaggle-maitam-v8/", "kaggle-huan-v2/")
    selected = {r["id"]: r for r in rows if r["id"].startswith(prefixes)}
    credits = [a for a in attrs if a["id"].startswith(prefixes)]
    if len(credits) != len(selected) or {a["id"] for a in credits} != selected.keys():
        raise ValueError("Kaggle attribution coverage differs from exported images")
    cache = set()
    for attr in credits:
        row = selected[attr["id"]]
        _source_evidence(row, root, cache)
        profile = _profile(row)
        if profile["prefix"] == "kaggle-huan-v2" and (
            row.get("train_only") is not True
            or row.get("new_split", "train") != "train"
        ):
            raise ValueError("Selected camera collection is restricted to training")
        if (
            row["source"] != f"kaggle:{profile['dataset']}"
            or row["id"]
            != f"{profile['prefix']}/{profile['name']}{Path(row['image']).suffix}"
        ):
            raise ValueError("Kaggle source identity or group changed")
        if (
            attr["annotation_license"] != profile["license"]
            or attr["sha256"] != row["sha256"]
            or attr["artist"] != profile["artist"]
        ):
            raise ValueError("Kaggle attribution/license changed")
        for key in (*_provenance_keys(profile), "source_image", "crop", "source_size"):
            if attr.get(key) != row.get(key):
                raise ValueError(f"Kaggle attribution differs: {key}")
