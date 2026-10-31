"""Import reviewed VnTraffic crops; tracking annotations never become helmet GT."""

import hashlib
import json
import math
import re
import zipfile
from pathlib import Path

import cv2
from curate_v2 import normalize_boxes
from PIL import Image, ImageChops
from prepare_dataset import digest, perceptual_hash
from training_crops import prepare_review_image

RECORD = "18195750"
PAGE = f"https://zenodo.org/records/{RECORD}"
GROUP = f"zenodo:{RECORD}:VNTraffic"
PREFIX = "zenodo-vntraffic/"
LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VIDEO_MEMBER = "VNTraffic/VNTraffic_Original-video.mp4"
DOWNLOAD = f"https://zenodo.org/api/records/{RECORD}/files/Vehicle_Tracking.zip/content"
KEYS = (
    "provider",
    "record_id",
    "record_url",
    "subset",
    "country",
    "location_evidence",
    "metadata_file",
    "metadata_sha256",
    "archive_file",
    "archive_sha256",
    "video_archive_member",
    "video_file",
    "video_sha256",
    "frame_file",
    "frame_sha256",
    "frame_index",
    "frame_time_seconds",
    "source_canvas",
    "source_group",
    "review_episode",
    "license",
    "license_url",
)


def _local(root, value):
    base = (Path(root) / "acquisition_2000").resolve()
    path = (base / value).resolve()
    if not path.is_relative_to(base / "research_zenodo"):
        raise ValueError("Zenodo source/evidence escapes research folder")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _credit(metadata):
    names = [x["name"] for x in metadata["creators"] + metadata.get("contributors", [])]
    if not names or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("Missing Zenodo depositor credit")
    return (
        "; ".join(names)
        + "; source video credited to Hanoi Radio and Television by depositor"
    )


def _source_evidence(entry, root, cache):
    if (
        entry["provider"] != "zenodo"
        or entry["record_id"] != RECORD
        or entry["record_url"] != PAGE
        or entry["subset"] != "VNTraffic"
        or entry["country"] != "VN"
        or not entry["location_evidence"].strip()
        or entry["source_group"] != GROUP
        or entry["video_archive_member"] != VIDEO_MEMBER
        or entry["license"] != LICENSE
        or entry["license_url"] != LICENSE_URL
        or type(entry["frame_index"]) is not int
        or entry["frame_index"] < 0
        or not re.fullmatch(r"[a-z0-9_-]{1,80}", entry["review_episode"])
    ):
        raise ValueError("Unverified VnTraffic source, country, frame, or license")
    metadata_file = _local(root, entry["metadata_file"])
    if digest(metadata_file) != entry["metadata_sha256"]:
        raise ValueError("Zenodo metadata changed")
    record = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata = record["metadata"]
    if (
        str(record["id"]) != RECORD
        or metadata["title"] != "Vehicle Tracking Dataset"
        or metadata["license"]["id"] != "cc-by-4.0"
        or metadata["access_right"] != "open"
        or "vntraffic" not in metadata["description"].lower()
        or "vietnam" not in metadata["description"].lower()
    ):
        raise ValueError("Metadata does not support VnTraffic country or license")
    credit = _credit(metadata)
    published = next(
        (f for f in record["files"] if f["key"] == "Vehicle_Tracking.zip"), None
    )
    if published is None or published["links"]["self"] != DOWNLOAD:
        raise ValueError("Missing authoritative Zenodo archive identity")
    archive = _local(root, entry["archive_file"])
    key = (archive, entry["archive_sha256"], entry["metadata_sha256"])
    if key not in cache:
        with archive.open("rb") as stream:
            md5 = hashlib.file_digest(stream, "md5").hexdigest()
        if (
            digest(archive) != entry["archive_sha256"]
            or archive.stat().st_size != published["size"]
            or published["checksum"] != "md5:" + md5
        ):
            raise ValueError("Zenodo archive differs from published checksum")
        cache.add(key)
    video = _local(root, entry["video_file"])
    with zipfile.ZipFile(archive) as z:
        member = z.getinfo(VIDEO_MEMBER)
        if member.file_size > 50 * 1024 * 1024:
            raise ValueError("Unexpectedly large VnTraffic video member")
        member_hash = hashlib.sha256(z.read(member)).hexdigest()
    if member_hash != entry["video_sha256"] or digest(video) != member_hash:
        raise ValueError("Original VnTraffic video differs from archive")
    frame_path = _local(root, entry["frame_file"])
    if digest(frame_path) != entry["frame_sha256"]:
        raise ValueError("Reviewed frame file changed")
    cap = cv2.VideoCapture(str(video))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if (
            not cap.isOpened()
            or not math.isfinite(fps)
            or fps <= 0
            or entry["frame_index"] >= count
            or not math.isclose(
                entry["frame_time_seconds"], entry["frame_index"] / fps, abs_tol=1e-6
            )
        ):
            raise ValueError("Frame index/timestamp outside original video")
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, entry["frame_index"]):
            raise ValueError("Cannot seek original video frame")
        ok, pixels = cap.read()
        if not ok:
            raise ValueError("Cannot decode original video frame")
        decoded = Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        with Image.open(frame_path) as saved:
            if (
                saved.size != decoded.size
                or list(saved.size) != entry["source_canvas"]
                or ImageChops.difference(decoded, saved.convert("RGB")).getbbox()
            ):
                raise ValueError("Decoded frame pixels differ from reviewed frame")
    finally:
        cap.release()
    return frame_path, credit


def load_zenodo_reviews(root):
    root = Path(root)
    rows, reviews, attrs, audit = [], {}, [], []
    seen_frames, seen_episodes, cache = set(), set(), set()
    for ledger in sorted(
        (root / "acquisition_2000").glob("zenodo_reviews_batch*.json")
    ):
        for entry in json.loads(ledger.read_text(encoding="utf-8")):
            if (
                entry["status"] != "corrected"
                or entry["visual_reviewed"] is not True
                or entry["source_verified"] is not True
                or not entry["reason"].strip()
            ):
                raise ValueError(
                    "Only complete explicit manual Zenodo reviews may export"
                )
            if (
                entry["frame_index"] in seen_frames
                or entry["review_episode"] in seen_episodes
            ):
                raise ValueError("Duplicate VnTraffic frame or actor/view episode")
            frame, credit = _source_evidence(entry, root, cache)
            seen_frames.add(entry["frame_index"])
            seen_episodes.add(entry["review_episode"])
            evidence = _local(root, entry["evidence"])
            if digest(evidence) != entry["evidence_sha256"]:
                raise ValueError("Manual Zenodo review evidence changed")
            image, lineage = prepare_review_image(
                frame,
                entry,
                root / "acquisition_2000/curated_zenodo_vntraffic",
                entry["frame_index"],
                provider="zenodo",
            )
            baseline = (
                root
                / "acquisition_2000/unlabeled_baseline"
                / f"zenodo_vntraffic_{entry['frame_index']:06}.txt"
            )
            baseline.parent.mkdir(parents=True, exist_ok=True)
            if not baseline.exists():
                baseline.write_text("", encoding="utf-8")
            if baseline.read_bytes():
                raise ValueError("Unlabeled Zenodo baseline changed")
            identifier = f"{PREFIX}frame_{entry['frame_index']:06}{image.suffix}"
            provenance = {key: entry[key] for key in KEYS}
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
                source=GROUP,
                source_url=PAGE,
                train_only=True,
                review_method="visual_manual_review",
                review_evidence="acquisition_2000/" + entry["evidence"],
                **provenance,
            )
            rows.append(row)
            reviews[identifier] = {
                "status": "corrected",
                "label_sha256": row["label_sha256"],
                "labels": normalize_boxes(entry["boxes"], entry["canvas"]),
                "source_verified": True,
                "reason": entry["reason"],
            }
            attrs.append(
                dict(
                    id=identifier,
                    title=f"VnTraffic original video frame {entry['frame_index']}",
                    artist=credit,
                    annotation_license=LICENSE,
                    sha256=row["sha256"],
                    page_url=PAGE,
                    original_url=PAGE,
                    download_url=DOWNLOAD,
                    license_basis="Zenodo depositor declaration; original broadcaster permission not independently established",
                    modifications=f"Native PNG crop {entry['crop']}; manually assigned motorcycle/rider labels.",
                    **provenance,
                )
            )
            audit.append(dict(batch=ledger.name, id=identifier, **entry))
    return rows, reviews, attrs, audit


def verify_zenodo_attribution(rows, attrs, root):
    selected = {r["id"]: r for r in rows if r["id"].startswith(PREFIX)}
    credits = [a for a in attrs if a["id"].startswith(PREFIX)]
    if len(credits) != len(selected) or {a["id"] for a in credits} != selected.keys():
        raise ValueError("Zenodo attribution coverage differs from exported frames")
    cache, frames, episodes = set(), set(), set()
    for attr in credits:
        row = selected[attr["id"]]
        _, credit = _source_evidence(row, root, cache)
        if row["frame_index"] in frames or row["review_episode"] in episodes:
            raise ValueError("Duplicate exported VnTraffic actor/frame")
        frames.add(row["frame_index"])
        episodes.add(row["review_episode"])
        if (
            row["source"] != GROUP
            or row.get("train_only") is not True
            or row.get("new_split", "train") != "train"
            or row["id"]
            != f"{PREFIX}frame_{row['frame_index']:06}{Path(row['image']).suffix}"
            or attr["artist"] != credit
            or attr["annotation_license"] != LICENSE
            or attr["sha256"] != row["sha256"]
        ):
            raise ValueError(
                "Zenodo identity, credit, license, or training restriction changed"
            )
        for key in (*KEYS, "source_image", "source_sha256", "source_size", "crop"):
            if attr.get(key) != row.get(key):
                raise ValueError(f"Zenodo attribution differs: {key}")
