"""Select separately acquired Commons originals without replacing thumbnails."""

import json
from pathlib import Path

from PIL import Image
from prepare_dataset import digest


def _confined_file(base, value, allowed):
    path = (base / value).resolve()
    if not path.is_relative_to(allowed.resolve()) or not path.is_file():
        raise ValueError(
            "Commons source/evidence must stay inside its acquisition folder"
        )
    return path


def resolve_commons_source(source, decision, root):
    base = Path(root).resolve() / "acquisition_2000"
    if source["pageid"] != decision["pageid"]:
        raise ValueError("Commons source page does not match review")
    variant = decision.get("source_variant", "thumbnail")
    if variant == "thumbnail":
        path = _confined_file(base, source["local_image"], base / "images")
        if (
            digest(path) != source["download_sha256"]
            or digest(path) != decision["image_sha256"]
        ):
            raise ValueError("Reviewed thumbnail bytes changed")
        return path, {}
    if variant != "original":
        raise ValueError("Unknown Commons source variant")
    folder = base / "commons_original_resolution_review"
    receipt_file = _confined_file(base, decision["original_acquisition_file"], folder)
    if digest(receipt_file) != decision["original_acquisition_sha256"]:
        raise ValueError("Original acquisition receipt changed")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    if (
        receipt.get("pageid") != source["pageid"]
        or receipt.get("requested_url") != source["url"]
        or receipt.get("status") != "downloaded_candidate_only"
        or receipt.get("http_status") != 200
        or receipt.get("canvas") != [source["width"], source["height"]]
    ):
        raise ValueError("Original receipt does not match source URL/page/canvas")
    path = _confined_file(base, decision["source_file"], folder)
    received_path = _confined_file(receipt_file.parent, receipt["file"], folder)
    if (
        path != received_path
        or digest(path) != receipt["sha256"]
        or receipt["sha256"] != decision["image_sha256"]
        or ("bytes" in receipt and path.stat().st_size != receipt["bytes"])
    ):
        raise ValueError("Original bytes/path do not match acquisition receipt")
    with Image.open(path) as image:
        image.load()
        if list(image.size) != receipt["canvas"] or image.getexif().get(274, 1) != 1:
            raise ValueError("Original pixels have unexpected dimensions/orientation")
    return path, {
        "commons_source_variant": "original",
        "original_acquisition_file": str(receipt_file),
        "original_acquisition_sha256": digest(receipt_file),
        "original_download_url": source["url"],
    }


def verify_commons_sources(rows, attribution, root):
    selected = [row for row in rows if row["id"].startswith("commons/")]
    if not selected:
        return 0
    sources = {}
    base = Path(root).resolve() / "acquisition_2000"
    for name in ("downloaded_partial.json", "sources.json"):
        file = base / name
        if file.exists():
            for source in json.loads(file.read_text(encoding="utf-8")):
                if "download_sha256" in source:
                    sources[source["pageid"]] = source
    attrs = {item["id"]: item for item in attribution}
    pinned_file = Path(root) / "dataset_v3/manifest.json"
    pinned = (
        {row["id"]: row for row in json.loads(pinned_file.read_text(encoding="utf-8"))}
        if pinned_file.exists()
        else {}
    )
    pinned_attrs_file = Path(root) / "dataset_v3/attribution.json"
    pinned_attrs = (
        {
            row["id"]: row
            for row in json.loads(pinned_attrs_file.read_text(encoding="utf-8"))
        }
        if pinned_attrs_file.exists()
        else {}
    )
    count = 0
    for row in selected:
        pageid = int(row["id"].split("_")[-1].split(".")[0])
        if row.get("original_pageid", pageid) != pageid:
            raise ValueError("Commons row page identity changed")
        attr = attrs[row["id"]]
        if pageid not in sources:
            # The first reviewed Commons photos predate acquisition_2000.
            # Keep those exact pinned images/credits; no new-source fallback.
            original_fields = (
                "commons_source_variant",
                "original_acquisition_file",
                "original_acquisition_sha256",
                "original_download_url",
            )
            if row["id"] not in pinned or any(
                key in obj for obj in (row, attr) for key in original_fields
            ):
                raise ValueError("Commons source missing from acquisition records")
            old = pinned[row["id"]]
            path = _confined_file(
                Path(root), old["image"], Path(root) / "external_commons/images"
            )
            if (
                path != Path(row["image"]).resolve()
                or digest(path) != old["sha256"]
                or row["sha256"] != old["sha256"]
            ):
                raise ValueError("Pinned legacy Commons image changed")
            old_attr = pinned_attrs[row["id"]]
            for key in (
                "sha256",
                "page_url",
                "original_url",
                "download_url",
                "license",
                "artist",
            ):
                if attr.get(key) != old_attr.get(key):
                    raise ValueError("Pinned legacy Commons credit changed")
            continue
        source = sources[pageid]
        variant = row.get("commons_source_variant", "thumbnail")
        decision = {
            "pageid": pageid,
            "source_variant": variant,
            "image_sha256": row.get("source_sha256", row.get("sha256")),
        }
        if variant == "original":
            decision.update(
                source_file=row["source_image"],
                original_acquisition_file=row["original_acquisition_file"],
                original_acquisition_sha256=row["original_acquisition_sha256"],
            )
        path, evidence = resolve_commons_source(source, decision, root)
        if path != Path(row.get("source_image", row.get("image"))).resolve():
            raise ValueError("Commons row points at a different source variant")
        for key in (
            "commons_source_variant",
            "original_acquisition_file",
            "original_acquisition_sha256",
            "original_download_url",
        ):
            if row.get(key) != evidence.get(key) or attr.get(key) != evidence.get(key):
                raise ValueError("Commons original provenance differs from attribution")
        if evidence:
            if (
                attr["download_url"] != source["url"]
                or attr["original_url"] != source["url"]
            ):
                raise ValueError(
                    "Original attribution references a different download URL"
                )
            count += 1
    return count
