"""Record one manually reviewed crop from verified Vietnam camera24 footage."""

from datetime import datetime, timezone
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


TRAIN = Path(__file__).resolve().parent
ACQ = TRAIN / "acquisition_2000"
RESEARCH = ACQ / "research_kaggle"
REVIEW = RESEARCH / "huanhuyle_v2_cam24_review"
EVIDENCE = REVIEW / "manual_batch41_review"
ARCHIVE = RESEARCH / "huanhuyle_vietnam_traffic_v2.zip"
METADATA = RESEARCH / "huanhuyle_vietnam-traffic-detection_metadata01.json"
COUNTRY_GATE = RESEARCH / "huanhuyle_v2_country_review/cam24_country_gate01.json"
LEDGER = ACQ / "kaggle_reviews_batch41.json"
MANUAL_LOG = REVIEW / "manual_batch31_reviews01.json"

IMAGE_ID = "00288"
CROP = [662, 404, 902, 644]
CANVAS = [240, 240]
BOXES = [
    [0, 77, 88, 174, 193],
    [1, 100, 46, 157, 158],
]


def digest(data):
    if isinstance(data, Path):
        data = data.read_bytes()
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    if LEDGER.exists() or MANUAL_LOG.exists():
        raise FileExistsError("Batch 41 already exists; review history is immutable")

    metadata = read_json(METADATA)
    if (
        metadata["ref"] != "huanhuyle/vietnam-traffic-detection"
        or metadata["currentVersionNumber"] != 2
        or metadata["licenseName"] != "Apache 2.0"
    ):
        raise ValueError("Kaggle metadata identity/license changed")
    gate = read_json(COUNTRY_GATE)
    if gate["country"] != "VN" or gate["camera_prefix"] != "day_cam_24":
        raise ValueError("Camera24 country gate does not verify Vietnam")

    archive_bytes = ARCHIVE.read_bytes()
    archive_hash = digest(archive_bytes)
    metadata_hash = digest(METADATA)
    gate_hash = digest(COUNTRY_GATE)
    proposal_scan = read_json(REVIEW / "range_unreviewed_proposals03/proposal_scan02.json")
    if proposal_scan.get("archive_sha256", proposal_scan.get("source_archive_sha256")) != archive_hash:
        raise ValueError("Proposal scan archive identity changed")
    proposals = {row["image_id"]: row for row in proposal_scan["proposals"]}
    proposal = proposals[IMAGE_ID]
    expected_member = "images/train/day_cam_24_" + IMAGE_ID + ".jpg"
    if proposal["archive_entry"] != expected_member:
        raise ValueError("Candidate is outside the verified camera24 training subset")

    reviewed_ids = set()
    reviewed_hashes = set()
    for ledger in sorted(ACQ.glob("kaggle_reviews_batch*.json")):
        for row in read_json(ledger):
            if row.get("dataset_ref") == "huanhuyle/vietnam-traffic-detection" and row.get("version") == 2:
                reviewed_ids.add(row["image_id"])
                reviewed_hashes.add(row["source_sha256"])
    if IMAGE_ID in reviewed_ids:
        raise ValueError("Source image already has a Kaggle review")

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        info = archive.getinfo(expected_member)
        source_bytes = archive.read(expected_member)
    source_hash = digest(source_bytes)
    if source_hash != proposal["source_sha256"] or source_hash in reviewed_hashes:
        raise ValueError("Source frame hash differs or has already been reviewed")
    if f"{info.CRC:08x}" != proposal["archive_crc32"]:
        raise ValueError("Archive CRC differs from the candidate scan")

    with Image.open(io.BytesIO(source_bytes)) as original:
        if original.size != (1280, 720) or original.getexif().get(274, 1) != 1:
            raise ValueError("Unexpected camera24 source geometry/orientation")
        crop = original.convert("RGB").crop(tuple(CROP))
    if list(crop.size) != CANVAS:
        raise ValueError("Reviewed crop geometry changed")
    for class_id, x1, y1, x2, y2 in BOXES:
        if class_id not in (0, 1, 2) or not (0 <= x1 < x2 <= 240 and 0 <= y1 < y2 <= 240):
            raise ValueError("Invalid manual box")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    source_path = REVIEW / ("day_cam_24_" + IMAGE_ID + ".jpg")
    source_path.write_bytes(source_bytes)
    if digest(source_path) != source_hash:
        raise ValueError("Saved source copy differs from the archive")
    crop_path = EVIDENCE / (IMAGE_ID + "_crop.png")
    crop.save(crop_path, format="PNG")

    overlay = crop.copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", 11)
    colors = {0: (20, 132, 255), 1: (22, 190, 80), 2: (244, 56, 56)}
    names = {0: "motorcycle", 1: "rider helmet", 2: "rider no helmet"}
    for class_id, x1, y1, x2, y2 in BOXES:
        color = colors[class_id]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        top = max(0, y1 - 13)
        label = str(class_id) + " " + names[class_id]
        bounds = draw.textbbox((x1 + 1, top), label, font=font)
        draw.rectangle((x1, top, min(240, bounds[2] + 2), max(y1, bounds[3] + 1)), fill=color)
        draw.text((x1 + 1, top), label, font=font, fill="white")
    overlay_path = EVIDENCE / (IMAGE_ID + "_overlay.png")
    overlay.save(overlay_path, format="PNG")

    head = crop.crop((112, 40, 151, 83)).resize((312, 344), Image.Resampling.NEAREST)
    head_path = EVIDENCE / (IMAGE_ID + "_head_detail.png")
    head.save(head_path, format="PNG")

    location_evidence = (
        "Camera day_cam_24 thuộc cùng video đã xác minh thủ công qua tuyến xe buýt 141, "
        "chữ tiếng Việt và biển đăng ký Việt Nam. Chỉ kết luận giao thông Việt Nam; "
        "không khẳng định thành phố hoặc tác giả video."
    )
    reason = (
        "Crop nguyên pixel 240x240 có một xe máy và một người ngồi lái nhìn từ phía sau. "
        "Mũ xanh có hình dáng vỏ rõ; không nhầm người đi bộ áo sọc ở góc dưới thành rider. "
        "Box xe và rider được vẽ tay trên crop sau khi đối chiếu frame native; proposal chỉ dùng để tìm vùng."
    )
    now = datetime.now(timezone.utc).isoformat()
    entry = {
        "provider": "kaggle",
        "dataset_ref": "huanhuyle/vietnam-traffic-detection",
        "version": 2,
        "image_id": IMAGE_ID,
        "archive_entry": expected_member,
        "archive_file": "research_kaggle/huanhuyle_vietnam_traffic_v2.zip",
        "archive_sha256": archive_hash,
        "archive_crc32": proposal["archive_crc32"],
        "metadata_file": "research_kaggle/huanhuyle_vietnam-traffic-detection_metadata01.json",
        "metadata_sha256": metadata_hash,
        "source_image": "research_kaggle/huanhuyle_v2_cam24_review/" + source_path.name,
        "source_sha256": source_hash,
        "source_canvas": [1280, 720],
        "crop": CROP,
        "canvas": CANVAS,
        "crop_sha256": digest(crop_path),
        "boxes": BOXES,
        "country": "VN",
        "location_evidence": location_evidence,
        "source_group": "kaggle:huanhuyle/vietnam-traffic-detection",
        "page_url": "https://www.kaggle.com/datasets/huanhuyle/vietnam-traffic-detection",
        "license": "Apache 2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "creator": "Original video author not supplied",
        "maintainer": "huan huy le (huanhuyle)",
        "visual_reviewed": True,
        "status": "corrected",
        "source_verified": True,
        "evidence": "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch41_review/" + overlay_path.name,
        "head_detail": "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch41_review/" + head_path.name,
        "review_log": "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch31_reviews01.json",
        "country_gate_file": "research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json",
        "country_gate_sha256": gate_hash,
        "camera_prefix": "day_cam_24",
        "evidence_sha256": digest(overlay_path),
        "head_detail_sha256": digest(head_path),
        "geometry_qa_passed": True,
        "geometry_qa": "Pixel XYXY boxes were drawn manually on the 240x240 crop and checked against the native 1280x720 archive frame; crop ROI is [662, 404, 902, 644]. The crop contains one complete motorcycle and one individually boxed helmeted rider. A visible pedestrian is outside the rider classes.",
        "reason": reason,
        "model_predictions_used_as_ground_truth": False,
        "reviewed_at_utc": now,
        "selected": {
            "proposal_scan": "research_kaggle/huanhuyle_v2_cam24_review/range_unreviewed_proposals03/proposal_scan02.json",
            "proposal_crop_used_as_label": False,
        },
    }
    log = {
        "batch": "manual_batch31",
        "dataset_ref": "huanhuyle/vietnam-traffic-detection",
        "dataset_version": 2,
        "review_method": "manual_native_full_frame_and_exact_pixel_crop_review",
        "model_predictions_used_as_ground_truth": False,
        "source_archive_sha256": archive_hash,
        "country_gate_file": "research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json",
        "country_gate_sha256": gate_hash,
        "reviewed_at_utc": now,
        "accepted_count": 1,
        "pending_count": 0,
        "accepted": [entry],
        "pending": [],
        "note": "One crop admitted after native-frame, helmet, person, vehicle, source, and geometry review.",
    }
    LEDGER.write_text(json.dumps([entry], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MANUAL_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": 1, "image_id": IMAGE_ID, "source_sha256": source_hash, "crop_sha256": digest(crop_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
