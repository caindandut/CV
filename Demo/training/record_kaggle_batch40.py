"""Record four manually reviewed Vietnam camera24 image crops.

Proposal windows only locate candidate objects. All exported boxes are drawn
manually from the exact archive pixels after checking the source frame and crop.
"""

from datetime import datetime, timezone
import argparse
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
EVIDENCE = REVIEW / "manual_batch40_review"
ARCHIVE = RESEARCH / "huanhuyle_vietnam_traffic_v2.zip"
METADATA = RESEARCH / "huanhuyle_vietnam-traffic-detection_metadata01.json"
COUNTRY_GATE = RESEARCH / "huanhuyle_v2_country_review/cam24_country_gate01.json"
LEDGER = ACQ / "kaggle_reviews_batch40.json"
MANUAL_LOG = REVIEW / "manual_batch30_reviews01.json"


SPECS = {
    "00058": {
        "scan": "range_unreviewed_proposals03/proposal_scan02.json",
        "crop": (540, 397, 780, 637),
        "canvas": (240, 240),
        "boxes": [(0, 88, 77, 183, 191), (1, 100, 48, 166, 174)],
        "head": (114, 47, 150, 84),
        "reason": (
            "Crop nguyên pixel có một xe máy và một người ngồi lái nhìn từ phía sau. "
            "Mũ đen có vỏ tròn rõ quanh đầu; đã dịch crop để loại người/xe ở mép "
            "phải. Box xe và rider được vẽ tay trên crop sau khi đối chiếu frame "
            "native; proposal chỉ dùng để tìm vùng."
        ),
    },
}

PENDING_SPECS = {
    "00149": {
        "scan": "range_unreviewed_proposals02/proposal_scan01.json",
        "crop": (100, 430, 340, 670),
        "canvas": (240, 240),
        "reason": (
            "Khung rộng tối thiểu 240x240 chứa một phần xe máy/người khác ở mép phải; "
            "khung hẹp hơn không đạt kích thước tối thiểu của pipeline. Không đưa crop "
            "có đối tượng mục tiêu bị cắt vào tập sạch."
        ),
    },
    "00729": {
        "scan": "range_unreviewed_proposals02/proposal_scan01.json",
        "crop": (910, 460, 1150, 700),
        "canvas": (240, 240),
        "reason": (
            "Khung 240x240 chứa phần xe/rider khác ở rìa trái hoặc phải; crop hẹp hơn "
            "không đạt kích thước tối thiểu của pipeline. Giữ chờ để tránh bỏ sót nhãn "
            "của đối tượng bị cắt."
        ),
    },
    "00866": {
        "scan": "range_unreviewed_proposals02/proposal_scan01.json",
        "crop": (340, 500, 520, 680),
        "canvas": (180, 180),
        "reason": (
            "Đối tượng và mũ nhìn rõ, nhưng crop sạch chỉ đạt 180x180, thấp hơn "
            "điều kiện tối thiểu 240x240 của pipeline; crop vuông mở rộng lại chứa "
            "các phần xe khác ở mép trên và dải che mờ dưới. Không đưa vào tập sạch."
        ),
    },
    "00325": {
        "scan": "range_unreviewed_proposals02/proposal_scan01.json",
        "crop": (663, 480, 903, 720),
        "canvas": (240, 240),
        "head": (100, 23, 142, 70),
        "reason": (
            "Người đang đi xe và box xe nhìn thấy rõ, nhưng vùng đầu quá tối để phân biệt "
            "chắc tóc với mũ đen. Giữ chờ, không suy đoán class 1/2."
        ),
    },
    "00238": {
        "scan": "range_unreviewed_proposals03/proposal_scan02.json",
        "crop": (604, 375, 844, 615),
        "canvas": (240, 240),
        "head": (112, 92, 150, 138),
        "reason": (
            "Có một rider và xe máy rõ, nhưng đầu tối trong frame/crop nên không xác định "
            "chắc mũ hay tóc. Không gán class 1/2 khi chưa đủ bằng chứng."
        ),
    },
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Create visual evidence but do not add review or dataset ledger records.",
    )
    args = parser.parse_args()
    if not args.preview and (LEDGER.exists() or MANUAL_LOG.exists()):
        raise FileExistsError("Batch 40 already exists; review history is immutable")

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
    archive_sha = digest(archive_bytes)
    metadata_sha = digest(METADATA.read_bytes())
    gate_sha = digest(COUNTRY_GATE.read_bytes())
    existing = []
    for ledger in sorted(ACQ.glob("kaggle_reviews_batch*.json")):
        if args.preview and ledger == LEDGER:
            continue
        existing.extend(read_json(ledger))
    reviewed_ids = {
        row["image_id"]
        for row in existing
        if row.get("dataset_ref") == "huanhuyle/vietnam-traffic-detection"
        and row.get("version") == 2
    }
    reviewed_hashes = {
        row["source_sha256"]
        for row in existing
        if row.get("dataset_ref") == "huanhuyle/vietnam-traffic-detection"
        and row.get("version") == 2
    }

    proposals = {}
    scan_names = {spec["scan"] for spec in SPECS.values()}
    scan_names.update(spec["scan"] for spec in PENDING_SPECS.values())
    for scan_name in scan_names:
        scan = read_json(REVIEW / scan_name)
        if scan.get("archive_sha256", scan.get("source_archive_sha256")) != archive_sha:
            raise ValueError(f"Proposal scan archive changed: {scan_name}")
        if scan.get("camera_prefix", "day_cam_24") != "day_cam_24":
            raise ValueError(f"Unexpected proposal camera: {scan_name}")
        for row in scan["proposals"]:
            proposals[row["image_id"]] = row

    font = ImageFont.truetype("arial.ttf", 10)
    colors = {0: (20, 132, 255), 1: (22, 190, 80), 2: (244, 56, 56)}
    names = {0: "motorcycle", 1: "rider_helmet", 2: "rider_no_helmet"}
    accepted = []
    accepted_detail = []
    reviewed_at = datetime.now(timezone.utc).isoformat()
    EVIDENCE.mkdir(exist_ok=True)

    with zipfile.ZipFile(ARCHIVE) as archive:
        for image_id, spec in SPECS.items():
            if image_id in reviewed_ids:
                raise ValueError(f"Source image already appears in a Kaggle ledger: {image_id}")
            proposal = proposals[image_id]
            member = proposal["archive_entry"]
            if not member.startswith(("images/train/day_cam_24_", "images/val/day_cam_24_")):
                raise ValueError(f"Image is outside the verified camera24 subset: {image_id}")
            source_bytes = archive.read(member)
            source_sha = digest(source_bytes)
            info = archive.getinfo(member)
            if source_sha != proposal["source_sha256"]:
                raise ValueError(f"Archive member hash changed: {image_id}")
            if source_sha in reviewed_hashes:
                raise ValueError(f"Source frame already appears in a Kaggle ledger: {image_id}")
            source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
            if source.size != (1280, 720):
                raise ValueError(f"Unexpected source canvas: {image_id}")

            rect = spec["crop"]
            if not (0 <= rect[0] < rect[2] <= source.width and 0 <= rect[1] < rect[3] <= source.height):
                raise ValueError(f"Crop extends outside native frame: {image_id}")
            crop = source.crop(rect)
            if crop.size != spec["canvas"]:
                raise ValueError(f"Crop canvas mismatch: {image_id}")
            for class_id, x1, y1, x2, y2 in spec["boxes"]:
                if class_id not in names or not (0 <= x1 < x2 <= crop.width and 0 <= y1 < y2 <= crop.height):
                    raise ValueError(f"Invalid manual box in {image_id}: {(class_id, x1, y1, x2, y2)}")

            crop_path = EVIDENCE / f"{image_id}_crop.png"
            crop.save(crop_path)
            local_source = REVIEW / f"day_cam_24_{image_id}.jpg"
            local_source.write_bytes(source_bytes)
            if digest(local_source.read_bytes()) != source_sha:
                raise ValueError(f"Saved source copy differs from archive: {image_id}")

            overlay = crop.copy()
            draw = ImageDraw.Draw(overlay)
            for class_id, x1, y1, x2, y2 in spec["boxes"]:
                draw.rectangle((x1, y1, x2, y2), outline=colors[class_id], width=2)
                label = f"{class_id} {names[class_id]}"
                top = max(0, y1 - 12)
                draw.rectangle((x1, top, min(crop.width, x1 + 92), max(10, y1)), fill=colors[class_id])
                draw.text((x1 + 1, max(0, y1 - 11)), label, font=font, fill="white")
            overlay_path = EVIDENCE / f"{image_id}_overlay.png"
            overlay.save(overlay_path)

            hx1, hy1, hx2, hy2 = spec["head"]
            if not (0 <= hx1 < hx2 <= crop.width and 0 <= hy1 < hy2 <= crop.height):
                raise ValueError(f"Invalid head evidence region: {image_id}")
            head = crop.crop((hx1, hy1, hx2, hy2)).resize(
                ((hx2 - hx1) * 6, (hy2 - hy1) * 6), Image.Resampling.NEAREST
            )
            head_path = EVIDENCE / f"{image_id}_head_detail.png"
            head.save(head_path)

            crop_sha = digest(crop_path.read_bytes())
            evidence_rel = (
                f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/"
                f"manual_batch40_review/{image_id}_overlay.png"
            )
            head_rel = (
                f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/"
                f"manual_batch40_review/{image_id}_head_detail.png"
            )
            canvas = [crop.width, crop.height]
            item = {
                "provider": "kaggle",
                "dataset_ref": "huanhuyle/vietnam-traffic-detection",
                "version": 2,
                "image_id": image_id,
                "archive_entry": member,
                "archive_file": "research_kaggle/huanhuyle_vietnam_traffic_v2.zip",
                "archive_sha256": archive_sha,
                "archive_crc32": f"{info.CRC:08x}",
                "metadata_file": "research_kaggle/huanhuyle_vietnam-traffic-detection_metadata01.json",
                "metadata_sha256": metadata_sha,
                "source_image": f"research_kaggle/huanhuyle_v2_cam24_review/day_cam_24_{image_id}.jpg",
                "source_sha256": source_sha,
                "source_canvas": [1280, 720],
                "crop": list(rect),
                "canvas": canvas,
                "crop_sha256": crop_sha,
                "boxes": [list(box) for box in spec["boxes"]],
                "country": "VN",
                "location_evidence": (
                    "Camera day_cam_24 trong cùng video; quốc gia Việt Nam được xác minh thủ công "
                    "qua tuyến xe buýt 141, chữ tiếng Việt và biển đăng ký Việt Nam. Chỉ kết luận "
                    "giao thông Việt Nam; không khẳng định thành phố hoặc tác giả video."
                ),
                "source_group": "kaggle:huanhuyle/vietnam-traffic-detection",
                "page_url": "https://www.kaggle.com/datasets/huanhuyle/vietnam-traffic-detection",
                "license": "Apache 2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "creator": "Original video author not supplied",
                "maintainer": "huan huy le (huanhuyle)",
                "visual_reviewed": True,
                "status": "corrected",
                "source_verified": True,
                "evidence": evidence_rel,
                "head_detail": head_rel,
                "review_log": "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch30_reviews01.json",
                "country_gate_file": "research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json",
                "country_gate_sha256": gate_sha,
                "camera_prefix": "day_cam_24",
                "evidence_sha256": digest(overlay_path.read_bytes()),
                "head_detail_sha256": digest(head_path.read_bytes()),
                "geometry_qa_passed": True,
                "geometry_qa": (
                    f"Pixel XYXY boxes were drawn manually on the {crop.width}x{crop.height} crop and "
                    f"checked against native 1280x720 archive frame; crop ROI is {list(rect)}. "
                    f"The crop contains one complete motorcycle and "
                    f"{sum(box[0] in (1, 2) for box in spec['boxes'])} individually boxed rider(s)."
                ),
                "reason": spec["reason"],
                "model_predictions_used_as_ground_truth": False,
                "reviewed_at_utc": reviewed_at,
            }
            accepted.append(item)
            accepted_detail.append(
                {
                    "image_id": image_id,
                    "archive_entry": member,
                    "source_sha256": source_sha,
                    "archive_crc32": f"{info.CRC:08x}",
                    "crop": list(rect),
                    "canvas": canvas,
                    "crop_sha256": crop_sha,
                    "boxes_pixel_xyxy": [list(box) for box in spec["boxes"]],
                    "status": "accepted_crop",
                    "visual_reviewed": True,
                    "source_verified": True,
                    "country": "VN",
                    "reason": spec["reason"],
                    "model_predictions_used_as_ground_truth": False,
                    "evidence": evidence_rel,
                    "evidence_sha256": item["evidence_sha256"],
                    "head_detail": head_rel,
                    "head_detail_sha256": item["head_detail_sha256"],
                    "reviewed_at_utc": reviewed_at,
                }
            )

    pending = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for image_id, spec in PENDING_SPECS.items():
            proposal = proposals[image_id]
            member = proposal["archive_entry"]
            source_bytes = archive.read(member)
            source_sha = digest(source_bytes)
            if source_sha != proposal["source_sha256"]:
                raise ValueError(f"Pending candidate source hash changed: {image_id}")
            source = Image.open(io.BytesIO(source_bytes)).convert("RGB")
            if source.size != (1280, 720):
                raise ValueError(f"Unexpected pending source canvas: {image_id}")
            crop = source.crop(spec["crop"])
            if crop.size != spec["canvas"]:
                raise ValueError(f"Pending crop canvas mismatch: {image_id}")
            crop_path = EVIDENCE / f"{image_id}_pending_crop.png"
            crop.save(crop_path)
            local_source = REVIEW / f"day_cam_24_{image_id}.jpg"
            local_source.write_bytes(source_bytes)
            detail = {
                "image_id": image_id,
                "archive_entry": member,
                "source_sha256": source_sha,
                "crop": list(spec["crop"]),
                "canvas": list(crop.size),
                "crop_sha256": digest(crop_path.read_bytes()),
                "status": "pending",
                "visual_reviewed": True,
                "source_verified": True,
                "country": "VN",
                "reason": spec["reason"],
                "model_predictions_used_as_ground_truth": False,
                "review_evidence": [
                    f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch40_review/{image_id}_pending_crop.png"
                ],
                "reviewed_at_utc": reviewed_at,
            }
            if "head" in spec:
                head = crop.crop(spec["head"]).resize(
                    (
                        (spec["head"][2] - spec["head"][0]) * 6,
                        (spec["head"][3] - spec["head"][1]) * 6,
                    ),
                    Image.Resampling.NEAREST,
                )
                head_path = EVIDENCE / f"{image_id}_pending_head_detail.png"
                head.save(head_path)
                detail["review_evidence"].append(
                    f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch40_review/{image_id}_pending_head_detail.png"
                )
                detail["pending_head_detail_sha256"] = digest(head_path.read_bytes())
            pending.append(detail)

    manual_log = {
        "batch": "manual_batch30",
        "dataset_ref": "huanhuyle/vietnam-traffic-detection",
        "dataset_version": 2,
        "review_method": "manual_native_full_frame_and_exact_pixel_crop_review",
        "model_predictions_used_as_ground_truth": False,
        "source_archive_sha256": archive_sha,
        "country_gate_file": "research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json",
        "country_gate_sha256": gate_sha,
        "reviewed_at_utc": reviewed_at,
        "accepted_count": len(accepted_detail),
        "pending_count": len(pending),
        "accepted": accepted_detail,
        "pending": pending,
        "note": (
            "Candidate windows only located images. Each accepted label was drawn manually in pixel "
            "XYXY after checking the exact archive source frame and crop. Images with ambiguous heads, "
            "target objects cut by the minimum crop size, or dimensions below 240x240 remain pending."
        ),
    }
    if not args.preview:
        MANUAL_LOG.write_text(json.dumps(manual_log, ensure_ascii=False, indent=2), encoding="utf-8")
        LEDGER.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "accepted": len(accepted),
                "pending": len(pending),
                "preview_only": args.preview,
                "archive_sha256": archive_sha,
                "reviewed_at_utc": reviewed_at,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
