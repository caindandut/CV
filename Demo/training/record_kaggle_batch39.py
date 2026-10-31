"""Record manually reviewed day_cam_24 crops for clean dataset batch 39.

Candidate labels were used only to locate image windows. This script writes
the review ledger and pixel evidence; it never imports Kaggle annotations as
training labels.
"""

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
EVIDENCE = REVIEW / "manual_batch29_review"
ARCHIVE = RESEARCH / "huanhuyle_vietnam_traffic_v2.zip"
METADATA = RESEARCH / "huanhuyle_vietnam-traffic-detection_metadata01.json"
COUNTRY_GATE = RESEARCH / "huanhuyle_v2_country_review/cam24_country_gate01.json"
LEDGER = ACQ / "kaggle_reviews_batch39.json"
MANUAL_LOG = REVIEW / "manual_batch29_reviews01.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if LEDGER.exists() or MANUAL_LOG.exists():
        raise FileExistsError("Batch 39 already exists; do not overwrite review history")
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

    rects = {
        "00714": (628, 461, 868, 701),
        "00302": (757, 468, 997, 708),
        "00303": (755, 476, 995, 716),
        # Shift right to crop out the adjacent car while retaining the full rider/bike.
        "00068": (720, 420, 960, 660),
        "00762": (628, 459, 868, 699),
    }
    boxes = {
        "00714": [(0, 68, 82, 174, 188), (1, 95, 37, 165, 159)],
        "00302": [(0, 77, 82, 165, 184), (1, 88, 42, 163, 153)],
        "00303": [(0, 78, 92, 167, 186), (1, 86, 70, 168, 164)],
        "00068": [(0, 69, 85, 175, 186), (1, 73, 48, 174, 163)],
        "00762": [(0, 76, 89, 164, 184), (1, 101, 35, 168, 150)],
    }
    head_regions = {
        "00714": (108, 36, 140, 72),
        "00302": (105, 38, 137, 75),
        "00303": (117, 65, 149, 99),
        "00068": (104, 44, 140, 82),
        "00762": (116, 30, 153, 71),
    }
    reasons = {
        "00714": (
            "Crop nguyên pixel 240x240 có một xe máy và một người đang ngồi lái. "
            "Mũ đen có vỏ tròn và dải phản sáng nhìn rõ; không có rider/xe máy "
            "mục tiêu thứ hai. Vòng sơn trên đường không phải đối tượng mục tiêu. "
            "Box được vẽ tay sau khi rà ảnh nguồn native và crop; không dùng nhãn "
            "Kaggle hoặc dự đoán model làm ground truth."
        ),
        "00302": (
            "Crop nguyên pixel 240x240 có một xe máy và một người đang ngồi lái. "
            "Mũ đen có vỏ cứng bo tròn và phản sáng rõ; không có rider/xe máy "
            "mục tiêu thứ hai trong crop. Box xe và người được vẽ tay theo pixel "
            "sau khi rà ảnh native toàn khung và crop; không dùng nhãn Kaggle hoặc "
            "dự đoán model làm ground truth."
        ),
        "00303": (
            "Crop nguyên pixel 240x240 có một xe máy và một người đang lái; mũ "
            "xanh lam có vỏ tròn và viền sáng nhìn thấy. Mép trái có một phần xe "
            "tải, không có xe máy/rider mục tiêu khác trong crop. Box rider và xe "
            "được vẽ thủ công theo ảnh native; không dùng nhãn Kaggle hoặc dự đoán "
            "model làm ground truth."
        ),
        "00068": (
            "Crop nguyên pixel dịch sang phải [720,420,960,660] cô lập một xe máy "
            "và một rider đội mũ đen; vỏ mũ và viền sáng thấy rõ. Crop không còn "
            "phần ô tô cạnh bên; xe và người nằm trọn trong khung. Box pixel được "
            "vẽ tay, không dùng nhãn Kaggle hoặc dự đoán model làm ground truth."
        ),
        "00762": (
            "Crop nguyên pixel 240x240 có một xe máy và một người đang ngồi lái. "
            "Mũ xanh có vỏ bo tròn và vành sáng rõ; không có rider/xe máy mục tiêu "
            "khác trong crop. Vòng sơn trên đường là nền. Box xe và rider được vẽ "
            "tay sau khi rà ảnh native và crop; không dùng nhãn Kaggle hoặc dự đoán "
            "model làm ground truth."
        ),
    }
    proposals = read_json(REVIEW / "range_isolated_proposals01/proposal_scan01.json")
    font = ImageFont.truetype("arial.ttf", 10)
    accepted = []
    accepted_detail = []
    reviewed_at = datetime.now(timezone.utc).isoformat()
    EVIDENCE.mkdir(exist_ok=True)

    with zipfile.ZipFile(ARCHIVE) as archive:
        for image_id, rect in rects.items():
            if image_id in reviewed_ids:
                raise ValueError(f"Source image already appears in a Kaggle ledger: {image_id}")
            proposal = next(
                row for row in proposals["proposals"] if row["image_id"] == image_id
            )
            member = proposal["archive_entry"]
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
            crop = source.crop(rect)
            crop_path = EVIDENCE / f"{image_id}_crop.png"
            crop.save(crop_path)
            local_source = REVIEW / f"day_cam_24_{image_id}.jpg"
            local_source.write_bytes(source_bytes)
            if digest(local_source.read_bytes()) != source_sha:
                raise ValueError(f"Saved source copy differs from archive: {image_id}")

            overlay = crop.copy()
            draw = ImageDraw.Draw(overlay)
            colors = {0: (20, 132, 255), 1: (22, 190, 80), 2: (244, 56, 56)}
            names = {0: "motorcycle", 1: "rider_helmet", 2: "rider_no_helmet"}
            for class_id, x1, y1, x2, y2 in boxes[image_id]:
                draw.rectangle((x1, y1, x2, y2), outline=colors[class_id], width=2)
                label = f"{class_id} {names[class_id]}"
                top = max(0, y1 - 12)
                draw.rectangle((x1, top, x1 + 65, max(10, y1)), fill=colors[class_id])
                draw.text((x1 + 1, max(0, y1 - 11)), label, font=font, fill="white")
            overlay_path = EVIDENCE / f"{image_id}_overlay.png"
            overlay.save(overlay_path)

            hx1, hy1, hx2, hy2 = head_regions[image_id]
            head = crop.crop((hx1, hy1, hx2, hy2)).resize(
                ((hx2 - hx1) * 6, (hy2 - hy1) * 6), Image.Resampling.NEAREST
            )
            head_path = EVIDENCE / f"{image_id}_head_detail.png"
            head.save(head_path)

            crop_sha = digest(crop_path.read_bytes())
            evidence_rel = (
                f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/"
                f"manual_batch29_review/{image_id}_overlay.png"
            )
            head_rel = (
                f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/"
                f"manual_batch29_review/{image_id}_head_detail.png"
            )
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
                "canvas": [240, 240],
                "crop_sha256": crop_sha,
                "boxes": [list(box) for box in boxes[image_id]],
                "country": "VN",
                "location_evidence": (
                    "Camera day_cam_24 trong cùng video; quốc gia Việt Nam được xác minh "
                    "thủ công qua tuyến xe buýt 141, chữ tiếng Việt và biển đăng ký Việt Nam. "
                    "Chỉ kết luận giao thông Việt Nam, không khẳng định thành phố hoặc tác giả video."
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
                "review_log": "acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/manual_batch29_reviews01.json",
                "country_gate_file": "research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json",
                "country_gate_sha256": gate_sha,
                "camera_prefix": "day_cam_24",
                "evidence_sha256": digest(overlay_path.read_bytes()),
                "head_detail_sha256": digest(head_path.read_bytes()),
                "geometry_qa_passed": True,
                "geometry_qa": (
                    f"Box pixel XYXY được vẽ tay trên crop 240x240 và đối chiếu ảnh native; "
                    f"crop khớp ROI {list(rect)} từ archive 1280x720. Crop có một xe máy và "
                    "một rider; đầu/mũ nhìn đủ rõ để xác định class 0 và class 1."
                ),
                "reason": reasons[image_id],
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
                    "canvas": [240, 240],
                    "crop_sha256": crop_sha,
                    "boxes_pixel_xyxy": [list(box) for box in boxes[image_id]],
                    "status": "accepted_crop",
                    "visual_reviewed": True,
                    "source_verified": True,
                    "country": "VN",
                    "reason": reasons[image_id],
                    "model_predictions_used_as_ground_truth": False,
                    "evidence": evidence_rel,
                    "evidence_sha256": item["evidence_sha256"],
                    "head_detail": head_rel,
                    "head_detail_sha256": item["head_detail_sha256"],
                    "reviewed_at_utc": reviewed_at,
                }
            )

    pending_specs = [
        ("00161", "range_isolated_proposals01", "Đầu người nhìn từ phía sau quá nhỏ/tối để phân biệt chắc tóc trần với mũ đen; không đoán class 1/2."),
        ("00057", "range_isolated_proposals01", "Hai người/xe xuất hiện gần nhau; crop dịch làm mất người ở mép dưới nhưng đầu và thân rider phía trước bị chồng lấp, chưa thể xác nhận từng rider/mũ."),
        ("00868", "range_isolated_proposals01", "Nhìn thấy hai mũ trên một xe nhưng thân người chồng lấp; chưa thể vẽ chắc hai box rider riêng."),
        ("00401", "range_unreviewed_proposals02", "Crop có một xe/rider nhưng đầu tối, không phân biệt chắc mũ nâu/đỏ với tóc trần."),
        ("00517", "range_unreviewed_proposals02", "Đầu quá nhỏ/tối để phân biệt chắc mũ đen với tóc; không suy đoán class."),
        ("00317", "range_unreviewed_proposals02", "Nhiều xe/rider chồng lấp; crop thử vẫn cắt một xe khác ở mép."),
        ("00803", "range_unreviewed_proposals02", "Crop đề xuất cắt một xe máy khác ở rìa; chưa có crop sạch bao trọn tất cả đối tượng mục tiêu."),
        ("00863", "range_unreviewed_proposals02", "Crop có phần xe máy khác bị cắt ở mép trên; không thể loại phần này mà vẫn giữ trọn hai rider/xe."),
    ]
    pending = []
    for image_id, folder, reason in pending_specs:
        scan_path = REVIEW / folder / "proposal_scan01.json"
        scan = read_json(scan_path)
        proposal = next(row for row in scan["proposals"] if row["image_id"] == image_id)
        pending.append(
            {
                "image_id": image_id,
                "archive_entry": proposal["archive_entry"],
                "source_sha256": proposal["source_sha256"],
                "archive_crc32": proposal.get("archive_crc32"),
                "crop": proposal.get("crop"),
                "canvas": [
                    proposal["crop"][2] - proposal["crop"][0],
                    proposal["crop"][3] - proposal["crop"][1],
                ],
                "status": "pending",
                "visual_reviewed": True,
                "source_verified": True,
                "country": "VN",
                "reason": reason,
                "model_predictions_used_as_ground_truth": False,
                "review_evidence": [
                    f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/{folder}/{proposal['full_native_file']}",
                    f"acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/{folder}/{proposal['crop_file']}",
                ],
                "reviewed_at_utc": reviewed_at,
            }
        )

    manual_log = {
        "batch": "manual_batch29",
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
            "Accepted boxes were drawn manually in pixel XYXY after checking archive source frames, "
            "exact crops, overlay alignment and head detail. Proposal windows only helped locate "
            "candidates. Pending images and full frames are excluded from clean export."
        ),
    }
    MANUAL_LOG.write_text(json.dumps(manual_log, ensure_ascii=False, indent=2), encoding="utf-8")
    LEDGER.write_text(json.dumps(accepted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(accepted), "pending": len(pending), "archive_sha256": archive_sha, "reviewed_at_utc": reviewed_at}))


if __name__ == "__main__":
    main()
