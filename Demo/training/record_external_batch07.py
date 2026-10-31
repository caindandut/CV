"""Two inspected Hanoi photos, with an explicit pending ledger for the batch."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 13,
        "pageid": 58081911,
        "crop": [644, 350, 939, 851],
        "canvas": [295, 501],
        "boxes": [[0, 15, 79, 234, 475]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Làng sinh viên Hacinco, Nguỵ Như Kon Tum, Hà Nội 001.JPG; mô tả Hà Nội, Việt Nam; Phan Minh Tuấn, CC BY-SA 4.0. Gom cùng nguồn tác giả/cảnh Hà Nội đã khóa test.",
        "reason": "Một xe tay ga đỗ có chân chống và hai gương rõ, một box lớp 0. Người phụ nữ áo xanh đứng cạnh xe nói điện thoại, cả hai chân đứng ngoài xe; không gán lớp 2 chỉ vì đầu không đội mũ. Toàn crop đã xem bằng mắt.",
    },
    {
        "proposal_index": 14,
        "pageid": 58081940,
        "crop": [43, 642, 345, 919],
        "canvas": [302, 277],
        "boxes": [[0, 13, 95, 289, 265], [1, 85, 14, 219, 238]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Junction of Dương Đình Nghệ and Trung Kính 14-01-2016 E2.jpg; mô tả Yên Hòa Ward, Hà Nội, Việt Nam; Phan Minh Tuấn, CC BY-SA 4.0.",
        "reason": "Một người ngồi lái xe tay ga, đội mũ vỏ cứng màu trắng lớp 1, mặc áo mưa xanh. Một box xe lớp 0. Xe buýt ở nền không phải xe máy; vùng cắt không có rider khác. Rà trực quan, không nhận box COCO làm nhãn đúng.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch07.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    records = []
    for decision in DECISIONS:
        source = ROOT / f"acquisition_2000/images/commons_{decision['pageid']}.jpg"
        image, _ = prepare_review_image(
            source,
            decision,
            ROOT / "acquisition_2000/curated_crops",
            decision["pageid"],
        )
        records.append(
            dict(
                country="VN",
                visual_reviewed=True,
                image_sha256=digest(source),
                evidence=image.relative_to(ROOT).as_posix(),
                proposal_run="acquisition_2000/coco_crop_proposals/batch04/run.json",
                **decision,
            )
        )
    write_json(output, records)
    write_screening(records)


def write_screening(records):
    proposals = json.loads(
        (
            ROOT / "acquisition_2000/coco_crop_proposals/batch04/proposals.json"
        ).read_text(encoding="utf-8")
    )
    accepted = {r["pageid"] for r in records}
    write_json(
        ROOT / "acquisition_2000/candidate_screening_batch04.json",
        [
            {
                "pageid": r["pageid"],
                "proposal_index": r["index"],
                "status": "accepted_with_manual_labels"
                if r["pageid"] in accepted
                else "pending",
                "review_level": "full_crop"
                if r["index"] in {13, 14, 35}
                else "thumbnail_only",
                "reason": (
                    "Nhật ký nhãn thủ công external_reviews_batch07.json."
                    if r["pageid"] in accepted
                    else "Người tiền cảnh đội mũ lưỡi trai nhưng còn xe/người nhỏ phía sau cần gán đầy đủ; yêu cầu ảnh native gặp HTTP 429, đã dừng tải. Không chấp nhận ảnh từ nhãn thiếu."
                    if r["index"] == 35
                    else "Chưa rà đầy đủ từng người/xe và nhãn trên toàn vùng cắt; không tính vào bộ sạch."
                ),
            }
            for r in proposals
        ],
    )
    print(json.dumps({"proposals": len(proposals), "accepted": len(records)}))


if __name__ == "__main__":
    main()
