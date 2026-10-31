"""Manual labels from five inspected COCO crop proposals; predictions are not GT."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 1,
        "pageid": 3101855,
        "crop": [0, 0, 524, 591],
        "canvas": [524, 591],
        "boxes": [[0, 0, 66, 413, 537]],
        "location_evidence": "Commons: Hanoi cycle rickshaw.jpg, danh mục Cycle rickshaws in Hanoi; không suy quốc gia từ nơi ở Dhaka của tác giả.",
        "reason": "Một xe máy đỗ, chân người đứng phía sau và ghế xích lô ở mép dưới không phải rider. COCO ghép người đứng với xe; nhãn thủ công chỉ có lớp 0.",
    },
    {
        "proposal_index": 7,
        "pageid": 16453438,
        "crop": [355, 547, 690, 960],
        "canvas": [335, 413],
        "boxes": [[0, 0, 138, 335, 413], [1, 94, 34, 312, 413], [1, 0, 37, 132, 413]],
        "location_evidence": "Commons ghi Nguyen Thi Minh Khai Rd., District 3, Ho Chi Minh City; danh mục Motorcycles in Ho Chi Minh City.",
        "reason": "Một xe có hai người ngồi, cả hai đội mũ bảo hiểm; mũ lưỡi trai nằm dưới mũ của người sau không làm đổi lớp 1. Thu hẹp vùng đề xuất để loại hẳn bánh xe của rider khác ở góc trên phải. Box từng người gán thủ công.",
    },
    {
        "proposal_index": 8,
        "pageid": 16493494,
        "crop": [427, 246, 842, 627],
        "canvas": [415, 381],
        "boxes": [
            [0, 119, 82, 395, 299],
            [0, 89, 152, 383, 368],
            [1, 174, 131, 326, 334],
        ],
        "location_evidence": "Commons CSGT.jpg mô tả Cảnh sát giao thông đang trao đổi công việc; danh mục Traffic police of Vietnam và Streets in Ho Chi Minh City.",
        "reason": "Hai xe máy cảnh sát, một người đang ngồi trên xe đội mũ lớp 1. Hai cảnh sát còn lại đứng trên vỉa hè dù đội mũ vẫn không gán rider. Không biến dự đoán person của COCO thành nhãn rider.",
    },
    {
        "proposal_index": 14,
        "pageid": 18088523,
        "crop": [731, 0, 988, 611],
        "canvas": [257, 611],
        "boxes": [[0, 0, 171, 224, 572]],
        "location_evidence": "Tiêu đề, mô tả và danh mục Commons ghi Bui Vien Street, Ho Chi Minh City.",
        "reason": "Phần xe máy đỗ nhìn rõ gồm đầu xe, giảm xóc và bánh trước. Người đàn ông cởi trần đứng phía sau không ngồi xe, không gán lớp 2.",
    },
    {
        "proposal_index": 22,
        "pageid": 18088550,
        "crop": [0, 956, 1001, 1928],
        "canvas": [1001, 972],
        "boxes": [[0, 0, 233, 819, 926]],
        "location_evidence": "Tiêu đề, mô tả và danh mục Commons ghi Bui Vien Street, Ho Chi Minh City, Women of Vietnam in 2011.",
        "reason": "Một xe máy Honda xanh đỗ bị người đi bộ che một phần; người đội nón lá đi ngang không gán rider. Xe đạp vàng phía sau không thuộc lớp 0; mũ treo ở xe không phải người.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch04.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch01/run.json",
                **decision,
            )
        )
    write_json(output, records)
    proposals = json.loads(
        (
            ROOT / "acquisition_2000/coco_crop_proposals/batch01/proposals.json"
        ).read_text()
    )
    accepted = {r["pageid"] for r in records}
    write_json(
        ROOT / "acquisition_2000/candidate_screening_batch02.json",
        [
            {
                "pageid": r["pageid"],
                "proposal_index": r["index"],
                "status": "accepted_with_manual_labels"
                if r["pageid"] in accepted
                else "pending",
                "review_level": "full_crop"
                if r["pageid"] in accepted
                else "thumbnail_only",
                "reason": "Nhãn được gán thủ công trong external_reviews_batch04.json."
                if r["pageid"] in accepted
                else "Chưa gán đầy đủ nhãn và xác minh nguồn; đề xuất COCO có thể chọn xe đạp/người đứng hoặc cắt rider khác. Không đưa vào train.",
            }
            for r in proposals
        ],
    )
    print(
        json.dumps(
            {
                "reviewed_proposals": len(proposals),
                "accepted_with_manual_labels": len(records),
            }
        )
    )


if __name__ == "__main__":
    main()
