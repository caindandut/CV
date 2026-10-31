"""Two Hanoi night/sidewalk cases, manually distinguished by seating position."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 1,
        "pageid": 184899753,
        "crop": [570, 524, 827, 875],
        "canvas": [257, 351],
        "boxes": [[0, 48, 105, 188, 331], [2, 72, 14, 210, 300]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Night stroll in Alley 110 of Trần Duy Hưng Street 23-02-2026 E1.jpg; mô tả Trung Kính Lake by night, danh mục Trần Duy Hưng Street, Hanoi, Hotels of Yên Hòa; Bohdan Phúc Đan, CC BY-SA 4.0. Giữ cùng nhóm bối cảnh Trung Kính/Yên Hòa test đã khóa.",
        "reason": "Một người tóc trần đang ngồi trên yên xe tay ga dừng trước khách sạn, chống chân phải: lớp 2, một box xe lớp 0. Nhìn rõ hông trên yên và đùi gập về phía trước, không gán chỉ dựa vào đứng gần xe. Nền không có rider khác; giữ ánh sáng đêm gốc, không làm nét hay đổi màu.",
    },
    {
        "proposal_index": 2,
        "pageid": 184976729,
        "crop": [76, 550, 431, 821],
        "canvas": [355, 271],
        "boxes": [[0, 79, 0, 337, 181]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Safe on the sidewalk of Vũ Phạm Hàm Street 25-02-2026.jpg; mô tả và danh mục Vũ Phạm Hàm Street, Hanoi, 2026 in Hanoi; Bohdan Phúc Đan, CC BY-SA 4.0. Giữ cùng nhóm nguồn/bối cảnh test TrungKinhHanoi.",
        "reason": "Người giao hàng ngồi trên bờ vỉa hè xem điện thoại, không ngồi trên xe. Xe tay ga đỗ có chân chống, yên không có người; mũ xanh đặt trên xe không phải đầu rider. Chỉ một box xe lớp 0, không gán lớp 2 cho người tóc trần ngồi bờ hè. Crop bỏ hẳn dải xe/rider xa trên đường ở phía trên, giữ pixel gốc; không tạo nền thiếu nhãn.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch11.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch08/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
