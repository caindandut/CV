"""Three helmeted seated riders on two motorcycles in Ho Chi Minh City."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 136201926,
        "crop": [735, 480, 1020, 990],
        "canvas": [285, 510],
        "boxes": [
            [0, 0, 102, 259, 450],
            [1, 40, 7, 238, 347],
            [1, 8, 1, 113, 340],
            [0, 163, 76, 285, 229],
            [1, 174, 21, 278, 179],
        ],
        "location_evidence": "Commons Nguyễn Thị Minh Khai Street, Ho Chi Minh City (25323591456), danh mục địa điểm và GPS 10.776975,106.692466; tác giả elyob, CC BY-SA 2.0.",
        "reason": "Kiểm tra ảnh toàn cảnh và crop pixel gốc: người lái mũ hồng và người ngồi sau mũ đen trên cùng xe tay ga vàng, mỗi người một box lớp 1; người lái áo đen mũ xanh trên xe đen phía sau là rider riêng lớp 1. Hai box xe lớp 0, dừng tại biên crop khi bị cắt. Crop bỏ toàn bộ nhóm xe nhỏ xa và các đầu bị che trong nền. Không dùng dự đoán làm nhãn; nguyên ảnh vẫn chưa đủ annotation.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch25.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_16.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
