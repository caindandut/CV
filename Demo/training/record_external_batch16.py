"""Manually reviewed parked motorcycles in Da Nang; hanging helmets are not riders."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 10854841,
        "canvas": [1280, 960],
        "boxes": [
            [0, 0, 548, 147, 803],
            [0, 94, 519, 282, 819],
            [0, 243, 532, 416, 800],
            [0, 528, 519, 689, 813],
            [0, 667, 524, 819, 798],
            [0, 781, 499, 975, 803],
            [0, 900, 529, 1128, 794],
            [0, 1050, 493, 1279, 795],
        ],
        "location_evidence": "Commons Motorbike wall Da Nang.JPG: mô tả bức tường có xe máy đỗ tại Đà Nẵng, Việt Nam; danh mục Motorcycles in Da Nang; Dragfyre, CC BY-SA 3.0. Giữ nhóm tác giả test đã khóa.",
        "reason": "Toàn ảnh có tám xe máy đỗ, từng xe một box lớp 0, kể cả xe bị cắt ở biên trái. Không có rider; mũ trắng trên xe trái, mũ bạc treo giữa dãy và mũ đen trên xe phải không phải người đội mũ. Kiểm tra toàn ảnh và lưới pixel gốc, không gán nhãn bằng dự đoán, không crop hay đổi kích thước.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch16.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_04.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
