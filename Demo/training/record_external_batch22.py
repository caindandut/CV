"""Man on a motorcycle wearing a pith hat, plus two parked motorcycles in Bui Vien."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 149613588,
        "crop": [900, 530, 1170, 830],
        "canvas": [270, 300],
        "boxes": [
            [0, 84, 79, 248, 226],
            [2, 99, 31, 195, 191],
            [0, 4, 15, 88, 94],
            [0, 76, 17, 151, 102],
        ],
        "location_evidence": "Commons Bui Vien Street, Ho Chi Minh City, by day, 2023 (08).jpg: danh mục Bui Vien Street, Ho Chi Minh City, Motor scooters in Vietnam, ngày 02/11/2023; tác giả Bahnfrend, CC BY-SA 4.0.",
        "reason": "Người đàn ông áo tối màu, quần đùi xanh ngồi trên xe máy đen, đội mũ cối xanh (không phải mũ bảo hiểm), một box lớp 2. Một box riêng xe đang chạy, cùng hai box xe đỗ đỏ và xanh phủ áo bên trái/phía sau. Mũ trắng trên gương xe đỏ không tạo rider. Crop bỏ hẳn người/xe xa phía trái và người đi bộ cao tuổi phía phải; kiểm tra toàn crop pixel gốc, không dùng dự đoán làm nhãn hoặc tăng độ phân giải.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch22.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_11.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
