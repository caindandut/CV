"""Two Hanoi photographs, each with separate annotations for driver and passenger."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 63463138,
        "canvas": [1280, 960],
        "boxes": [
            [0, 884, 730, 1178, 958],
            [1, 972, 701, 1105, 915],
            [1, 1072, 684, 1178, 897],
        ],
        "location_evidence": "Commons Along the circular West Lake road 18 - Temple.jpg: mô tả West Lake, Hanoi; danh mục Motorcycles in Hanoi, Ỷ La Temple; tác giả Amenoc, CC0.",
        "reason": "Toàn ảnh có một xe máy chạy qua cổng đền, người lái áo đỏ mũ đỏ và người ngồi sau áo trắng mũ đen cứng bóng, mỗi người một box lớp 1. Một box xe lớp 0 chỉ phủ phần nhìn thấy, dừng ở biên dưới ảnh. Giữ chuyển động mờ nguyên bản, không làm nét. Ảnh 63463139 chụp cùng cặp người liền cảnh không nhận thêm để tránh lặp.",
    },
    {
        "pageid": 105761228,
        "crop": [580, 450, 836, 706],
        "canvas": [256, 256],
        "boxes": [
            [0, 148, 31, 244, 165],
            [1, 180, 4, 240, 129],
            [1, 140, 10, 214, 131],
        ],
        "location_evidence": "Commons Au Co Road, Tây Hồ District, Hanoi (7067709779).jpg: danh mục Motorcycles in Hanoi, Âu Cơ Street, Hanoi; tác giả David McKelvey, CC BY 2.0. Giữ nhóm tác giả test đã khóa.",
        "reason": "Crop pixel gốc giữ một xe máy và hai người: người lái áo đen đội mũ đen cứng, người ngồi sau áo khoác be đội mũ đỏ; hai box lớp 1 riêng và một box xe lớp 0. Crop bỏ hẳn các xe phía xa và xe chở hàng/ba bánh bên trái, không để các đối tượng chưa rõ thành nền thiếu nhãn. Không nâng độ phân giải hoặc dùng dự đoán làm nhãn.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch18.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_06.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
