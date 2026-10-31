"""Three separate motorcycles on Hang Dau Street: two helmeted riders and one bareheaded rider."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 147777952,
        "crop": [0, 530, 256, 850],
        "canvas": [256, 320],
        "boxes": [
            [0, 0, 46, 99, 150],
            [1, 12, 24, 76, 120],
            [0, 77, 70, 156, 145],
            [2, 76, 25, 140, 119],
            [0, 91, 68, 230, 170],
            [1, 123, 16, 210, 154],
        ],
        "location_evidence": "Commons Hàng Dầu Street, Hanoi, 20240204 1443 5822.jpg: danh mục Hàng Dầu Street, Hanoi, Scooters in Vietnam; tác giả Jakub Hałun, CC BY 4.0.",
        "reason": "Crop trái có ba rider trên ba xe khác nhau: người áo đen mũ xanh/trắng lớp 1; người áo xanh đậm tóc trần trên Vespa trắng lớp 2; người áo gile xanh mũ trắng lớp 1 trên xe số. Không gộp người tóc trần và người gile thành lái/ngồi sau cùng xe; kiểm tra nguyên ảnh và crop phóng to chỉ để phân biệt ba xe. Sáu box thủ công, từng người và xe một box; box xe bị cắt ở biên trái dừng tại crop. Crop loại hẳn nhóm xe/rider giữa và phải chưa đủ nhãn; bản train giữ pixel gốc, không nâng độ phân giải.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch24.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_14.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
