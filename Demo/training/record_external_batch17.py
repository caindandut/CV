"""Two helmeted riders beside a bus on Vo Van Tan, Ho Chi Minh City."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 176070812,
        "crop": [0, 520, 256, 850],
        "canvas": [256, 330],
        "boxes": [
            [0, 0, 85, 27, 301],
            [1, 0, 37, 39, 166],
            [0, 6, 70, 115, 258],
            [1, 10, 33, 118, 233],
        ],
        "location_evidence": "Commons (VNM-Ho Chi Minh City) SaigonBus 50H-365.23 06 2025-10-04.jpg: mô tả Samco City Hi-Class H.76 CNG @ Võ Văn Tần; danh mục Buses in Ho Chi Minh City; tác giả S5A-0043, CC BY 4.0. Xem ảnh gốc và crop pixel gốc trước khi gán nhãn.",
        "reason": "Hai rider trên hai xe máy cạnh xe buýt, mũ xanh ở biên trái và mũ cam của người mặc áo chống nắng; từng người một box lớp 1. Xe ở biên trái bị cắt, box chỉ phủ phần nhìn thấy; xe tay ga trắng phía trước có box riêng lớp 0. Không gán tài xế xe buýt thành rider. Crop giữ nguyên pixel, không nhân bản hay làm nét.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch17.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_05.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
