"""A helmeted seated rider crossing a flooded road in Binh Tan."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 126481406,
        "crop": [800, 250, 1130, 590],
        "canvas": [330, 340],
        "boxes": [[0, 24, 85, 319, 295], [1, 102, 20, 205, 246]],
        "location_evidence": "Commons Street flood in Saigon (10729260963), mô tả Bình Tân District, Ho Chi Minh City, danh mục Kinh Dương Vương Road; tác giả Bem photography:, CC BY 2.0.",
        "reason": "Kiểm tra nguyên ảnh và crop pixel gốc: một người áo xanh đậm đang ngồi lái xe máy qua đường ngập, mũ bảo hiểm trắng nhìn rõ. Một box người lớp 1 và một box xe lớp 0, phần bánh chìm chỉ gán vùng nhìn thấy. Crop loại toàn bộ xe chở hàng, rider tóc trần và nhóm xa chưa đủ nhãn phía trái. Người trong ô tô/xe khách và phản chiếu kính không thuộc lớp rider. Không dùng dự đoán model làm nhãn.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch26.json"
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
