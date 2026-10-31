"""Orange-clad scooter rider in a cloth brimmed hat on Hang But Street, Hanoi."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 120728081,
        "crop": [0, 400, 376, 903],
        "canvas": [376, 503],
        "boxes": [[0, 0, 124, 376, 448], [2, 47, 5, 295, 370]],
        "location_evidence": "Commons Hanoi (6021614463).jpg: mô tả Hanoi 2011, the old quarter; danh mục Hàng Bút Street, Hanoi; tác giả Xiaojun Deng, CC BY 2.0.",
        "reason": "Người mặc áo cam ngồi trên xe tay ga trắng, đội mũ vải trắng vành rộng, không phải mũ bảo hiểm, một box lớp 2. Một box xe lớp 0 gồm phần xe/túi hàng nhìn thấy và dừng ở biên crop. Crop kết thúc x=376 để bỏ hẳn xe/rider khác ở giữa ảnh (bắt đầu khoảng x=378); người đứng dưới ô phía sau không phải rider, xe bánh mảnh ở biên trái là xe đạp. Đã xem toàn ảnh và crop pixel gốc; giữ nhòe chuyển động và hạt ảnh phim, không làm nét/nâng độ phân giải hay dùng dự đoán làm nhãn.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch23.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_13.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
