"""Bicyclists are not motorcycle riders; review one parked scooter and one negative crop."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 105763301,
        "crop": [630, 340, 1280, 721],
        "canvas": [650, 381],
        "boxes": [[0, 399, 77, 526, 298]],
        "location_evidence": "Commons Cycling around West Lake, Tây Hồ District, Hanoi (6914729944).jpg: danh mục Bicycles in Hanoi, Tây Hồ; mô tả tour xe đạp Urban Discovery Hà Nội; David McKelvey, CC BY 2.0. Giữ tác giả trong test đã khóa.",
        "reason": "Một xe máy đỗ lớp 0 bên phải. Người đội mũ đỏ ở trái đi xe đạp có bàn đạp, xích và khung xe đạp rõ, không gán lớp rider dù đội mũ. Crop bỏ hẳn người/xe nhỏ bên trái và vật đỏ xa chưa xác nhận; giữ nguyên pixel, không dùng dự đoán làm nhãn.",
    },
    {
        "pageid": 29976063,
        "crop": [500, 400, 1000, 800],
        "canvas": [500, 400],
        "boxes": [],
        "location_evidence": "Commons Cycling in Xuan Mai Town.jpg: mô tả National Route 21, Xuân Mai, Hà Nội, Việt Nam; danh mục Cycling in Vietnam, Bicycles in Hanoi; Obakeneko, CC BY 3.0.",
        "reason": "Ảnh âm tính đã rà: hai người trùm áo chống nắng ngồi xe đạp, thấy rõ bàn đạp/xích/khung và bánh xe, không phải xe máy. Không có đối tượng của ba lớp trong crop nên nhãn rỗng có chủ ý. Crop bỏ hẳn vật xanh không rõ ở biên trái toàn ảnh; không biến đối tượng xe máy chưa rõ thành nền. Không nhân bản hay nâng độ phân giải.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch20.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_09.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
