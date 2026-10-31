"""One full Bui Vien photograph: Honda Wave beside a pedal bicycle."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 18088546,
        "canvas": [1280, 850],
        "boxes": [[0, 114, 350, 689, 799]],
        "location_evidence": "Commons Bui Vien St.ブイヴィエン通り Thành phố Hồ Chí Minh 城舗胡志明 ホーチミン DSCF1634.JPG: danh mục Bui Vien Street, Ho Chi Minh City; tác giả 松岡明芳, CC BY-SA 3.0.",
        "reason": "Toàn ảnh có một Honda Wave xanh đỗ, gồm gương và chân chống nhìn thấy, một box lớp 0. Xe vàng bên cạnh là xe đạp có bàn đạp/xích/khung; không gán xe máy. Không có rider; quần áo phơi và mũ để trên xe không phải người. Xe/ảnh không crop, không tăng độ phân giải. Không nhận thêm ảnh 18088547 cùng xe/góc gần liền nhau.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch21.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_10.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
