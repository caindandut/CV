"""One native crop of a helmeted rider on Mac Thai Tong Street, Hanoi."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 185835678,
        "crop": [500, 125, 925, 960],
        "canvas": [425, 835],
        "boxes": [[0, 24, 403, 425, 815], [1, 275, 7, 425, 450]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Crisis of dragons blood at Mạc Thái Tông Street 10-03-2026 E7.jpg: danh mục Mạc Thái Tông Street, Hanoi, Motorcycles in Hanoi; tác giả Baophucminh53G, CC BY-SA 4.0. Gom thận trọng vào nhóm khu vực Hà Nội TrungKinhHanoi đã khóa test; không khẳng định cùng camera/tác giả với các ảnh khác.",
        "reason": "Một rider đang ngồi trên Honda Vision, mũ đen cứng sọc trắng, lớp 1; box phủ phần đầu/thân nhìn thấy và dừng ở mép crop. Một box lớp 0 cho phần xe phía sau nhìn thấy. Crop kết thúc x=925 trước phần người áo nâu bên phải (bắt đầu khoảng x=932), bỏ hẳn các rider nền phía trái; không giữ người bị cắt đầu chưa rõ thành nền. Không vẽ thêm phần bị khuất, không nâng độ phân giải; ảnh toàn cảnh chưa đủ nhãn vẫn pending.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch19.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_08.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
