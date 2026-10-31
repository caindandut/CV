"""Manually reviewed hard negative: standing vendor beside a pedal-powered cart."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 12402834,
        "crop": [70, 100, 880, 850],
        "canvas": [810, 750],
        "boxes": [],
        "location_evidence": "Commons A street seller in Ho Chi Minh City (Saigon), danh mục Street vendors in Ho Chi Minh City; tác giả Avenue, CC BY-SA 3.0.",
        "reason": "Ảnh âm tính khó đã kiểm tra nguyên ảnh và crop: người bán ổi đang đứng cạnh xe ba bánh đạp bằng bàn đạp, không phải xe máy và không phải rider. Mũ vải đen trên người đứng không tạo nhãn không đội mũ. Crop loại hoàn toàn hai xe máy đỗ phía phải; phần bánh/khung xe nhìn qua cửa kính phía trên là xe đạp, không gán xe máy. Không có đối tượng thuộc ba lớp trong crop, nhãn rỗng có chủ đích để giảm nhầm người bán hàng thành vi phạm. Một crop duy nhất cho nguồn, không nhân bản ảnh nền.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch27.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_17.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
