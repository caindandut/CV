"""Manually reviewed Hue cargo rider, retaining the locked author test group."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 16,
        "pageid": 199856776,
        "crop": [528, 95, 810, 917],
        "canvas": [282, 822],
        "boxes": [[0, 0, 183, 282, 790], [1, 0, 39, 193, 387]],
        "location_evidence": "Commons A funny way of carrying ducks (1).jpg: mô tả Ducks being transported on a motorcycle in Huế (Vietnam); tác giả Gzzz, CC BY-SA 4.0. Giữ cùng nhóm tác giả test đã khóa, không chuyển sang train.",
        "reason": "Rider ngồi trên xe, mũ cứng đen trên đầu: lớp 1. Một xe máy chở vịt, box xe gồm gương và giỏ hàng gắn xe; vịt không phải người. Crop bỏ hẳn giao thông xa chưa rõ ở bên trái; đầu người rõ, phần thân dưới bị hàng che. Đã kiểm tra toàn crop PNG 282x822 ở pixel gốc, không nâng độ phân giải.",
    }
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch12.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch09/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
