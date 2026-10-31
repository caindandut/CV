"""Nine manually inspected original-frame crops, one crop per source frame."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    280: [
        [0, 0, 192, 277, 509],
        [0, 223, 144, 445, 472],
        [2, 34, 25, 264, 499],
        [1, 226, 24, 442, 404],
    ],
    281: [
        [0, 0, 185, 250, 493],
        [0, 260, 145, 460, 446],
        [2, 25, 28, 234, 479],
        [1, 279, 31, 460, 425],
    ],
    282: [
        [0, 0, 146, 208, 441],
        [0, 184, 151, 375, 407],
        [2, 13, 0, 206, 432],
        [1, 191, 0, 375, 405],
    ],
    283: [
        [0, 0, 157, 203, 435],
        [0, 190, 156, 350, 401],
        [2, 0, 4, 215, 429],
        [1, 193, 0, 350, 398],
    ],
    284: [[0, 14, 152, 264, 464], [1, 34, 24, 255, 406]],
    285: [[0, 52, 163, 260, 424], [1, 82, 37, 261, 423]],
    287: [
        [0, 0, 151, 216, 504],
        [0, 227, 136, 463, 450],
        [0, 470, 132, 650, 414],
        [1, 0, 4, 207, 432],
        [1, 260, 26, 453, 437],
        [1, 464, 28, 650, 403],
    ],
    288: [
        [0, 16, 172, 275, 492],
        [0, 282, 147, 499, 446],
        [0, 550, 128, 725, 407],
        [1, 18, 19, 273, 481],
        [1, 303, 44, 489, 409],
        [1, 520, 31, 719, 399],
    ],
    289: [
        [0, 0, 132, 150, 407],
        [0, 74, 136, 319, 494],
        [1, 0, 43, 143, 402],
        [1, 81, 10, 320, 504],
    ],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch07.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    crops = {
        r["proposal"]: r
        for r in json.loads(
            (ROOT / "acquisition_2000/original_crop_review/proposals.json").read_text()
        )
    }
    decisions = []
    for number in range(280, 292):
        row = inventory[proposals[number]["index"]]
        base = {
            "id": row["id"],
            "proposal": number,
            "label_sha256": row["label_sha256"],
            "image_sha256": row["sha256"],
        }
        if number not in BOXES:
            decisions.append(
                dict(
                    **base,
                    status="pending",
                    reason="Ảnh toàn cảnh cần rà thêm rider bị che hoặc box xe thiếu gương; chưa đưa vào tập sử dụng.",
                    evidence=f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
                )
            )
            continue
        crop = crops[number]
        labels = normalize_boxes(BOXES[number], crop["canvas"])
        reason = (
            "Chỉ vùng cắt đã rà bằng mắt được chấp nhận; giữ ảnh toàn cảnh riêng, không bỏ box mơ hồ thành nền. "
            "Từng người ngồi xe có box riêng, mũ vỏ cứng thuộc lớp 1, tóc trần thấy rõ thuộc lớp 2; người trong chốt/đi bộ không gán rider. "
            "Các phần xe/người ngoài crop không xuất hiện trong dữ liệu train. Một frame chỉ được tính một lần; không dùng detector làm nhãn."
        )
        decisions.append(
            dict(
                **base,
                status="corrected",
                labels=labels,
                source_verified=True,
                reason=reason,
                crop=crop["crop"],
                canvas=crop["canvas"],
                source_canvas=[row["width"], row["height"]],
                evidence=f"acquisition_2000/original_crop_review/R{number}.png",
            )
        )
    write_json(output, decisions)
    print(json.dumps({"reviewed": len(decisions), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
