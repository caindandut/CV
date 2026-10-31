"""Nine more original frames: full-crop visual review and manual labels."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    292: [[0, 27, 168, 249, 490], [2, 45, 22, 251, 416]],
    293: [[0, 18, 135, 279, 456], [1, 64, 35, 253, 393]],
    294: [[0, 24, 156, 279, 522], [1, 51, 15, 280, 431]],
    295: [[0, 31, 146, 285, 445], [1, 90, 22, 278, 378]],
    296: [[0, 0, 218, 316, 490], [1, 59, 49, 298, 475]],
    297: [[0, 39, 184, 242, 486], [1, 49, 77, 237, 474]],
    298: [[0, 47, 122, 249, 415], [1, 62, 15, 251, 408]],
    299: [
        [0, 35, 170, 298, 519],
        [0, 360, 169, 571, 466],
        [0, 599, 193, 739, 430],
        [1, 39, 36, 289, 510],
        [1, 395, 56, 564, 405],
        [1, 600, 78, 739, 414],
    ],
    302: [
        [0, 5, 149, 246, 465],
        [0, 139, 188, 374, 432],
        [0, 352, 135, 562, 403],
        [1, 35, 35, 237, 455],
        [1, 199, 55, 375, 423],
        [1, 399, 72, 561, 369],
    ],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch08.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text()
    )
    folder = ROOT / "acquisition_2000/original_crop_review/batch08"
    crops = {
        r["proposal"]: r for r in json.loads((folder / "proposals.json").read_text())
    }
    decisions = []
    for number in range(292, 304):
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
                    reason="Đã xem contact sheet nhưng còn người trùm đầu hoặc người bị che chưa xác định rõ mũ/ngồi xe; cần rà riêng, chưa chấp nhận nhãn.",
                    evidence=f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
                )
            )
            continue
        crop = crops[number]
        decisions.append(
            dict(
                **base,
                status="corrected",
                labels=normalize_boxes(BOXES[number], crop["canvas"]),
                source_verified=True,
                crop=crop["crop"],
                canvas=crop["canvas"],
                source_canvas=[row["width"], row["height"]],
                evidence=f"acquisition_2000/original_crop_review/batch08/R{number}.png",
                reason="Đã xem toàn vùng cắt ở độ phân giải gốc, gán thủ công từng xe và từng người ngồi. R292 tóc trần lớp 2; các rider còn lại có mũ bảo hiểm vỏ cứng lớp 1. Người đứng chốt/đi bộ không gán rider. Box xe gồm gương nhìn thấy và phần xe bị cắt ở mép ảnh. Giữ nguyên ảnh toàn cảnh có vùng chưa rõ ở pending; mỗi frame chỉ tính một lần, không dùng dự đoán model làm nhãn.",
            )
        )
    write_json(output, decisions)
    print(json.dumps({"reviewed": len(decisions), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
