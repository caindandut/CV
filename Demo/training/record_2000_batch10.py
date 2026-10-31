"""Manual foreground review of seven public-road frames; never copy COCO labels."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    317: [[0, 44, 184, 320, 493], [1, 12, 27, 338, 465]],
    319: [[0, 25, 163, 298, 700], [2, 33, 18, 293, 541]],
    320: [[0, 28, 260, 411, 676], [1, 27, 28, 398, 540]],
    321: [[0, 12, 118, 367, 559], [1, 37, 20, 349, 425]],
    322: [[0, 65, 242, 414, 649], [1, 32, 30, 416, 541]],
    324: [[0, 27, 134, 348, 570], [1, 25, 20, 335, 459]],
    326: [[0, 24, 108, 229, 489], [1, 25, 20, 218, 390]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch10.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inv = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    crops = {
        c["proposal"]: c
        for c in json.loads(
            (
                ROOT / "acquisition_2000/original_crop_review/batch10/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for number in range(316, 328):
        row = inv[proposals[number]["index"]]
        entry = {
            "id": row["id"],
            "proposal": number,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
            "status": "pending",
            "evidence": f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
            "reason": "Còn người/xe bị che hoặc xe đỗ ở nền chưa có nhãn đầy đủ; không đưa cả ảnh vào tập sử dụng.",
        }
        if number in crops:
            entry["evidence"] = (
                f"acquisition_2000/original_crop_review/batch10/R{number}.png"
            )
        if number == 325:
            entry["reason"] = (
                "Rà crop cho thấy hai người trên xe, nhãn cũ gộp một box. Tóc người ngồi sau rõ nhưng đầu người lái bị che nhiều; chưa xác nhận đủ cả hai nhãn nên giữ pending."
            )
        if number == 327:
            entry["reason"] = (
                "Rider tiền cảnh đội mũ rõ nhưng vùng cắt còn chi tiết xe/người rất nhỏ sát tường ở nền chưa đủ rõ; không bỏ qua nhãn nền."
            )
        if number in BOXES:
            c = crops[number]
            entry.update(
                status="corrected",
                source_verified=True,
                labels=normalize_boxes(BOXES[number], c["canvas"]),
                crop=c["crop"],
                canvas=c["canvas"],
                source_canvas=[row["width"], row["height"]],
                reason="Rà toàn crop ở pixel gốc, mỗi xe và người ngồi có box riêng. R319 tóc trần rõ lớp 2; các rider còn lại đội mũ vỏ cứng lớp 1. Sửa box xe để gồm gương/tay lái nhìn thấy. Người chống chân vẫn ngồi trên yên thuộc rider. Giữ toàn cảnh gốc pending; không nhân nhiều crop của một frame và không dùng dự đoán làm nhãn chuẩn.",
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"reviewed": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
