"""Five foreground crops from the train forecourt source, manually annotated."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    343: [[0, 21, 110, 229, 315], [1, 17, 15, 215, 271]],
    344: [[0, 85, 129, 429, 625], [1, 30, 38, 411, 487]],
    348: [[0, 12, 218, 388, 510], [1, 53, 70, 360, 473]],
    350: [[0, 0, 206, 311, 600], [1, 0, 115, 300, 504]],
    354: [[0, 13, 59, 211, 308], [1, 0, 5, 203, 248]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch12.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch12/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(342, 356):
        row = inv[proposals[n]["index"]]
        entry = {
            "id": row["id"],
            "proposal": n,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
            "status": "pending",
            "reason": "Đã xem contact sheet; toàn cảnh còn dãy xe đỗ thiếu nhãn hoặc rider bị che/chuyển động chưa rà đủ. Chưa đưa vào tập sử dụng.",
            "evidence": f"acquisition_2000/original_review/sheet_{n // 4:03d}.jpg",
        }
        if n in BOXES:
            c = crops[n]
            entry.update(
                status="corrected",
                source_verified=True,
                labels=normalize_boxes(BOXES[n], c["canvas"]),
                crop=c["crop"],
                canvas=c["canvas"],
                source_canvas=[row["width"], row["height"]],
                evidence=f"acquisition_2000/original_crop_review/batch12/R{n}.png",
                reason="Rà toàn crop ở pixel gốc: mỗi ảnh một xe và một người ngồi đội mũ vỏ cứng lớp 1, có box riêng. Giữ chuyển động mờ thật của nguồn, không làm nét/tăng cỡ để tính thêm ảnh. Người đứng cạnh tòa nhà không gán rider. Crop chỉ giữ vùng đã xác định, ảnh toàn cảnh vẫn pending; phần xe sát mép ảnh có box cắt đúng biên. Một crop mỗi frame, không dùng nhãn dự đoán.",
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
