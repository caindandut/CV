"""Four manually reviewed helmet riders; unresolved full frames stay pending."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    408: [[0, 0, 150, 405, 650], [1, 0, 19, 384, 572]],
    414: [[0, 23, 118, 178, 257], [1, 30, 22, 175, 225]],
    426: [[0, 0, 90, 128, 322], [1, 0, 22, 136, 297]],
    429: [[0, 0, 190, 183, 372], [1, 0, 24, 152, 308]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch16.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch16/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(400, 432):
        row = inv[proposals[n]["index"]]
        entry = {
            "id": row["id"],
            "proposal": n,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
            "status": "pending",
            "reason": "Contact sheet còn xe đỗ/rider nhỏ chưa có nhãn đủ. Các frame sát nhau của cùng cặp người chưa được chấp nhận thêm chỉ để tăng số lượng; cần rà đầy đủ riêng.",
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
                evidence=f"acquisition_2000/original_crop_review/batch16/R{n}.png",
                reason=(
                    "Đã kiểm tra toàn cảnh, crop và lưới tọa độ ở pixel gốc. "
                    "R408 rider mũ tím, R414 rider mũ trắng cứng có vành, R426 mũ cứng có tai trang trí, R429 mũ đen: lớp 1. "
                    "Crop loại hẳn xe đỗ/rider nền chưa đủ nhãn; box giới hạn ở phần nhìn thấy, kể cả đối tượng bị cắt biên. "
                    "R426 người bảo vệ đi bộ không phải rider. Mỗi người ngồi xe một box, mỗi xe một box gồm gương nhìn thấy. "
                    "Các frame sát nhau cùng người không được cộng thêm; R425 đầu đen chưa rõ mũ mềm/cứng giữ pending. "
                    "Toàn cảnh chưa đủ nhãn vẫn pending. Nhãn gán thủ công, không dùng dự đoán làm nhãn chuẩn."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
