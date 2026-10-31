"""Five manually reviewed source frames; separate bare head from hard helmets."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    386: [[0, 24, 103, 404, 545], [1, 83, 27, 380, 431]],
    389: [[0, 0, 236, 372, 540], [2, 0, 23, 370, 540]],
    391: [[0, 0, 110, 107, 291], [1, 0, 20, 118, 257]],
    394: [[0, 43, 219, 211, 405], [1, 37, 92, 209, 346]],
    398: [[0, 65, 165, 457, 610], [1, 37, 27, 454, 558]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch15.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch15/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(384, 400):
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
                evidence=f"acquisition_2000/original_crop_review/batch15/R{n}.png",
                reason=(
                    "Kiểm tra crop ở pixel gốc, mỗi người ngồi xe một box. "
                    "R386 mũ đen hoa, R391 mũ đen, R394 mũ đen đỏ và R398 mũ hoa trắng: lớp 1. "
                    "R389 thấy tóc trần: lớp 2. R391 cắt bỏ toàn bộ dãy xe nền; người bảo vệ đi bộ không phải rider. "
                    "Box xe gồm gương nhìn thấy, các phần bị cắt dừng đúng biên crop. "
                    "Các frame sát nhau cùng người không được chấp nhận thêm chỉ để tăng số lượng; toàn cảnh thiếu nhãn vẫn pending. "
                    "Nhãn thủ công, không lấy dự đoán model làm nhãn chuẩn."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
