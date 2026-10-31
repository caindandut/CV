"""Six manually reviewed frames, including two driver/passenger pairs."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    455: [[0, 0, 137, 205, 414], [1, 0, 25, 202, 352]],
    460: [[0, 72, 90, 250, 240], [1, 117, 41, 232, 202]],
    463: [[0, 45, 68, 160, 198], [1, 63, 33, 128, 168], [1, 88, 33, 159, 190]],
    464: [[0, 36, 93, 192, 222], [1, 54, 43, 133, 184], [1, 85, 46, 180, 189]],
    476: [[0, 34, 87, 196, 225], [1, 49, 33, 152, 183]],
    478: [[0, 30, 60, 214, 204], [1, 72, 27, 164, 167]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch18.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch18/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(448, 480):
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
                evidence=f"acquisition_2000/original_crop_review/batch18/R{n}.png",
                reason=(
                    "Đã mở toàn cảnh và crop ở pixel gốc, kiểm tra lưới tọa độ; nhãn thủ công. "
                    "R455 một người mũ cứng đen be; R460 người lái mũ đen, người đứng cầm mũ đỏ không phải rider. "
                    "R463 tách nhãn gộp: người lái áo be mũ đen đỏ và người ngồi sau ba lô đỏ mũ đen trắng, hai box lớp 1. "
                    "R464 người lái áo xám mũ trắng và người ngồi sau áo tím mũ đen trắng, hai box lớp 1 trên một xe. "
                    "R476 mũ cứng xám, R478 mũ đen: lớp 1. Bao gồm gương nhìn thấy trong box xe, không tính mũ cầm tay/treo trên xe thành người. "
                    "Crop loại hẳn rider xa chưa rõ, không đưa họ thành nền thiếu nhãn; toàn cảnh vẫn pending. "
                    "R449 tuy thấy tóc trần tiền cảnh nhưng đầu/rider nền chưa xác nhận đủ, không chấp nhận toàn cảnh hoặc crop này. "
                    "Không dùng dự đoán model hay nhãn cũ làm nhãn đúng mặc định; không nhận thêm frame sát nhau cùng người chỉ để tăng số lượng."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
