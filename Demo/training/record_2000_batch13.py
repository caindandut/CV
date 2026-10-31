"""Four manually inspected train crops; split two bareheaded women in R358."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    356: [[0, 23, 100, 228, 339], [1, 20, 32, 219, 283]],
    358: [
        [0, 6, 100, 358, 508],
        [2, 33, 27, 280, 292],
        [2, 45, 61, 362, 426],
    ],
    363: [[0, 144, 218, 402, 550], [1, 40, 34, 398, 513]],
    366: [[0, 25, 114, 405, 570], [1, 42, 30, 405, 480]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch13.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch13/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(356, 368):
        row = inv[proposals[n]["index"]]
        entry = {
            "id": row["id"],
            "proposal": n,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
            "status": "pending",
            "reason": "Contact sheet còn xe đỗ thiếu nhãn hoặc rider phía trước bị che đầu; cần rà pixel gốc và gán đủ, chưa chấp nhận cả ảnh.",
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
                evidence=f"acquisition_2000/original_crop_review/batch13/R{n}.png",
                reason=(
                    "Đã mở toàn cảnh và crop ở pixel gốc. R358 có hai người: lái áo be tóc đen trần, ngồi sau áo đen tóc nâu búi trần; sửa box gộp thành hai box lớp 2 theo phần thân thấy được, một xe trắng lớp 0. Các crop khác mỗi ảnh một người đội mũ vỏ cứng lớp 1 và một xe lớp 0. R366 tóc nâu bay ra bên mũ của cùng một người, không tự nhân thành rider thứ hai. Crop gồm các phần gương nhìn thấy; vùng xe/người chưa rõ ngoài crop không xuất hiện trong ảnh sử dụng. Toàn cảnh giữ pending, một crop mỗi frame, không lấy dự đoán làm nhãn."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
