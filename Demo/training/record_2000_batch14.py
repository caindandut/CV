"""Four source-frame crops; correct mixed helmet states and hanging helmet."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    369: [[0, 29, 133, 324, 450], [1, 43, 26, 289, 378]],
    373: [[0, 0, 151, 377, 552], [2, 4, 27, 372, 439]],
    377: [[0, 0, 370, 290, 635], [1, 0, 26, 290, 635]],
    383: [
        [0, 109, 214, 578, 665],
        [1, 250, 31, 611, 665],
        [2, 15, 29, 506, 613],
    ],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch14.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch14/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(368, 384):
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
                evidence=f"acquisition_2000/original_crop_review/batch14/R{n}.png",
                reason=(
                    "Đã mở toàn cảnh và crop ở pixel gốc. R369 một rider mũ xanh lớp 1; R373 người áo trắng tóc trần lớp 2, crop bỏ rider xa ở nền. R377 chỉ giữ rider mũ trắng và phần xe nhìn thấy, cắt hoàn toàn người áo be phía trước chưa chắc tư thế; không đổi người này thành nền không nhãn. R383 sửa nhãn sai/box gộp: lái mũ trắng lớp 1, ngồi sau tóc trần lớp 2, mỗi người một box; mũ đỏ treo trước xe không tạo rider lớp 1. Box xe gồm các phần gương/cargo nhìn thấy và cắt đúng biên ảnh. Toàn cảnh chưa đủ nhãn giữ pending, một crop mỗi frame; không dùng dự đoán làm nhãn chuẩn."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
