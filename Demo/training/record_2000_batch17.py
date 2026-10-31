"""Three manually reviewed frames, including a bareheaded rider mislabelled as helmeted."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    433: [[0, 20, 227, 304, 489], [1, 24, 37, 255, 420]],
    441: [[0, 244, 244, 700, 635], [0, 298, 142, 357, 226], [2, 333, 45, 660, 516]],
    447: [[0, 0, 164, 198, 475], [0, 208, 76, 260, 212], [1, 0, 26, 198, 392]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch17.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch17/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(432, 448):
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
                evidence=f"acquisition_2000/original_crop_review/batch17/R{n}.png",
                reason=(
                    "Đã kiểm tra toàn cảnh/crop ở pixel gốc và lưới tọa độ; xem phóng to phần xe nền để đếm đủ. "
                    "R433 rider mũ đen đỏ lớp 1, hai người đi bộ không phải rider. "
                    "R441 sửa người lái áo thể dục tóc trần từ lớp 1 sang lớp 2; mũ trắng treo trước xe không tạo nhãn người đội mũ. "
                    "Người áo tay kem đứng cạnh giữ tay lên đuôi xe, hai chân chạm đất theo tư thế đứng, không ngồi sau; không gán rider. "
                    "R441 còn một xe đỗ đen thấy phần đuôi/biển số phía sau; có box riêng phần nhìn thấy. "
                    "R447 rider mũ đen lớp 1 và một xe đỗ bị cắt ở biên phải, hai xe hai box. "
                    "Box dừng đúng phần nhìn thấy và biên crop. Toàn cảnh thiếu nhãn vẫn pending, mỗi frame chỉ một crop. "
                    "Không dùng dự đoán làm nhãn chuẩn; frame sát nhau cùng người giữ pending."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
