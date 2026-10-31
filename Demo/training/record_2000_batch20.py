"""Seven manually reviewed frames, separating two successive motorcycles."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    514: [
        [0, 0, 85, 170, 354],
        [0, 0, 82, 34, 199],
        [0, 143, 85, 226, 193],
        [1, 8, 26, 170, 294],
    ],
    516: [[0, 35, 140, 174, 295], [2, 27, 36, 177, 276]],
    517: [[0, 17, 126, 265, 438], [1, 40, 33, 252, 403]],
    518: [[0, 19, 120, 267, 440], [1, 25, 29, 266, 363]],
    521: [
        [0, 0, 132, 236, 463],
        [0, 191, 150, 344, 310],
        [1, 20, 26, 236, 390],
        [1, 201, 64, 342, 278],
    ],
    523: [
        [0, 35, 122, 129, 269],
        [0, 39, 156, 309, 379],
        [1, 34, 34, 143, 244],
        [2, 38, 19, 244, 318],
    ],
    524: [[0, 28, 107, 194, 310], [1, 29, 28, 189, 252]],
}
REASONS = {
    514: "Một rider mũ đen có logo trắng, lớp 1. Ba xe: xe đang chạy và hai xe đỗ chỉ thấy một phần ở hai biên crop, có box riêng. Không gán người trong poster là rider.",
    516: "Một rider áo xanh xám, ba lô xanh, tóc đen không có vỏ mũ cứng, lớp 2; mũ áo màu kem nằm dưới đầu. Không gán các phụ nữ đi bộ ở toàn cảnh thành rider.",
    517: "Một rider áo đỏ, mũ trắng sọc hồng cứng, lớp 1. Crop bỏ hẳn rider xa phía sau trụ cổng chưa đủ nhãn.",
    518: "Một rider áo khoác xám, mũ xanh đậm cứng, lớp 1; một xe. Người bảo vệ đứng gần cổng không phải rider.",
    521: "Hai người trên hai xe khác nhau, không phải driver/passenger cùng xe: người áo kem mũ đen và người áo hồng mũ đỏ, hai box lớp 1 và hai box xe. Crop bỏ người chạy xe trắng ở nền chưa rõ.",
    523: "Sửa người áo trắng tóc trần bị gán lớp 1 thành lớp 2. Người áo kem mũ đen đi trên xe phía trước là rider khác, lớp 1; hai xe khác nhau có hai box riêng, không gộp thành một xe chở hai người. Người đi bộ áo đỏ không phải rider, nằm ngoài crop.",
    524: "Một rider áo thể dục mũ xanh lá cứng, lớp 1. Crop bỏ rider áo hồng rất xa ở lối ra; một xe trong crop.",
}
PENDING = {
    527: "Còn người có tóc dài phía trước rider tiền cảnh chưa xác nhận chắc đang ngồi xe hay đi bộ; toàn ảnh pending, không đoán thành passenger hoặc nền không nhãn.",
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch20.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch20/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(512, 528):
        row = inv[proposals[n]["index"]]
        entry = {
            "id": row["id"],
            "proposal": n,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
            "status": "pending",
            "reason": PENDING.get(
                n,
                "Contact sheet còn xe đỗ/rider nhỏ chưa có nhãn đủ hoặc frame lặp cùng người. Chưa rà đầy đủ nhãn ở pixel gốc; không chấp nhận thêm chỉ để tăng số lượng.",
            ),
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
                evidence=f"acquisition_2000/original_crop_review/batch20/R{n}.png",
                reason=(
                    REASONS[n]
                    + " Đã kiểm tra toàn cảnh, crop và lưới tọa độ ở pixel gốc. Box gồm gương/phần xe nhìn thấy, dừng đúng biên crop. Crop bỏ hẳn đối tượng xa chưa rõ; toàn cảnh chưa đủ nhãn vẫn pending. Một crop mỗi frame; nhãn thủ công, không dùng dự đoán làm nhãn chuẩn."
                ),
            )
        records.append(entry)
    write_json(output, records)
    print(json.dumps({"screened": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
