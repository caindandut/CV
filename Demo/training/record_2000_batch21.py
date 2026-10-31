"""Four manually reviewed frames, including a seated-to-standing hard negative."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    529: [[0, 22, 184, 440, 650], [1, 135, 33, 433, 521], [2, 0, 36, 390, 547]],
    535: [
        [0, 43, 210, 290, 500],
        [0, 234, 188, 349, 367],
        [1, 20, 35, 288, 408],
        [1, 201, 28, 315, 220],
    ],
    536: [[0, 48, 164, 260, 430], [1, 14, 34, 260, 416]],
    539: [[0, 76, 48, 260, 415], [0, 0, 344, 181, 445]],
}
REASONS = {
    529: "Sửa nhãn gộp/sai: người lái mũ trắng lớp 1, người ngồi sau tóc trần lớp 2, hai box trên một xe. Mũ đỏ treo bên xe không tạo thêm rider lớp 1. R528/R530 cùng cặp này không được cộng thêm.",
    535: "Hai người đội mũ đi trên hai xe: người mũ hoa ba lô xanh ở trước ảnh và người mũ đen trên xe đi trước. Hai xe, hai box rider lớp 1, không gộp thành cùng xe. Người đi bộ tiền cảnh ngoài crop, không gán rider.",
    536: "Người mũ đen sọc trắng vẫn ngồi trên yên xe, chống chân: lớp 1. Một xe trong crop. Chọn trạng thái còn ngồi xe để đối chiếu với R539 đã xuống xe, không lấy nhiều frame cùng tư thế chỉ để tăng số lượng.",
    539: "Sửa nhãn rider giả: người đội mũ đã xuống xe, đứng bên trái xem điện thoại, không ngồi trên yên. Không có lớp 1/2 trong crop này. Một xe đỗ nghiêng có chân chống và một phần xe khác ở biên dưới, hai box lớp 0. Giữ người đứng cạnh xe làm nền có kiểm chứng.",
}
PENDING = {
    538: "Mũ xanh cứng và quai đỏ của rider tiền cảnh nhìn rõ, nhãn cũ lớp 2 sai; nhưng toàn cảnh còn xe đỗ/rider nền chưa đủ nhãn nên cả ảnh pending.",
    540: "Đầu đen chưa phân biệt chắc mũ mềm/cứng như R425; không mặc định lớp 2 theo nhãn cũ. Toàn cảnh pending.",
    545: "Mũ áo che đầu người lái chưa rõ mũ bên trong; toàn cảnh pending.",
    553: "Người ngồi sau mũ áo xanh còn che đầu như R489; chưa xác nhận trạng thái mũ. Toàn cảnh pending.",
    555: "Mũ áo hồng che đầu, chưa xác nhận trạng thái mũ; toàn cảnh pending.",
    556: "Mũ áo hồng che đầu, chưa xác nhận trạng thái mũ; toàn cảnh pending.",
    557: "Cùng cặp R508, tay vẫn che đầu người ngồi sau; toàn cảnh pending.",
}
for n in [
    528,
    530,
    531,
    532,
    533,
    534,
    537,
    541,
    542,
    543,
    544,
    546,
    549,
    550,
    551,
    552,
    554,
    558,
    559,
]:
    PENDING[n] = (
        "Frame gần cùng người/cặp người đã được chọn ở lô này hoặc lô trước. Không cộng thêm chỉ để tăng số lượng; toàn cảnh còn xe/rider nền chưa đủ nhãn, pending."
    )


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch21.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch21/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(528, 560):
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
                evidence=f"acquisition_2000/original_crop_review/batch21/R{n}.png",
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
