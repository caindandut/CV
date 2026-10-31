"""Six manually reviewed frames, correcting a mixed helmet pair."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    484: [[0, 37, 128, 221, 318], [1, 48, 27, 186, 269]],
    491: [[0, 20, 90, 230, 290], [1, 66, 24, 200, 244]],
    493: [[0, 36, 83, 259, 351], [1, 121, 25, 234, 295], [2, 18, 20, 190, 318]],
    500: [[0, 42, 91, 149, 225], [2, 52, 31, 135, 193]],
    506: [[0, 44, 93, 275, 344], [1, 95, 26, 250, 281]],
    510: [[0, 24, 101, 239, 305], [1, 107, 32, 213, 254], [1, 40, 27, 157, 250]],
}
REASONS = {
    484: "Một rider mũ xanh cứng, áo caro, ba lô đen; lớp 1. Một xe nhìn rõ, không gán người đi bộ ở toàn cảnh thành rider.",
    491: "Một rider mũ đen cứng nhìn rõ trên mũ áo xanh nhạt, lớp 1. Chỉ một xe trong crop, toàn cảnh có rider xa nên giữ pending.",
    493: "Sửa passenger bị gán lớp 1: người lái mũ đen cứng lớp 1; người ngồi sau áo be tóc trần/búi tóc lớp 2. Hai người hai box, một xe. Không lấy tóc dài phía sau làm bằng chứng duy nhất; nhìn rõ phần tóc trần trên đỉnh đầu và búi tóc.",
    500: "Người lái áo đen tóc trần lớp 2, một xe xanh. Người áo đỏ đứng cạnh đường ở toàn cảnh không phải hành khách. Không gộp người đứng vào rider.",
    506: "Một rider mũ xanh đậm cứng, áo sọc ngắn tay, quần nâu: lớp 1. Chọn một frame của chuỗi R505–R507; không đếm các frame sát nhau cùng người.",
    510: "Hai người mũ xanh dương và xanh lá, áo xanh/caro: hai box lớp 1 trên một xe. Không gộp hai rider.",
}
PENDING = {
    489: "Người ngồi sau trùm mũ áo xanh, chưa xác nhận mũ cứng; cả ảnh pending, không mặc định lớp 1 theo nhãn cũ.",
    490: "Người áo sọc trùm mũ áo, chưa xác nhận phần mũ bên trong; giữ cả ảnh pending, không mặc định lớp 2 theo nhãn cũ.",
    495: "Cùng người trùm mũ áo sọc R490; trạng thái mũ chưa rõ, pending.",
    508: "Người lái tóc trần rõ nhưng người ngồi sau lấy tay che phần lớn đầu, chưa xác nhận đủ trạng thái mũ; cả ảnh pending, không chấp nhận chỉ riêng driver.",
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch19.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch19/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(480, 512):
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
                evidence=f"acquisition_2000/original_crop_review/batch19/R{n}.png",
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
