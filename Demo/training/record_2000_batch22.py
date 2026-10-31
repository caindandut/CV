"""Four manually reviewed frames; finish screening original proposals R560-R590."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    561: [
        [0, 163, 108, 455, 423],
        [0, 29, 108, 198, 289],
        [1, 182, 28, 421, 359],
        [1, 34, 48, 171, 263],
    ],
    562: [[0, 32, 79, 211, 340], [1, 18, 27, 211, 277]],
    577: [[0, 0, 313, 340, 605], [2, 75, 18, 340, 520]],
    578: [[0, 0, 117, 490, 720], [1, 0, 0, 490, 720]],
}
REASONS = {
    561: "Hai xe khác nhau: rider tiền cảnh áo thể dục mũ đen sọc trắng, ba lô đen có hình xanh; rider đi trước áo đỏ mũ trắng. Hai box lớp 1 và hai box xe. Không gộp hai xe thành một cặp driver/passenger.",
    562: "Một rider mũ đen cứng, áo thể dục, ba lô đỏ và tóc buộc sau mũ: lớp 1. Tóc dài lộ sau mũ không phải bằng chứng không đội mũ; người đi bộ ở toàn cảnh không phải rider.",
    577: "Một rider áo đen tóc trần, ba lô đen chữ nhật, lớp 2. Crop chỉ giữ phần người/xe ở tiền cảnh nhìn rõ, cắt hoàn toàn xe/rider nền chưa đủ nhãn; đối tượng bị cắt biên có box dừng đúng biên ảnh. Không lấy nhãn cũ hay dự đoán làm chuẩn.",
    578: "Một rider áo phao xanh lá, quần xanh, mũ cứng xám nhìn rõ phía trên mũ áo: lớp 1. Crop ở pixel gốc giữ trường hợp người/xe bị cắt biên; hai box có thể chồng nhiều do hình chiếu/che khuất, không phải nhãn trùng cùng lớp. Xe và người chỉ có phần nhìn thấy trong crop.",
}
PENDING = {
    563: "Mũ áo/đầu đen còn cần xác nhận và toàn cảnh còn nhiều xe đỗ chưa đủ nhãn; pending.",
    564: "Người tóc trần đang chuyển tư thế lên xe, chưa xác nhận đã ngồi trên yên; không đoán thành rider hoặc nền. Toàn ảnh pending.",
    585: "Đầu người trong mũ áo chưa xác nhận trạng thái mũ, không mặc định lớp 2 theo nhãn cũ; toàn ảnh pending.",
    589: "Người ngồi sau trùm mũ áo xanh như R489, đầu chưa xác nhận trạng thái mũ; toàn ảnh pending.",
}
for n in [
    560,
    565,
    566,
    567,
    568,
    569,
    570,
    571,
    572,
    573,
    574,
    575,
    576,
    579,
    580,
    581,
    583,
    584,
    587,
    588,
    590,
]:
    PENDING[n] = (
        "Cùng chuỗi người/cặp người đã được chọn ở lô này hoặc lô trước; không nhận thêm frame sát nhau chỉ để tăng số lượng. Toàn cảnh còn rider/xe nền chưa đủ nhãn, pending."
    )


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch22.json"
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
                ROOT / "acquisition_2000/original_crop_review/batch22/proposals.json"
            ).read_text(encoding="utf-8")
        )
    }
    records = []
    for n in range(560, 591):
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
                evidence=f"acquisition_2000/original_crop_review/batch22/R{n}.png",
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
