"""Manual decisions R112..R159 from sheets 028..039."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

ACCEPTED = {
    112,
    113,
    120,
    121,
    122,
    127,
    131,
    139,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    158,
    159,
}
REPLACE = {
    112: [0, 425, 135, 574, 421],
    113: [0, 389, 80, 500, 296],
    120: [0, 0, 196, 294, 540],
    121: [0, 405, 138, 675, 537],
    122: [0, 476, 122, 626, 399],
    127: [0, 532, 235, 760, 540],
    131: [0, 444, 304, 662, 540],
    148: [0, 507, 219, 673, 359],
    149: [0, 0, 251, 174, 517],
    150: [0, 447, 195, 492, 265],
    154: [0, 211, 428, 361, 526],
    158: [0, 272, 145, 378, 375],
    159: [0, 312, 173, 418, 329],
}


def main():
    out = ROOT / "acquisition_2000/original_reviews_batch03.json"
    if out.exists():
        raise FileExistsError("Review log exists")
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = []
    for number in range(112, 160):
        proposal = proposals[number]
        row = original[proposal["index"]]
        if number not in ACCEPTED:
            reason = (
                "Cảnh nền ngày khác: đề xuất xe đỗ cũ trúng ghế/ô tô, cần rà và bổ sung các xe đỗ thực tế."
                if number
                in {117, 118, 135, 136, 137, 141, 142, 143, 144, 145, 146, 147}
                else "Phần đầu/người bị cắt nhiều hoặc không xác nhận được tư thế ngồi xe; giữ cả ảnh chờ duyệt."
            )
            entry = {"status": "pending", "reason": reason}
        else:
            boxes = copy.deepcopy(proposal["labels"])
            reason = "Rà toàn ảnh: một người đang ngồi trên xe; phân lớp theo mũ trên đầu, không suy từ frame liền kề."
            if number in {112, 113, 120, 121, 122, 127, 131, 139}:
                boxes = boxes[: proposal["original_count"]]
                reason += " Bỏ đề xuất hình học của cảnh ngày khác, không dùng box trúng ghế/ô tô."
            if number in REPLACE:
                assert boxes[0][0] == 0
                boxes[0] = normalize_boxes([REPLACE[number]], [960, 540])[0]
                reason += " Chỉnh box xe bao phần gương và thân xe thấy được."
            if number == 153:
                assert boxes[2][0] in {1, 2}
                del boxes[2]
                reason += " Xóa nhãn rider trên xe đỏ đỗ trống; giữ box xe đỗ lớp 0 đã kiểm tra."
            if number == 155:
                assert boxes[1][0] == 0
                del boxes[1]
                reason += " Xóa box xe máy đặt lên ô tô bên trái."
            entry = {
                "status": "corrected",
                "labels": boxes,
                "source_verified": True,
                "reason": reason,
            }
        decisions.append(
            dict(
                id=row["id"],
                proposal=number,
                label_sha256=row["label_sha256"],
                image_sha256=row["sha256"],
                evidence=f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
                **entry,
            )
        )
    write_json(out, decisions)
    print(json.dumps({"reviewed": len(decisions), "accepted": len(ACCEPTED)}))


if __name__ == "__main__":
    main()
