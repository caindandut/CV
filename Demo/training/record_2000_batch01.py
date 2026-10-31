"""Manual decisions for R000..R095; all 24 sheets inspected visually."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

PENDING = {10, 28, 29, *range(41, 65), 67, 78, 84, 85, 87, 92}
REPLACE = {
    5: [0, 395, 166, 462, 286],
    6: [0, 380, 201, 458, 327],
    7: [0, 405, 174, 497, 325],
    8: [0, 396, 165, 496, 354],
    9: [0, 364, 173, 488, 404],
    11: [0, 586, 234, 729, 370],
    12: [0, 343, 228, 509, 350],
    13: [0, 222, 241, 303, 337],
    14: [0, 270, 213, 342, 303],
    16: [0, 440, 199, 491, 271],
    23: [0, 276, 243, 383, 436],
    24: [0, 251, 260, 371, 482],
    27: [0, 156, 210, 420, 442],
    30: [0, 481, 166, 910, 505],
    33: [0, 402, 230, 650, 434],
    34: [0, 497, 300, 602, 425],
    35: [0, 564, 310, 623, 411],
    36: [0, 430, 171, 535, 268],
    37: [0, 338, 175, 449, 268],
    38: [0, 251, 174, 319, 265],
    39: [0, 249, 181, 295, 260],
    40: [0, 634, 218, 777, 420],
    68: [0, 510, 198, 668, 460],
    71: [0, 355, 222, 439, 340],
    72: [0, 365, 191, 471, 326],
    73: [0, 443, 193, 545, 309],
    74: [0, 538, 205, 596, 295],
    75: [0, 534, 193, 598, 278],
    76: [0, 376, 188, 503, 278],
    82: [0, 144, 380, 302, 540],
    86: [0, 576, 337, 842, 538],
    88: [0, 515, 159, 725, 428],
    89: [0, 578, 172, 722, 401],
    90: [0, 554, 191, 667, 354],
    91: [0, 610, 203, 690, 311],
    93: [0, 481, 173, 554, 281],
    94: [0, 547, 207, 604, 311],
    95: [0, 472, 200, 578, 294],
}


def main():
    out = ROOT / "acquisition_2000/original_reviews_batch01.json"
    if out.exists():
        raise FileExistsError(
            "Manual review log already recorded; use a new batch for changes."
        )
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = []
    for number in range(96):
        proposal = proposals[number]
        row = original[proposal["index"]]
        boxes = copy.deepcopy(proposal["labels"])
        reason = f"Rà trực quan toàn ảnh R{number:03d}, từng người ngồi trên xe, mũ và xe đỗ."
        if number in PENDING:
            reason = (
                "Ảnh bị cắt nhiều phần đầu/người hoặc có nhiễu tăng cường mạnh; giữ cả ảnh để rà lại, "
                "không suy lớp từ frame liền kề và không tính vào mục tiêu 2.000."
            )
            decision = {"status": "pending", "reason": reason}
        else:
            if number in REPLACE:
                assert boxes[0][0] == 0
                boxes[0] = normalize_boxes([REPLACE[number]], [960, 540])[0]
                reason += " Nới box xe bao gương nhìn thấy."
            if number == 20:
                assert boxes[1][0] == 2
                del boxes[1]
                boxes += normalize_boxes([[0, 239, 229, 279, 294]], [960, 540])
                reason += (
                    " Xóa nhãn người lớp 2 trên xe đỗ; thêm box xe đang chạy bị bỏ sót."
                )
            if number in {21, 79, 80, 81}:
                index = {21: 2, 79: 4, 80: 4, 81: 2}[number]
                assert boxes[index][0] == 2
                del boxes[index]
                reason += " Xóa lớp 2 trên xe đỏ đỗ trống."
            if number == 80:
                boxes[0] = normalize_boxes([[0, 732, 185, 894, 404]], [960, 540])[0]
                reason += " Bổ sung gương vào box xe đỏ."
            if number == 81:
                boxes += normalize_boxes([[0, 904, 265, 960, 436]], [960, 540])
                reason += " Bổ sung xe nhìn thấy một phần ở mép phải; không có đầu/người thấy được trong ảnh tại xe này."
            reason += " Xe đỗ được đề xuất bằng phép chiếu hình học và kiểm tra bằng mắt; không dùng dự đoán detector."
            decision = {
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
                **decision,
            )
        )
    write_json(out, decisions)
    print(
        json.dumps(
            {
                "reviewed": len(decisions),
                "accepted": sum(r["status"] != "pending" for r in decisions),
            }
        )
    )


if __name__ == "__main__":
    main()
