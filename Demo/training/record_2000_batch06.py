"""Manual review R240..R279 with full-image evidence for gate corrections."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

ACCEPTED = {
    242,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    258,
    259,
    263,
    270,
    273,
    278,
    279,
}
DROP = {245: {4, 5}, 253: {6}, 254: {6}, 258: {6, 7}, 259: {5}, 273: {5}}
REPLACE = {
    242: {
        0: [0, 262, 161, 410, 354],
        1: [0, 49, 154, 235, 392],
        2: [0, 440, 130, 566, 326],
    },
    245: {0: [0, 402, 190, 516, 413], 1: [0, 505, 188, 636, 393]},
    246: {0: [0, 632, 199, 735, 395], 1: [0, 374, 211, 533, 438]},
    247: {0: [0, 96, 215, 286, 505], 1: [0, 577, 183, 705, 405]},
    248: {0: [0, 453, 221, 595, 414]},
    249: {0: [0, 224, 216, 393, 485], 1: [0, 650, 247, 770, 412]},
    252: {1: [0, 578, 252, 706, 462]},
    253: {
        0: [0, 0, 279, 199, 540],
        1: [0, 450, 273, 592, 467],
        2: [0, 679, 274, 765, 430],
    },
    258: {0: [0, 75, 285, 246, 505], 1: [0, 313, 214, 513, 526]},
    259: {0: [0, 171, 237, 436, 540], 1: [0, 670, 244, 864, 422]},
    263: {1: [0, 638, 238, 754, 438]},
    273: {
        0: [0, 0, 275, 206, 526],
        1: [0, 419, 244, 589, 450],
        2: [0, 642, 215, 747, 419],
    },
    278: {1: [0, 525, 266, 690, 443]},
    279: {0: [0, 272, 318, 469, 540], 1: [0, 654, 350, 721, 500]},
}
APPEND = {
    252: [[0, 839, 266, 935, 420]],
    247: [[0, 711, 181, 818, 396]],
    248: [[0, 640, 184, 760, 379]],
    249: [[0, 828, 223, 934, 396]],
    251: [[0, 856, 296, 947, 450]],
    255: [[0, 849, 289, 939, 431]],
    256: [[0, 694, 279, 819, 439]],
    263: [[0, 884, 269, 953, 411]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch06.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = []
    for number in range(240, 280):
        proposal = proposals[number]
        row = inventory[proposal["index"]]
        detail = ROOT / f"acquisition_2000/original_review/detail_R{number}.png"
        evidence = (
            detail.relative_to(ROOT).as_posix()
            if detail.exists()
            else f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg"
        )
        if number not in ACCEPTED:
            reason = "Có rider bị cắt đầu, áo trùm che mũ, box gộp người hoặc người/xe phía sau chưa xác nhận tư thế; giữ cả ảnh chờ duyệt."
            if 264 <= number <= 269 or number in {271, 272}:
                reason = "Có xe trắng/người phía sau thanh chắn chưa xác nhận đủ nhãn và tư thế; giữ cả ảnh chờ duyệt, không bỏ riêng đối tượng thành nền."
            entry = {"status": "pending", "reason": reason}
        else:
            labels = copy.deepcopy(proposal["labels"])
            reason = "Rà toàn ảnh cổng trường, đối chiếu mũ/tóc và tư thế ngồi; người đi bộ/đứng cạnh không thuộc lớp rider."
            for index, box in REPLACE.get(number, {}).items():
                assert labels[index][0] == box[0] == 0
                labels[index] = normalize_boxes([box], [960, 540])[0]
            if number in REPLACE:
                reason += " Chỉnh box xe bao gương nhìn thấy."
            if number in DROP:
                labels = [
                    box for index, box in enumerate(labels) if index not in DROP[number]
                ]
                reason += " Xóa box giả/trùng trên biển số, người đi bộ, áo hoặc bó hoa; giữ các người ngồi xe thực tế."
            if number in APPEND:
                labels += normalize_boxes(APPEND[number], [960, 540])
                reason += " Bổ sung xe máy nhìn thấy sau thanh chắn, kể cả phần bị che."
            if number == 270:
                assert labels[3][0] == 1
                labels[3][0] = 2
                reason += " Người bên phải đội mũ lưỡi trai vải có vành, không phải mũ bảo hiểm; đổi lớp 1 thành 2."
            if number in {258, 259}:
                reason += " Từng người lái/ngồi sau có box riêng; không gộp hai người vào một box."
            entry = {
                "status": "corrected",
                "labels": labels,
                "source_verified": True,
                "reason": reason,
            }
        decisions.append(
            dict(
                id=row["id"],
                proposal=number,
                label_sha256=row["label_sha256"],
                image_sha256=row["sha256"],
                evidence=evidence,
                **entry,
            )
        )
    write_json(output, decisions)
    print(json.dumps({"reviewed": len(decisions), "accepted": len(ACCEPTED)}))


if __name__ == "__main__":
    main()
