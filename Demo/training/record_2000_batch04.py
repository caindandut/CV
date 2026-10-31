"""Manual whole-image review R160..R199, contact sheets 040..049."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

ACCEPTED = (
    set(range(160, 166))
    | set(range(170, 176))
    | {177, 178, 179, 181, 188}
    | set(range(191, 200))
)
REPLACE = {
    160: [0, 458, 143, 565, 277],
    161: [0, 518, 161, 600, 257],
    162: [0, 478, 311, 569, 404],
    163: [0, 245, 177, 296, 250],
    164: [0, 265, 166, 305, 242],
    165: [0, 279, 164, 337, 235],
    170: [0, 317, 168, 456, 332],
    171: [0, 390, 166, 501, 317],
    172: [0, 479, 198, 568, 302],
    174: [0, 493, 206, 590, 281],
    175: [0, 271, 164, 354, 279],
    177: [0, 613, 210, 710, 338],
    178: [0, 255, 207, 320, 340],
    191: [0, 688, 418, 779, 540],
    192: [0, 406, 150, 501, 335],
    193: [0, 214, 217, 276, 325],
    194: [0, 418, 198, 485, 275],
    195: [0, 419, 209, 473, 282],
    196: [0, 249, 210, 300, 294],
    199: [0, 559, 319, 614, 409],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch04.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = []
    for number in range(160, 200):
        proposal = proposals[number]
        row = inventory[proposal["index"]]
        if number not in ACCEPTED:
            entry = {
                "status": "pending",
                "reason": "Đầu bị cắt khỏi khung hoặc cảnh ngày khác có xe đỗ chưa được gán đầy đủ; giữ cả ảnh chờ duyệt.",
            }
        else:
            labels = copy.deepcopy(proposal["labels"])
            reason = "Kiểm tra toàn ảnh: một người ngồi xe, mũ/tóc thấy rõ; xe đỗ có box riêng, người đi bộ không gán rider."
            if number == 164:
                # The parked red bike has both an old bike box and a false rider.
                labels = [
                    box for index, box in enumerate(labels) if index not in {1, 3}
                ]
                reason += " Xóa rider giả và box xe trùng trên xe đỏ đỗ trống; giữ một box xe đã đối chiếu."
            if number == 175:
                assert labels[2][0] == 2
                del labels[2]
                reason += " Xóa box rider thứ hai chồng lên cùng một người; ảnh chỉ có một người đang ngồi xe."
            if number in REPLACE:
                assert labels[0][0] == 0
                labels[0] = normalize_boxes([REPLACE[number]], [960, 540])[0]
                reason += " Chỉnh box xe gồm phần gương nhìn thấy."
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
                evidence=f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
                **entry,
            )
        )
    write_json(output, decisions)
    print(json.dumps({"reviewed": len(decisions), "accepted": len(ACCEPTED)}))


if __name__ == "__main__":
    main()
