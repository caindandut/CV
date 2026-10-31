"""Whole-image manual review R200..R239, sheets 050..059."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

ACCEPTED = {203, 204, 205, 206, 208, 209, 210, 211, 212, 217, 218}
REPLACE = {
    203: [0, 544, 228, 649, 428],
    205: [0, 354, 183, 473, 281],
    206: [0, 340, 197, 438, 293],
    209: [0, 539, 225, 599, 306],
    210: [0, 497, 202, 544, 290],
    212: [0, 543, 172, 714, 438],
    217: [0, 526, 44, 623, 256],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch05.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = []
    for number in range(200, 240):
        proposal = proposals[number]
        row = inventory[proposal["index"]]
        if number not in ACCEPTED:
            reason = (
                "Cảnh cổng có rider/xe bị cắt ở mép, box thiếu hoặc trùng và mũ đen/áo trùm chưa xác nhận rõ; giữ cả ảnh để rà chi tiết."
                if number >= 221
                else "Đầu bị cắt nhiều hoặc nền ngày khác có xe đỗ chưa được gán đầy đủ; giữ cả ảnh chờ duyệt."
            )
            entry = {"status": "pending", "reason": reason}
        else:
            labels = copy.deepcopy(proposal["labels"])
            reason = "Rà toàn ảnh, người ngồi xe có tóc trần hoặc mũ bảo hiểm nhìn thấy trực tiếp; không suy nhãn từ frame khác."
            if number in REPLACE:
                assert labels[0][0] == 0
                labels[0] = normalize_boxes([REPLACE[number]], [960, 540])[0]
                reason += " Chỉnh box xe bao gương nhìn thấy."
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
