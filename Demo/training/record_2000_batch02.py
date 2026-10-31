"""Visual decisions R096..R111 after inspection of sheets 024..027."""

import copy
import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json


def main():
    out = ROOT / "acquisition_2000/original_reviews_batch02.json"
    if out.exists():
        raise FileExistsError("Manual review log exists")
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text(
            encoding="utf-8"
        )
    )
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    pending = {97, 100, 101, 103, 104, 107, 108, 109}
    replacements = {
        96: [0, 376, 196, 504, 289],
        99: [0, 263, 176, 381, 267],
        102: [0, 67, 215, 751, 540],
        110: [0, 341, 227, 532, 478],
        111: [0, 145, 211, 294, 399],
    }
    result = []
    for number in range(96, 112):
        proposal = proposals[number]
        row = original[proposal["index"]]
        reason = "Rà trực quan toàn ảnh, rider đầu trần/có mũ và các xe nhìn thấy; không dùng detector."
        if number in pending:
            reason = (
                "Cảnh nền đã thay đổi xe đỗ, phép chiếu cũ trúng người đứng/ô tô; cần rà nhãn lại cả ảnh."
                if number in {107, 108, 109}
                else "Đầu bị cắt ở mép trên; giữ cả ảnh, không suy lớp từ frame liền kề."
            )
            entry = {"status": "pending", "reason": reason}
        else:
            boxes = copy.deepcopy(proposal["labels"])
            if number in replacements:
                assert boxes[0][0] == 0
                boxes[0] = normalize_boxes([replacements[number]], [960, 540])[0]
                reason += " Chỉnh box xe bao gương nhìn thấy."
            entry = {
                "status": "corrected",
                "labels": boxes,
                "source_verified": True,
                "reason": reason,
            }
        result.append(
            dict(
                id=row["id"],
                proposal=number,
                label_sha256=row["label_sha256"],
                image_sha256=row["sha256"],
                evidence=f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg",
                **entry,
            )
        )
    write_json(out, result)
    print(
        json.dumps(
            {
                "reviewed": len(result),
                "accepted": sum(r["status"] != "pending" for r in result),
            }
        )
    )


if __name__ == "__main__":
    main()
