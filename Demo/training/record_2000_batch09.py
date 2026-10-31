"""Separate two seated riders; leave obscured heads and hoods pending."""

import json

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json

BOXES = {
    304: [
        [0, 40, 189, 265, 487],
        [1, 109, 47, 264, 407],
        [1, 17, 34, 238, 415],
    ],
    305: [[0, 48, 184, 265, 457], [1, 10, 25, 257, 447]],
    310: [[0, 76, 196, 340, 556], [1, 31, 17, 339, 460]],
    313: [[0, 29, 206, 263, 557], [1, 35, 28, 266, 448]],
}


def main():
    output = ROOT / "acquisition_2000/original_reviews_batch09.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    inventory = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    proposals = json.loads(
        (ROOT / "acquisition_2000/original_review/proposals.json").read_text()
    )
    crops = {
        r["proposal"]: r
        for r in json.loads(
            (
                ROOT / "acquisition_2000/original_crop_review/batch09/proposals.json"
            ).read_text()
        )
    }
    records = []
    for number in range(304, 316):
        row = inventory[proposals[number]["index"]]
        base = {
            "id": row["id"],
            "proposal": number,
            "image_sha256": row["sha256"],
            "label_sha256": row["label_sha256"],
        }
        if number not in BOXES:
            detail = (
                "R312 có hai người trên xe, đầu người phía trước bị che; không giữ box gộp của nhãn cũ."
                if number == 312
                else "R306 đã xem crop đầy đủ nhưng chi tiết trắng trên đầu chưa đủ rõ để phân biệt mũ bảo hiểm với phụ kiện vải."
                if number == 306
                else "Đã xem contact sheet, còn rider xa/che khuất hoặc trùm mũ áo chưa xác nhận được."
            )
            records.append(
                dict(
                    **base,
                    status="pending",
                    reason=detail
                    + " Giữ cả ảnh chờ duyệt, không dùng nhãn model làm chuẩn.",
                    evidence=(
                        f"acquisition_2000/original_crop_review/batch09/R{number}.png"
                        if number in crops
                        else f"acquisition_2000/original_review/sheet_{number // 4:03d}.jpg"
                    ),
                )
            )
            continue
        crop = crops[number]
        records.append(
            dict(
                **base,
                status="corrected",
                labels=normalize_boxes(BOXES[number], crop["canvas"]),
                source_verified=True,
                crop=crop["crop"],
                canvas=crop["canvas"],
                source_canvas=[row["width"], row["height"]],
                evidence=f"acquisition_2000/original_crop_review/batch09/R{number}.png",
                reason="Rà toàn vùng cắt bằng mắt. R304 sửa box gộp thành hai box rider lớp 1: lái mũ đen, ngồi sau mũ hồng. R305, R310 và R313 mỗi xe một người đội mũ vỏ cứng; box xe gồm phần gương/tay lái nhìn thấy. Không có đối tượng mơ hồ trong crop; giữ toàn cảnh gốc pending và chỉ tính một ảnh mỗi frame.",
            )
        )
    write_json(output, records)
    print(json.dumps({"reviewed": len(records), "accepted_crops": len(BOXES)}))


if __name__ == "__main__":
    main()
