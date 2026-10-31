"""Three visually reviewed VN photos, including four clearly unhelmeted riders."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 18,
        "pageid": 61901825,
        "crop": [169, 88, 1072, 853],
        "canvas": [903, 765],
        "boxes": [[0, 34, 215, 864, 758]],
        "location_evidence": "Commons Ho Chi Minh City, Vietnam (Unsplash 5oRaXQqV6CM).jpg; mô tả Ho Chi Minh City, Vietnam và danh mục Motorcycles in Ho Chi Minh City. Tao Xanh Kim, CC0 theo metadata Commons.",
        "reason": "Một xe Vespa trắng đỗ trên chân chống, một box lớp 0. Người trong quán ngồi ghế riêng, không ngồi trên xe nên không gán rider. Toàn crop đã xem ở pixel gốc; không coi người không đội mũ bất kỳ là lớp 2.",
    },
    {
        "proposal_index": 21,
        "pageid": 63784305,
        "crop": [55, 480, 570, 815],
        "canvas": [515, 335],
        "boxes": [
            [0, 0, 128, 220, 313],
            [0, 65, 86, 207, 198],
            [0, 319, 88, 508, 205],
            [2, 0, 50, 144, 287],
            [2, 54, 35, 180, 198],
            [2, 97, 34, 184, 169],
            [2, 342, 24, 478, 194],
        ],
        "location_evidence": "Commons RMK-BRJ HQ Saigon.jpg: mô tả trụ sở tại 2 Duy Tân, nay là Phạm Ngọc Thạch, Quận 1, Sài Gòn, tháng 1/1972; tác giả Overlain, CC BY-SA 4.0. Đây là ảnh lịch sử, không giả làm ảnh giao thông hiện đại.",
        "reason": "Mở ảnh toàn cảnh để kiểm tra các đầu người bị crop đề xuất ban đầu cắt mất. Crop mới có ba xe máy và bốn rider: người tiền cảnh mũ vải mềm, hai người trên xe phía sau đội mũ vải/beret, người xe đỏ tóc trần; cả bốn không có mũ bảo hiểm và có box riêng lớp 2. Người đạp xe nhỏ ở nền không gán motorcycle/rider. Crop loại phần người không xác định ở mép trái nguồn; xe tiền cảnh bị cắt mép trái nhưng đầu rider còn rõ. Nhãn thủ công, không từ COCO. Giữ màu và pixel ảnh lịch sử đúng nguồn.",
    },
    {
        "proposal_index": 26,
        "pageid": 69181468,
        "crop": [317, 57, 1163, 674],
        "canvas": [846, 617],
        "boxes": [[0, 265, 30, 827, 539]],
        "location_evidence": "Commons Street vendor (4849545362).jpg; mô tả Old quarter, Hanoi và danh mục Hàng Bút Street, Hanoi. Caitriana Nicholson, CC BY-SA 2.0; giữ cùng nhóm tác giả test đã khóa.",
        "reason": "Một xe tay ga đỗ, mũ bảo hiểm đặt trên tay lái không phải đầu người. Người bán hàng đội nón lá đứng/đi bộ gánh hàng, không gán rider. Đã xem cả ảnh toàn cảnh để xác nhận phía phải chỉ là hàng trong cửa tiệm, không có xe máy thứ hai.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch08.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    records = []
    for decision in DECISIONS:
        source = ROOT / f"acquisition_2000/images/commons_{decision['pageid']}.jpg"
        image, _ = prepare_review_image(
            source,
            decision,
            ROOT / "acquisition_2000/curated_crops",
            decision["pageid"],
        )
        records.append(
            dict(
                country="VN",
                visual_reviewed=True,
                image_sha256=digest(source),
                evidence=image.relative_to(ROOT).as_posix(),
                proposal_run="acquisition_2000/coco_crop_proposals/batch05/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
