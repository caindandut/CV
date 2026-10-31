"""Six manually reviewed HCM photos, including hanging helmet and night lighting."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 0,
        "pageid": 147357850,
        "crop": [0, 618, 755, 1395],
        "canvas": [755, 777],
        "boxes": [
            [0, 0, 211, 688, 705],
            [1, 167, 37, 452, 743],
            [1, 354, 70, 638, 589],
        ],
        "location_evidence": "Commons BHHA ng27th8n2022 (xe máy dừng chờ bờ rào miếu Ngũ hành Nương nương) (2).jpg, mô tả Road transport in Ho Chi Minh City; Phương Huy, CC0.",
        "reason": "Hai người ngồi trên xe tay ga, lái mũ đen có kính, ngồi sau mũ vàng: hai box lớp 1, một xe lớp 0. Người lái chống chân nhưng hông vẫn trên yên. Bánh xe bên phải thuộc xe đẩy quầy bánh mì, không gán motorcycle. Toàn crop đã xem trực quan, không nhận dự đoán COCO làm nhãn.",
    },
    {
        "proposal_index": 1,
        "pageid": 147357876,
        "crop": [382, 844, 862, 1479],
        "canvas": [480, 635],
        "boxes": [
            [0, 25, 99, 457, 602],
            [1, 142, 31, 388, 487],
            [0, 331, 153, 414, 363],
            [1, 244, 18, 394, 234],
        ],
        "location_evidence": "Commons SG ng13th11n2022 (chạy xe máy trên đường).jpg, mô tả Road transport in Ho Chi Minh City; Phương Huy, CC0.",
        "reason": "Xe SH trắng tiền cảnh có người lái đội mũ đen lớp 1; phía sau là xe khác và người lái áo đen quay lưng đội mũ đỏ lớp 1. Gán hai xe, hai người riêng, box người sau theo phần thân nhìn thấy khi chân bị xe tiền cảnh che. Hình người trên tường bên phải không phải rider thật.",
    },
    {
        "proposal_index": 2,
        "pageid": 147357877,
        "crop": [125, 995, 569, 1526],
        "canvas": [444, 531],
        "boxes": [
            [0, 22, 152, 422, 509],
            [1, 92, 22, 335, 496],
            [0, 237, 103, 370, 235],
            [1, 240, 25, 310, 167],
        ],
        "location_evidence": "Commons SG Tân Phú ng5th11n2022 (xe máy trên đường).jpg, mô tả Road transport in Ho Chi Minh City; Phương Huy, CC0.",
        "reason": "Người tiền cảnh áo xanh đội mũ xanh lái một xe, người áo trắng đội mũ phía sau lái xe đỏ khác. Hai box xe lớp 0, hai box rider lớp 1 theo phần nhìn thấy. Ô tô xanh ở mép trái và cọc phân cách không thuộc ba lớp; đã rà toàn crop.",
    },
    {
        "proposal_index": 3,
        "pageid": 147357881,
        "crop": [93, 152, 934, 1017],
        "canvas": [841, 865],
        "boxes": [[0, 29, 276, 799, 826], [1, 172, 40, 578, 824]],
        "location_evidence": "Commons SG th1n2023 (xe máy dừng chờ ở xóm).jpg, mô tả Road transport in Ho Chi Minh City; Phương Huy, CC0.",
        "reason": "Người đội mũ tím đang ngồi xe dừng, hai chân chống xuống đất, hông trên yên; một box lớp 1 và một box xe lớp 0. Tay cầm điện thoại không làm mất trạng thái ngồi xe. Hàng trái cây phía sau không phải người hoặc xe khác.",
    },
    {
        "proposal_index": 10,
        "pageid": 149613583,
        "crop": [666, 558, 923, 853],
        "canvas": [257, 295],
        "boxes": [[0, 9, 48, 244, 292], [2, 64, 18, 189, 236]],
        "location_evidence": "Commons Bui Vien Street, Ho Chi Minh City, by day, 2023 (05).jpg, mô tả và danh mục Bui Vien Street, Ho Chi Minh City; Bahnfrend, CC BY-SA 4.0.",
        "reason": "Người giao hàng đang ngồi lái xe, tóc trần rõ, lớp 2. Mũ trắng treo dưới tay lái không nằm trên đầu, không tạo box rider có mũ. Một box xe lớp 0 gồm thùng giao hàng gắn phía sau. Nền chỉ là bàn ghế nhà hàng, không có rider khác.",
    },
    {
        "proposal_index": 14,
        "pageid": 149619467,
        "crop": [914, 407, 1280, 853],
        "canvas": [366, 446],
        "boxes": [
            [0, 22, 151, 270, 379],
            [1, 78, 31, 250, 320],
            [1, 188, 26, 265, 312],
        ],
        "location_evidence": "Commons Bui Vien Street, Ho Chi Minh City, at night, 2023 (14).jpg, mô tả và danh mục Bui Vien Street, Ho Chi Minh City; Bahnfrend, CC BY-SA 4.0.",
        "reason": "Mở toàn cảnh và crop: hai người ngồi trên cùng xe, cả hai có mũ vỏ cứng, mỗi người một box lớp 1; một box xe lớp 0. Những người áo cam/đen và khách quán đang đi bộ/đứng/ngồi ghế riêng, không gán rider. Giữ ánh sáng xanh tím và chuyển động ban đêm thật, không chỉnh màu hay làm nét để tạo ảnh mới.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch10.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch07/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
