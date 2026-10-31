"""Three manually reviewed Vietnam photographs; preserve Hanoi location test group."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 31951085,
        "crop": [474, 340, 730, 720],
        "canvas": [256, 380],
        "boxes": [[0, 52, 120, 199, 339], [1, 67, 39, 194, 282]],
        "location_evidence": "Commons Lascar Ho Chi Minh City - The capital of motorbikes (4607953274).jpg: danh mục Streets in Ho Chi Minh City, Motorcycles in Ho Chi Minh City; Jorge Láscar from Australia, CC BY 2.0.",
        "reason": "Người lái mũ trắng cứng trên xe xanh, lớp 1; một xe lớp 0, gồm phần gương/ghi đông nhìn thấy. Toàn crop được kiểm tra ở pixel gốc. Crop loại hẳn rider xa và người bán hàng đội nón lá ở toàn cảnh, không gán họ thành rider hay để thiếu nhãn trong ảnh train. Không nâng độ phân giải.",
    },
    {
        "pageid": 186362450,
        "crop": [545, 545, 850, 815],
        "canvas": [305, 270],
        "boxes": [[0, 0, 0, 264, 204]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons Honda motorcycle in Hanoi, 17-03-2026 E1.jpg: mô tả Nguyệt Hà Teich 2026, danh mục Nguyệt Hà Pond, Motorcycles in Hanoi; HaCong4439u, CC BY-SA 4.0. Metadata Commons pageid 184446683 liên kết Nguyệt Hà Pond và Trung Kính Hạ Temple; gom bảo thủ cùng khu vực test TrungKinhHanoi đã khóa, không khẳng định cùng camera/tác giả.",
        "reason": "Một xe máy thể thao Honda đỗ cạnh cây, không có rider. Crop loại hẳn các xe xa ở góc trên trái và người ở nền, giữ nguyên ánh sáng đêm và pixel gốc. Box lớp 0 dừng ở biên crop nơi một phần xe bị cắt. Giữ cùng nhóm khu vực test, không chuyển sang train để khớp tỷ lệ.",
    },
    {
        "pageid": 181139479,
        "canvas": [1280, 960],
        "boxes": [[0, 18, 70, 1267, 911]],
        "source_group": "TrungKinhHanoi",
        "location_evidence": "Commons A motorbike of New Year's Day 01-01-2026 E2.jpg: mô tả Rue Trung Kính, danh mục Trung Kính Street, Hanoi, Yamaha motorcycles in Vietnam; Hesebitium298e, CC BY-SA 4.0. Giữ cùng khu vực test TrungKinhHanoi đã khóa.",
        "reason": "Toàn ảnh có một Yamaha NVX đỗ có chân chống, không có người. Một box xe máy lớp 0 gồm gương và chân chống nhìn thấy; không tạo rider trên yên trống. Không crop, không đổi ảnh nguồn, không dùng dự đoán làm nhãn chuẩn.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch14.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_02.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
