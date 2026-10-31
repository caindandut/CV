"""Two manually reviewed Vietnam rider photographs: rain cargo and historical patrol."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 12239452,
        "crop": [175, 45, 1010, 850],
        "canvas": [835, 805],
        "boxes": [[0, 16, 18, 788, 758], [1, 421, 161, 692, 676]],
        "location_evidence": "Commons Motorcycle Thành phố Hồ Chí Minh ホーチミン市のオートバイの群れ DSCF2198.jpg: danh mục Motorcycles in Ho Chi Minh City, Honda Super Cub; tác giả 兵庫胡志明倶楽部, CC BY-SA 3.0.",
        "reason": "Một rider ngồi trên xe, đội mũ trắng cứng có quai rõ, mặc áo mưa xanh, lớp 1. Một box xe lớp 0 gồm xe và các bao hàng buộc trên xe, không coi bao/pink plastic là người. Giữ mưa/ánh sáng gốc, không làm nét. Crop bỏ hẳn xe/người ở biên trái chưa đủ nhãn, nền cửa hàng không gán rider. Toàn crop đã kiểm tra ở pixel gốc.",
    },
    {
        "pageid": 119996564,
        "crop": [85, 85, 640, 360],
        "canvas": [555, 275],
        "boxes": [[0, 0, 86, 253, 275], [1, 0, 15, 130, 275]],
        "location_evidence": "Commons Australian MP and South Vietnamese motorbike policeman in Saigon.jpg: metadata Sài Gòn tháng 12/1967, mô tả cảnh sát Việt Nam trên xe máy; Crothers, Richard William, CC0. Giữ đây là ảnh lịch sử, không coi là ảnh camera hiện đại.",
        "reason": "Cảnh sát áo trắng mũ cứng ngồi trên xe máy lớp 1, một phần xe máy lớp 0 gồm kính chắn gió và đèn nhìn thấy. Người quân phục mũ beret ngồi trong Land Rover không phải rider, không gán lớp 2 cho người này. Crop bỏ hẳn phần phương tiện phía dưới chưa tách đủ; người/xe dừng đúng biên crop. Nguồn toàn ảnh giữ nguyên, kể cả dòng ghi nguồn, lưu attribution CC0; không tô màu hoặc nâng độ phân giải.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch15.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_03.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
