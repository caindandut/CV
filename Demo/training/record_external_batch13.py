"""Three manually reviewed Vietnam crops: seated helmet rider, pedal bicycle, parked bike."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "pageid": 34228148,
        "crop": [758, 175, 1185, 853],
        "canvas": [427, 678],
        "boxes": [[0, 0, 197, 427, 637], [1, 112, 23, 401, 326]],
        "location_evidence": "Commons Motorcycle passengers (14488394599).jpg: mô tả Hội An, Vietnam, danh mục Motorcycles in Hoi An; Erwin Verbruggen from Amsterdam, The Netherlands, CC BY-SA 2.0.",
        "reason": "Rider áo cam ngồi trên yên Suzuki, mũ vàng cứng trên đầu, lớp 1. Thân dưới che bởi thân xe/túi hàng, box người chỉ phần nhìn thấy. Một xe máy lớp 0, gồm túi hàng gắn trên xe và phần bị cắt biên. Crop loại hẳn người/xe phía phải và xe đạp phía trái; người đi bộ/nón lá nền không phải rider. Không gộp xe khác vào Suzuki.",
    },
    {
        "pageid": 142998583,
        "crop": [800, 500, 1100, 780],
        "canvas": [300, 280],
        "boxes": [],
        "location_evidence": "Commons On the motorbike (1346180455).jpg: danh mục 2007 in Huế, Vietnam photographs taken on 2007-08-29; thalling55, CC BY 2.0. Giữ cùng nhóm tác giả đã khóa.",
        "reason": "Một người đạp xe đạp có bàn đạp/khung xe đạp và bánh mảnh, không phải xe máy. Không gán lớp 0 và không gán lớp 2 cho người đạp xe. Crop không có đối tượng thuộc ba lớp, nhãn rỗng có kiểm chứng; cắt hoàn toàn phần người/xe chụp từ xe ở toàn cảnh. Không nâng độ phân giải.",
    },
    {
        "pageid": 142998601,
        "crop": [0, 600, 280, 960],
        "canvas": [280, 360],
        "boxes": [[0, 0, 83, 129, 337]],
        "location_evidence": "Commons Helmeting before the motorbike tour of Hue (1346058107).jpg: danh mục 2007 in Huế, Vietnam photographs taken on 2007-08-29; thalling55, CC BY 2.0. Giữ cùng nhóm tác giả đã khóa.",
        "reason": "Một phần xe máy đỗ nhìn rõ yên, tay vịn, đuôi và bánh sau, box lớp 0 dừng đúng biên crop. Crop loại hẳn nhóm người đang chuẩn bị mũ phía phải, không tự gán những người đứng này thành rider. PNG giữ nguyên pixel, mỗi ảnh nguồn chỉ một crop.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch13.json"
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
                review_log="acquisition_2000/cached_full_review_20261015_01.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
