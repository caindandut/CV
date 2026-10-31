"""Manually reviewed Hanoi crops from an existing train source (RG72)."""

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 9,
        "pageid": 98773040,
        "crop": [356, 247, 950, 756],
        "canvas": [594, 509],
        "boxes": [
            [0, 26, 140, 569, 479],
            [1, 237, 24, 380, 493],
            [1, 130, 0, 274, 389],
            [0, 0, 81, 60, 279],
            [0, 0, 145, 146, 331],
        ],
        "location_evidence": "Commons Motorciklistoj en strato Cầu Gỗ (Hanojo).jpg; mô tả Hanojo, Vjetnamio và danh mục Cầu Gỗ Street, Hanoi. RG72, CC BY-SA 4.0.",
        "reason": "Hai phụ nữ đang ngồi cùng xe, mỗi người một box lớp 1: người lái mũ đen, người ngồi sau mũ đỏ. Người lái chống chân nhưng thân vẫn ngồi trên yên. Hai xe đỗ phía trái có box riêng lớp 0; hình phụ nữ trên áp phích không phải đối tượng thật. Rà toàn bộ crop bằng mắt.",
    },
    {
        "proposal_index": 10,
        "pageid": 98773048,
        "crop": [814, 127, 1280, 677],
        "canvas": [466, 550],
        "boxes": [[0, 38, 167, 466, 437]],
        "location_evidence": "Commons Bovloj ĵus lavitaj kaj lasitaj sur trotuaro en Hanojo 01.jpg; mô tả Hanojo, Vjetnamio và danh mục Hàng Bạc Street, Hanoi. RG72, CC BY-SA 4.0.",
        "reason": "Một xe máy đỗ chở hàng, thân xe bị người đi bộ che một phần. Phụ nữ đội nón lá đang đi bộ phía trước và phụ nữ phía sau dắt xe đạp không thuộc lớp rider. Chỉ một box xe máy, không biến nón lá của người đi bộ thành lớp 2.",
    },
    {
        "proposal_index": 12,
        "pageid": 98773568,
        "crop": [0, 865, 398, 1680],
        "canvas": [398, 815],
        "boxes": [[0, 0, 403, 360, 782]],
        "location_evidence": "Commons Strato post pluvo en Hanojo.jpg; mô tả Hanoi, Vietnam, GPS 21.036619,105.846886 và danh mục Motorcycles in Hanoi. RG72, CC BY-SA 4.0.",
        "reason": "Phần đuôi xe máy đỗ nhìn rõ, gồm yên, đèn hậu, biển số và bánh sau; một box lớp 0 bị cắt bởi mép ảnh trái. Người phụ nữ phía sau đang đi bộ, không có rider. Các túi hàng phía trên không phải xe/người ngồi xe.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch06.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/train_batch01/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(f"Recorded {len(records)} manually annotated photos")


if __name__ == "__main__":
    main()
