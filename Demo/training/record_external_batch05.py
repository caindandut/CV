"""Three inspected VN crops; includes clear cloth-cap riders, not guessed helmets."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 0,
        "pageid": 20900076,
        "crop": [171, 102, 1063, 852],
        "canvas": [892, 750],
        "boxes": [
            [0, 34, 267, 885, 750],
            [1, 225, 61, 529, 750],
            [2, 14, 62, 305, 750],
        ],
        "location_evidence": "Commons Bác xe ôm chở khách.jpg, danh mục Motorcycle taxis in Vietnam; biển chữ Việt trong ảnh. Tác giả Bảo Trí Nguyễn, CC BY 2.0.",
        "reason": "Một xe với người lái đội mũ bảo hiểm vỏ cứng có quai cằm lớp 1; người ngồi sau đội mũ vải mềm trắng, không có mũ bảo hiểm lớp 2. Hai người có box riêng, giữ ảnh đen trắng đúng nguồn.",
    },
    {
        "proposal_index": 1,
        "pageid": 24297289,
        "crop": [109, 139, 511, 712],
        "canvas": [402, 573],
        "boxes": [[0, 6, 162, 388, 563], [2, 65, 37, 384, 507]],
        "location_evidence": "Commons SAIGON (2041927237).jpg; danh mục People of Ho Chi Minh City, Streets in Ho Chi Minh City và Street vendors in Ho Chi Minh City.",
        "reason": "Một người ngồi xe mặc áo mưa vàng, đội mũ lưỡi trai vải trắng lớp 2; có box xe riêng. Người bán hàng đội nón lá đứng bên trái và người đẩy xe đạp phía sau không gán rider. Rà toàn vùng cắt bằng mắt.",
    },
    {
        "proposal_index": 10,
        "pageid": 30491978,
        "crop": [980, 524, 1237, 799],
        "canvas": [257, 275],
        "boxes": [[0, 0, 7, 219, 263], [1, 65, 14, 192, 232], [0, 228, 62, 257, 135]],
        "location_evidence": "Commons Hanoi bus 09.jpg mô tả Hanoi bus - Thaco, danh mục Buses in Hanoi; biển số và cảnh nguồn Hà Nội.",
        "reason": "Người chở hàng mặc đồ xanh đội mũ bảo hiểm lớp 1 trên một xe; xe máy đỗ nhìn thấy một phần ở mép phải có box lớp 0. Người trong cabin ô tô không phải rider; không gán nhãn từ dự đoán COCO.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch05.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch02/run.json",
                **decision,
            )
        )
    write_json(output, records)
    proposals = json.loads(
        (
            ROOT / "acquisition_2000/coco_crop_proposals/batch02/proposals.json"
        ).read_text()
    )
    accepted = {r["pageid"] for r in records}
    write_json(
        ROOT / "acquisition_2000/candidate_screening_batch03.json",
        [
            {
                "pageid": r["pageid"],
                "proposal_index": r["index"],
                "status": "accepted_with_manual_labels"
                if r["pageid"] in accepted
                else "pending",
                "review_level": "full_crop"
                if r["index"] in {0, 1, 10, 12}
                else "thumbnail_only",
                "reason": (
                    "Nhãn thủ công trong external_reviews_batch05.json."
                    if r["pageid"] in accepted
                    else "Có người ngồi sau bị người phía trước che phần đầu, chưa xác nhận mũ; giữ cả ảnh chờ duyệt."
                    if r["index"] == 12
                    else "Chưa có nhãn đầy đủ và xác minh nguồn cho toàn vùng cắt; không đưa vào train."
                ),
            }
            for r in proposals
        ],
    )
    print(
        json.dumps(
            {
                "reviewed_proposals": len(proposals),
                "accepted_with_manual_labels": len(records),
            }
        )
    )


if __name__ == "__main__":
    main()
