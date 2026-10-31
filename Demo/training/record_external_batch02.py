"""Manual annotations of four inspected Vietnamese Commons photographs."""

import json

from prepare_dataset import ROOT, digest, write_json

DECISIONS = [
    {
        "pageid": 155062346,
        "canvas": [1280, 1672],
        "boxes": [
            [0, 380, 1169, 1157, 1672],
            [1, 444, 873, 837, 1628],
            [1, 601, 874, 980, 1653],
        ],
        "location_evidence": "Commons: tiêu đề và mô tả ghi tunnel in Saigon; danh mục Ho Chi Minh City và Vietnam photographs taken on 2024-11-08.",
        "reason": "Một xe có người lái và người ngồi sau, mỗi người có mũ trên đầu và box lớp 1 riêng. Cảnh hầm tối; không có rider khác trong ảnh.",
    },
    {
        "pageid": 69181456,
        "canvas": [1280, 960],
        "boxes": [
            [0, 399, 72, 538, 340],
            [0, 742, 158, 1067, 362],
            [0, 1116, 48, 1240, 307],
            [0, 716, 566, 1168, 829],
            [1, 829, 551, 1043, 798],
            [0, 200, 699, 662, 919],
            [1, 278, 686, 483, 905],
        ],
        "location_evidence": "Mô tả Commons: Old quarter, Hanoi; danh mục Motorcycles in Hanoi. Tác giả Caitriana Nicholson; không suy địa điểm từ nơi cư trú của tác giả.",
        "reason": "Hai người lái đội mũ, ba xe đỗ không người. Mũ đặt ở giỏ xe không gán rider; xe đạp ở góc phải không gán xe máy. Kiểm tra vùng tối bằng ảnh tăng sáng để đếm xe.",
    },
    {
        "pageid": 69181470,
        "canvas": [1280, 960],
        "boxes": [
            [0, 417, 98, 542, 334],
            [0, 758, 160, 1083, 359],
            [0, 1132, 75, 1248, 313],
            [0, 200, 559, 586, 815],
            [2, 241, 543, 424, 811],
            [1, 135, 556, 303, 781],
        ],
        "location_evidence": "Mô tả Commons: Old quarter, Hanoi; danh mục Motorcycles in Hanoi. Cùng cảnh và tác giả với commons_69181456, được gom nguồn trước khi chia tập.",
        "reason": "Người lái đội mũ vải mềm có đường may/vành, không phải mũ bảo hiểm; người ngồi sau đội mũ bảo hiểm có vỏ cứng. Một xe đang đi, ba xe đỗ; xe đạp không gán lớp 0.",
    },
    {
        "pageid": 86207935,
        "canvas": [1280, 853],
        "boxes": [],
        "location_evidence": "Tiêu đề và mô tả Commons ghi Hai Van Pass, Vietnam, north of Da Nang; danh mục Hai Van Pass và Men of Vietnam.",
        "reason": "Ảnh âm tính khó: người đội mũ đứng bên công sự, không ngồi xe; toàn ảnh không thấy xe máy. Không gán rider chỉ vì người có mũ.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch02.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    records = []
    for decision in DECISIONS:
        path = ROOT / f"acquisition_2000/images/commons_{decision['pageid']}.jpg"
        records.append(
            dict(
                country="VN",
                visual_reviewed=True,
                image_sha256=digest(path),
                evidence=path.relative_to(ROOT).as_posix(),
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"reviewed": len(records), "accepted": len(records)}))


if __name__ == "__main__":
    main()
