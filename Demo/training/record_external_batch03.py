"""Six manually reviewed VN photographs; four use logged foreground crops."""

from prepare_dataset import ROOT, digest, write_json

DECISIONS = [
    {
        "pageid": 80522495,
        "crop": [205, 110, 1275, 1700],
        "canvas": [1070, 1590],
        "boxes": [[0, 0, 239, 1026, 1517], [2, 169, 38, 974, 1491]],
        "location_evidence": "Commons ghi Downtown Hanoi - Vietnam, danh mục Motorcycles in Hanoi và Vietnam photographs taken on 2019-06-16.",
        "reason": "Người ngồi trên xe chở xi măng đội mũ cối bằng vải, không phải mũ bảo hiểm xe máy: lớp 2. Cắt bỏ rider bị cắt đầu ở góc trái của ảnh nguồn; toàn ảnh cắt có một xe và một người rõ.",
    },
    {
        "pageid": 140483165,
        "crop": [237, 272, 1051, 728],
        "canvas": [814, 456],
        "boxes": [[0, 34, 0, 802, 441]],
        "location_evidence": "Commons ghi Police motorcycle in Saigon, danh mục District 1, Ho Chi Minh City và Motorcycles in Ho Chi Minh City.",
        "reason": "Cắt vùng xe cảnh sát đỗ không người; các rider xa ở phần trên nằm hoàn toàn ngoài vùng cắt. Mũ gắn trên cọc không gán rider.",
    },
    {
        "pageid": 105761425,
        "crop": [580, 125, 935, 664],
        "canvas": [355, 539],
        "boxes": [[0, 0, 126, 290, 511], [0, 240, 154, 354, 398], [1, 0, 12, 259, 528]],
        "location_evidence": "Tiêu đề Commons ghi Mark on scooter, Hanoi; danh mục Lý Nam Đế Street, Hanoi; biển Viettel và biển số Hà Nội trong ảnh nguồn.",
        "reason": "Một người đội mũ bảo hiểm đang ngồi xe, một xe đỗ phía sau không người. Vùng cắt loại các người đứng ngoài cửa hàng và tay lái của phương tiện ở sát máy ảnh; không biến họ thành rider.",
    },
    {
        "pageid": 49057126,
        "crop": [529, 164, 800, 806],
        "canvas": [271, 642],
        "boxes": [[0, 0, 145, 262, 638], [1, 0, 0, 166, 420], [1, 60, 79, 201, 292]],
        "location_evidence": "Mô tả gốc Bộ Ngoại giao Mỹ ghi streets of Ho Chi Minh City, Vietnam; danh mục Mothers and children in Vietnam. Public domain theo nguồn ảnh.",
        "reason": "Một xe với mẹ và trẻ ngồi phía trước, cả hai có mũ bảo hiểm nhìn thấy trực tiếp; mỗi người một box lớp 1. Cắt tiền cảnh, bỏ các rider xa và người đi bộ ở ảnh nguồn; không suy nhãn từ model.",
    },
    {
        "pageid": 86671235,
        "canvas": [1280, 960],
        "boxes": [[0, 333, 342, 1241, 919]],
        "location_evidence": "Commons mô tả Vietnam, EXIF GPS 23.097747 N, 105.250053 E và ảnh được dùng trên trang Ha Giang của Wikivoyage; đã đối chiếu trang nguồn Semiauto motorbike.jpg.",
        "reason": "Một xe Honda đỗ trên đường núi, không người. Túi buộc ở yên không phải người; rà toàn ảnh, không có rider cần gán nhãn.",
    },
    {
        "pageid": 185288696,
        "canvas": [1280, 960],
        "source_group": "TrungKinhHanoi",
        "boxes": [[0, 346, 472, 786, 778]],
        "location_evidence": "Commons ghi Yên Hòa Parkview City, danh mục Motorcycles in Hanoi, Bicycles in Hanoi và Banyan Alley, YHPV. Gom thận trọng cùng khu vực Hà Nội với nguồn Trung Kính.",
        "reason": "Một xe máy đỏ đỗ; xe đạp cạnh bên không thuộc lớp 0. Mũ đặt trên xe đạp và túi hàng không phải rider; không có người trong ảnh.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch03.json"
    if output.exists():
        raise FileExistsError("Review log already exists")
    records = []
    for decision in DECISIONS:
        source = ROOT / f"acquisition_2000/images/commons_{decision['pageid']}.jpg"
        evidence = (
            f"acquisition_2000/curated_crops/commons_{decision['pageid']}.png"
            if "crop" in decision
            else source.relative_to(ROOT).as_posix()
        )
        records.append(
            dict(
                country="VN",
                visual_reviewed=True,
                image_sha256=digest(source),
                evidence=evidence,
                **decision,
            )
        )
    write_json(output, records)
    print(f"Accepted {len(records)} manually reviewed original photographs")


if __name__ == "__main__":
    main()
