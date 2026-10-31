"""Seven manually reviewed VN crops; cloth caps and cyclists are distinguished."""

import json

from prepare_dataset import ROOT, digest, write_json
from training_crops import prepare_review_image

DECISIONS = [
    {
        "proposal_index": 2,
        "pageid": 105751953,
        "crop": [167, 154, 592, 835],
        "canvas": [425, 681],
        "boxes": [
            [0, 13, 229, 411, 650],
            [0, 0, 174, 109, 346],
            [0, 334, 138, 425, 476],
            [2, 78, 52, 251, 557],
            [1, 99, 30, 347, 578],
        ],
        "location_evidence": "Commons Hanoi (2824822210).jpg, danh mục Hàng Bút Street, Hanoi; Nam-ho Park, CC BY 2.0.",
        "reason": "Xem toàn cảnh và crop: người lái áo be giữ mũ vải caro trên đầu, lớp 2; người ngồi sau áo đen đội mũ vỏ cứng xanh có quai, lớp 1. Mỗi người một box, một xe đang chạy và hai phần xe đỗ ở nền có box lớp 0. Người gánh hàng nón lá phía trước là người đi bộ, không phải rider. Xe/người ngoài crop không dùng làm nền thiếu nhãn.",
    },
    {
        "proposal_index": 10,
        "pageid": 105762706,
        "crop": [628, 389, 906, 651],
        "canvas": [278, 262],
        "boxes": [[0, 0, 105, 267, 251], [1, 77, 13, 196, 212]],
        "location_evidence": "Commons Peak hour traffic on Duong Au Co, Hanoi, Vietnam (7060641081).jpg; danh mục Âu Cơ Street, Hanoi. David McKelvey, CC BY 2.0.",
        "reason": "Một người ngồi lái xe chở hàng, đội mũ vỏ cứng xanh lớp 1; một box xe lớp 0. Túi hàng màu xanh không phải người. Cột phía sau không che trạng thái đầu; toàn crop không có rider khác.",
    },
    {
        "proposal_index": 11,
        "pageid": 105762826,
        "crop": [438, 303, 695, 628],
        "canvas": [257, 325],
        "boxes": [[0, 45, 106, 210, 311], [1, 47, 15, 139, 282]],
        "location_evidence": "Commons Bottleneck, Urban Discovery Tour, Hanoi (7060657055).jpg; danh mục Hàng Bút Street, Hanoi. David McKelvey, CC BY 2.0.",
        "reason": "Một người ngồi xe tay ga, đầu có mũ bảo hiểm vỏ cứng có lỗ thông gió, lớp 1; tác vụ phân biệt có/không mũ, không suy đoán chứng nhận an toàn của mũ. Bao trắng là hàng trên xe, không phải rider. Người bị tường che một phần vẫn thấy rõ đầu và chân ngồi xe. Một box xe lớp 0, không thêm box người từ bao hàng.",
    },
    {
        "proposal_index": 13,
        "pageid": 105763249,
        "crop": [806, 260, 1196, 721],
        "canvas": [390, 461],
        "boxes": [[0, 0, 179, 374, 461], [2, 14, 43, 175, 345]],
        "location_evidence": "Commons Stopping at Traffic lights, Urban Discovery Tour, Hanoi (7060764699).jpg; danh mục Hàng Đậu Street, Hanoi, biển đường trong toàn cảnh. David McKelvey, CC BY 2.0.",
        "reason": "Người tóc trần đang ngồi trên yên Honda Dream, nghe điện thoại khi dừng đèn đỏ: lớp 2, có box riêng xe lớp 0. Mở toàn cảnh để xác nhận tư thế ngồi và nhận ra bánh/yên ở tiền cảnh thuộc xe đạp của nhóm đi tour, không phải xe máy thứ hai. Người đứng quầy đội nón lá cũng không thuộc rider.",
    },
    {
        "proposal_index": 48,
        "pageid": 134444089,
        "crop": [1024, 515, 1280, 942],
        "canvas": [256, 427],
        "boxes": [[0, 129, 127, 256, 408], [1, 107, 20, 256, 329]],
        "location_evidence": "Commons Bus 150 in front of Binh Tay market 2.jpg; danh mục Binh Tay Market, Buses in Ho Chi Minh City; Syced, CC0.",
        "reason": "Một người ngồi lái xe tay ga đỏ, đội mũ vỏ cứng và đeo khẩu trang lớp 1; một box xe lớp 0. Phần người/xe bị cắt sát mép phải được giới hạn đúng biên ảnh. Xe buýt bên trái không thuộc ba lớp; không có rider khác trong crop.",
    },
    {
        "proposal_index": 49,
        "pageid": 134444108,
        "crop": [0, 550, 256, 856],
        "canvas": [256, 306],
        "boxes": [[0, 6, 65, 180, 292], [1, 8, 16, 138, 254], [1, 107, 13, 173, 224]],
        "location_evidence": "Commons Bus 150 in front of Binh Tay market 3.jpg; danh mục Binh Tay Market, Buses in Ho Chi Minh City; Syced, CC0. Giữ cùng nhóm tác giả test đã khóa.",
        "reason": "Hai người trên cùng một xe tay ga, cả hai đội mũ vỏ cứng, mỗi người một box lớp 1; xe có box lớp 0 gồm gương thấy rõ. Ô tô phía sau và người trong ô tô không gán rider. Đây là ảnh khác với ảnh bus 150 số 2, không phải crop thứ hai của cùng ảnh nguồn.",
    },
    {
        "proposal_index": 56,
        "pageid": 142999883,
        "crop": [0, 239, 760, 1707],
        "canvas": [760, 1468],
        "boxes": [
            [0, 0, 425, 670, 1468],
            [2, 0, 129, 697, 1468],
            [1, 510, 100, 760, 1468],
        ],
        "location_evidence": "Commons Hue, Vietnam (1347057020).jpg; mô tả Hue, Vietnam, danh mục Motorcycles in Huế và ngày chụp 29/08/2007; thalling55, CC BY 2.0.",
        "reason": "Cặp người trên xe Honda ở đường cạnh ruộng: lái đội mũ lưỡi trai vải trắng có đường may, lớp 2; người ngồi sau đội mũ bảo hiểm đỏ có quai cằm, lớp 1. Hai box riêng và một box xe gồm gương. Thu hẹp mép phải so với đề xuất để loại hoàn toàn chi tiết người/xe rất xa chưa rõ ở nền; đầu cả hai rider vẫn đủ rõ, không bổ sung bằng dự đoán. Giữ pixel gốc PNG, không nhân thêm crop của ảnh này.",
    },
]


def main():
    output = ROOT / "acquisition_2000/external_reviews_batch09.json"
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
                proposal_run="acquisition_2000/coco_crop_proposals/batch06/run.json",
                **decision,
            )
        )
    write_json(output, records)
    print(json.dumps({"accepted": len(records)}))


if __name__ == "__main__":
    main()
