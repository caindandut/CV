# Nhiệm vụ Cloud Codex: hoàn thiện 5.000 ảnh sạch

## Mục tiêu

Tiếp tục bộ dữ liệu xe máy tại Việt Nam từ Demo/training/dataset_v165 cho tới khi có 5.000 ảnh được rà thủ công, nhãn đúng, có nguồn và giấy phép truy xuất được. Không tính ảnh chờ duyệt vào con số 5.000. Không hứa rằng việc thu thập và rà nhãn thủ công sẽ hoàn thành trong một khoảng thời gian cố định.

Trạng thái tại ngày 31/10/2026:

- 690 ảnh đã được chấp nhận: 546 train, 76 validation, 68 test.
- 1.497 ảnh trong snapshot đang chờ duyệt.
- Còn 4.310 ảnh sạch để đạt mục tiêu; cần xem xét ảnh chờ duyệt trước khi chấp nhận chúng.
- Báo cáo v165: Demo/training/verification/dataset_v165.json.
- Bản gốc của 1.803 ảnh và nhãn phải được giữ nguyên.

## Khởi động phiên cloud

1. Đồng bộ nhánh hiện tại và tải dữ liệu LFS: chạy git lfs install, rồi git lfs pull.
2. Kiểm tra ảnh trong Demo/training/dataset_v165/{train,val,test,pending}/images/ là ảnh thật, không phải chuỗi pointer LFS.
3. Chạy python Demo/training/cloud_handoff/prepare_portable_manifest.py. Script đối chiếu hash ảnh/nhãn và tạo Demo/training/cloud_handoff/dataset_v165_manifest_portable.json, dùng đường dẫn tương đối. Giữ nguyên manifest gốc của snapshot.
4. Đọc dataset_v165/ATTRIBUTION.md, dataset_v165/usage_restrictions.json, dataset_v165/review_audit_2000.json, dataset_v165/changes.json và báo cáo kiểm chứng trước khi sửa dữ liệu.

## Quy tắc nhãn

- Class 0: xe máy, gồm xe đang chạy hoặc đỗ nhưng nhìn đủ rõ.
- Class 1: từng người đang ngồi/lái xe máy và đội mũ trên đầu.
- Class 2: từng người đang ngồi/lái xe máy và không đội mũ.
- Mỗi người trên xe có box riêng, bao gồm người lái và người ngồi sau. Người đi bộ/đứng cạnh xe không được gán class 1 hoặc 2.
- Không gán theo mũ treo trên xe, màu/quần áo, hoặc dự đoán của mô hình. Box người phải bao quanh đúng người; box xe phải bao quanh xe.
- Nếu không thể nhìn rõ tư thế, người có đang ngồi trên xe hay không, hoặc trạng thái mũ, để cả ảnh ở trạng thái chờ duyệt. Không biến vùng chưa rõ thành nhãn âm tính.

## Thu thập ảnh mới và ghi nguồn

- Chỉ dùng ảnh thể hiện giao thông Việt Nam. Ghi URL trực tiếp của ảnh/trang nguồn, tác giả, giấy phép và bằng chứng đủ để xác định bối cảnh Việt Nam. Không suy đoán quốc gia chỉ từ một đặc điểm mơ hồ.
- Kiểm tra điều khoản sử dụng trước khi tải/cắt. Giữ nguyên ghi công và giới hạn giấy phép; các mục liệt kê trong usage_restrictions.json bị giới hạn phi thương mại.
- Loại bản sao giống hệt; dùng perceptual hash chỉ để tìm ứng viên gần trùng, sau đó kiểm tra nội dung. Giữ frame cùng video/cùng cảnh trong một nhóm nguồn.
- Không đưa ảnh thiếu nguồn, ảnh không rõ giấy phép, hoặc ảnh chưa được người rà vào tập sạch. Không tính crop và ảnh gốc cùng cảnh là hai ảnh độc lập nếu chúng biểu diễn cùng một mẫu.
- Ghi mọi sửa nhãn bằng giá trị trước/sau, lý do, người/phiên rà, nguồn và đường dẫn bằng chứng. Giữ ảnh mơ hồ riêng, không xoá bằng chứng.

## Chia tập và kiểm soát chất lượng

- Không sửa, di chuyển hoặc dùng lại test đang khóa trong dataset_v165.
- Chia ảnh mới theo nhóm nguồn/video/cảnh để không rò rỉ cảnh gần trùng giữa train, validation và test. Ưu tiên tách nhóm nguồn hơn ép tỷ lệ.
- Chỉ tăng số ảnh sạch sau khi xác minh ảnh mở được, nhãn parse được, class ID nằm trong 0–2, nguồn/giấy phép/bằng chứng hiện diện, trùng lặp đã xử lý và split hợp lệ.
- Duy trì số liệu riêng cho ảnh đã chấp nhận, chờ duyệt và bị loại. Báo số ảnh và số box theo từng lớp; không lấy số box làm số ảnh.
- Cập nhật manifest, audit, attribution, thống kê, trạng thái và báo cáo kiểm chứng cho mỗi batch. Tạo snapshot phiên bản mới; không sửa snapshot cha.
- Không dùng pseudo-label từ model làm ground truth. Không chạy train/benchmark hoặc thay checkpoint demo trong nhiệm vụ làm sạch dữ liệu.

## Bàn giao mỗi phiên

Commit từng batch nhỏ đã hoàn tất cùng nhật ký nguồn và thay đổi nhãn. Trong phần tóm tắt, ghi số ảnh sạch mới được chấp nhận, số còn chờ, số bị loại, tổng tiến độ trên 5.000, phân bố class, nguồn/giấy phép, kết quả kiểm tra trùng và mọi mục chưa xác minh. Nếu nguồn ảnh/giấy phép hoặc nhãn còn mơ hồ, giữ ảnh chờ duyệt và nêu rõ phần còn thiếu.

## Prompt để bắt đầu

Tiếp tục mục tiêu 5.000 ảnh theo Demo/training/CLOUD_CODEX_5000.md. Trước hết tải Git LFS, kiểm tra snapshot v165 và manifest portable. Sau đó rà ảnh pending theo từng batch; chỉ nhận ảnh giao thông Việt Nam có nguồn/giấy phép và nhãn đã được kiểm tra. Giữ nguyên dữ liệu gốc và test đã khóa, ghi nhật ký đầy đủ, tạo snapshot phiên bản mới cho từng batch, rồi báo tiến độ chính xác. Không train model hoặc tự gán nhãn bằng dự đoán.
