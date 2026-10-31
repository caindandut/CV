# Chuẩn bị dữ liệu nhận diện xe máy và người đội mũ

## Trạng thái hiện tại

Snapshot dùng để tiếp tục trên GitHub là [dataset_v165](dataset_v165/). Có **690 ảnh đã được rà và chấp nhận** trên mục tiêu 5.000; còn **4.310 ảnh** cần bổ sung hoặc rà duyệt. Snapshot có thêm 1.497 ảnh đang chờ duyệt. Ảnh chờ duyệt không được tính vào mục tiêu sạch hoặc dùng để train cho tới khi được kiểm tra và ghi nguồn.

Class ID: 0 = xe máy; 1 = người đi xe máy có đội mũ; 2 = người đi xe máy không đội mũ. Các box người phải tách riêng từng người ngồi trên xe. Người đi bộ không thuộc hai lớp rider. Xe máy đỗ vẫn được gán nếu nhìn rõ.

## Tiếp tục bằng Cloud Codex

1. Mở [CLOUD_CODEX_5000.md](CLOUD_CODEX_5000.md) và dùng nó làm chỉ dẫn công việc.
2. Đảm bảo các ảnh Git LFS đã tải xuống: chạy git lfs install rồi git lfs pull trong bản clone.
3. Chạy python Demo/training/cloud_handoff/prepare_portable_manifest.py để tạo manifest với đường dẫn tương đối, phù hợp Linux và Codespaces.
4. Chỉ tăng bộ đếm ảnh sạch khi nhãn, nguồn, giấy phép, bằng chứng giao thông Việt Nam và kiểm tra trùng đã được ghi nhận.

## Những gì được đưa lên GitHub

Snapshot v165 được lưu bằng Git LFS để Codex trên cloud có ảnh và nhãn hiện tại. Manifest, nhật ký rà nhãn, thống kê, attribution, giới hạn sử dụng và báo cáo kiểm chứng đi cùng snapshot. File dataset_v165/manifest.json là bản lưu nguyên trạng và còn chứa đường dẫn máy Windows; cloud dùng manifest portable do script tạo.

Archive tải nguồn, thư mục dữ liệu gốc, checkpoint, kết quả huấn luyện, .env và mật khẩu không được commit. Bộ dữ liệu có hai ảnh bị giới hạn phi thương mại; xem dataset_v165/usage_restrictions.json và attribution trước khi tái sử dụng.

Các notebook và bundle SCConv cũ trong colab_bundles/ là lịch sử, chưa cập nhật để train từ mục tiêu 5.000. Không chạy train và không thay checkpoint demo khi đang làm công việc làm sạch dữ liệu.
