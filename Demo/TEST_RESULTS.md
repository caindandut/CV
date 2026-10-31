# Kết quả kiểm thử — 13/10/2026

Cập nhật đánh giá mở rộng: [model audit](evaluation/model_audit_20261013/REPORT.md). Bộ kiểm thử hiện có **22 passed**; đã chạy 150 job audit và validation AP trên 140 ảnh. Các mục bên dưới mô tả kiểm tra tích hợp trước đợt audit; chất lượng detection và vấn đề nhãn/split mới được trình bày trong báo cáo mở rộng.

## Trạng thái hiện tại

**Demo đã chạy inference thật với checkpoint hiện tại và MySQL80 đang cấu hình.** Người dùng đã chọn giữ checkpoint head 80 kênh; ứng dụng giới hạn kết quả ở ba nhãn 0, 1, 2. Không chỉnh sửa trọng số, cắt head, đổi baseline hoặc huấn luyện lại.

Checkpoint: `../Report/helmet_yolo26s_scconv_768_dataset2/weights/best.pt`, SHA256 `f6b6b0ab4bfe794f52b446107cf2b79d548070641bb805440c3f4a976cbc8f39`; SHA256 sau thay đổi ứng dụng giống trước thay đổi.

## Đã chạy và thành công

| Kiểm tra | Kết quả |
|---|---|
| Backend unit/API tests | **17 passed**, một cảnh báo deprecated httpx của Starlette TestClient |
| Rà soát lỗi Python | Ruff `--select F`: tất cả kiểm tra qua |
| Frontend | TypeScript + Vite build thành công; npm audit báo 0 lỗ hổng; frontend không thay đổi trong lần cập nhật bộ lọc này |
| Custom SCConv | AST của 10 định nghĩa khớp notebook huấn luyện gốc |
| Tương thích checkpoint | Head 80 kênh chỉ được phép với đúng SHA256 nêu trên; names phải đúng thứ tự; checkpoint khác không tự được chấp nhận |
| Lọc lớp | `classes=[0,1,2]` được truyền vào Ultralytics; kiểm tra lại ID trước ánh xạ/vẽ, không gán nhãn cho ID khác |
| 8 ảnh validation thật | 48 detection mục tiêu, 0 detection nhãn 3–79 vượt ngưỡng 0.25; confidence/toạ độ mục tiêu khớp kết quả không lọc |
| Smoke ảnh thật | 2 detection; ảnh xuất PNG có nhãn tiếng Việt, confidence thực và dấu phẩy thập phân |
| Smoke video | Clip kỹ thuật 3 frame từ ảnh validation thật; H.264/yuv420p, âm thanh AAC giữ lại, file trung gian được xóa |
| MySQL80 người dùng | Đăng nhập, tạo database/bảng, health `database_ready=true`, API lịch sử HTTP 200 |
| Inference qua API/MySQL80 | **2 job ảnh + 1 job video hoàn thành**, một job mỗi lần, API health tiếp tục phản hồi |
| Tải kết quả | Nội dung tải khớp tệp đầu ra, attachment và HTTP Range 206 hoạt động |
| Khởi động lại/MySQL80 | Các job hoàn thành còn trong lịch sử; hai hàng metadata kiểm thử queued/processing chuyển failed với lỗi gián đoạn |
| Lỗi đầu vào | Ảnh giả, quá giới hạn kể cả body streaming, ID không tồn tại, traversal và origin POST ngoài danh sách bị từ chối |

Kiểm tra 8 ảnh chỉ là mẫu tích hợp nhỏ, không phải đánh giá mAP hoặc chứng minh 77 kênh dư không bao giờ tạo detection. Không thay threshold, không dùng tập test để tinh chỉnh.

Các báo cáo chi tiết không commit, nằm trong `storage/outputs/`:

- `class-filter-validation.json`: kết quả từng ảnh và kiểm tra bộ lọc.
- `smoke-report.json`: suy luận ảnh/video thật theo bộ lọc hiện tại.
- `configured-mysql-acceptance-report.json`: ba job thật, tiến độ, thời gian xử lý và kiểm tra phục hồi với MySQL80 đang cấu hình.
- Các báo cáo MySQL riêng trước đây chỉ áp dụng instance kiểm thử cổng 3307, không thay thế báo cáo MySQL80 mới.

Trong lịch sử có thể thấy hai hàng `KIEM_THU_GIAN_DOAN.jpg` thất bại: đây là metadata tạo riêng để kiểm chứng xử lý gián đoạn, không phải lỗi inference hoặc dự đoán giả.

## Cách tái lập từ Demo

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH='backend'
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\ruff.exe check backend --exclude scconv.py --select F
.\.venv\Scripts\python.exe backend/validate_class_filter.py
.\.venv\Scripts\python.exe backend/smoke_inference.py
```

Kiểm tra API + MySQL80 và phục hồi: **dừng backend trước** để chỉ có một worker, rồi chạy:

```powershell
.\.venv\Scripts\python.exe backend/integration_mysql.py --configured-db
```

Script sử dụng `.env`, tạo các job kiểm thử trong database đã cấu hình, khởi động backend kiểm thử ở cổng 8001, kiểm tra phục hồi rồi dừng backend đó. Sau khi chạy, khởi động lại backend bình thường ở cổng 8000. Script không thay tài khoản/mật khẩu và không xóa lịch sử có sẵn.

Mặc định không có `--configured-db`, script dùng instance MySQL kiểm thử riêng ở cổng 3307, root mật khẩu rỗng, database `helmet_demo_acceptance`; phải tự khởi động instance disposable này trước. Không dùng cấu hình đó cho ứng dụng thật. Instance kiểm thử của phiên trước đã dừng. `--metadata-only` chỉ kiểm tra database/khôi phục, không tạo dự đoán.

## Làm rõ metadata ba nhãn và head 80 kênh

`data.yaml`, `model.names` và `model.nc` ghi ba nhãn 0,1,2. Tuy nhiên `Detect.nc=80`, cả sáu convolution phân loại có shape `[80,128,1,1]`; forward thô 768 px trả scores `[1,80,12096]` cho hai nhánh. `best.pt` và `epoch0.pt` cùng thí nghiệm SCConv 768 đều như vậy. Điều này không có nghĩa dataset của người dùng chứa 80 nhãn.

Notebook nạp `YOLO(MODEL_NAME)` rồi thay SCConv; custom trainer trả lại mạng có sẵn thay vì dựng lại `DetectionModel(..., nc=self.data['nc'])`. Framework sau đó gán model.nc/model.names theo dataset, nhưng convolution phân loại vẫn có 80 kênh. Xem `checkpoint_inspection.json` và `selected_model.json`.

Bản trước từ chối head 80 theo yêu cầu ban đầu. Người dùng sau đó chọn “demo hiện tại”, nên bản hiện tại áp dụng tương thích cho đúng checkpoint đã xác nhận, giữ nguyên kiến trúc và lọc lớp đầu ra. Không mô tả head này thành head ba lớp.

## Chưa xác minh

- Giao diện tương tác và phát video trực tiếp trong trình duyệt: phiên công cụ không có trình duyệt kết nối. File video đã kiểm tra codec/giải mã; chưa tuyên bố browser QA thành công.
- GPU/CUDA: môi trường kiểm thử dùng CPU.
- Video giao thông thực tế và timestamp VFR: clip smoke là kiểm thử kỹ thuật từ ảnh thật, không đại diện video giao thông.
- Metric validation riêng của best.pt, xếp hạng checkpoint, chất lượng phát hiện và ngưỡng tối ưu: chưa đánh giá; không khẳng định checkpoint tốt nhất hoặc cam kết hiệu năng.
# Bổ sung: chuẩn bị SCConv 70 epoch (2026-10-13)

- 32 kiểm thử qua: 22 backend hiện có và 10 kiểm thử dữ liệu/train mới. Ruff kiểm tra lỗi Python phần training qua.
- Kiểm tra cú pháp và giải mã 1.803 ảnh; sàng lọc bằng mắt qua 76 contact sheet. Đây là lượt rà đầu, không chứng nhận mọi nhãn đúng ở ảnh gốc. 6 ảnh có sửa được kiểm tra chi tiết; ảnh còn thiếu nhãn giữ pending.
- Hash xác nhận toàn bộ ảnh/nhãn gốc không đổi; loại 7 bản sao giống byte. Kiểm tra 201 liên kết gần trùng, group và hash không giao giữa các tập.
- Bản sao: 630 train (hai cảnh), 97 val (một cảnh), 31 test (một cảnh), 1.038 pending. Test không có GT lớp 2; chưa thể đánh giá độ chính xác người không đội mũ trên test.
- Pretrained chính thức chuyển 696/708 tensor sang mạng mới nc=3 trước khi thay SCConv; sáu convolution ra 3 lớp, raw score hai nhánh đều 3 kênh ở 768.
- Smoke CPU thật trên bản sao train/val, random initialization: lưu sau epoch 1, khôi phục 386 optimizer states/LR/scheduler/epoch, chạy tới 3 epoch tổng; hai khối SCConv nhận gradient. Snapshot backup trước và checkpoint lớp 2 được tạo. Đây không phải thí nghiệm chất lượng.
- Checkpoint mới của smoke nạp được trong process mới chỉ dùng định nghĩa custom của backend. Metadata run mới ghi yolo26s.yaml; chưa thay loader/đường dẫn model đang chạy.
- Notebook có 7 ô code, tất cả parse/compile được. ZIP 1.530 tệp được kiểm tra CRC, checksum và code khớp bản nguồn; không chứa ảnh pending hoặc .env.
- FE 5173 và health BE 8000 đều trả 200 cuối phiên; SHA256 checkpoint demo hiện tại không đổi.
- Chưa chạy: CUDA preflight 8/4/2, Colab/Google Drive thật, run chính 70 epoch, chọn confidence trên model mới, test cuối độc lập hoặc đo mức cải thiện. Xem [hướng dẫn](training/README.md) và [báo cáo nhãn](training/review_report.html).

