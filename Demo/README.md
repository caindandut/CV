# Demo nhận diện người đi xe máy không đội mũ bảo hiểm

Đánh giá mở rộng ngày 13/10/2026: [báo cáo trực quan](evaluation/model_audit_20261013/report.html) và [phân tích, hướng train tiếp theo](evaluation/model_audit_20261013/REPORT.md). Đã kiểm tra 140 ảnh validation, 150 job qua API và đo AP riêng; phát hiện trùng train/validation và nhãn cần rà soát. Chưa train lại hoặc đổi checkpoint/ngưỡng demo.

Ứng dụng cục bộ gồm React + TypeScript + Vite, FastAPI + PyTorch/Ultralytics custom SCConv, MySQL + SQLAlchemy/PyMySQL, OpenCV, Pillow và FFmpeg. Chỉ có nhận diện ảnh/video tải lên, lịch sử và tải kết quả. Không có đăng nhập, camera live, tracking hoặc nhận diện biển số.

## Trạng thái cần biết trước khi chạy

**Demo dùng được với checkpoint hiện tại theo lựa chọn của người dùng.** Checkpoint `../Report/helmet_yolo26s_scconv_768_dataset2/weights/best.pt` được giữ nguyên trọng số và head 80 kênh. Dataset và names có ba nhãn; demo truyền `classes=[0,1,2]` khi suy luận và kiểm tra lại ID trước khi vẽ/hiển thị. Không cắt head, sửa checkpoint, đổi baseline hoặc huấn luyện lại.

Ngoại lệ head 80 kênh chỉ áp dụng checkpoint có SHA256 `f6b6b0ab4bfe794f52b446107cf2b79d548070641bb805440c3f4a976cbc8f39`, đã được kiểm tra custom SCConv, kiến trúc và thứ tự names. Checkpoint khác vẫn phải có head ba lớp hoặc được kiểm tra và chấp thuận riêng; chỉ đổi `model.nc` không đủ. Xem [bằng chứng cấu trúc](checkpoint_inspection.json) và [manifest](selected_model.json).

Đã kiểm tra 8 ảnh validation: 48 detection mục tiêu, không thấy nhãn dư vượt ngưỡng 0.25, toạ độ/confidence giữ nguyên sau lọc. Đây là kiểm tra tích hợp nhỏ, không phải đánh giá chất lượng hay khẳng định 77 kênh dư không bao giờ phát hiện.

Kết nối MySQL80 đã được sửa cho khớp Workbench: `DB_HOST=localhost`, `DB_USER=root`, giữ nguyên mật khẩu trong `.env`. Đã tạo database/bảng thành công; health báo `database_ready=true`, API lịch sử trả HTTP 200. Các kiểm tra phục hồi job trước đó dùng instance riêng ở cổng 3307. Xem [báo cáo kiểm thử](TEST_RESULTS.md).

## Cấu trúc

```text
Demo/
  frontend/                 # React, TypeScript, Vite; package-lock.json
  backend/
    app/                    # Cấu hình, API, MySQL, worker, detector, vẽ nhãn
    tests/                  # Kiểm thử tự động
    requirements.txt        # Toàn bộ phiên bản môi trường đã kiểm tra
    init_db.py
    inspect_checkpoint.py
    smoke_inference.py
    integration_mysql.py
  assets/fonts/             # DejaVu Sans và giấy phép
  storage/{uploads,outputs,temp}/
  scripts/                  # Khởi chạy trên Windows
  .env.example
  selected_model.json
  checkpoint_inspection.json
```

`.env`, `.venv`, cache, checkpoint, upload và kết quả không đưa vào Git. Không có `Demo/Demo`; mọi đường dẫn tương đối của cấu hình được tính từ `Demo`, không phụ thuộc thư mục terminal.

## Cài đặt trên Windows

Yêu cầu Python 3.11, Node.js 22.15 trở lên phù hợp Vite 6, MySQL 8.0 và công cụ `uv` chính thức. Môi trường kiểm tra thực tế: Python 3.11.15, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, Ultralytics 8.4.155, OpenCV 5.0.0.93, Pillow 12.3.0; phiên bản còn lại được ghim trong `backend/requirements.txt`. Node đã dùng: 22.15.0. Không nâng Ultralytics tùy ý.

Từ thư mục dự án:

```powershell
cd 'D:\Nam 4\CV\Demo'
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe --index-strategy unsafe-best-match -r backend/requirements.txt
cd frontend
npm.cmd ci --ignore-scripts --cache ../.cache/npm
cd ..
```

Nếu môi trường `.venv` đã được tạo trong phiên này, có thể dùng ngay. `requirements.txt` chỉ ghim các phiên bản từ PyPI và kho CPU chính thức của PyTorch; `unsafe-best-match` cho phép uv tìm đúng bản `+cpu` trên kho PyTorch khi tất cả phiên bản đã được ghim.

Bản mặc định dùng CPU. Muốn GPU cần cài cặp PyTorch/torchvision tương thích CUDA từ nguồn chính thức trong môi trường riêng, giữ Ultralytics 8.4.155 và chạy lại chẩn đoán/kiểm thử; không cam kết GPU hoạt động khi chưa kiểm tra. `DEVICE=auto` chọn CUDA khi `torch.cuda.is_available()` trả true, nếu không chọn CPU. Có thể đặt `DEVICE=cpu` hoặc `DEVICE=0`.

### FFmpeg và font

Dependency `imageio-ffmpeg==0.6.0` cung cấp executable FFmpeg cho Windows; môi trường đã kiểm tra dùng FFmpeg 7.1, có `libx264` và AAC. Không cần FFmpeg trong PATH khi executable này dùng được. Có thể tự cài FFmpeg, thêm thư mục `bin` vào PATH hoặc đặt `FFMPEG_PATH` tới `ffmpeg.exe`. Backend ưu tiên cấu hình explicit, tiếp theo PATH, rồi executable của imageio-ffmpeg.

Font mặc định `assets/fonts/DejaVuSans.ttf` hỗ trợ dấu tiếng Việt, đi kèm `LICENSE_DEJAVU`. Có thể đặt `FONT_PATH` tới một font Unicode khác. Pillow vẽ nhãn, không dùng nhãn mặc định tiếng Anh của Ultralytics.

### MySQL và .env

Nếu chưa có `.env`, sao chép; **không ghi đè tệp đã có**:

```powershell
Copy-Item .env.example .env
```

Điền tài khoản thật trong `.env`; không gửi mật khẩu qua chat hoặc commit. Mật khẩu chứa `#` hoặc khoảng trắng cần đặt trong dấu nháy theo cú pháp dotenv.

Ví dụ quản trị viên MySQL tạo database/tài khoản riêng; thay giá trị mật khẩu minh họa trước khi chạy:

```sql
CREATE DATABASE IF NOT EXISTS helmet_demo
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS 'helmet_demo'@'127.0.0.1'
  IDENTIFIED BY 'THAY_BANG_MAT_KHAU_RIENG';
GRANT ALL PRIVILEGES ON helmet_demo.* TO 'helmet_demo'@'127.0.0.1';
```

Nếu tài khoản đã tồn tại, `CREATE USER IF NOT EXISTS` không đổi mật khẩu. Dùng đúng tài khoản/mật khẩu đã có hoặc nhờ quản trị viên xử lý. Lỗi 1045 là đăng nhập bị từ chối; kiểm tra tài khoản, mật khẩu và quyền theo host, không thay bằng SQLite.

Các biến bắt buộc nằm trong `.env.example`: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `MODEL_PATH`, `DEFAULT_IMGSZ`, `CONFIDENCE_THRESHOLD`, `MAX_UPLOAD_MB`. Bổ sung `DEVICE`, `QUEUE_SIZE`, `MAX_IMAGE_PIXELS`, `FONT_PATH`, `FFMPEG_PATH`. Mặc định upload tối đa 1 GB (1024 MB, `MAX_UPLOAD_MB=1024`), 40 triệu pixel/ảnh hoặc frame, hàng đợi 8 job. Giới hạn dung lượng dùng chung cho ảnh và video; tệp được đọc theo từng khối, không nạp toàn bộ vào RAM.

Tạo database và bảng bằng tài khoản đã cấu hình:

```powershell
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe backend/init_db.py
```

Script chỉ tạo database theo tên cấu hình và bảng `jobs`; không thay đổi tài khoản. Nếu không có quyền tạo database, quản trị viên tạo trước rồi cấp quyền cho database đó.

## Mô hình và custom SCConv

Mã `backend/app/scconv.py` trích nguyên 10 định nghĩa cần thiết từ `../Train3_YOLO26s_SCConv_768.ipynb`, gồm SRU, CRU, ScConv và C3k2_SCConv. Kiểm thử AST đối chiếu định nghĩa với notebook gốc. Khi dùng checkpoint hiện tại, việc lọc lớp chỉ tác động kết quả suy luận, không thay cấu trúc hoặc trọng số mô hình. Trước khi nạp, backend đăng ký đúng các tên `__main__.*` đã được serialize trong checkpoint.

Mặc định dùng ứng viên SCConv 768, chưa khẳng định tốt nhất; args của bản SCConv còn lại ghi 640, không phải 648. Không dùng test để chọn mô hình/ngưỡng, không lấy threshold 0.35 của Stage2. `CONFIDENCE_THRESHOLD=0.25` chưa được tối ưu trên validation. `DEFAULT_IMGSZ=768`, đầu vào suy luận vuông với `rect=False`. Mỗi job lưu cấu hình và SHA256 mô hình thực tế; đổi cấu hình cần khởi động lại.

Chẩn đoán checkpoint mà không sửa trọng số:

```powershell
.\.venv\Scripts\python.exe backend/inspect_checkpoint.py
```

Để thay checkpoint, sửa `MODEL_PATH` rồi chạy lại chẩn đoán. Chỉ dùng checkpoint đáng tin cậy do PyTorch serialized model có thể thực thi mã khi nạp. Checkpoint mới phải tương thích đúng custom, detection YOLO26s, hai C3k2_SCConv và thứ tự ba tên lớp. Head ba lớp được chấp nhận bình thường; head 80 kênh chỉ được chấp nhận cho đúng checkpoint đã phê duyệt ở trên. Demo chỉ hiển thị:

| ID | Tên nội bộ | Nhãn hiển thị | Màu |
|---|---|---|---|
| 0 | motorcycle | Xe máy | Xanh dương |
| 1 | rider_with_helmet | Người đội mũ bảo hiểm | Xanh lá |
| 2 | rider_without_helmet | Người không đội mũ bảo hiểm | Đỏ |

## Chạy ứng dụng từ Demo

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
```

Terminal 2:

```powershell
cd frontend
npm.cmd run dev
```

Mở **http://127.0.0.1:5173**. Hoặc chạy `scripts/start-backend.ps1` và `scripts/start-frontend.ps1`. API docs tại http://127.0.0.1:8000/docs. Chỉ chạy một process backend; không dùng nhiều workers hoặc auto-reload trong demo. Có khóa worker để tránh hai process cùng chạy trên một storage.

Chọn Ảnh/Video, chọn hoặc kéo thả tệp, xem trước rồi bấm Nhận diện. Frontend polling trạng thái mỗi giây; video tiến độ tối đa 95% trước bước chuyển mã và chỉ đạt 100% khi kiểm tra đầu ra thành công. Khi mô hình/MySQL chưa sẵn sàng, giao diện báo rõ và khóa gửi. Thiếu FFmpeg chỉ khóa video. Lịch sử phân trang, thời gian hiển thị theo Việt Nam.

Ảnh chuẩn hóa EXIF, xuất PNG có nhãn tiếng Việt và confidence thật. Video đọc từng frame, xuất MP4 H.264/yuv420p với `faststart`, giữ FPS; kích thước lẻ được đệm tối đa một pixel mỗi chiều. **Giữ âm thanh gốc nếu có, chuyển mã sang AAC**, không tạo âm thanh cho video không có audio. Không cộng detection giữa các frame thành số người vi phạm.

## API

Dữ liệu JSON dùng snake_case. Lỗi có dạng `{"error":{"code":"503","message":"Thông báo tiếng Việt"}}`.

| Endpoint | Hành vi |
|---|---|
| `GET /api/health` | Trạng thái MySQL, mô hình, font, FFmpeg, thiết bị và cấu hình |
| `POST /api/jobs` | Multipart `file` và `media_type=image/video`; trả 202 cùng job |
| `GET /api/jobs?page=1&page_size=10` | `{items,page,page_size,total}`, tối đa 50/trang |
| `GET /api/jobs/{id}` | Trạng thái, tiến độ, cấu hình, thời gian, lỗi và detection ảnh |
| `GET /api/jobs/{id}/input` | Tệp gốc qua ID, không nhận đường dẫn người dùng |
| `GET /api/jobs/{id}/output` | Kết quả hoàn thành; hỗ trợ HTTP Range |

Thêm `?download=true` để tải tệp. Trạng thái nội bộ: `queued`, `processing`, `completed`, `failed`. Detection ảnh gồm `class_id`, `class_name`, `label`, `confidence` (0–1), `xyxy`. API không trả đường dẫn storage. 413 khi quá dung lượng, 415 khi sai/không giải mã được định dạng, 429 khi đầy hàng đợi, 503 khi dịch vụ chưa sẵn sàng. Job đang chờ/đang xử lý lúc khởi động lại được đánh dấu thất bại do gián đoạn; có thể tải tệp lên chạy lại.

## Kiểm thử và giới hạn

```powershell
$env:PYTHONPATH='backend'
$env:PYTHONUTF8='1'
.\.venv\Scripts\python.exe -m pytest backend/tests -q
.\.venv\Scripts\ruff.exe check backend --exclude scconv.py --select F
cd frontend
npm.cmd run build
npm.cmd audit --audit-level=high
```

`backend/smoke_inference.py` đã chạy thành công với checkpoint hiện tại theo chế độ tương thích và bộ lọc 0,1,2. Video smoke dùng ba frame lặp từ ảnh validation thật và âm thanh thử nghiệm, chỉ để kiểm tra pipeline, không đại diện video giao thông thực tế.

`backend/integration_mysql.py --metadata-only` kiểm thử metadata/khôi phục với **instance MySQL kiểm thử riêng** ở cổng 3307; không chạy với database thật của người dùng. Chi tiết cách chạy trong `TEST_RESULTS.md`. Chế độ không có `--metadata-only` kiểm tra inference thật và yêu cầu video smoke đã tạo. Muốn kiểm tra MySQL đang cấu hình trong `.env`, dừng backend đang chạy rồi dùng `backend/integration_mysql.py --configured-db`; script tạo các job kiểm thử và kiểm tra khởi động lại, không sửa tài khoản hoặc mật khẩu.

Giới hạn: ứng dụng cục bộ, không xác thực và không công bố ra mạng; một job inference mỗi lần; chưa đo thời gian đủ để cam kết hiệu năng; video FPS biến đổi được ghi lại theo FPS OpenCV báo, không bảo toàn mọi timestamp VFR; không tracking. Chưa kiểm tra trực tiếp phát video/giao diện trong trình duyệt do phiên công cụ không có trình duyệt kết nối. Ứng dụng dùng ba nhãn đầu ra trên checkpoint head 80 kênh theo lựa chọn hiện tại; không mô tả kiến trúc này thành head ba lớp. Chưa đánh giá đầy đủ chất lượng detection.
# Chuẩn bị huấn luyện lại trên Colab

Bộ chuẩn bị mới nằm trong [training/README.md](training/README.md), gồm notebook SCConv có head 3 lớp thật, ngân sách 70 epoch, lưu/resume qua Drive, dataset sàng lọc và nhật ký sửa nhãn. Mở [báo cáo rà dữ liệu](training/review_report.html) trước khi train: 1.038 ảnh còn chờ duyệt và test hiện thiếu lớp không đội mũ. Model đang phục vụ demo chưa được thay.

