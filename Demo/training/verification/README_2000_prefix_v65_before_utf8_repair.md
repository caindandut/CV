# Tiến độ bộ dữ liệu mục tiêu 2.000 ảnh

Cập nhật ngày 16/10/2026. **Chưa hoàn thành mục tiêu 2.000 ảnh sạch.**

## Bản đã xuất

`dataset_v65` có **485 ảnh được chấp nhận**, đạt **24,25% về số lượng**, còn **1.515 ảnh**. Đây không phải phần trăm độ chính xác của mô hình hoặc chứng nhận nhãn không thể có sai sót.

| Tập | Ảnh | Box xe máy | Box người đội mũ | Box người không đội mũ |
|---|---:|---:|---:|---:|
| Train | 348 | 750 | 239 | 142 |
| Validation | 69 | 128 | 111 | 21 |
| Test | 68 | 92 | 59 | 9 |

Trong 485 ảnh có 377 frame/ảnh từ dữ liệu gốc và 96 ảnh bổ sung (88 Commons, 2 NCpedia, 18 Kaggle) có bằng chứng chụp tại Việt Nam. Có bốn ảnh nhãn rỗng đã kiểm tra: người đội mũ đứng cạnh công sự, người đạp xe đạp hai người trùm áo chống nắng đi xe đạp ở Xuân Mai và người bán ổi đứng cạnh xe hàng có bàn đạp; các crop này không có đối tượng thuộc ba lớp của đề tài. Không nhân bản, xoay hoặc đổi kích thước để tăng bộ đếm.

**1.381 ảnh có mã Commons đã tải về là ứng viên**, còn phải rà nhãn và nguồn; không cộng toàn bộ vào số ảnh sạch. Danh sách tải gồm cả ảnh đường phố đông người, vật thể quá nhỏ, ô tô và trường hợp metadata tìm kiếm chưa đủ để chứng minh địa điểm. Chỉ các ảnh có quyết định chấp nhận mới vào tập sử dụng. `dataset_v65/pending` có 1.508 ảnh gốc/ảnh bổ sung cũ chờ rà, không bao gồm tất cả ứng viên mới tải.

## Hồ sơ kiểm tra

- `dataset_v65/manifest.json`: ảnh, nhãn, nguồn, hash, quyết định và tập tương ứng.
- `dataset_v65/changes.json` và `review_audit_2000.json`: nhãn trước/sau, lý do, liên kết bằng chứng.
- `dataset_v65/ATTRIBUTION.md`: tác giả, URL và giấy phép ảnh bổ sung; dữ liệu gốc giữ nguyên điều kiện của chủ sở hữu.
- [review_v65.html](review_v65.html): đối chiếu box trước/sau. Đối với ảnh Commons/NCpedia, phía trước chưa có nhãn, không phải ảnh âm tính.
- `verification/dataset_v65.json`: kết quả kiểm tra tự động; `acquisition_2000/progress.json`: bộ đếm mới nhất.
- `acquisition_2000/original_reviews_batch01..25.json`: đã sàng lọc/rà R000–R590. R328–R341 mới được sàng lọc contact sheet và vẫn pending. Batch12/13 bổ sung các crop đã kiểm tra pixel gốc; các ảnh toàn cảnh thiếu nhãn vẫn pending. Đã sàng lọc hết 591 đề xuất R000–R590; `original_proposal_screening_completion.json` kiểm tra không thiếu/trùng chỉ số trong nhật ký. Chỉ số tiếp theo 591 trong progress là điểm kết thúc danh sách, không phải ảnh mới. Các ảnh pending vẫn chưa được xác nhận sạch. R400–R431 đã sàng lọc, bốn crop R408/R414/R426/R429 có nhãn thủ công; ảnh toàn cảnh chưa đủ nhãn vẫn pending; không xem ảnh pending là đã làm sạch.
- `acquisition_2000/external_reviews_batch01..30.json`: nhãn thủ công các ảnh mới được chấp nhận; dự đoán model không được dùng làm nhãn chuẩn.

Có 56 ảnh Commons, 2 ảnh NCpedia, 6 ảnh Kaggle và 89 frame gốc sử dụng vùng cắt, lưu PNG giữ nguyên pixel; không thay thế ảnh nguồn. Manifest có `source_image`, `source_sha256`, `source_size`, `crop`; frame gốc còn có `source_id`, `source_labels` để lưu nhãn trước khi cắt. Box được gán thủ công và kiểm tra trên toàn vùng cắt. Mỗi ảnh nguồn chỉ được chấp nhận một lần; không dùng nhiều crop của một ảnh để tăng số lượng. Bộ kiểm chứng đối chiếu từng pixel với vùng cắt từ ảnh nguồn. Các vùng ngoài crop không xuất hiện trong ảnh train nên không bị biến thành nền âm tính có đối tượng thiếu nhãn; ảnh toàn cảnh gốc vẫn giữ pending.

## Camera11: b?n v65

B? sung m?t crop t? frame550: hai ng??i ng?i tr?n m?t xe, m?i ng??i ??i m? c? box ri?ng; th?m m?t xe ?? ch? c?n ph?n ph?a sau ? m?p tr?n tr?i. ?? xem native550/551 v? ki?m tra geometry to?n crop, kh?ng d?ng d? ?o?n model l?m nh?n. B?y ?nh m?u kh?c v? b?n frame ph? v?n kh?ng ???c nh?n; ca ??u t?i m?u, ng??i ?i b? v? ng??i ??p xe kh?ng t? chuy?n th?nh class2.

Importer ch? m? camera11/24 khi ?? b?ng ch?ng c?a ??ng camera v? hash ngu?n; kh?ng d?ng gate24 thay cho11, c?c camera kh?c ch?a ???c m?. To?n b? ngu?n Huan v?n m?t nh?m train. **89 test training pass**, ki?m tra lint pass; xem `verification/kaggle_cam11_tests01.txt`, `verification/kaggle_cam11_export_v65.json`, `verification/dataset_v65.json`. G?i Colab v?n v56 (460 ?nh), ch?a train ch?nh.

VnTraffic509 c? b?n nh?p m?t rider t?c tr?n tr?n xe d?ng, ?? ??i chi?u l?i pixel t? video trong ZIP v? s?a box t?i ch?n/b?nh xe nh?n th?y. Draft ch?a ???c nh?n v? importer ri?ng ch?a ho?n thi?n; kh?ng c?ng v?o485. Kho Zenodo39,5MB ?? c? t? tr??c v? ???c t?i s? d?ng, kh?ng t?i tr?ng. Th? t?i Commons4080px b? HTTP429, ?? d?ng; thumbnail c? kh?ng b? thay.

## L? camera24 b? sung: b?n v64

Th?m 6 ?nh v?i **6 xe m?y, 9 ng??i ??i m?**, g?m ba c?p rider c? box ri?ng. Kh?ng c? nh?n l?p2 m?i trong l? n?y. ?nh680/860/900 c? hai ng??i m?i xe; trang ph?c v? bal? kh?c nhau ???c ??i chi?u ?? kh?ng nh?n c?c frame g?n nhau c?a c?ng ??i t??ng. ?nh700/820/900 c? ph?n xe ch?m/c?t m?p ?nh; ch? g?n box ph?n nh?n th?y, kh?ng ??c l??ng ph?n ngo?i khung. C?c v?ng privacy overlay gi? trong n?n kh?ng ph? ??i t??ng; crop820 d?ng tr??c strip.

?nh660/661, 800/801 v? 840/839 v?n ch? duy?t v? kh? x?c ??nh ??u ng??i l?i. Crop ph?ng g?n ban ??u c?a661 th?c ch?t l? ??u ng??i ng?i sau; ki?m tra l?i ?? lo?i c? ?? xu?t660 tr??c xu?t. B?n nh?p b? lo?i ???c l?u `batch03/rejected_drafts01.json`, kh?ng d?ng l?m ground truth. Kh?ng ?p nh?n kh?ng ??i m? t? c?c ca ch?a r?. ?nh720/740/780 ch?a c? crop ?? nh?n, v?n pending.

H? s?: `acquisition_2000/kaggle_reviews_batch05.json`, `acquisition_2000/research_kaggle/huanhuyle_v2_cam24_review/batch03/review_status01.json`, `verification/kaggle_cam24_export_v64.json`, `verification/dataset_v64.json`. G?i Colab v?n v56 (460 ?nh); ch?a train ch?nh ho?c thay model demo.

## L? camera24 b? sung: b?n v63

Th?m 4 ?nh: 4 xe m?y, 4 rider ??i m? v? 1 rider kh?ng ??i m?. Ca t?c tr?n ?o ?en ch?n frame254 v?i ph?n xe/bi?n s? ??y ??, kh?ng nh?n th?m frame253/255 c?a c?ng ng??i. Ca ch? hai ng??i c? hai box rider ri?ng; ?nh ph?402 l?m r? chi?c m? xanh ???c c?m b?n c?nh, kh?ng ph?i ??u ng??i th? ba. ?nh48 v?n pending v? m?/t?c ng??i l?i ch?a ph?n bi?t ch?c. C?c v?ng c?t c?n d?nh rider n?n ho?c ??u ng??i kh?c ch?a ???c nh?n.

?? ki?m ch?ng ngu?n/ZIP/metadata/crop, nh?n YOLO kh?p t?a ?? th? c?ng, kh?ng tr?ng ngu?n gi?a split, 68 ID/nh?n test gi? nguy?n t? v62 v? to?n b? 1.803 ?nh/nh?n g?c kh?ng ??i. Xem `verification/dataset_v63.json`, `verification/kaggle_cam24_export_v63.json`, `acquisition_2000/kaggle_reviews_batch04.json`. ??y l? ki?m tra t?nh nh?t qu?n d? li?u, kh?ng ph?i mAP ho?c b?o ??m nh?n ho?n h?o. G?i Colab hi?n v?n v56 (460 ?nh); ch?a train ch?nh.

## Ngu?n camera m?i: b?n v62

?? b? sung m?t crop t? `huanhuyle/vietnam-traffic-detection` v2, camera `day_cam_24`: m?t xe m?y v? m?t ng??i ??i m?. ?nh g?c, th?nh vi?n ZIP, metadata, b?ng ch?ng ??a ?i?m v? crop ??u c? SHA256. Ki?m tra nh?n sau xu?t kh?p t?a ?? th? c?ng; test v61 gi? nguy?n c? 68 ID v? nh?n.

B?ng ch?ng Vi?t Nam ch? ?p d?ng cho c?nh camera24 ?? ??i chi?u: bi?n s? xe bu?t 51B-037.49 v? ch? tuy?n xe ti?ng Vi?t. Kh?ng suy ra to?n b? 51.282 ?nh trong g?i ??u ? Vi?t Nam; nh?m Helsinki b? lo?i, c?c camera ch?a x?c minh v?n ch?. Gi?y ph?p Apache 2.0 l? khai b?o c?a nh? ph?t h?nh Kaggle huan huy le; t?c gi? video g?c kh?ng ???c cung c?p, kh?ng tuy?n b? ?? ki?m ch?ng quy?n s? h?u camera. To?n b? ngu?n n?y ???c gi? chung nh?m **train**, kh?ng k? th?a th? m?c train/val c?a Kaggle.

H? s?: `acquisition_2000/kaggle_reviews_batch03.json`, `acquisition_2000/research_kaggle/huanhuyle_v2_country_review/cam24_country_gate01.json`, `verification/kaggle_cam24_export_v62.json`. C? 8 frame camera24 ?? r? ? pixel g?c; ch? crop ?? ?? nh?n ???c ch?p nh?n. Ca t?c tr?n v?i xe b? c?t kh?i khung v?n pending. B? ki?m th? training hi?n c? **86 test pass**, bao g?m ki?m tra ngu?n camera, b?ng ch?ng thay ??i, gi?i h?n train v? h?i quy c?c ngu?n c?; kh?ng ph?i ph?p ?o ?? ch?nh x?c m? h?nh.

