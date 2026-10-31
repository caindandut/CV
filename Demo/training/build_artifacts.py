"""Build a self-contained Colab bundle, notebook and Vietnamese review report."""
import ast
import hashlib
import html
import json
from pathlib import Path
from urllib.parse import quote
import zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent


def cell(kind,source):
    result=dict(cell_type=kind,metadata={},source=source.strip().splitlines(keepends=True))
    result['source']=[s if s.endswith('\n') else s+'\n' for s in result['source']]
    if kind=='code':
        result.update(execution_count=None,outputs=[])
    return result


def notebook():
    cells=[cell('markdown','''# SCConv đúng 3 lớp — tối đa 70 epoch

Chọn **Runtime → Change runtime type → GPU**. Tải `training_colab_70.zip` vào `MyDrive/HelmetSCConv70`.

**Dữ liệu là bản sàng lọc lượt đầu:** 630 train / 97 val / 31 test; 1.038 ảnh mơ hồ đang giữ riêng tại máy. Train chỉ có hai nhóm cảnh, nhiều frame cùng một người; test chưa có GT lớp 2. Chưa đủ để kết luận chất lượng nhận diện vi phạm trên camera mới. Xem README và báo cáo rà, tiếp tục hoàn thiện nhãn trước nghiên cứu cuối cùng.

Chạy SCConv trước. 70 là **tổng ngân sách**, có early stopping và resume; không bảo đảm tránh giới hạn Colab. Notebook không thay model demo.'''),
    cell('code','''import sys, subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.4.155'])
# Giữ PyTorch CUDA do Colab cung cấp; kiểm chứng runtime/preflight ở các ô sau.'''),
    cell('code','''from google.colab import drive
from pathlib import Path
import os, json, zipfile, hashlib
drive.mount('/content/drive')
DRIVE = Path('/content/drive/MyDrive/HelmetSCConv70')
ARCHIVE = DRIVE / 'training_colab_70.zip'
LOCAL = Path('/content/helmet_scconv70')
LOCAL.mkdir(parents=True, exist_ok=True)
assert ARCHIVE.is_file(), f'Hãy tải training_colab_70.zip vào {DRIVE}'
with zipfile.ZipFile(ARCHIVE) as archive:
    for entry in archive.infolist():
        target = (LOCAL / entry.filename).resolve()
        assert target.is_relative_to(LOCAL.resolve()), 'Đường dẫn ZIP không hợp lệ'
    archive.extractall(LOCAL)
for name, expected in json.loads((LOCAL/'files_sha256.json').read_text()).items():
    assert hashlib.sha256((LOCAL/name).read_bytes()).hexdigest() == expected, name
CODE = LOCAL / 'code'
DATA = LOCAL / 'dataset_v1'
os.chdir(LOCAL)
sys.path.insert(0, str(CODE))
import torch, ultralytics
from train_colab import verify_dataset, preflight_batch, run, validate_resume_checkpoint, PRETRAINED_SHA256
assert ultralytics.__version__ == '8.4.155'
assert torch.cuda.is_available(), 'Chọn runtime GPU trước khi chạy chính'
print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU:', torch.cuda.get_device_name(0))
verify_dataset(DATA/'data.yaml')
print(json.loads((DATA/'statistics.json').read_text()))'''),
    cell('markdown','''## Kiểm tra train/save/resume trước run chính
Chỉ vài ảnh train/val, 3 epoch tổng, kích thước 64 để kiểm tra kỹ thuật; đồng thời kiểm tra head forward 768. Kết quả không dùng đánh giá chất lượng. Snapshot smoke được ghi vào Drive.'''),
    cell('code','''from datetime import datetime
SMOKE_NAME = 'smoke_' + datetime.now().strftime('%Y%m%d_%H%M%S')
subprocess.check_call([sys.executable, str(CODE/'smoke_test.py'), '--data', str(DATA),
                       '--output', str(LOCAL/SMOKE_NAME), '--backup-dir', str(DRIVE/SMOKE_NAME)])
assert json.loads((LOCAL/SMOKE_NAME/'result.json').read_text())['passed']'''),
    cell('markdown','''## Cấu hình run
Lần đầu `RESUME=False`. Sau khi bị ngắt: giữ nguyên tên run/dataset, chọn `RESUME=True`. Snapshot `resume.pt` khôi phục optimizer/LR/epoch. `resume.previous.pt` dùng khi snapshot chính hỏng. Đổi tên run khi thử baseline; không chạy hai mô hình cùng lúc.'''),
    cell('code','''VARIANT = 'scconv'  # 'baseline' để chạy đối chứng sau, dùng tên run khác
RUN_NAME = 'scconv_clean70_v1'
RESUME = False
RUN_DIR = LOCAL/'runs'/RUN_NAME
BACKUP_DIR = DRIVE/RUN_NAME
RESUME_FILE = BACKUP_DIR/'resume.pt'  # hoặc 'resume.previous.pt'
PRETRAINED = LOCAL/'pretrained/yolo26s.pt'
assert hashlib.sha256(PRETRAINED.read_bytes()).hexdigest() == PRETRAINED_SHA256
if RESUME:
    payload = torch.load(RESUME_FILE, map_location='cpu', weights_only=False)
    validate_resume_checkpoint(payload, 70)
    assert payload['helmet_training']['variant'] == VARIANT
    assert payload['train_args']['imgsz'] == 768
    BATCH = int(payload['train_args']['batch'])
    print('Tiếp tục từ epoch', payload['epoch']+2, 'đến tối đa 70')
    del payload
    preflight_batch(str(PRETRAINED), VARIANT, candidates=(BATCH,))
else:
    BATCH = preflight_batch(str(PRETRAINED), VARIANT)
print('Batch đã chốt:', BATCH, '| imgsz=768 | epochs=70 | patience=20')'''),
    cell('code','''trainer = run(DATA/'data.yaml', RUN_DIR, BACKUP_DIR, str(PRETRAINED),
              variant=VARIANT, resume=RESUME_FILE if RESUME else None, batch=BATCH)
print('Checkpoint và CSV trên Drive:', BACKUP_DIR)
print('Không dùng checkpoint smoke để demo.')'''),
    cell('markdown','''## Validation và chọn ngưỡng
Ưu tiên checkpoint có AP50–95 lớp 2 tốt nhất. `best.pt` vẫn giữ best theo fitness tổng. AP, P/R, ảnh lỗi và ngưỡng F1 lớp 2 chỉ được tính từ validation. Xem ảnh lỗi trước khi quyết định tích hợp; confidence cao không tự chứng minh dự đoán đúng.'''),
    cell('code','''BEST = BACKUP_DIR/'best-class2.pt'
assert BEST.exists(), 'Chưa có checkpoint lớp 2 hợp lệ'
EVALUATION_NAME = 'val_' + datetime.now().strftime('%Y%m%d_%H%M%S')
VAL_OUTPUT = DRIVE/RUN_NAME/EVALUATION_NAME
subprocess.check_call([sys.executable, str(CODE/'evaluate_model.py'), '--model', str(BEST),
                       '--data', str(DATA/'data.yaml'), '--output', str(VAL_OUTPUT)])
print((VAL_OUTPUT/'metrics.json').read_text())
print('Đề xuất ngưỡng:', VAL_OUTPUT/'selection.json')'''),
    cell('markdown','''## Test cuối — chỉ sau khi khóa cấu hình
Ô dưới tắt mặc định. Test hiện không có GT lớp 2: không dùng để kết luận độ chính xác người không đội mũ. Cần holdout mới có nhãn sạch trước báo cáo cuối. Không dùng test để chọn ngưỡng/epoch. Code đánh dấu lần test theo hash model và dataset; nếu lần chạy lỗi, đọc log trước khi xử lý marker, không chạy lặp để chọn điểm đẹp.'''),
    cell('code','''RUN_FINAL_TEST = False
if RUN_FINAL_TEST:
    TEST_OUTPUT = DRIVE/RUN_NAME/'final_test'
    subprocess.check_call([sys.executable, str(CODE/'evaluate_model.py'), '--model', str(BEST),
                           '--data', str(DATA/'data.yaml'), '--output', str(TEST_OUTPUT),
                           '--final-test', '--selection', str(VAL_OUTPUT/'selection.json')])
else:
    print('Chưa chạy test cuối. Giữ test khóa; hoàn thiện holdout lớp 2 trước kết luận cuối.')''')]
    nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                                    language_info=dict(name='python'),colab=dict(name='Train_SCConv_Clean_70_Colab.ipynb')),
            nbformat=4,nbformat_minor=4)
    for c in cells:
        if c['cell_type']=='code':
            ast.parse(''.join(c['source']))
    (ROOT/'Train_SCConv_Clean_70_Colab.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')


def review_report():
    dataset=ROOT/'dataset_v1'
    rows=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    before={r['id']:r for r in json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))}
    changes=json.loads((dataset/'changes.json').read_text(encoding='utf-8'))
    assets=ROOT/'review_examples'
    assets.mkdir(exist_ok=True)
    font=ImageFont.truetype(str(ROOT.parent/'assets/fonts/DejaVuSans.ttf'),18)
    cards=[]
    for i,change in enumerate(changes):
        row=before[change['id']]
        for side in ['before','after']:
            im=Image.open(row['image']).convert('RGB')
            im.thumbnail((960,540))
            draw=ImageDraw.Draw(im)
            w,h=im.size
            for cls,x,y,bw,bh in change[side]:
                color=['#1689ff','#18b84b','#fa3030'][cls]
                draw.rectangle(((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h),outline=color,width=3)
                draw.text(((x-bw/2)*w,max(0,(y-bh/2)*h-20)),f'Lớp {cls}',fill=color,font=font)
            im.save(assets/f'{i:02d}-{side}.jpg',quality=93)
        cards.append(f'<article><h3>{html.escape(change["id"])}</h3><p>{html.escape(change["reason"])}</p>'
                     f'<div class="pair"><figure><figcaption>Annotation trước sửa</figcaption><img src="review_examples/{i:02d}-before.jpg"></figure>'
                     f'<figure><figcaption>Annotation sau sửa (ảnh có thể vẫn pending)</figcaption><img src="review_examples/{i:02d}-after.jpg"></figure></div></article>')
    pending=[]
    for row in rows:
        if row['new_split']=='pending':
            url=quote('dataset_v1/pending/images/'+row['id'].replace('/','__'))
            pending.append(f'<tr><td><a href="{url}">{html.escape(row["id"])}</a></td><td>{html.escape(row["reason"])}</td></tr>')
    stats=(dataset/'statistics.json').read_text(encoding='utf-8')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Rà nhãn và chuẩn bị SCConv 70 epoch</title>
<style>body{font:16px/1.6 system-ui;max-width:1180px;margin:30px auto;padding:0 20px;color:#19313b}h1,h2{color:#096b60}.notice{background:#fff1cf;padding:18px;border-radius:12px}.pair{display:flex;gap:12px}figure{margin:0;flex:1;min-width:0}img{width:100%}article{border-top:1px solid #ddd;padding:20px 0}h3{overflow-wrap:anywhere}td{padding:12px;border-bottom:1px solid #ddd;vertical-align:top;overflow-wrap:anywhere}table{width:100%;table-layout:fixed}pre{overflow:auto;background:#eef5f5;padding:18px}@media(max-width:700px){.pair{display:block}}</style>
<h1>Rà nhãn và chuẩn bị SCConv 70 epoch</h1>
<p>Đã kiểm tra cú pháp 1.803 ảnh, rà lượt đầu 76 contact sheet; sửa rõ 6 ảnh, loại 7 bản sao. Dữ liệu gốc và model demo được giữ nguyên.</p>
<div class="notice"><b>Bản sàng lọc ban đầu, chưa phải nhãn chuẩn đã chứng nhận.</b> 1.038 ảnh chờ duyệt được tách cả ảnh. Train tập trung vào hai cảnh và nhiều frame cùng một người. Test không có GT lớp 2 nên không thể báo độ chính xác người không đội mũ trên test. Tiếp tục gán nhãn/bổ sung holdout trước nghiên cứu cuối cùng.</div>
<p><a href="README.md">Hướng dẫn Colab</a> · <a href="Train_SCConv_Clean_70_Colab.ipynb">Notebook 70 epoch</a> · <a href="reviews.json">Quyết định từng ảnh</a> · <a href="dataset_v1/changes.json">Nhật ký sửa</a> · <a href="dataset_v1/split_lock.json">Khóa split</a></p>
<h2>Thống kê bản sao đã chia theo nhóm cảnh</h2><pre>STATS</pre>
<h2>Các sửa được kiểm tra trên ảnh lớn</h2>CARDS
<h2>Toàn bộ ảnh chờ duyệt</h2><p>Bấm tên ảnh để xem bản sao; kiểm tra từng người/xe theo README. Không suy nhãn từ dự đoán model.</p><table><thead><tr><th>Ảnh</th><th>Lý do</th></tr></thead><tbody>PENDING</tbody></table></html>'''
    page=page.replace('STATS',html.escape(stats)).replace('CARDS',''.join(cards)).replace('PENDING',''.join(pending))
    (ROOT/'review_report.html').write_text(page,encoding='utf-8')


def bundle():
    files={}
    for name in ['train_colab.py','scconv.py','train_config.json','smoke_test.py','evaluate_model.py','evaluation_metrics.py']:
        files['code/'+name]=ROOT/name
    files['README.md']=ROOT/'README.md'
    files['pretrained/yolo26s.pt']=ROOT/'.cache/yolo26s.pt'
    for path in (ROOT/'dataset_v1').rglob('*'):
        if path.is_file() and 'pending' not in path.relative_to(ROOT/'dataset_v1').parts:
            files[path.relative_to(ROOT).as_posix()]=path
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()}
    target=ROOT/'training_colab_70.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in files.items():
            z.write(path,name)
        z.writestr('files_sha256.json',json.dumps(hashes,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    info=dict(path=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              files=len(files),pending_images_included=False)
    (ROOT/'bundle_info.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info))


if __name__=='__main__':
    notebook()
    review_report()
    bundle()
