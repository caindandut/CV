import zipfile
import json
import math
from pathlib import Path

archive_path = Path("Demo/training/acquisition_2000/research_kaggle/huanhuyle_vietnam_traffic_v2.zip")

used = set()
for b in Path("Demo/training/acquisition_2000").glob("kaggle_reviews_batch*.json"):
    for item in json.loads(b.read_text(encoding="utf-8")):
        if "huan" in item.get("dataset_ref", ""):
            used.add(item["image_id"])

with zipfile.ZipFile(archive_path, 'r') as zf:
    txt_files = [n for n in zf.namelist() if n.startswith("labels/train/day_cam_24_") and n.endswith(".txt")]
    
    candidates = []
    for txt_file in sorted(txt_files):
        cid = txt_file.split("_")[-1].split(".")[0]
        if cid in used:
            continue
        
        lines = zf.read(txt_file).decode('utf-8').strip().splitlines()
        boxes = []
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = [float(x) for x in parts[1:5]]
                px_xc, px_yc = xc * 1280, yc * 720
                px_w, px_h = w * 1280, h * 720
                boxes.append({
                    "cls": cls_id,
                    "xc": px_xc, "yc": px_yc,
                    "w": px_w, "h": px_h,
                    "x1": px_xc - px_w/2, "y1": px_yc - px_h/2,
                    "x2": px_xc + px_w/2, "y2": px_yc + px_h/2
                })
        
        # We want yc between 440 and 525 so a 240x240 box has y2 <= 640 and y1 >= 320
        # xc between 650 and 900
        for b in boxes:
            if 650 <= b["xc"] <= 920 and 430 <= b["yc"] <= 535 and 65 <= b["h"] <= 145:
                # distance to all other boxes
                min_dist = float('inf')
                for other in boxes:
                    if other is b:
                        continue
                    d = math.hypot(b["xc"] - other["xc"], b["yc"] - other["yc"])
                    if d < min_dist:
                        min_dist = d
                
                if min_dist > 150:
                    candidates.append({
                        "cid": cid,
                        "xc": round(b["xc"], 1),
                        "yc": round(b["yc"], 1),
                        "w": round(b["w"], 1),
                        "h": round(b["h"], 1),
                        "min_dist": round(min_dist, 1)
                    })

candidates.sort(key=lambda x: x["min_dist"], reverse=True)
print(f"Found {len(candidates)} candidates:")
for c in candidates[:25]:
    # propose 240x240 crop centered on xc, yc
    x1 = int(round(c["xc"] - 120))
    y1 = int(round(c["yc"] - 120))
    x2 = x1 + 240
    y2 = y1 + 240
    print(f"cid: {c['cid']}, crop: [{x1}, {y1}, {x2}, {y2}], xc={c['xc']}, yc={c['yc']}, h={c['h']}, min_dist={c['min_dist']}")
