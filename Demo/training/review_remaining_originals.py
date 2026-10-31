"""Choose one original-frame variant and prepare visual review; no auto admission."""

import json
from collections import defaultdict

import cv2
import numpy as np
from curate_v2 import normalize_boxes
from expand_review_v3 import artifact_score
from PIL import Image, ImageDraw, ImageFont
from prepare_dataset import ROOT, write_json


def main():
    out = ROOT / "acquisition_2000/original_review"
    out.mkdir(exist_ok=True)
    if (out / "proposals.json").exists():
        raise FileExistsError(
            "Review index already exists; do not renumber recorded decisions."
        )
    original = json.loads((ROOT / "inventory.json").read_text(encoding="utf-8"))
    v3 = json.loads((ROOT / "dataset_v3/manifest.json").read_text(encoding="utf-8"))
    expansion = json.loads(
        (ROOT / "expansion_decisions_v3.json").read_text(encoding="utf-8")
    )
    used = {
        r["id"].split("/", 1)[1].split(".rf.")[0]
        for r in v3
        if r["new_split"] != "pending" and not r["id"].startswith("commons/")
    }
    used.update(r["id"].split("/", 1)[1].split(".rf.")[0] for r in expansion)
    frames = defaultdict(list)
    for index, row in enumerate(original):
        frame = row["id"].split("/", 1)[1].split(".rf.")[0]
        if frame not in used:
            frames[frame].append(index)
    selected = [
        min(items, key=lambda index: artifact_score(original[index]["image"]))
        for items in frames.values()
    ]
    # Review isolated parking riders first; dense forecourt needs more manual work.
    order = {
        "scene:campus_parking": 0,
        "scene:campus_gate": 1,
        "scene:public_road": 2,
        "scene:campus_forecourt": 3,
    }
    selected.sort(key=lambda index: (order.get(original[index]["source"], 4), index))
    sift = cv2.SIFT_create(nfeatures=6000)
    k1, d1 = sift.detectAndCompute(cv2.imread(original[585]["image"], 0), None)
    matcher = cv2.BFMatcher()
    proposals = []
    for index in selected:
        row = original[index]
        projected = []
        if row["source"] == "scene:campus_parking":
            k2, d2 = sift.detectAndCompute(cv2.imread(row["image"], 0), None)
            good = [
                a
                for pair in matcher.knnMatch(d1, d2, k=2)
                if len(pair) == 2
                for a, b in [pair]
                if a.distance < 0.68 * b.distance and k1[a.queryIdx].pt[1] < 240
            ]
            if len(good) >= 12:
                src = np.float32([k1[m.queryIdx].pt for m in good])
                dst = np.float32([k2[m.trainIdx].pt for m in good])
                matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3)
                if matrix is not None and mask.sum() >= 10:
                    for x1, y1, x2, y2 in [(81, 112, 124, 174), (239, 108, 275, 149)]:
                        points = cv2.perspectiveTransform(
                            np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]),
                            matrix,
                        )[0]
                        a, b = points.min(0)
                        c, d = points.max(0)
                        a, b, c, d = max(0, a), max(0, b), min(1280, c), min(720, d)
                        if c - a > 5 and d - b > 5:
                            projected.append(
                                [0, *[round(float(v), 2) for v in [a, b, c, d]]]
                            )
        proposals.append(
            {
                "index": index,
                "id": row["id"],
                "source": row["source"],
                "labels": row["labels"] + normalize_boxes(projected, [1280, 720]),
                "original_count": len(row["labels"]),
                "projected_count": len(projected),
                "status": "proposal_only",
            }
        )
    write_json(out / "proposals.json", proposals)
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 18)
    for start in range(0, len(proposals), 4):
        sheet = Image.new("RGB", (1920, 1140), "#222")
        for slot, proposal in enumerate(proposals[start : start + 4]):
            im = (
                Image.open(original[proposal["index"]]["image"])
                .convert("RGB")
                .resize((960, 540))
            )
            draw = ImageDraw.Draw(im)
            for box_index, (cls, x, y, w, h) in enumerate(proposal["labels"]):
                color = ["#1689ff", "#18b84b", "#fa3030"][cls]
                draw.rectangle(
                    (
                        (x - w / 2) * 960,
                        (y - h / 2) * 540,
                        (x + w / 2) * 960,
                        (y + h / 2) * 540,
                    ),
                    outline=color,
                    width=2,
                )
                draw.text(
                    ((x - w / 2) * 960, max(0, (y - h / 2) * 540 - 18)),
                    f"{cls}:{box_index}",
                    fill=color,
                    font=font,
                )
            x, y = slot % 2 * 960, slot // 2 * 570
            sheet.paste(im, (x, y))
            ImageDraw.Draw(sheet).text(
                (x + 5, y + 542),
                f"R{start + slot:03d} | original {proposal['index']} | {proposal['source']}",
                fill="white",
                font=font,
            )
        sheet.save(out / f"sheet_{start // 4:03d}.jpg", quality=96)
    print(json.dumps({"proposals": len(proposals), "accepted": 0}))


if __name__ == "__main__":
    main()
