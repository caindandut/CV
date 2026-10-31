"""Display independent COCO geometry plus native head details for visual review."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from sprint_proposals import ROOT, ACQ, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    folder = (ACQ / "sprint_5000_5h" / args.batch).resolve()
    if not folder.is_relative_to((ACQ / "sprint_5000_5h").resolve()):
        raise ValueError("Batch escapes review folder")
    proposals = json.loads((folder / "proposals.json").read_text(encoding="utf-8"))
    screened = json.loads((folder / "screening.json").read_text(encoding="utf-8"))
    key = "pageid" if proposals and "pageid" in proposals[0] else "image_id"
    screening = {r[key]: r for r in screened}
    font = ImageFont.truetype(str(ROOT.parent / "assets/fonts/DejaVuSans.ttf"), 12)
    geometries = []
    for proposal in proposals:
        name = str(proposal[key])
        im = Image.open(folder / (name + "_crop.png")).convert("RGB")
        window = proposal["crop"]
        overlay = im.copy()
        draw = ImageDraw.Draw(overlay)
        boxes = []
        for prediction in screening[proposal[key]]["coco_proposals_NOT_GT"]:
            x1, y1, x2, y2 = prediction["xyxy"]
            box = [max(0, round(x1 - window[0])), max(0, round(y1 - window[1])), min(im.width, round(x2 - window[0])), min(im.height, round(y2 - window[1]))]
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            if (box[2] - box[0]) * (box[3] - box[1]) < 0.05 * (x2 - x1) * (y2 - y1):
                continue
            kind = "P" if prediction["class_id"] == 0 else "B"
            index = len(boxes)
            boxes.append({"index": index, "kind": kind, "box_xyxy": box, "confidence": prediction["confidence"], "status": "proposal_not_ground_truth"})
            color = "#facc15" if kind == "P" else "#1689ff"
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], max(0, box[1] - 14)), kind + str(index), fill=color, font=font)
        panel = Image.new("RGB", (320, 460), "#171c23")
        overlay.thumbnail((312, 320))
        panel.paste(overlay, (4, 25))
        pd = ImageDraw.Draw(panel)
        pd.text((4, 4), name + " PENDING", fill="white", font=font)
        heads = []
        for obj in boxes:
            if obj["kind"] != "P":
                continue
            box = obj["box_xyxy"]
            rect = [max(0, box[0] - 3), max(0, box[1] - 4), min(im.width, box[2] + 3), min(im.height, box[1] + max(16, round((box[3] - box[1]) * 0.25)))]
            heads.append({"person_index": obj["index"], "native_rect": rect})
            if len(heads) <= 3:
                head = im.crop(rect)
                scale = min(98 / head.width, 86 / head.height)
                head = head.resize((max(1, round(head.width * scale)), max(1, round(head.height * scale))), Image.Resampling.NEAREST)
                # Pixel repetition is for inspection only; source/crop pixels stay native.
                x = (len(heads) - 1) * 105
                panel.paste(head, (x + 4, 362))
                pd.text((x + 4, 345), "P" + str(obj["index"]), fill="#facc15", font=font)
        panel.save(folder / (name + "_geometry_review.png"))
        geometries.append({key: proposal[key], "canvas": list(im.size), "crop": window, "coco_boxes_NOT_GT": boxes, "head_review_regions": heads, "review_status": "pending"})
    save_json(folder / "geometry_proposals.json", geometries)
    for start in range(0, len(proposals), 12):
        sheet = Image.new("RGB", (1280, 1380), "#171c23")
        for slot, proposal in enumerate(proposals[start:start + 12]):
            panel = Image.open(folder / (str(proposal[key]) + "_geometry_review.png"))
            sheet.paste(panel, (slot % 4 * 320, slot // 4 * 460))
        sheet.save(folder / f"geometry_sheet_{start // 12:03d}.png")
    print(json.dumps({"batch": args.batch, "geometry_candidates": len(geometries), "accepted": 0}))


if __name__ == "__main__":
    main()
