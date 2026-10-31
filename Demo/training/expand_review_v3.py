"""Create annotation proposals for visual review, NOT an admitted dataset.

Select one low-artifact variant per source frame, then cap each video to avoid
counting offline augmentations as independent evidence. No detector is used.
"""
import json
import re
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from curate_v2 import normalize_boxes
from prepare_dataset import ROOT, write_json


def artifact_score(path):
    im = cv2.imread(path, 0)
    median = cv2.medianBlur(im, 3)
    impulse = np.abs(im.astype(float)-median)>90
    corners = np.concatenate([im[:10,:10].ravel(), im[-10:,:10].ravel(),
                              im[:10,-10:].ravel(), im[-10:,-10:].ravel()])
    return float(impulse.mean()+.2*(corners<3).mean())


def pixel(box):
    c,x,y,w,h=box
    return [c,(x-w/2)*1280,(y-h/2)*720,(x+w/2)*1280,(y+h/2)*720]


def main():
    rows=json.loads((ROOT/'inventory.json').read_text(encoding='utf-8'))
    initial={r['id']:r for r in json.loads((ROOT/'dataset_v1/manifest.json').read_text(encoding='utf-8'))}
    core=json.loads((ROOT/'dataset_v2/manifest.json').read_text(encoding='utf-8'))
    used={r['id'].split('.rf.')[0] for r in core if r['new_split']!='pending'}
    by_frame=defaultdict(list)
    for i,row in enumerate(rows):
        if row['id'] not in initial or initial[row['id']]['new_split'] not in ['train','val']:
            continue
        if row['source']=='scene:campus_parking' and not re.search(r'IMG_507[056]',row['id']):
            continue
        if row['source'] not in ['scene:campus_parking','scene:campus_gate']:
            continue
        base=row['id'].split('/',1)[1].split('.rf.')[0]
        if any(u.split('/',1)[1]==base for u in used):
            continue
        by_frame[base].append(i)
    by_video=defaultdict(list)
    for indices in by_frame.values():
        i=min(indices,key=lambda j:artifact_score(rows[j]['image']))
        by_video[rows[i]['original_source']].append(i)
    selected=[]
    for video,indices in sorted(by_video.items()):
        indices.sort()
        cap=24 if rows[indices[0]]['source']=='scene:campus_parking' else 5
        positions=np.linspace(0,len(indices)-1,min(cap,len(indices))).astype(int)
        selected.extend(indices[p] for p in positions)
    selected.sort()
    sift=cv2.SIFT_create(nfeatures=6000)
    ref=cv2.imread(rows[585]['image'],0)
    k1,d1=sift.detectAndCompute(ref,None)
    matcher=cv2.BFMatcher()
    proposals=[]
    for i in selected:
        row=rows[i]
        labels=initial[row['id']]['labels']
        fixes=[]
        if row['source']=='scene:campus_parking':
            im=cv2.imread(row['image'],0)
            k2,d2=sift.detectAndCompute(im,None)
            good=[a for a,b in matcher.knnMatch(d1,d2,k=2)
                  if a.distance<.68*b.distance and k1[a.queryIdx].pt[1]<240]
            if len(good)<12:
                continue
            src=np.float32([k1[m.queryIdx].pt for m in good]);dst=np.float32([k2[m.trainIdx].pt for m in good])
            H,mask=cv2.findHomography(src,dst,cv2.RANSAC,3)
            if H is None or mask.sum()<10:
                continue
            for x1,y1,x2,y2 in [(81,112,124,174),(239,108,275,149)]:
                q=cv2.perspectiveTransform(np.float32([[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]]),H)[0]
                a,b=q.min(0);c,d=q.max(0)
                a,b,c,d=max(0,a),max(0,b),min(1280,c),min(720,d)
                if c-a>5 and d-b>5:
                    fixes.append([0,*[round(float(v),2) for v in [a,b,c,d]]])
        # Keep every original box for review; the reviewer must resolve conflicts.
        boxes=[pixel(b) for b in labels]+fixes
        proposals.append(dict(index=i,labels=normalize_boxes(boxes,[1280,720]),
            original_count=len(labels),status='proposal_only',source=row['source']))
    target=ROOT/'review_round3/expansion'
    target.mkdir(exist_ok=True)
    write_json(target/'proposals.json',proposals)
    font=ImageFont.truetype(str(ROOT.parent/'assets/fonts/DejaVuSans.ttf'),18)
    for start in range(0,len(proposals),4):
        canvas=Image.new('RGB',(1920,1140),'#222')
        for j,proposal in enumerate(proposals[start:start+4]):
            im=Image.open(rows[proposal['index']]['image']).convert('RGB').resize((960,540))
            draw=ImageDraw.Draw(im)
            for k,(cls,x,y,w,h) in enumerate(proposal['labels']):
                col=['#1689ff','#18b84b','#fa3030'][cls]
                draw.rectangle(((x-w/2)*960,(y-h/2)*540,(x+w/2)*960,(y+h/2)*540),outline=col,width=2)
                draw.text(((x-w/2)*960,max(0,(y-h/2)*540-18)),f'{cls}:{k}',fill=col,font=font)
            x=j%2*960;y=j//2*570;canvas.paste(im,(x,y));ImageDraw.Draw(canvas).text((x+5,y+542),
                f"proposal {start+j} | original {proposal['index']} | {proposal['source']}",fill='white',font=font)
        canvas.save(target/f'sheet_{start//4:02}.jpg',quality=96)
    print('PROPOSALS ONLY:',len(proposals),'sheets:',(len(proposals)+3)//4)


if __name__=='__main__':
    main()
