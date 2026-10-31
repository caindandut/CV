"""Validation metrics and confidence selection; explicit locked final-test entrypoint."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO
from train_colab import assert_contract, verify_dataset, sha256
from evaluation_metrics import evaluate_image, summarize


def fixed_metrics(model, data_path, split, confidence, output):
    """Class-aware one-to-one IoU>=0.5 counts at exactly the chosen confidence."""
    root=data_path.resolve().parent
    rows=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    totals=[dict(tp=0,fp=0,fn=0,gt=0) for _ in range(3)]
    errors=[]
    gallery=output/'errors'
    gallery.mkdir(parents=True,exist_ok=True)
    for row in rows:
        if row['new_split']!=split:
            continue
        path=root/split/'images'/row['id'].replace('/','__')
        result=model.predict(str(path),imgsz=768,conf=confidence,rect=False,verbose=False,
                             device=0 if torch.cuda.is_available() else 'cpu')[0]
        predictions=[dict(class_id=int(b.cls.item()),confidence=float(b.conf.item()),xyxy=b.xyxy[0].tolist()) for b in result.boxes]
        w,h=row['width'],row['height']
        gt=[dict(class_id=c,xyxy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]) for c,x,y,bw,bh in row['labels']]
        diagnostics=evaluate_image(gt,predictions)
        for total,counts in zip(totals,diagnostics['counts']):
            for k,v in counts.items():
                total[k]+=v
        if diagnostics['missed_gt'] or diagnostics['false_positives']:
            errors.append(dict(id=row['id'],**diagnostics,predictions=predictions))
            if len(errors)<=20:
                result.save(filename=str(gallery/f'{len(errors):02d}-prediction.jpg'))
                im=Image.open(path).convert('RGB')
                draw=ImageDraw.Draw(im)
                for box in gt:
                    color=['blue','green','red'][box['class_id']]
                    draw.rectangle(box['xyxy'],outline=color,width=3)
                    draw.text(box['xyxy'][:2],f'GT {box["class_id"]}',fill=color)
                im.save(gallery/f'{len(errors):02d}-annotation.jpg')
    (output/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    return summarize(totals)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--final-test',action='store_true')
    p.add_argument('--selection',type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new evaluation output directory.')
    lock=verify_dataset(args.data)
    ckpt=torch.load(args.model,map_location='cpu',weights_only=False)
    assert_contract((ckpt.get('ema') or ckpt['model']).float(),imgsz=768)
    del ckpt
    model=YOLO(str(args.model))
    split='test' if args.final_test else 'val'
    selection=None
    if args.final_test:
        if not args.selection:
            raise ValueError('Final test needs selection.json produced on validation.')
        selection=json.loads(args.selection.read_text(encoding='utf-8'))
        assert selection.get('confidence') is not None, 'Validation did not produce a meaningful confidence threshold'
        assert selection['model_sha256']==sha256(args.model)
        assert selection['dataset_lock']==lock
        # One marker per frozen model+dataset prevents accidental repeated test tuning.
        marker=args.data.parent/f'final_test_{sha256(args.model)[:16]}.json'
        if marker.exists():
            raise FileExistsError(f'Final test already invoked: {marker}')
        marker.write_text(json.dumps({'output':str(args.output.resolve()),'status':'started'}),encoding='utf-8')
    metrics=model.val(data=str(args.data.resolve()),split=split,imgsz=768,batch=1,rect=False,
                      conf=.001,iou=.7,device=0 if torch.cuda.is_available() else 'cpu',
                      project=str(args.output.parent.resolve()),name=args.output.name,
                      plots=True,save_json=True)
    ids=list(metrics.box.ap_class_index)
    rows=[]
    for cls,name in model.names.items():
        if cls in ids:
            index=ids.index(cls)
            rows.append(dict(id=cls,name=name,precision=float(metrics.box.p[index]),recall=float(metrics.box.r[index]),
                             ap50=float(metrics.box.ap50[index]),ap50_95=float(metrics.box.ap[index])))
        else:
            rows.append(dict(id=cls,name=name,precision=None,recall=None,ap50=None,ap50_95=None,
                             note='No ground-truth instances in this split; cannot evaluate this class.'))
    report=dict(split=split,model_sha256=sha256(args.model),dataset_lock=lock,
                map50=float(metrics.box.map50),map50_95=float(metrics.box.map),classes=rows,
                precision_recall_note='Ultralytics operating point for maximal mean F1; AP uses full low-confidence curve.',
                speed=metrics.speed)
    report['map_class_ids']=[int(i) for i in ids]
    if not args.final_test:
        assert 2 in ids, 'Validation lacks class 2'
        index=ids.index(2)
        eligible=np.flatnonzero(metrics.box.px>=.001)
        best=int(eligible[np.argmax(metrics.box.f1_curve[index,eligible])])
        if float(metrics.box.f1_curve[index,best])<=0:
            report['selection_error']='Class-2 F1 is zero at every threshold; cannot calibrate confidence.'
            (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            raise ValueError(report['selection_error'])
        selection=dict(model_sha256=sha256(args.model),dataset_lock=lock,
                       confidence=float(metrics.box.px[best]),criterion='maximum class-2 F1 on validation',
                       class2_precision=float(metrics.box.p_curve[index,best]),
                       class2_recall=float(metrics.box.r_curve[index,best]),
                       note='Proposed threshold; review false-positive/false-negative tradeoff before demo integration.')
        (args.output/'selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    else:
        report['selected_confidence']=selection['confidence']
        report['selected_threshold_note']='Fixed confidence, class-aware greedy matching at IoU >= 0.5; separate from full-curve AP.'
    report['selected_threshold_metrics']=fixed_metrics(model,args.data,split,selection['confidence'],args.output)
    if args.final_test:
        marker.write_text(json.dumps({'output':str(args.output.resolve()),'status':'completed'}),encoding='utf-8')
    (args.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
