"""Tiny CPU training test on real copied images; never a quality benchmark."""
import json
import argparse
from pathlib import Path
import shutil
import torch
import yaml
from train_colab import ROOT, run, assert_contract


def main():
    torch.set_num_threads(2)
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'verification'/'smoke')
    parser.add_argument('--data',type=Path,default=ROOT/'dataset_v1')
    parser.add_argument('--backup-dir',type=Path)
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError('Smoke output exists; archive it before repeating.')
    source=args.data.resolve()
    rows=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    data=out/'data'
    # Technical smoke uses copies of reviewed train/val, never the locked test.
    for split,n in [('train',4),('val',2)]:
        selected=[]
        for cls in [2,1]:
            limit=n//2 if cls==2 else n
            for row in rows:
                if row['new_split']==split and row not in selected and any(b[0]==cls for b in row['labels']):
                    selected.append(row)
                    if len(selected)>=limit:
                        break
        for r in selected:
            name=r['id'].replace('/','__')
            for folder,file_name in [('images',name),('labels',Path(name).stem+'.txt')]:
                target=data/split/folder/file_name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/split/folder/file_name,target)
    cfg=data/'data.yaml'
    cfg.write_text(yaml.safe_dump(dict(path=str(data.resolve()),train='train/images',val='val/images',nc=3,
                                      names=['motorcycle','rider_with_helmet','rider_without_helmet'])),encoding='utf-8')
    local,backup=out/'run',args.backup_dir.resolve() if args.backup_dir else out/'durable'
    first=run(cfg,local,backup,None,batch=2,device='cpu',epochs=3,imgsz=64,smoke=True,stop_after=1)
    ckpt=torch.load(backup/'resume.pt',map_location='cpu',weights_only=False)
    assert ckpt['epoch']==0 and ckpt['optimizer']['state']
    assert_contract(ckpt['ema'].float(),imgsz=768)
    del ckpt,first
    resumed=run(cfg,local,backup,None,resume=backup/'resume.pt',batch=2,device='cpu',epochs=3,imgsz=64,smoke=True)
    assert resumed.start_epoch==1 and resumed.epoch==2 and resumed.epochs==3
    state=json.loads((backup/'resume_state.json').read_text())
    assert state['resume_verified']['optimizer_states']>0
    assert state['resume_verified']['start_epoch']==1
    assert (backup/'resume.previous.pt').exists()
    assert (backup/'best-class2.pt').exists()
    summary=dict(passed=True,note='CPU technical test, random initialization, no quality claims; 3 TOTAL epochs',
                 first_saved_epoch=0,resumed_start_epoch=resumed.start_epoch,final_epoch=resumed.epoch,
                 resumed=state['resume_verified'],head_768_verified=True)
    (out/'result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    main()
