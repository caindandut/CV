"""Three-class SCConv trainer, bounded resume and durable epoch checkpoints."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / '.yolo'))
import torch
import ultralytics
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import unwrap_model

from scconv import C3k2_SCConv
import scconv

# Match the demo's explicit __main__ registration, independent of Colab folder names.
for _name in ['GroupBatchnorm2d','SRU','CRU','ScConv','Bottleneck_SCConv','C3k2_SCConv']:
    _class = getattr(scconv,_name)
    _class.__module__ = '__main__'
    setattr(sys.modules['__main__'],_name,_class)

ROOT = Path(__file__).resolve().parent
NAMES = {0: 'motorcycle', 1: 'rider_with_helmet', 2: 'rider_without_helmet'}
CONFIG = json.loads((ROOT/'train_config.json').read_text(encoding='utf-8'))
PRETRAINED_SHA256 = '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_copy(source, dest, backup=False):
    """Write sibling temp, verify bytes, then replace; keep prior complete version."""
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_name(dest.name+'.uploading')
    shutil.copy2(source, temp)
    if sha256(source) != sha256(temp):
        raise IOError(f'Copy verification failed: {dest}')
    if backup and dest.exists():
        previous = dest.with_name(dest.stem+'.previous'+dest.suffix)
        previous_temp = previous.with_name(previous.name+'.uploading')
        shutil.copy2(dest, previous_temp)
        os.replace(previous_temp, previous)
    os.replace(temp, dest)


def build_model(variant='scconv', pretrained='yolo26s.pt', verbose=True):
    base = YOLO(pretrained).model if pretrained else None
    if base is not None:
        assert base.model[-1].nc==80 and not any(isinstance(m,C3k2_SCConv) for m in base.modules()), 'Fresh run needs original COCO pretrained weights'
    cfg = deepcopy(base.yaml) if base is not None else 'yolo26s.yaml'
    model = DetectionModel(cfg, nc=3, ch=3, verbose=verbose)
    model.names = NAMES.copy()
    model.nc = 3
    model.args = get_cfg(overrides={**CONFIG, 'model':'yolo26s.yaml', 'data':None})
    if base is not None:
        # Copy only equal-shaped tensors. Original COCO classification heads are excluded.
        state, target = base.state_dict(), model.state_dict()
        compatible = {k:v for k,v in state.items() if k in target and v.shape==target[k].shape}
        model.load_state_dict(compatible, strict=False)
        print(f'Pretrained transfer: {len(compatible)}/{len(target)} tensors; source SHA256={sha256(pretrained)}')
    if variant == 'scconv':
        head = model.model[-1]
        inputs = [int(i) for i in head.f]
        targets = [i for i, stride in zip(inputs, head.stride.tolist()) if int(stride) in (8,16)]
        if len(targets) != 2:
            raise ValueError(f'Expected P3/P4 targets, got {targets}')
        for i in targets:
            original = model.model[i]
            if original.__class__.__name__ != 'C3k2':
                raise ValueError(f'Unexpected neck block {i}: {type(original)}')
            replacement = C3k2_SCConv(original)
            for key in ['i', 'f', 'type']:
                setattr(replacement, key, getattr(original, key))
            replacement.np = sum(p.numel() for p in replacement.parameters())
            model.model[i] = replacement
    elif variant != 'baseline':
        raise ValueError(variant)
    return model


def assert_contract(model, imgsz=64, raw=True):
    model = unwrap_model(model)
    head = model.model[-1]
    assert model.names == NAMES and model.nc == head.nc == 3, 'Class contract mismatch'
    outputs = [branch[-1].out_channels for key in ['cv3','one2one_cv3'] for branch in getattr(head,key)]
    assert outputs == [3]*6, outputs
    blocks = sum(isinstance(m, C3k2_SCConv) for m in model.modules())
    assert blocks in {0,2}, blocks
    channels = []
    if raw:
        was_training = model.training
        model.eval()
        try:
            parameter = next(model.parameters())
            with torch.no_grad():
                output = model(torch.zeros(1,3,imgsz,imgsz,device=parameter.device,dtype=parameter.dtype))
            def visit(value):
                if isinstance(value, dict):
                    if 'scores' in value:
                        channels.append(int(value['scores'].shape[1]))
                    for key,item in value.items():
                        if key != 'scores':
                            visit(item)
                elif isinstance(value,(tuple,list)):
                    for item in value:
                        visit(item)
            visit(output)
            assert channels == [3,3], channels
        finally:
            model.train(was_training)
    return dict(classification_outputs=outputs, raw_score_channels=channels, custom_blocks=blocks)


def validate_resume_checkpoint(ckpt, total=70):
    if ckpt.get('optimizer') is None or not 0 <= ckpt.get('epoch',-1) < total-1:
        raise ValueError('Checkpoint stripped/finished: use a resumable snapshot from an unfinished run.')
    if ckpt.get('train_args',{}).get('epochs') != total:
        raise ValueError('Resume must preserve the original total epoch budget.')
    if ckpt.get('helmet_training',{}).get('finished'):
        raise ValueError('This run already finished or early-stopped; start a separate experiment.')


def restore_csv(checkpoint, path):
    results=checkpoint.get('train_results',{})
    if not results:
        raise ValueError('Resumable checkpoint lacks its own CSV history')
    keys=list(results)
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        writer.writeheader()
        for index,epoch in enumerate(results['epoch']):
            if float(epoch)<=checkpoint['epoch']+1:
                writer.writerow({key:results[key][index] for key in keys})


class HelmetTrainer(DetectionTrainer):
    variant = 'scconv'
    backup_dir = None
    stop_after = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        if self.resume:
            model = deepcopy(weights).float()
            assert_contract(model, raw=False)
            return model
        return build_model(self.variant, self.args.pretrained if isinstance(self.args.pretrained,str) else None, verbose)

    def resume_training(self, ckpt):
        if self.resume:
            validate_resume_checkpoint(ckpt, self.epochs)
        super().resume_training(ckpt)
        if self.resume:
            assert self.start_epoch == ckpt['epoch']+1
            assert len(self.optimizer.state)==len(ckpt['optimizer']['state'])
            assert [g['lr'] for g in self.optimizer.param_groups]==[g['lr'] for g in ckpt['optimizer']['param_groups']]
            self.resume_verified = dict(start_epoch=self.start_epoch, total=self.epochs,
                                        optimizer_states=len(self.optimizer.state),
                                        optimizer_lr_restored=True,
                                        scheduler_last_epoch=self.start_epoch-1,
                                        ema_updates=self.ema.updates)
            assert self.resume_verified['optimizer_states'] > 0
            state=ckpt.get('helmet_training',{})
            self.best_class2=state.get('best_class2_ap',-1)
            if state.get('stopper'):
                self.stopper.best_epoch=state['stopper']['best_epoch']
                self.stopper.best_fitness=state['stopper']['best_fitness']

    def save_model(self):
        result = super().save_model()
        # Save resumable copy before final_eval strips optimizer from last/best.
        if result and self.backup_dir:
            dest = Path(self.backup_dir)
            metric = self.validator.metrics.box
            ids = list(metric.ap_class_index)
            new_class2_best=False
            if 2 in ids:
                score = float(metric.all_ap[ids.index(2)].mean())
                if score > getattr(self,'best_class2',-1):
                    self.best_class2 = score
                    new_class2_best=True
            checkpoint=torch.load(self.last,map_location='cpu',weights_only=False)
            checkpoint['helmet_training']=dict(variant=self.variant,best_class2_ap=getattr(self,'best_class2',-1),
                                               finished=bool(self.stop or self.epoch+1>=self.epochs),
                                               stopper=dict(best_epoch=self.stopper.best_epoch,best_fitness=self.stopper.best_fitness))
            candidate=self.wdir/'resume-candidate.pt'
            torch.save(checkpoint,candidate)
            del checkpoint
            atomic_copy(candidate, dest/'resume.pt', backup=True)
            if new_class2_best:
                atomic_copy(candidate,dest/'best-class2.pt')
            candidate.unlink()
            if self.best.exists():
                atomic_copy(self.best, dest/'best-unstripped.pt')
            for name in ['results.csv','args.yaml','runtime.json','effective_config.json','dataset_lock.json']:
                path = self.save_dir/name
                if path.exists():
                    atomic_copy(path, dest/name)
            state = dict(epoch_completed=self.epoch+1, total_epochs=self.epochs, batch=self.batch_size,
                         optimizer_states=len(self.optimizer.state), lr=[g['lr'] for g in self.optimizer.param_groups],
                         resume_sha256=sha256(dest/'resume.pt'),
                         best_class2_ap=getattr(self,'best_class2',None),
                         resume_verified=getattr(self,'resume_verified',None))
            (self.save_dir/'resume_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
            atomic_copy(self.save_dir/'resume_state.json',dest/'resume_state.json')
        if self.stop_after and self.epoch+1 >= self.stop_after:
            self.stop = True
        return result


def preflight_batch(pretrained, variant, device='cuda:0', imgsz=768, candidates=(8,4,2)):
    if not torch.cuda.is_available() or not str(device).startswith('cuda'):
        raise RuntimeError('Main training requires a Colab CUDA GPU.')
    for batch in candidates:
        model = optimizer = loss = sample = None
        try:
            model = build_model(variant,pretrained,verbose=False).to(device).train()
            assert_contract(model, imgsz=imgsz)
            optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
            sample = dict(img=torch.rand(batch,3,imgsz,imgsz,device=device),
                          batch_idx=torch.arange(batch,device=device).repeat_interleave(3),
                          cls=torch.tensor([[0],[1],[2]],device=device).repeat(batch,1),
                          bboxes=torch.tensor([[.5,.6,.5,.6],[.4,.4,.2,.4],[.6,.4,.2,.4]],device=device).repeat(batch,1))
            scaler = torch.amp.GradScaler('cuda', enabled=CONFIG['amp'], init_scale=128)
            with torch.autocast(device_type='cuda',enabled=CONFIG['amp']):
                loss = model.loss(sample)[0].sum()
            assert torch.isfinite(loss), 'Nonfinite preflight loss'
            scaler.scale(loss).backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Nonfinite preflight gradients'
            scaler.step(optimizer)
            scaler.update()
            # Validation loader defaults to 2*batch; probe its inference activation memory too.
            model.eval()
            with torch.no_grad(),torch.autocast(device_type='cuda',enabled=CONFIG['amp']):
                model(torch.rand(batch*2,3,imgsz,imgsz,device=device))
            print(f'Preflight batch={batch}: OK, imgsz={imgsz}')
            return batch
        except torch.cuda.OutOfMemoryError:
            print(f'Preflight batch={batch}: OOM; trying next candidate')
        finally:
            del model, optimizer, loss, sample
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError('Even batch=2 failed at 768; stop and use a GPU with more memory.')


def verify_dataset(data_path):
    root = Path(data_path).resolve().parent
    config=yaml.safe_load(Path(data_path).read_text(encoding='utf-8'))
    if config.get('path'):
        assert Path(config['path']).is_absolute() and Path(config['path']).resolve()==root, 'Dataset root must match manifest'
    names=config['names']
    if isinstance(names,list):
        names=dict(enumerate(names))
    assert config['nc']==3 and names==NAMES
    assert config['train']=='train/images' and config['val']=='val/images' and config['test']=='test/images'
    manifest_path, lock_path = root/'manifest.json', root/'split_lock.json'
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert sha256(manifest_path)==lock['manifest_sha256'], 'Frozen manifest changed'
    rows = json.loads(manifest_path.read_text(encoding='utf-8'))
    groups, hashes = {}, {}
    counts = {split:[0,0,0] for split in ['train','val','test']}
    expected_files = {s:set() for s in counts}
    for row in rows:
        split = row['new_split']
        if split=='pending':
            continue
        assert row['review_status'] != 'pending'
        if split != 'train':
            assert row['source_verified'], 'Unverified provenance in evaluation'
        for lookup,key in [(groups,row['group']),(hashes,row['sha256'])]:
            assert lookup.setdefault(key,split)==split, 'Split leakage'
        name = row['id'].replace('/','__')
        expected_files[split].add(name)
        image = root/split/'images'/name
        label = root/split/'labels'/(Path(name).stem+'.txt')
        assert sha256(image)==row['sha256'], image
        parsed = [[int(p[0]),*map(float,p[1:])] for line in label.read_text().splitlines() if (p:=line.split())]
        assert parsed == row['labels'], label
        for b in row['labels']:
            counts[split][b[0]] += 1
    for split,names in expected_files.items():
        assert {p.name for p in (root/split/'images').iterdir() if p.is_file()}==names, 'Untracked image in split'
        assert {p.name for p in (root/split/'labels').glob('*.txt')}=={Path(n).stem+'.txt' for n in names}, 'Untracked label in split'
    assert sorted(r['id'] for r in rows if r['new_split']=='test') == lock['test_ids']
    assert all(all(v>0 for v in counts[s]) for s in ['train','val']), f'Train/val need all 3 classes: {counts}'
    if not all(v>0 for v in counts['test']):
        print(f'WARNING: independent test lacks some classes: {counts["test"]}; no accuracy claims for absent classes.')
    return lock


def run(data, run_dir, backup_dir, pretrained, variant='scconv', resume=None, batch=8,
        device=0, epochs=70, imgsz=768, smoke=False, stop_after=None):
    assert ultralytics.__version__=='8.4.155', ultralytics.__version__
    if not smoke:
        assert epochs==70 and imgsz==768 and torch.cuda.is_available()
        assert sha256(pretrained)==PRETRAINED_SHA256, 'Use the verified original yolo26s.pt in the bundle'
        lock = verify_dataset(data)
    else:
        lock = {'smoke_test':True}
    run_dir = Path(run_dir).resolve()
    backup_dir = Path(backup_dir).resolve()
    cfg = {**CONFIG,'epochs':epochs,'imgsz':imgsz,'batch':batch,'device':device,'data':str(Path(data).resolve()),
           'model':'yolo26s.yaml','pretrained':str(Path(pretrained).resolve()) if pretrained else False,
           'project':str(run_dir.parent),'name':run_dir.name,'exist_ok':False,'save':True,'save_period':-1,
           'plots':True,'val':True,'cache':False,'workers':0 if smoke else 2}
    if smoke:
        cfg.update(amp=False, close_mosaic=0)
    if resume:
        resume = Path(resume).resolve()
        ckpt = torch.load(resume,map_location='cpu',weights_only=False)
        validate_resume_checkpoint(ckpt,epochs)
        previous_lock = json.loads((resume.parent/'dataset_lock.json').read_text(encoding='utf-8'))
        assert previous_lock==lock, 'Dataset lock differs from checkpoint run'
        if not smoke:
            assert ckpt['train_args']['imgsz']==768
        cfg.update(resume=str(resume),model=str(resume),exist_ok=True)
        assert ckpt.get('helmet_training',{}).get('variant')==variant, 'Resume variant differs'
        resume_payload=ckpt
    elif run_dir.exists():
        raise FileExistsError(f'Use another run name or resume: {run_dir}')
    trainer_class = type('ConfiguredHelmetTrainer',(HelmetTrainer,),dict(variant=variant,backup_dir=backup_dir,stop_after=stop_after))
    trainer = trainer_class(overrides=cfg)
    # check_resume restores save_dir in train_args; insist on the requested local location.
    if trainer.save_dir.resolve()!=run_dir:
        raise ValueError(f'Resume run location must be {trainer.save_dir}; requested {run_dir}')
    if resume:
        restore_csv(resume_payload,run_dir/'results.csv')
        del resume_payload
    runtime = dict(ultralytics=ultralytics.__version__,torch=torch.__version__,cuda=torch.version.cuda,
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',variant=variant,
                   smoke=smoke,pretrained_sha256=sha256(pretrained) if pretrained else None)
    (run_dir/'runtime.json').write_text(json.dumps(runtime,indent=2),encoding='utf-8')
    (run_dir/'dataset_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    gradient_seen = set()
    handles = []
    start_time = time.monotonic()
    def setup(t):
        result = assert_contract(t.model,imgsz=imgsz)
        assert result['custom_blocks']==(2 if variant=='scconv' else 0)
        if t.resume:
            assert t.scheduler.last_epoch == t.start_epoch-1
            t.resume_verified['scheduler_last_epoch'] = t.scheduler.last_epoch
        model = unwrap_model(t.model)
        for name,module in model.named_modules():
            if isinstance(module,C3k2_SCConv):
                for p in module.parameters():
                    if p.requires_grad:
                        def record(grad, block=name):
                            if torch.isfinite(grad).all() and grad.abs().sum()>0:
                                gradient_seen.add(block)
                            return grad
                        handles.append(p.register_hook(record))
        effective = dict(args=vars(t.args),optimizer=type(t.optimizer).__name__,
                         groups=[{k:v for k,v in g.items() if k!='params'} for g in t.optimizer.param_groups],contract=result)
        (run_dir/'effective_config.json').write_text(json.dumps(effective,indent=2,default=str),encoding='utf-8')
    def epoch_end(t):
        if variant=='scconv':
            assert len(gradient_seen)==2, f'Missing SCConv gradients: {gradient_seen}'
        elapsed = time.monotonic()-start_time
        completed = t.epoch-t.start_epoch+1
        print(f'Estimate: {(elapsed/completed)*(t.epochs-t.epoch-1)/3600:.2f} hours remaining; excludes interruptions')
        for h in handles:
            h.remove()
        handles.clear()
    def enforce_batch(t):
        if int(t.batch_size)!=batch or int(t.args.batch)!=batch:
            raise RuntimeError('Batch changed after preflight. Stop this run; do not silently alter the experiment.')
    trainer.add_callback('on_train_start',setup)
    trainer.add_callback('on_train_batch_start',enforce_batch)
    trainer.add_callback('on_train_epoch_end',epoch_end)
    trainer.train()
    for path in [trainer.best,trainer.last]:
        if path.exists():
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            assert_contract((ckpt.get('ema') or ckpt['model']).float(),imgsz=imgsz)
            atomic_copy(path,backup_dir/path.name)
    return trainer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--backup-dir',type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--variant',choices=['scconv','baseline'],default='scconv')
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    batch=8
    if args.resume:
        ckpt=torch.load(args.resume,map_location='cpu',weights_only=False)
        validate_resume_checkpoint(ckpt)
        batch=ckpt['train_args']['batch']
        assert ckpt['train_args']['imgsz']==768
        del ckpt
        preflight_batch(str(args.pretrained),args.variant,candidates=(batch,))
    else:
        batch=preflight_batch(str(args.pretrained),args.variant)
    run(args.data,args.run_dir,args.backup_dir,str(args.pretrained),args.variant,args.resume,batch)


if __name__=='__main__':
    main()
