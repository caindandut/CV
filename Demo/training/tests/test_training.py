import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_colab import build_model, assert_contract, validate_resume_checkpoint, atomic_copy, restore_csv


@pytest.mark.parametrize('variant,blocks',[('scconv',2),('baseline',0)])
def test_real_three_channel_head_and_custom_blocks(variant,blocks):
    torch.set_num_threads(2)
    model = build_model(variant, pretrained=None, verbose=False)
    result = assert_contract(model, imgsz=64)
    assert result['classification_outputs'] == [3]*6
    assert result['raw_score_channels'] == [3, 3]
    assert result['custom_blocks'] == blocks


def test_resume_rejects_stripped_or_finished_checkpoints():
    with pytest.raises(ValueError):
        validate_resume_checkpoint({'epoch': -1, 'optimizer': None}, 70)
    with pytest.raises(ValueError):
        validate_resume_checkpoint({'epoch': 69, 'optimizer': {'state': {}}, 'train_args': {'epochs':70}}, 70)
    with pytest.raises(ValueError):
        validate_resume_checkpoint({'epoch': 2, 'optimizer': {'state': {}}, 'train_args': {'epochs':50}}, 70)


def test_atomic_copy_retains_previous_generation(tmp_path):
    source, dest = tmp_path/'s.pt', tmp_path/'drive'/'last.pt'
    source.write_bytes(b'first')
    atomic_copy(source, dest, backup=True)
    source.write_bytes(b'second')
    atomic_copy(source, dest, backup=True)
    assert dest.read_bytes() == b'second'
    assert dest.with_name('last.previous.pt').read_bytes() == b'first'


def test_csv_restores_checkpoint_epoch_not_latest_drive_log(tmp_path):
    checkpoint={'epoch':1,'train_results':{'epoch':[1,2,3],'loss':[3.0,2.0,1.0]}}
    path=tmp_path/'results.csv'
    restore_csv(checkpoint,path)
    assert path.read_text().splitlines()==['epoch,loss','1,3.0','2,2.0']
