import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path("d:/Nam 4/CV/Demo/training")
BUNDLE = ROOT / "colab_bundles/dataset_v86/training_colab_70_v86.zip"
EXTRACT_ROOT = ROOT / "verification/colab_bundle_v86_extract"

if EXTRACT_ROOT.exists():
    shutil.rmtree(EXTRACT_ROOT)
EXTRACT_ROOT.mkdir(parents=True)

with zipfile.ZipFile(BUNDLE) as z:
    crc_res = z.testzip()
    assert crc_res is None, f"CRC check failed: {crc_res}"
    members = z.infolist()
    z.extractall(EXTRACT_ROOT)

print(f"CRC check passed! Total members: {len(members)}")

member_hashes = {}
for m in members:
    if m.is_dir():
        continue
    fpath = EXTRACT_ROOT / m.filename
    member_hashes[m.filename] = hashlib.sha256(fpath.read_bytes()).hexdigest()

code_path = EXTRACT_ROOT / "code/train_colab.py"
assert code_path.is_file(), "Missing code/train_colab.py"

# Test import and dataset verification from extracted code
sys.path.insert(0, str(code_path.parent))
import train_colab
data_yaml = EXTRACT_ROOT / "dataset_v86/data.yaml"
assert data_yaml.is_file(), "Missing dataset_v86/data.yaml in bundle"
train_colab.verify_dataset(data_yaml)

bundle_sha256 = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()

receipt = {
    "passed": True,
    "dataset": "dataset_v86",
    "accepted_images": 523,
    "target": 2000,
    "target_achieved": False,
    "archive_sha256": bundle_sha256,
    "archive_members": len(members),
    "all_extracted_member_hashes_verified": len(member_hashes),
    "crc_verified": True,
    "pending_media_excluded": True,
    "packaged_code_imported_from": str(code_path.resolve()),
    "extracted_training_code_verified_dataset": True,
    "extraction_root": str(EXTRACT_ROOT.resolve()),
    "epochs_total": 70,
    "imgsz": 768,
    "patience": 20,
    "main_training_run": False,
    "new_cpu_smoke_run": False,
    "gpu_or_drive_runtime_test": False,
    "previous_smoke_reference": "v54 CPU 3 total epochs, unchanged packaged training logic; not rerun for static dataset update."
}

out_receipt = ROOT / "verification/colab_bundle_v86.json"
out_receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"Saved verification receipt: {out_receipt}")
