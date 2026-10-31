import os
from pathlib import Path

root = Path("d:/Nam 4/CV/Demo/training/acquisition_2000")

items = []
for entry in os.scandir(root):
    if entry.is_dir(follow_symlinks=False):
        total = sum(f.stat().st_size for f in Path(entry.path).rglob('*') if f.is_file())
        items.append((entry.name, total, True))
    else:
        items.append((entry.name, entry.stat().st_size, False))

items.sort(key=lambda x: x[1], reverse=True)
for name, size, is_dir in items[:30]:
    kind = "DIR " if is_dir else "FILE"
    print(f"[{kind}] {name:35} : {size / (1024**3):.2f} GB ({size / (1024**2):.1f} MB)")
