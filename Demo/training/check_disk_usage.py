import os
from pathlib import Path

root = Path("d:/Nam 4/CV/Demo")

def get_dir_size(path):
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
            elif entry.is_dir(follow_symlinks=False):
                total += get_dir_size(entry.path)
    except Exception:
        pass
    return total

print("--- Top Subdirectories in Demo ---")
items = []
for entry in os.scandir(root):
    if entry.is_dir(follow_symlinks=False):
        size = get_dir_size(entry.path)
        items.append((entry.name, size))
    else:
        items.append((entry.name, entry.stat().st_size))

items.sort(key=lambda x: x[1], reverse=True)
for name, size in items:
    print(f"{name:35} : {size / (1024**3):.2f} GB ({size / (1024**2):.1f} MB)")
