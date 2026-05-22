"""
Сканирует картинки в датасете и печатает уникальные размеры (ширина × высота)
и сколько файлов каждого размера.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Windows-консоль часто cp1252 — без этого русский текст в print падает.
# _reconfigure = getattr(sys.stdout, "reconfigure", None)
# if callable(_reconfigure):
#     try:
#         _reconfigure(encoding="utf-8", errors="replace")
#     except (OSError, ValueError):
#         pass

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_dataset_root() -> Path:
    here = Path(__file__).resolve().parent
    for root in (here, here.parent):
        candidate = root / "dataset"
        if candidate.is_dir():
            return candidate
    return here / "dataset"


def iter_image_paths(dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for split in ("train", "test"):
        img_dir = dataset_root / split / "img"
        if not img_dir.is_dir():
            continue
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(p)
    return sorted(paths)


def main() -> None:
    dataset_root = find_dataset_root()
    if not dataset_root.is_dir():
        print(f"Папка датасета не найдена: {dataset_root}")
        return

    paths = iter_image_paths(dataset_root)
    if not paths:
        print(f"Картинки не найдены под {dataset_root}")
        return

    print(f"Корень датасета: {dataset_root}")
    print(f"Всего файлов: {len(paths)}\n")

    sizes: Counter[tuple[int, int]] = Counter()
    errors: list[tuple[Path, str]] = []

    for path in tqdm(paths, desc="Scanning images"):
        try:
            with Image.open(path) as im:
                w, h = im.size
            sizes[(w, h)] += 1
        except (OSError, ValueError, UnidentifiedImageError) as e:
            errors.append((path, str(e)))

    # Сортировка: сначала по числу картинок (убывание), потом по площади
    def sort_key(item: tuple[tuple[int, int], int]) -> tuple[int, int, int]:
        (w, h), cnt = item
        return (-cnt, w * h, w)

    print("Уникальные размеры (ширина × высота) и количество:\n")
    f = open("./tmp_size.txt", "w")
    for (w, h), cnt in sorted(sizes.items(), key=sort_key):
        # print(f"  {w} X {h}:  {cnt}")
        f.write(f"  {w} X {h}:  {cnt}\n")
    print(f"\nВсего уникальных размеров: {len(sizes)}")

    if errors:
        print(f"\nНе удалось прочитать {len(errors)} файл(ов), примеры:")
        for p, msg in errors[:10]:
            print(f"  {p}: {msg}")
        if len(errors) > 10:
            print(f"  ... и ещё {len(errors) - 10}")


if __name__ == "__main__":
    main()
