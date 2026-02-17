#!/usr/bin/env python3
import csv
import os
from pathlib import Path
import sys

# ====== CONFIGURATION ======
root_dir = "data_old"  # folder containing .npy files
output_csv = "data_old/data_labels.csv"  # output csv file
# ===========================

ATOMIC = {"normal", "manhole", "speed_bump", "other", "pothole"}

MAP_TO_NUM = {
    "normal": 0,
    "manhole": 1,
    "bump": 2,
    "other": 3,
    "pothole": 4,
    # mixed labels -> 5
}


def normalize_component(comp: str) -> str:
    c = comp.strip().lower().lstrip("_")  # tolerate a leading underscore
    if c in {"speedbump", "speed-bump", "speed_bump"}:
        return "bump"
    return c


def extract_label_from_name(name: str) -> str | None:
    """
    Extract the label from the filename (token after the last underscore, before .npy).
    Mixed labels are separated by '+'.
    """
    stem = Path(name).stem
    if "_" not in stem:
        return None
    label_part = stem.rsplit("_", 1)[-1]
    parts = [normalize_component(p) for p in label_part.split("+")]
    return "+".join(parts)


def to_numeric_label(label: str) -> int | None:
    """
    Convert a normalized label string to the numeric class.
    - Single atomic -> MAP_TO_NUM
    - Mixed (contains '+') -> 5
    """
    if "+" in label:
        parts = label.split("+")
        if not all(p in ATOMIC for p in parts):
            return None
        return 5
    return MAP_TO_NUM.get(label, None)


def main():
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        print(f"Error: root path does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    skipped = 0

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.lower().endswith(".npy"):
                continue
            fpath = Path(dirpath) / fname
            label_str = extract_label_from_name(fname)
            if not label_str:
                skipped += 1
                continue

            y = to_numeric_label(label_str)
            if y is None:
                print(f"Warning: Unrecognized label '{label_str}' in '{fpath}'. Skipping.", file=sys.stderr)
                skipped += 1
                continue

            rows.append((str(fpath), y))

    out_path = Path(output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    if skipped:
        print(f"Skipped {skipped} files due to missing or invalid labels.", file=sys.stderr)


if __name__ == "__main__":
    main()
