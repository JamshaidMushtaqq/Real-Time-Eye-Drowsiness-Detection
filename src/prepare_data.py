
from pathlib import Path
import argparse
import shutil
import cv2
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "raw"
OUT_DEFAULT = ROOT / "data" / "processed"
IMG_SIZE = (64, 64)
VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm"}

def parse_mrl_name(path: Path):
    parts = path.stem.split("_")
    # Example: s0018_03749_0_0_1_0_1_01
    if len(parts) < 8:
        return None
    subject = parts[0]
    try:
        eye_state = int(parts[4])
    except (ValueError, IndexError):
        return None
    if eye_state not in (0, 1):
        return None
    label = "closed" if eye_state == 0 else "open"
    return subject, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--clear", action="store_true", help="remove old processed data first")
    args = parser.parse_args()

    images = [p for p in args.raw.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]
    records = []
    for p in images:
        parsed = parse_mrl_name(p)
        if parsed:
            subject, label = parsed
            records.append({"path": str(p), "subject": subject, "label": label})

    if len(records) < 100:
        raise SystemExit(
            f"Only {len(records)} valid MRL images found under {args.raw}. "
            "Download and extract the MRL archive first. See README.md."
        )

    df = pd.DataFrame(records).drop_duplicates("path").reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise SystemExit("Both closed and open classes must be present.")

    # Subject-wise split: same participant will not appear in multiple splits.
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    trainval_idx, test_idx = next(splitter1.split(df, groups=df["subject"]))
    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()

    splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=43)
    train_idx, val_idx = next(splitter2.split(trainval, groups=trainval["subject"]))
    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    splits = {"train": train, "val": val, "test": test}
    if args.clear and args.out.exists():
        shutil.rmtree(args.out)
    for split_name, split_df in splits.items():
        for label in ("closed", "open"):
            (args.out / split_name / label).mkdir(parents=True, exist_ok=True)

    saved = []
    for split_name, split_df in splits.items():
        for row in split_df.itertuples(index=False):
            src = Path(row.path)
            image = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
            dest = args.out / split_name / row.label / f"{src.stem}.png"
            if not dest.exists():
                if not cv2.imwrite(str(dest), image):
                    continue
            saved.append({
                "path": str(dest.relative_to(ROOT)),
                "subject": row.subject,
                "label": row.label,
                "split": split_name,
                "source": str(src)
            })

    manifest = pd.DataFrame(saved)
    manifest.to_csv(args.out / "manifest.csv", index=False)
    print("\nPreprocessing complete")
    print(manifest.groupby(["split", "label"]).size().unstack(fill_value=0))
    print(f"\nSaved {len(manifest)} images to: {args.out}")
    print("Images are grayscale, resized to 64x64, and normalized to [0,1] during training.")

if __name__ == "__main__":
    main()
