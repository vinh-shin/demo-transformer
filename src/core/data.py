import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

DATA_URL = "https://raw.githubusercontent.com/sonlam1102/vihsd/main/data/vihsd.zip"


def load_vihsd(data_dir="../data/vihsd"):
    """Download (if needed) and load the ViHSD train/dev/test CSVs.

    Returns (train_raw, dev_raw, test_raw) as pandas DataFrames.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "vihsd.zip"
    train_csv = data_dir / "vihsd" / "train.csv"
    dev_csv = data_dir / "vihsd" / "dev.csv"
    test_csv = data_dir / "vihsd" / "test.csv"

    if not train_csv.exists():
        print("Đang tải ViHSD dataset...")
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
        print("Đã tải và giải nén vào:", data_dir.resolve())
    else:
        print("Dataset đã tồn tại, bỏ qua bước tải.")

    assert train_csv.exists() and dev_csv.exists() and test_csv.exists(), "Không tìm thấy đủ 3 file csv!"

    train_raw = pd.read_csv(train_csv)
    dev_raw = pd.read_csv(dev_csv)
    test_raw = pd.read_csv(test_csv)
    return train_raw, dev_raw, test_raw
