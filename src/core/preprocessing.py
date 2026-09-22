import re

from underthesea import word_tokenize


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # loại URL
    text = re.sub(r"@\w+", " ", text)               # loại mention
    text = re.sub(r"#", " ", text)                   # loại ký tự #
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def segment_text(text: str) -> str:
    return word_tokenize(text, format="text")


def preprocess_df(df, drop_empty: bool = False, keep_free_text: bool = False):
    """Clean + word-segment `free_text`, rename `label_id` -> `labels`.

    If `drop_empty`, rows whose segmented text is empty (e.g. comments that
    were only a URL/emoji) are dropped — otherwise an all-<pad> sequence can
    turn the attention mask into all -inf and blow up training with NaN loss.
    """
    out = df.copy()
    out["clean_text"] = out["free_text"].progress_apply(clean_text)
    out["segmented_text"] = out["clean_text"].progress_apply(segment_text)
    out = out.rename(columns={"label_id": "labels"})

    if drop_empty:
        n_before = len(out)
        out = out[out["segmented_text"].str.strip() != ""]
        n_dropped = n_before - len(out)
        if n_dropped > 0:
            print(f"  Đã loại {n_dropped} dòng rỗng sau khi làm sạch.")
        out = out.reset_index(drop=True)

    cols = ["segmented_text", "labels"]
    if keep_free_text:
        cols = ["free_text"] + cols
    return out[cols]
