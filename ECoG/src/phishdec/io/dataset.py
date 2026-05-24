import pandas as pd


def load_eval_csv(path: str, text_col: str = "text", label_col: str = "label"):
    df = pd.read_csv(path)
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"eval_set must contain columns: {missing}")

    if text_col != "text":
        df = df.rename(columns={text_col: "text"})
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    return df