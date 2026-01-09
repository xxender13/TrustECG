import ast
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/ptb-xl")

def load_metadata():
    return pd.read_csv(DATA_DIR / "ptbxl_database.csv")

def load_scp_statements():
    return pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)

def extract_superclasses(scp_codes, scp_map):
    superclasses = set()
    for code in scp_codes:
        if code in scp_map.index:
            sc = scp_map.loc[code, "diagnostic_class"]
            if isinstance(sc, str):
                superclasses.add(sc)
    return list(superclasses)

def main():
    df = load_metadata()
    scp_map = load_scp_statements()

    # parse scp_codes from string → dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # extract diagnostic superclasses
    df["diagnostic_superclass"] = df["scp_codes"].apply(
        lambda x: extract_superclasses(x.keys(), scp_map)
    )

    # define labels
    def assign_label(classes):
        if classes == ["NORM"]:
            return 0  # Normal
        if any(c in ["MI", "STTC", "CD", "HYP"] for c in classes):
            return 1  # Abnormal
        return None  # ambiguous / exclude

    df["label"] = df["diagnostic_superclass"].apply(assign_label)

    # drop ambiguous
    df_clean = df.dropna(subset=["label"]).copy()
    df_clean["label"] = df_clean["label"].astype(int)

    # save
    out_path = Path("data/processed/ptbxl_labels.csv")
    df_clean.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(df_clean["label"].value_counts())

if __name__ == "__main__":
    main()
