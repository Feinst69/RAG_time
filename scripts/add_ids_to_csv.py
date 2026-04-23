"""Add a stable `id` column to the dataset CSV.

Uses the same formula as qdrant_utils.py: uuid5(NAMESPACE_DNS, subject + first_chunk),
so IDs match Qdrant ref_ids without any pipeline changes.
Rows that already have an id are left untouched.
"""

import argparse
from uuid import uuid5, NAMESPACE_DNS

import pandas as pd

from rag_time.config.settings import settings
from rag_time.embeddings import TextSplitter


def compute_ref_id(subject: str, body: str, splitter: TextSplitter) -> str:
    text = f"{subject}\n{body}" if subject else body
    first_chunk = splitter.split_text(text)[0]
    return str(uuid5(NAMESPACE_DNS, subject + first_chunk))


def main():
    parser = argparse.ArgumentParser(description="Add stable IDs to dataset CSV")
    parser.add_argument("-i", "--input", required=True, help="Path to CSV file (modified in place)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if "id" not in df.columns:
        df["id"] = None

    missing = df["id"].isna()
    count = int(missing.sum())
    if count == 0:
        print("All rows already have an id — nothing to do.")
        return

    print(f"Computing IDs for {count} rows...")
    splitter = TextSplitter(chunk_size=settings.chunk_size)

    for idx in df[missing].index:
        subject = str(df.at[idx, "subject"]) if pd.notna(df.at[idx, "subject"]) else ""
        body = str(df.at[idx, "body"]) if pd.notna(df.at[idx, "body"]) else ""
        df.at[idx, "id"] = compute_ref_id(subject, body, splitter)

    df.to_csv(args.input, index=False)
    print(f"Done. {count} IDs added to {args.input}")


if __name__ == "__main__":
    main()
