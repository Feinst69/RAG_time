import json
import argparse
from rag_time.config.settings import settings
from rag_time.qdrant_utils import create_collection, upload_tickets_to_qdrant, delete_collection


def _detect_vector_size(jsonl_path: str) -> int:
    """Read the first JSONL record and infer the token embedding dimension."""
    with open(jsonl_path) as f:
        first = json.loads(f.readline())
    # embeddings: list[per-chunk matrices], each matrix: list[token vectors]
    return len(first["embeddings"][0][0])


def main():
    parser = argparse.ArgumentParser(description="Upload RAG data to Qdrant")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    vector_size = _detect_vector_size(args.input)
    print(f"Detected vector size: {vector_size}")

    print(f"Deleting collection '{settings.collection_name}' from Qdrant...")
    delete_collection(settings.collection_name)

    print(f"Creating collection '{settings.collection_name}' in Qdrant...")
    create_collection(settings.collection_name, vector_size)

    print(f"Uploading tickets from {args.input} to Qdrant...")
    upload_tickets_to_qdrant(args.input)
    print("Upload completed.")

if __name__ == "__main__":
    main()
