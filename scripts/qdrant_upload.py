from rag_time.config.settings import settings
from rag_time.qdrant_utils import create_collection, upload_tickets_to_qdrant, delete_collection
import argparse

def main():
    parser = argparse.ArgumentParser(description="Upload RAG data to Qdrant")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input JSONL file")
    args = parser.parse_args()

    print(f"Deleting collection '{settings.collection_name}' from Qdrant...")
    delete_collection(settings.collection_name)

    print(f"Creating collection '{settings.collection_name}' in Qdrant...")
    create_collection(settings.collection_name, settings.embedding_dimension)
 
    print(f"Uploading tickets from {args.input} to Qdrant...")
    upload_tickets_to_qdrant(args.input)
    print("Upload completed.")
