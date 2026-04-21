import rag_time.retrieval as ragrt
from rag_time.config.settings import settings
import argparse
import json


def main():
    parser= argparse.ArgumentParser(description="RAG Time - Semantic Search")
    parser.add_argument("-q","--query", type=str, required=True, help="The query to search for")
    parser.add_argument("-m","--method", type=str, choices=["semantic", "lexical", "hybrid"], default="hybrid", help="The search method to use")
    parser.add_argument("-f","--filter", type=str, help="The filter to apply to the search results in the format {\"key\":\"value\", ...}")
    args = parser.parse_args()
    query = args.query
    method = args.method
    filter = None if not args.filter else json.loads(args.filter)
    print(f"Filter: {filter}")
   
    results = ragrt.rag_search(query, settings.collection_name, method=method, filter=filter, limit=settings.top_k)
    print(f"RAG Search Results:")
    for key, val in results.items():
        print(f"ID: {key}\n\nScore: {val['score']}\n\nSujet: {val['subject']}\n\nQuestion: {val['body']}\
              \n\nRéponse: {val['answer']}\n\nQueue: {val['queue']}\n\nType: {val['type']}\n\n{'-'*50}\n\n")
   