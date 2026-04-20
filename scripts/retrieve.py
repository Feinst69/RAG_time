from rag_time.retrieval import do_semantic_search, do_lexical_search, do_hybrid_search
import argparse

def main():
    parser= argparse.ArgumentParser(description="RAG Time - Semantic Search")
    parser.add_argument("-q","--query", type=str, required=True, help="The query to search for")
    args = parser.parse_args()
    query = args.query
    
    """
    print("="*50)
    results = do_semantic_search(query)
    print("="*50)
    print(f"Semantic Search Results:")
    for key, val in results.items():
        print(f"ID: {key}\n\n Score: {val['score']}\n\nSujet: {val['subject']}\n\nQuestion: {val['body']}\n\nRéponse; {val['answer']}\n\n{'-'*50}\n\n")
   
    print("="*50)
    results = do_lexical_search(query)
    print("="*50)
    print(f"Lexical Search Results:")
    for key, val in results.items():
        print(f"ID: {key}\n\n Score: {val['score']}\n\nSujet: {val['subject']}\n\nQuestion: {val['body']}\n\nRéponse; {val['answer']}\n\n{'-'*50}\n\n")
    """
    print("="*50)
    results = do_hybrid_search(query)
    print("="*50)
    print(f"Hybrid Search Results:")
    for key, val in results.items():
        print(f"ID: {key}\n\n Score: {val['score']}\n\nSujet: {val['subject']}\n\nQuestion: {val['body']}\n\nRéponse; {val['answer']}\n\n{'-'*50}\n\n")