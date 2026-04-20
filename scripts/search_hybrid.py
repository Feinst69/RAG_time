"""CLI for hybrid retrieval over ticket embeddings stored in JSONL."""

from __future__ import annotations

import argparse

from rag_time.retriever import HybridRetriever, SearchFilters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recherche hybride BM25 + vectorielle sur un fichier JSONL"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Chemin du fichier JSONL avec chunks et embeddings"
    )
    parser.add_argument("-q", "--query", required=True, help="Question utilisateur")
    parser.add_argument(
        "-k", "--top-k", default=5, type=int, help="Nombre de resultats a afficher"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Modele Hugging Face utilise pour encoder la requete",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device pour l'encodage de la requete",
    )
    parser.add_argument(
        "--vector-weight",
        default=0.5,
        type=float,
        help="Poids du score vectoriel entre 0.0 et 1.0",
    )
    parser.add_argument("--type", dest="ticket_type", help="Filtre pre-search sur le type")
    parser.add_argument("--queue", help="Filtre pre-search sur la queue")
    parser.add_argument("--priority", help="Filtre pre-search sur la priorite")
    parser.add_argument("--language", help="Filtre pre-search sur la langue")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    retriever = HybridRetriever(
        jsonl_path=args.input,
        embedding_model=args.model,
        device=args.device,
        vector_weight=args.vector_weight,
    )
    filters = SearchFilters(
        ticket_type=args.ticket_type,
        queue=args.queue,
        priority=args.priority,
        language=args.language,
    )

    results = retriever.search(query=args.query, top_k=args.top_k, filters=filters)
    if not results:
        print("Aucun resultat pour cette requete avec les filtres fournis.")
        return

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        print(f"[{rank}] score={result.score:.4f}")
        print(
            f"    bm25={result.bm25_score:.4f} vector={result.vector_score:.4f} "
            f"line={chunk.source_line} chunk={chunk.chunk_index}"
        )
        print(
            f"    type={chunk.ticket_type} queue={chunk.queue} "
            f"priority={chunk.priority} language={chunk.language}"
        )
        if chunk.subject:
            print(f"    subject={chunk.subject}")
        print(f"    chunk={chunk.text}")
        if chunk.answer:
            print(f"    answer={chunk.answer}")
        print()


if __name__ == "__main__":
    main()
