"""Streamlit UI for the hybrid retriever."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_time.retriever import HybridRetriever, SearchFilters


DEFAULT_JSONL_PATH = os.getenv("RAG_JSONL_PATH", "data/aa_embeddings_hg.jsonl")
DEFAULT_MODEL = os.getenv("RAG_QUERY_MODEL", "intfloat/multilingual-e5-small")
DEFAULT_DEVICE = os.getenv("RAG_DEVICE", "cpu")


def collect_filter_options(retriever: HybridRetriever) -> dict[str, list[str]]:
    """Build distinct metadata values from indexed chunks."""
    return {
        "ticket_type": sorted({chunk.ticket_type for chunk in retriever.chunks if chunk.ticket_type}),
        "queue": sorted({chunk.queue for chunk in retriever.chunks if chunk.queue}),
        "priority": sorted({chunk.priority for chunk in retriever.chunks if chunk.priority}),
        "language": sorted({chunk.language for chunk in retriever.chunks if chunk.language}),
    }


@st.cache_resource(show_spinner=False)
def load_retriever(
    jsonl_path: str,
    embedding_model: str,
    device: str,
    vector_weight: float,
) -> HybridRetriever:
    return HybridRetriever(
        jsonl_path=jsonl_path,
        embedding_model=embedding_model,
        device=device,
        vector_weight=vector_weight,
    )


def build_optional_filter(
    label: str,
    options: list[str],
    key_prefix: str,
) -> str | None:
    enabled = st.checkbox(f"Activer {label}", value=False, key=f"{key_prefix}_enabled")
    if not enabled or not options:
        return None
    return st.selectbox(label, options=options, key=f"{key_prefix}_value")


def main() -> None:
    st.set_page_config(
        page_title="RAG Time Search",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("RAG Time")
    st.caption("Recherche hybride BM25 + vectorielle avec pré-filtres sur les métadonnées.")

    with st.sidebar:
        st.subheader("Configuration")
        jsonl_path = st.text_input("Fichier JSONL", value=DEFAULT_JSONL_PATH)
        model_name = st.text_input("Modèle de requête", value=DEFAULT_MODEL)
        device_options = ["cpu", "cuda"]
        default_device_index = device_options.index(DEFAULT_DEVICE) if DEFAULT_DEVICE in device_options else 0
        device = st.selectbox("Device", options=device_options, index=default_device_index)
        vector_weight = st.slider(
            "Poids vectoriel",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        top_k = st.number_input("Top K", min_value=1, max_value=50, value=5, step=1)

    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        st.error(f"Fichier introuvable: {jsonl_file}")
        st.stop()

    try:
        retriever = load_retriever(jsonl_path, model_name, device, vector_weight)
    except Exception as exc:
        st.error(f"Impossible de charger le retriever: {exc}")
        st.stop()

    filter_options = collect_filter_options(retriever)

    with st.sidebar:
        st.subheader("Pré-filtres")
        ticket_type = build_optional_filter("Type", filter_options["ticket_type"], "ticket_type")
        queue = build_optional_filter("Queue", filter_options["queue"], "queue")
        priority = build_optional_filter("Priorité", filter_options["priority"], "priority")
        language = build_optional_filter("Langue", filter_options["language"], "language")

    filters = SearchFilters(
        ticket_type=ticket_type,
        queue=queue,
        priority=priority,
        language=language,
    )

    query = st.text_input(
        "Requête",
        value="incident cloud",
        placeholder="Exemple: incident de sécurité cloud",
    )

    active_filters = {
        "type": ticket_type,
        "queue": queue,
        "priority": priority,
        "language": language,
    }
    active_filters = {name: value for name, value in active_filters.items() if value}

    meta_col, action_col = st.columns([3, 1])
    with meta_col:
        if active_filters:
            st.write("Filtres actifs :", active_filters)
        else:
            st.write("Filtres actifs : aucun")
    with action_col:
        run_search = st.button("Rechercher", use_container_width=True, type="primary")

    if not run_search:
        st.info("Choisis une requête, active les filtres voulus, puis lance la recherche.")
        st.stop()

    try:
        results = retriever.search(query=query, top_k=int(top_k), filters=filters)
    except Exception as exc:
        st.error(f"Erreur pendant la recherche: {exc}")
        st.stop()

    if not results:
        st.warning("Aucun résultat pour cette requête avec les filtres actuels.")
        st.stop()

    st.success(f"{len(results)} résultat(s) trouvé(s).")

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk
        title = f"#{rank} · score={result.score:.4f} · {chunk.subject or 'Sans sujet'}"
        with st.expander(title, expanded=(rank == 1)):
            left_col, right_col = st.columns([2, 3])
            with left_col:
                st.metric("Score hybride", f"{result.score:.4f}")
                st.metric("BM25", f"{result.bm25_score:.4f}")
                st.metric("Vectoriel", f"{result.vector_score:.4f}")
                st.write(
                    {
                        "type": chunk.ticket_type,
                        "queue": chunk.queue,
                        "priority": chunk.priority,
                        "language": chunk.language,
                        "line": chunk.source_line,
                        "chunk": chunk.chunk_index,
                    }
                )
            with right_col:
                if chunk.subject:
                    st.markdown(f"**Sujet**\n\n{chunk.subject}")
                st.markdown(f"**Chunk**\n\n{chunk.text}")
                if chunk.answer:
                    st.markdown(f"**Réponse**\n\n{chunk.answer}")


if __name__ == "__main__":
    main()
