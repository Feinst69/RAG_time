import argparse
from time import time

from langchain_huggingface import HuggingFaceEmbeddings

from rag_time.data_loader import DataLoader, Ticket
from rag_time.embeddings import TextSplitter


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def chrono(func):
    def wrapper(*args, **kwargs):
        started_at = time()
        result = func(*args, **kwargs)
        return result, time() - started_at

    return wrapper


class HuggingFaceDenseEmbeddings:
    """Dense embeddings generator for small Hugging Face retrieval models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize_embeddings},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        passages = [f"{self.passage_prefix}{text}" for text in texts]
        return self.model.embed_documents(passages)

    def embed_query(self, query: str) -> list[float]:
        return self.model.embed_query(f"{self.query_prefix}{query}")


@chrono
def handle_ticket(line, splitter: TextSplitter, generator: HuggingFaceDenseEmbeddings):
    text = f"{line.subject}\n{line.body}" if line.subject else line.body
    ticket = Ticket(
        subject=line.subject,
        body=line.body,
        answer=line.answer,
        type=line.type,
        queue=line.queue,
        priority=line.priority,
        language=line.language,
        chunks=splitter.split_text(text),
    )
    ticket.embeddings = generator.embed_documents(ticket.chunks)
    return ticket


def build_rag_data(
    input_path: str,
    output_path: str,
    model_name: str = DEFAULT_MODEL,
    chunk_size: int = 512,
    limit: int | None = None,
    device: str = "cpu",
):
    print(f"Loading dataset from {input_path}...")
    df = DataLoader(input_path).load(limit=limit)
    df_len = len(df)

    print(f"Loading Hugging Face embedding model: {model_name}...")
    generator = HuggingFaceDenseEmbeddings(model_name=model_name, device=device)

    print(f"Loading Text Splitter with chunk_size={chunk_size}...")
    splitter = TextSplitter(chunk_size=chunk_size)

    with open(output_path, "w", encoding="utf-8") as output_file:
        processed = 0
        total_elapsed = 0.0
        print(f"Processing {df_len} tickets")

        for line in df.itertuples():
            processed += 1
            try:
                ticket, line_elapsed = handle_ticket(line, splitter, generator)
                ticket.time = line_elapsed
                total_elapsed += line_elapsed
                print(
                    f"Processed {processed}/{df_len} tickets in "
                    f"{line_elapsed:.2f} s (total: {total_elapsed:.2f} s)",
                    end="\r",
                )
                output_file.write(ticket.model_dump_json() + "\n")
            except Exception as exc:
                print(f"\nError processing ticket {processed}: {exc}")
                raise

    print(f"\nFinished processing {df_len} tickets in {total_elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(
        description="Build JSONL ticket embeddings with a small Hugging Face model"
    )
    parser.add_argument(
        "-i", "--input", required=True, type=str, help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=str, help="Path to the output JSONL file"
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face embedding model name",
    )
    parser.add_argument(
        "--chunk-size",
        default=512,
        type=int,
        help="Chunk size used to split ticket texts before embedding",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Optional limit on number of rows to process",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device used to run the embedding model",
    )
    args = parser.parse_args()

    build_rag_data(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        chunk_size=args.chunk_size,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
