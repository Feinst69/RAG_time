import argparse
from rag_time.embeddings import EmbeddingsGenerator, TextSplitter, SparseEmbeddingsGenerator
from rag_time.data_loader import DataLoader, Ticket
from rag_time.config.settings import settings
from rag_time.decorators import chrono

@chrono
def handle_ticket(line, splitter, generator, sparse_generator):
    ticket = Ticket(
                subject=line.subject,
                body=line.body,
                answer=line.answer,
                type=line.type,
                queue=line.queue,
                priority=line.priority,
                language=line.language,
                chunks=splitter.split_text(f"{line.subject}\n{line.body}" if line.subject else line.body)
            )
    ticket.embeddings = generator.generate_embeddings(ticket.chunks)
    ticket.sparse_embeddings = sparse_generator.generate_vector(ticket.chunks)
    return ticket

def build_rag_data(input_path: str, output_path: str):
    print(f"Loading dataset from {input_path}...")
    df = DataLoader(input_path).load()
    df_len = len(df)
    print(f"Loading Embeddings model...")
    generator  = EmbeddingsGenerator(settings.embedding_model, max_length=settings.embedding_dimension)
    sparse_generator = SparseEmbeddingsGenerator(settings.sparse_model)
    print("Loading Text Splitter...")
    splitter = TextSplitter(chunk_size=settings.chunk_size)
    with open(output_path, "w") as f:
        current_index = 0
        elapsed = 0.0
        print(f"Processing {df_len} tickets")
        for line in df.itertuples():
            current_index += 1
            try:
                ticket, line_elapsed = handle_ticket(line, splitter, generator, sparse_generator)
                ticket.time = line_elapsed
                elapsed += line_elapsed
                print(f"Processed {current_index}/{df_len} tickets...",end="")
                print(f"in {line_elapsed:.2f} s (total: {elapsed:.2f} s)",end="\r")
                f.write(ticket.model_dump_json() + "\n")
            except Exception as e:
                print(f"Error processing ticket {current_index}: {e}")
                exit(1)
    print(f"\nFinished processing {df_len} tickets in {elapsed:.2f} seconds.")

def main():
    parser = argparse.ArgumentParser(description="Build RAG data from dataset")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to the input CSV file")
    parser.add_argument("-o", "--output", required=True, type=str, help="Path to the output JSONL file")
    args = parser.parse_args()
    build_rag_data(args.input, args.output)

if __name__ == "__main__":
    main()